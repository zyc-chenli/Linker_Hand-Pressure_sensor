import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import torch
import torch.nn as nn
import threading
import queue # 导入队列模块

# --- PyTorch模型结构 (保持不变) ---
class SimpleCNN(nn.Module):
    # ... (这部分代码完全不变) ...
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 3 * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 10)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

# --- ROS 2 节点 ---
class DigitRecognizerNode(Node):
    def __init__(self, model_path):
        super().__init__('digit_recognizer_node')

        # --- 加载模型 (不变) ---
        if not os.path.exists(model_path):
            self.get_logger().error(f"模型文件未找到: {model_path}! 关闭节点。")
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.get_logger().info(f"成功从 {model_path} 加载模型到 {self.device}")

        # --- 参数和状态机 (不变) ---
        self.declare_parameter('start_threshold', 15.0)
        self.declare_parameter('stop_timeout_sec', 0.5)
        self.start_threshold = self.get_parameter('start_threshold').get_parameter_value().double_value
        self.stop_timeout_sec = self.get_parameter('stop_timeout_sec').get_parameter_value().double_value
        self.state = 'IDLE'
        self.pressure_frames = []
        self.last_pressure_time = self.get_clock().now()

        # --- ROS 接口 (不变) ---
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/fsr_array/raw_pressure_image', self.image_callback, 10)
        self.result_publisher = self.create_publisher(Int32, '/fsr_array/recognized_digit', 10)
        self.debug_image_publisher = self.create_publisher(Image, '/fsr_array/aggregated_image', 10)

        # +++ 新增：处理队列和工作线程 +++
        self.processing_queue = queue.Queue(maxsize=5) # 创建一个最大长度为5的队列
        self.worker_thread = threading.Thread(target=self.processing_worker)
        self.worker_thread.daemon = True # 设置为守护线程，主程序退出时它也退出
        self.worker_thread.start()

        self.get_logger().info("数字识别节点已启动 (带工作线程)。请在传感器上书写...")

    def image_callback(self, msg):
        # 这个回调函数现在非常轻量，只做状态判断和数据收集
        gray_image = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        current_pressure_sum = np.sum(gray_image)
        now = self.get_clock().now()

        if self.state == 'IDLE' and current_pressure_sum > self.start_threshold:
            self.get_logger().info("检测到动作 -> 开始记录...")
            self.state = 'RECORDING'
            self.pressure_frames.clear()
            self.pressure_frames.append(gray_image)
            self.last_pressure_time = now
        
        elif self.state == 'RECORDING':
            if current_pressure_sum > 0:
                self.pressure_frames.append(gray_image)
                self.last_pressure_time = now
            
            if (now - self.last_pressure_time).nanoseconds / 1e9 > self.stop_timeout_sec:
                self.get_logger().info(f"动作结束 -> 将 {len(self.pressure_frames)} 帧数据送入处理队列。")
                
                # *** 关键改动：不直接处理，而是把数据副本放入队列 ***
                if not self.processing_queue.full():
                    frames_copy = list(self.pressure_frames) # 创建一个副本
                    self.processing_queue.put(frames_copy)
                else:
                    self.get_logger().warn("处理队列已满，丢弃当前动作。")

                self.pressure_frames.clear()
                self.state = 'IDLE' # 立刻重置状态，准备接收下一个动作

    def processing_worker(self):
        """这个函数在独立的后台线程中运行"""
        while rclpy.ok():
            try:
                # 阻塞式地从队列中获取数据，带1秒超时
                frames_to_process = self.processing_queue.get(timeout=1.0)
                
                # 拿到数据后，调用原来的处理函数
                self.process_and_recognize(frames_to_process)

            except queue.Empty:
                # 队列为空是正常情况，继续等待
                pass
            except Exception as e:
                self.get_logger().error(f"处理线程出错: {e}")

    def process_and_recognize(self, frames):
        """这个函数现在由工作线程调用，不再由回调函数调用"""
        if not frames: return

        aggregated_image = np.max(np.array(frames), axis=0).astype(np.uint8)
        self.debug_image_publisher.publish(self.bridge.cv2_to_imgmsg(aggregated_image, 'mono8'))

        img_normalized = aggregated_image.astype('float32') / 255.0
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            recognized_digit = predicted_idx.item()

        result_msg = Int32()
        result_msg.data = int(recognized_digit)
        self.result_publisher.publish(result_msg)
        self.get_logger().info(f"======> 识别结果: {recognized_digit} (置信度: {confidence.item():.2f}) <======")

def main(args=None):
    rclpy.init(args=args)
    
    from ament_index_python.packages import get_package_share_directory
    import os

    package_share_dir = get_package_share_directory('fsr_array_publisher')
    model_filename = 'fsr_digit_classifier.pth'
    model_path = os.path.join(package_share_dir, 'resource', model_filename)

    try:
        recognizer_node = DigitRecognizerNode(model_path=model_path)
        # rclpy.spin() 会阻塞在这里，直到节点被关闭(例如Ctrl+C)
        # 它会自动处理所有事件循环
        rclpy.spin(recognizer_node)
    except FileNotFoundError as e:
        print(f"启动失败: {e}")
    except KeyboardInterrupt:
        # 用户按下了Ctrl+C，spin会退出，我们在这里什么都不用做
        # 只需要让程序自然结束即可
        pass
    
    # 当 spin 退出后，rclpy 会自动处理大部分清理工作
    # 我们不再需要手动的 destroy_node 和 shutdown
    print("\n节点已关闭。")

if __name__ == '__main__':
    main()
