import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
import threading
import numpy as np
from rclpy.executors import SingleThreadedExecutor
from datetime import datetime # <-- 放在文件顶部

# 定义工作空间的绝对路径
WORKSPACE_PATH = '/home/zyc/Desktop/hand_ws'
# 数据集将被保存在工作空间下的 'dataset' 文件夹中
DATASET_PATH = os.path.join(WORKSPACE_PATH, 'dataset')

class DynamicDataCollectorNode(Node):
    def __init__(self):
        super().__init__('dynamic_data_collector_node')
        
        self.declare_parameter('start_threshold', 200.0)
        self.declare_parameter('stop_timeout_sec', 0.5)

        self.start_threshold = self.get_parameter('start_threshold').get_parameter_value().double_value
        self.stop_timeout_sec = self.get_parameter('stop_timeout_sec').get_parameter_value().double_value

        self.state = 'IDLE'
        self.pressure_frames = []
        self.last_pressure_time = self.get_clock().now()
        self.image_lock = threading.Lock()
        self.last_pressure_sum = 0.0

        self.subscription = self.create_subscription(
            Image,
            '/fsr_array/raw_pressure_image',
            self.image_callback,
            10)
        self.bridge = CvBridge()

        os.makedirs(DATASET_PATH, exist_ok=True)
        self.get_logger().info("--- Enhanced Dynamic FSR Data Collector (Timestamp Naming) ---")
        self.get_logger().info(f"Dataset will be saved to: {DATASET_PATH}")
        self.get_logger().info(f"Current Start Threshold: {self.start_threshold}")
        
        self.feedback_timer = self.create_timer(1.0, self.provide_feedback)

    def provide_feedback(self):
        with self.image_lock:
            self.get_logger().info(f"State: {self.state} | Current Pressure Sum: {self.last_pressure_sum:.2f}")

    def image_callback(self, msg):
        try:
            gray_image = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            current_pressure_sum = np.sum(gray_image)
            now = self.get_clock().now()

            with self.image_lock:
                self.last_pressure_sum = current_pressure_sum

                if self.state == 'IDLE':
                    if current_pressure_sum > self.start_threshold:
                        self.get_logger().warn("ACTION STARTED -> RECORDING.")
                        self.state = 'RECORDING'
                        self.pressure_frames.clear()
                        self.pressure_frames.append(gray_image)
                        self.last_pressure_time = now
                
                elif self.state == 'RECORDING':
                    if current_pressure_sum > 0:
                        self.pressure_frames.append(gray_image)
                        self.last_pressure_time = now
                    
                    if (now - self.last_pressure_time).nanoseconds / 1e9 > self.stop_timeout_sec:
                        self.get_logger().warn(f"ACTION FINISHED. {len(self.pressure_frames)} frames recorded. Ready to save.")
                        self.state = 'IDLE'

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def save_aggregated_image(self, label):
        with self.image_lock:
            if not self.pressure_frames:
                self.get_logger().error("No action recorded to save. Perform an action first.")
                return
            
            aggregated_image = np.max(np.array(self.pressure_frames), axis=0).astype(np.uint8)
            self.pressure_frames.clear()

        label_path = os.path.join(DATASET_PATH, str(label))
        os.makedirs(label_path, exist_ok=True)
        
        # +++ 使用ROS时钟和datetime生成唯一文件名 +++
        current_time = self.get_clock().now()
        seconds = current_time.nanoseconds // 10**9
        nanoseconds = current_time.nanoseconds % 10**9
        dt_object = datetime.fromtimestamp(seconds)
        timestamp_str = dt_object.strftime('%Y%m%d_%H%M%S') + f'_{nanoseconds // 10**6:03d}'
        filename = f"{timestamp_str}.png"
        filepath = os.path.join(label_path, filename)
        # +++ 修改结束 +++
        
        cv2.imwrite(filepath, aggregated_image)
        self.get_logger().info(f"Saved aggregated image for label '{label}' to {filepath}")

# main 函数保持不变
def main(args=None):
    rclpy.init(args=args)
    collector_node = DynamicDataCollectorNode()
    executor = SingleThreadedExecutor()
    executor.add_node(collector_node)
    executor_thread = threading.Thread(target=executor.spin)
    executor_thread.daemon = True
    executor_thread.start()
    try:
        print("\n--- INSTRUCTIONS ---")
        print("1. Watch the 'Current Pressure Sum' in the log.")
        print("2. Press on the sensor to see how high the value goes.")
        print("3. If needed, restart this node with a new threshold: ros2 run ... --ros-args -p start_threshold:=<new_value>")
        print("4. To save: perform an action, wait for 'ACTION FINISHED', then enter the label.")
        print("--------------------\n")
        while rclpy.ok():
            user_input = input("Enter label (0-9) to save last action, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
            elif user_input.isdigit() and 0 <= int(user_input) <= 9:
                collector_node.save_aggregated_image(int(user_input))
            else:
                print("Invalid input.")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("\nShutting down...")
        rclpy.shutdown()
        executor_thread.join()

if __name__ == '__main__':
    main()