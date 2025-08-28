import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState 

class GestureControllerNode(Node):
    def __init__(self):
        super().__init__('gesture_controller_node')

        # --- 1. 定义手势数据 ---
        
        # 假设的关节名列表。如果控制失败，我们首先要回来修改这里。
        # 顺序必须和position数组对应。
        self.joint_names = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 
            'joint_6', 'joint_7', 'joint_8', 'joint_9', 'joint_10'
        ]

        # 使用你提供的精确数据填充手势数据库
        self.gesture_database = {
            # 数字: [10个关节的position值(0-255)]
            1: [5.0, 241.0, 254.0, 2.0, 2.0, 0.0, 254.0, 252.0, 254.0, 29.0],
            2: [5.0, 241.0, 254.0, 254.0, 2.0, 0.0, 254.0, 252.0, 254.0, 29.0],
            3: [5.0, 241.0, 254.0, 254.0, 254.0, 0.0, 254.0, 252.0, 254.0, 29.0],
            4: [5.0, 241.0, 254.0, 254.0, 254.0, 254.0, 254.0, 252.0, 254.0, 29.0],

    
             # +++ 新增的手势 +++
            5: [205.0, 251.0, 254.0, 254.0, 254.0, 254.0, 254.0, 252.0, 254.0, 36.0],
            6: [205.0, 251.0, 0.0, 1.0, 2.0, 254.0, 254.0, 252.0, 254.0, 36.0],
            7: [254.0, 254.0, 254.0, 0.0, 2.0, 217.0, 254.0, 252.0, 253.0, 53.0],
            8: [254.0, 254.0, 254.0, 0.0, 1.0, 0.0, 254.0, 252.0, 135.0, 51.0],
            9: [7.0, 254.0, 137.0, 0.0, 1.0, 0.0, 174.0, 252.0, 135.0, 33.0],
        }
        self.get_logger().info(f"已加载 {len(self.gesture_database)} 个手势。")

        # --- 2. 创建订阅者和发布者 ---
        self.subscription = self.create_subscription(
            Int32,
            '/fsr_array/recognized_digit',
            self.digit_callback,
            10)
        
        # 确认使用右手、0-255范围值的话题
        self.hand_command_publisher = self.create_publisher(
            JointState,
            '/cb_right_hand_control_cmd',
            10)

        self.get_logger().info("手势控制器节点已启动，等待识别结果...")

    def digit_callback(self, msg):
        recognized_digit = msg.data
        self.get_logger().info(f"接收到识别结果: {recognized_digit}")

        if recognized_digit in self.gesture_database:
            joint_positions = self.gesture_database[recognized_digit]
            
            # --- 3. 创建并发布指令消息 ---
            command_msg = JointState()
            command_msg.header.stamp = self.get_clock().now().to_msg()
            # 即使状态话题的name是空的，我们发送指令时通常也必须提供
            command_msg.name = self.joint_names
            command_msg.position = [float(p) for p in joint_positions]
            
            self.hand_command_publisher.publish(command_msg)
            self.get_logger().info(f"已发布手势 '{recognized_digit}' 的控制指令。")
        else:
            self.get_logger().warn(f"在手势数据库中未找到数字 '{recognized_digit}' 对应的手势。")

def main(args=None):
    rclpy.init(args=args)
    node = GestureControllerNode()
    rclpy.spin(node)
    # 在Ctrl+C后，spin会退出，下面的代码会执行
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
