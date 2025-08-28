import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np
import serial
import threading
import time
import cv2
import os
from ament_index_python.packages import get_package_share_directory

class FingerSensorNode(Node):
    def __init__(self):
        super().__init__('finger_sensor_node')

        # (参数定义保持不变)
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baseline_adc_value', 1320)
        self.declare_parameter('max_pressure_adc_value', 1800)
        self.declare_parameter('deadzone_offset', 20)

        self.serial_port_name = self.get_parameter('serial_port').get_parameter_value().string_value
        self.baseline_adc = self.get_parameter('baseline_adc_value').get_parameter_value().integer_value
        self.max_adc = self.get_parameter('max_pressure_adc_value').get_parameter_value().integer_value
        self.deadzone_offset = self.get_parameter('deadzone_offset').get_parameter_value().integer_value
        
        self.marker_publisher_ = self.create_publisher(MarkerArray, '/finger_sensor/visualization_markers', 10)
        
        self.get_logger().info("--- Finger Sensor (Hand Shape) Node Started ---")

        self.num_channels = 25
        self.pressure_values = np.zeros(self.num_channels, dtype=np.uint16)
        self.received_frames = set()

        # ######################################################################
        # ############## 关键修改 1: 定义新的、准确的布局和映射 #################
        # ######################################################################

        # --- 1. 定义25个点在RViz中的显示位置 (4-4-4-4-4-5 布局) ---
        # 你需要根据你的背景图，仔细微调这些 (x, y) 坐标
        self.sensor_positions = [
            # 小指 (4个点)
            (0.24, 0.35), (0.22, 0.25), (0.20, 0.15), (0.18, 0.05),
            # 无名指 (4个点)
            (0.13, 0.50), (0.12, 0.40), (0.11, 0.30), (0.10, 0.20),
            # 中指 (4个点)
            (-0.02, 0.52), (-0.02, 0.42), (-0.02, 0.32), (-0.02, 0.22),
            # 食指 (4个点)
            (-0.18, 0.50), (-0.17, 0.40), (-0.16, 0.30), (-0.15, 0.20),
            # 拇指 (4个点)
            (-0.36, 0.28), (-0.30, 0.23), (-0.24, 0.12), (-0.17, 0.01),
            # 手掌 (5个点)
            (-0.10, 0.05), (-0.07,-0.03), (0.13, -0.02), # 上
            (0.01, -0.1), (0.085, -0.09)  # 下
        ]

        # --- 2. 定义通道映射表 ---
        # 这是一个 crucial list。列表的第 i 个元素，存储的是“显示在第 i 个位置的传感器”
        # 对应的“通信协议中的通道号 (从1开始)”。
        # !!! 你必须通过实验，用你自己的结果来填充这个列表 !!!
        # 下面这个只是一个【完全虚构的例子】
        self.channel_map = [
            # 小指 (4个点) - 对应显示位置 0, 1, 2, 3
            5, 10, 15, 20,
            # 无名指 (4个点) - 对应显示位置 4, 5, 6, 7
            4, 9, 14, 19,
            # 中指 (4个点) - 对应显示位置 8, 9, 10, 11
            3, 8, 13, 18,
            # 食指 (4个点) - 对应显示位置 12, 13, 14, 15
            2, 7, 12, 17,
            # 拇指 (4个点) - 对应显示位置 16, 17, 18, 19
            1, 6, 11, 16,
            # 手掌 (5个点) - 对应显示位置 20, 21, 22, 23, 24
            22, 21, 25, 23, 24
        ]

        # --- 串口和状态机 (保持不变) ---
        self.serial_port = None
        self.state = "WAITING_FOR_HANDSHAKE"
        # ... (后面的串口初始化和线程启动代码保持不变)
        try:
            self.serial_port = serial.Serial(
                port=self.serial_port_name,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=2.0
            )
            self.get_logger().info(f"Successfully opened serial port: {self.serial_port_name}")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to open serial port {self.serial_port_name}: {e}")
            rclpy.shutdown()
            return

        self.reader_thread = threading.Thread(target=self.serial_reader_loop)
        self.reader_thread.daemon = True
        self.reader_thread.start()
        
        self.create_timer(1.0, self.publish_background_marker)

    # (handle_handshake, handle_data_frames, get_color_from_value, publish_background_marker 保持不变)
    def serial_reader_loop(self):
        while rclpy.ok() and self.serial_port.is_open:
            try:
                if self.state == "WAITING_FOR_HANDSHAKE":
                    self.handle_handshake()
                elif self.state == "READING_DATA":
                    self.handle_data_frames()
            except Exception as e:
                self.get_logger().error(f"Error in serial reader loop: {e}")
                time.sleep(1)

    def handle_handshake(self):
        if self.serial_port.read(1) == b'\x25':
            init_frame = self.serial_port.read(18)
            if len(init_frame) == 18 and init_frame[-1] == 0xFF:
                self.get_logger().info("Handshake frame received from sensor.")
                response_frame = bytes([0xAA, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF])
                self.serial_port.write(response_frame)
                self.state = "READING_DATA"
                self.get_logger().info("Handshake response sent. Switched to data reading mode.")

    def handle_data_frames(self):
        while rclpy.ok() and self.serial_port.is_open:
            header = self.serial_port.read(1)
            if header == b'\x24':
                break
        if not rclpy.ok() or not self.serial_port.is_open: return
        data_frame = self.serial_port.read(18)
        if len(data_frame) == 18 and data_frame[-1] == 0xFF:
            sequence_num = data_frame[0]
            start_channel_offset = -1
            if sequence_num == 0x00: start_channel_offset = 0
            elif sequence_num == 0x08: start_channel_offset = 8
            elif sequence_num == 0x10: start_channel_offset = 16
            elif sequence_num == 0x18: start_channel_offset = 24
            else:
                self.get_logger().warn(f"Unknown sequence number: {hex(sequence_num)}")
                return
            if sequence_num == 0x00:
                self.received_frames.clear()
                self.pressure_values.fill(0)
            for i in range(8):
                channel_index = start_channel_offset + i
                if channel_index < self.num_channels:
                    high_byte = data_frame[1 + i*2]
                    low_byte = data_frame[2 + i*2]
                    adc_value = (high_byte << 8) | low_byte
                    self.pressure_values[channel_index] = adc_value
            self.received_frames.add(sequence_num)
            if len(self.received_frames) == 4:
                self.publish_pressure_markers()
                self.received_frames.clear()
        else:
            self.get_logger().warn("Received incomplete or invalid data frame.")

    def get_color_from_value(self, value):
        val_8bit = int(np.clip(value, 0, 1) * 255)
        bgr_color = cv2.applyColorMap(np.array([[val_8bit]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
        return ColorRGBA(r=float(bgr_color[2])/255.0, g=float(bgr_color[1])/255.0, b=float(bgr_color[0])/255.0, a=1.0)

    def publish_background_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "background"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        width = 1.0
        height = 1.4
        marker.pose.position.z = -0.01
        p1 = [-width/2, -height/2, 0.0]
        p2 = [width/2, -height/2, 0.0]
        p3 = [width/2, height/2, 0.0]
        p4 = [-width/2, height/2, 0.0]
        marker.points.append(Point(x=p1[0], y=p1[1], z=p1[2]))
        marker.points.append(Point(x=p2[0], y=p2[1], z=p2[2]))
        marker.points.append(Point(x=p3[0], y=p3[1], z=p3[2]))
        marker.points.append(Point(x=p1[0], y=p1[1], z=p1[2]))
        marker.points.append(Point(x=p3[0], y=p3[1], z=p3[2]))
        marker.points.append(Point(x=p4[0], y=p4[1], z=p4[2]))
        pkg_share = get_package_share_directory('finger_sensor_publisher')
        marker.mesh_resource = "file://" + os.path.join(pkg_share, 'media', 'hand_background.png')
        marker.mesh_use_embedded_materials = True
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_publisher_.publish(marker_array)
        self.get_logger().info("Published background marker.")

    # ######################################################################
    # ############## 关键修改 2: 使用映射表来获取正确的数据 ##################
    # ######################################################################
    def publish_pressure_markers(self):
        marker_array = MarkerArray()
        
        # --- 数据处理 (不变) ---
        raw_values = self.pressure_values.copy()
        relative_pressure = raw_values.astype(np.float32) - self.baseline_adc
        relative_pressure[relative_pressure < self.deadzone_offset] = 0
        scale = self.max_adc - self.baseline_adc
        if scale > 0:
            normalized_values = relative_pressure / scale
        else:
            normalized_values = np.zeros_like(relative_pressure)
        normalized_values = np.clip(normalized_values, 0, 1)

        # --- 为每个通道创建 Marker ---
        # 我们现在循环 25 次，i 代表的是“显示位置”的索引 (0-24)
        for i in range(self.num_channels):
            # --- 使用映射表获取正确的数据 ---
            # 1. 找到这个显示位置对应的通信协议通道号
            protocol_channel = self.channel_map[i]
            # 2. 通道号从1开始，但数组索引从0开始，所以要减1
            data_index = protocol_channel - 1
            
            # 3. 从原始数据和处理后数据中，按正确的索引取出值
            current_normalized_value = normalized_values[data_index]
            current_relative_pressure = relative_pressure[data_index]

            # --- 创建代表压力点的圆柱 Marker ---
            point_marker = Marker()
            point_marker.header.frame_id = "map"
            point_marker.header.stamp = self.get_clock().now().to_msg()
            point_marker.ns = "pressure_points"
            point_marker.id = i
            point_marker.type = Marker.CYLINDER
            point_marker.action = Marker.ADD
            
            # 设置位置 (使用第 i 个显示位置的坐标)
            pos = self.sensor_positions[i]
            point_marker.pose.position.x = pos[0]
            point_marker.pose.position.y = pos[1]
            point_marker.pose.position.z = 0.0
            
            point_marker.scale.x = 0.05
            point_marker.scale.y = 0.05
            point_marker.scale.z = 0.01
            
            # 设置颜色 (使用正确索引的值)
            if current_normalized_value > 0:
                point_marker.color = self.get_color_from_value(current_normalized_value)
            else:
                point_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.5)

            marker_array.markers.append(point_marker)

            # --- 创建显示数值的文字 Marker ---
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "pressure_values"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = pos[0] + 0.05
            text_marker.pose.position.y = pos[1]
            text_marker.pose.position.z = 0.0
            
            # 设置文字内容 (使用正确索引的值)
            text_marker.text = str(int(current_relative_pressure))
            text_marker.scale.z = 0.04
            text_marker.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)

            marker_array.markers.append(text_marker)

        self.marker_publisher_.publish(marker_array)

    def destroy_node(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.get_logger().info("Serial port closed.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FingerSensorNode()
    if rclpy.ok():
        rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
