import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
# from std_msgs.msg import String  <-- 不再需要，可以删除
import numpy as np
import serial
import threading
import time
import cv2

class Fsr5x5HardwareNode(Node):
    def __init__(self):
        super().__init__('fsr_5x5_hardware_node')

        # (参数定义部分保持不变)
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baseline_adc_value', 1320)
        self.declare_parameter('max_pressure_adc_value', 1800)
        self.declare_parameter('deadzone_offset', 20)

        self.serial_port_name = self.get_parameter('serial_port').get_parameter_value().string_value
        self.baseline_adc = self.get_parameter('baseline_adc_value').get_parameter_value().integer_value
        self.max_adc = self.get_parameter('max_pressure_adc_value').get_parameter_value().integer_value
        self.deadzone_offset = self.get_parameter('deadzone_offset').get_parameter_value().integer_value
        
        # 只保留热力图的发布者
        self.heatmap_publisher_ = self.create_publisher(Image, '/fsr_array/pressure_heatmap', 10)
        self.raw_image_publisher_ = self.create_publisher(Image, '/fsr_array/raw_pressure_image', 10)

        self.get_logger().info("--- Sensor Calibration Parameters ---")
        self.get_logger().info(f"Baseline (Zero Pressure) ADC: {self.baseline_adc}")
        self.get_logger().info(f"Max Pressure ADC: {self.max_adc}")
        self.get_logger().info(f"Deadzone Offset: {self.deadzone_offset}")
        self.get_logger().info("------------------------------------")

        # (后面的初始化代码保持不变)
        self.rows = 5
        self.cols = 5
        self.pressure_values = np.zeros(self.rows * self.cols, dtype=np.uint16)
        self.received_frames = set()
        self.serial_port = None
        self.state = "WAITING_FOR_HANDSHAKE"
        
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

    # (handle_handshake 和 handle_data_frames 函数保持不变)
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
                if channel_index < 25:
                    high_byte = data_frame[1 + i*2]
                    low_byte = data_frame[2 + i*2]
                    adc_value = (high_byte << 8) | low_byte
                    self.pressure_values[channel_index] = adc_value
            self.received_frames.add(sequence_num)
            if len(self.received_frames) == 4:
                self.publish_pressure_data()
                self.received_frames.clear()
        else:
            self.get_logger().warn("Received incomplete or invalid data frame.")

    # ######################################################################
    # ############## 关键修改: 直接打印，不再发布String消息 ##################
    # ######################################################################
    def publish_pressure_data(self):
        # 1. 数据处理 (不变)
        raw_values = self.pressure_values.copy()
        relative_pressure = raw_values.astype(np.float32) - self.baseline_adc
        relative_pressure[relative_pressure < self.deadzone_offset] = 0
        
        # 2. 归一化 (不变)
        scale = self.max_adc - self.baseline_adc
        if scale > 0:
            normalized_values = relative_pressure / scale
        else:
            normalized_values = np.zeros_like(relative_pressure)
        normalized_values = np.clip(normalized_values, 0, 1)
        
        # 3. 发布热力图 (Image) (不变)
        pressure_grid = normalized_values.reshape((self.rows, self.cols))
        heatmap_gray = (pressure_grid * 255).astype(np.uint8)

        # +++ 新增部分：发布原始灰度图 +++
        raw_image_msg = Image()
        raw_image_msg.header.stamp = self.get_clock().now().to_msg()
        raw_image_msg.header.frame_id = 'fsr_array_link'
        raw_image_msg.height = self.rows
        raw_image_msg.width = self.cols
        raw_image_msg.encoding = 'mono8' # 使用8位单通道编码
        raw_image_msg.is_bigendian = 0
        raw_image_msg.step = self.cols * 1 # 每个像素1个字节
        raw_image_msg.data = heatmap_gray.tobytes()
        self.raw_image_publisher_.publish(raw_image_msg)

        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        background = np.full(heatmap_color.shape, 255, dtype=np.uint8)
        mask = pressure_grid > 0
        background[mask] = heatmap_color[mask]
        final_rgb_image = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        
        image_msg = Image()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = 'fsr_array_link'
        image_msg.height = self.rows
        image_msg.width = self.cols
        image_msg.encoding = 'rgb8'
        image_msg.is_bigendian = 0
        image_msg.step = self.cols * 3
        image_msg.data = final_rgb_image.tobytes()
        self.heatmap_publisher_.publish(image_msg)

        # --- 4. (新) 直接在节点终端打印格式化数据 ---
        # 构建一个清爽的、多行的字符串用于打印
        log_string = "\n--- FSR Array Data ---\n" # 开头加一个换行，让格式更清晰
        log_string += "--- 原始数据 ---      --- 校准后数据 ---\n"
        
        raw_grid = raw_values.reshape((self.rows, self.cols))
        calibrated_grid = relative_pressure.astype(np.int32).reshape((self.rows, self.cols))

        for i in range(self.rows):
            # 使用 f-string 和格式化指定符来确保对齐
            raw_row_str = " ".join([f"{val:5d}" for val in raw_grid[i]])
            calibrated_row_str = " ".join([f"{val:5d}" for val in calibrated_grid[i]])
            log_string += f"{raw_row_str}   |   {calibrated_row_str}\n"
        
        # 使用 logger 打印整个多行字符串
        self.get_logger().info(log_string)

    def destroy_node(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.get_logger().info("Serial port closed.")
        super().destroy_node()

# (main 函数保持不变)
def main(args=None):
    rclpy.init(args=args)
    node = Fsr5x5HardwareNode()
    if rclpy.ok():
        rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
