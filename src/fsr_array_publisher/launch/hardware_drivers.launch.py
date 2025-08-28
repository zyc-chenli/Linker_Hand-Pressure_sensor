import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    
    # 包含外部的灵巧手驱动launch文件
    linker_hand_pkg_share = get_package_share_directory('linker_hand_ros2_sdk')
    linker_hand_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(linker_hand_pkg_share, 'launch', 'linker_hand.launch.py')
        )
    )

    # 在 fsr_publisher_node 的定义中
    fsr_publisher_node = Node(
        package='fsr_array_publisher',
        executable='fsr_publisher',
        name='fsr_publisher_node',
        output='screen',
        # +++ 添加QoS参数 +++
        # ROS 2中，可以通过参数来覆盖代码里的QoS设置
        # 我们使用 sensor_data QoS profile
        parameters=[{'qos_overrides': {
            '/fsr_array/raw_pressure_image': {
                'reliability': 'best_effort',
                'depth': 1
            }
        }}]
    )

    return LaunchDescription([
        linker_hand_launch,
        fsr_publisher_node,
    ])
