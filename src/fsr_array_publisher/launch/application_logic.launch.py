from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # AI识别节点，并指定其Python虚拟环境
    venv_python_path = '/home/zyc/Desktop/hand_ws/.venv/bin/python'
    # 在 digit_recognizer_node 的定义中
    digit_recognizer_node = Node(
        package='fsr_array_publisher',
        executable='digit_recognizer',
        name='digit_recognizer_node',
        output='screen',
        prefix=f'{venv_python_path} ',
        # +++ 添加QoS参数 +++
        parameters=[{'qos_overrides': {
            '/fsr_array/raw_pressure_image': {
                'reliability': 'best_effort',
                'depth': 1
            }
        }}]
    )

    # 手势控制器节点
    gesture_controller_node = Node(
        package='fsr_array_publisher',
        executable='gesture_controller',
        name='gesture_controller_node',
        output='screen'
    )

    return LaunchDescription([
        digit_recognizer_node,
        gesture_controller_node,
    ])
