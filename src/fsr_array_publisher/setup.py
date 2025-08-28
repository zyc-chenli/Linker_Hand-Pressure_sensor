from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'fsr_array_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 安装 resource 目录下的所有文件 (包括我们的 .pth 模型)
        (os.path.join('share', package_name, 'resource'), glob('resource/*')),
        # +++ 添加这一行来安装launch文件 +++
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zyc',
    maintainer_email='zyc@todo.todo',
    description='A ROS 2 package for FSR array and digit recognition.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fsr_publisher = fsr_array_publisher.fsr_5x5_node:main',
            'data_collector = fsr_array_publisher.data_collector_node:main',
            # 添加新的识别节点入口点
            'digit_recognizer = fsr_array_publisher.digit_recognizer_node:main',
            'gesture_controller = fsr_array_publisher.gesture_controller_node:main',
        ],
    },
)