from setuptools import setup

package_name = 'finger_sensor_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='A ROS 2 publisher node for a finger-shaped FSR sensor array.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    # !!! 这是你需要添加的关键部分 !!!
    entry_points={
        'console_scripts': [
            'finger_sensor = finger_sensor_publisher.finger_sensor_node:main',
        ],
    },
)
