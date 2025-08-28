# LinkerHand ROS2 Project

The LinkerHand ROS2 SDK supports Ubuntu 22.04+ with ROS2 Humble/Jazzy and Python 3.10+.

**Current Environment:** Ubuntu 24.04 LTS + ROS2 Jazzy


## Before you start, please create the uv environment and install all the independencies.

### First choice, I recommend you to create the uv environment
```bash
########### Step 1: Install uv (if not already installed)   ##############

curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv

########### Step 2: Create and setup virtual environment   ###############

# Create uv virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install all dependencies
uv pip install -r requirements.txt

########## Step 3: Source ROS2 environment #############################
# For ROS2 Jazzy (Ubuntu 24.04)
source /opt/ros/jazzy/setup.bash

# For ROS2 Humble (Ubuntu 22.04)
source /opt/ros/humble/setup.bash
```

### Second choice, you can have a try to use conda environment
```bash
 创建conda环境
conda create -n hand_ws python=3.10 -y

# 激活环境
conda activate hand_ws

# 安装依赖
pip install -r requirements.txt

# 激活ROS2环境
source /opt/ros/jazzy/setup.bash  # 或者 humble
```



## After prepare the environment, the following is writed by myself  !!!!!!!!!!!!!!!!!!!!!!
```bash
# First you need source the enervirment 

$ source .venv/bin/activate
$ source install/setup.bash

# To view all the dependencies in the environment 
$ uv pip list 


# Initialize the linker_hand, you can refer to the github Website: https://github.com/linkerbotai/linker_hand_ros2_sdk .


$ sudo /usr/sbin/ip link set can0 up type can bitrate 1000000 #USB转CAN设备蓝色灯常亮状态
$ ros2 launch linker_hand.launch.py（启动灵巧手）
$ ros2 launch gui_control gui_control.launch.py  （用来启动UI页面，可以滑动来控制灵巧手的自由度）

# Pressure_senser Initialize + collect data + recognize digit
$ colcon build --packages-select fsr_array_publisher
```

Ps: 这边有一个bug，每次colon build之后，你需要把install/fsr_array_publisher/lib/fsr_array_publisher/digit_recognizer.py文件中开头
#!/usr/bin/python3
替换成 
#!/home/zyc/Desktop/hand_ws/.venv/bin/python3
不然启动digit_recognizer节点会报错，当时这个问题没办法解决，因为用的uv的虚拟环境。

### The solution to solve it

```bash
方法1：手动修改
# 每次 colcon build 后
nano install/fsr_array_publisher/lib/fsr_array_publisher/digit_recognizer
# 将第一行 #!/usr/bin/python3 
# 改为 #!/home/zyc/Desktop/hand_ws/.venv/bin/python3

方法2：使用自动脚本

#我在写README.md的时候，ai帮我写了一个fix_shebang_auto.sh的脚本来自动在colon build之后修复这个问题，不知道有没有用
chmod +x fix_shebang_auto.sh (可能只要配置一次就好了)

# 构建项目
colcon build --packages-select fsr_array_publisher

# 自动修复shebang
./fix_shebang_auto.sh

# 检查shebang行
head -n 1 install/fsr_array_publisher/lib/fsr_array_publisher/digit_recognizer

# 应该显示：#!/home/zyc/Desktop/hand_ws/.venv/bin/python3 (或您的虚拟环境路径)
```



### train and run the project

```bash
对5*5压力传感器
$ ros2 run fsr_array_publisher fsr_publisher （启动）
$ ros2 run fsr_array_publisher data_collector --ros-args -p start_threshold:=15.0   （数据收集）
$ ros2 run fsr_array_publisher digit_recognizer --ros-args -p start_threshold:=20.0   （数字识别检测）


# The script to train data

$ python3 train_model_pytorch.py
$ mv ~/Desktop/hand_ws/fsr_digit_classifier.pth ~/Desktop/hand_ws/src/fsr_array_publisher/resource/

# How to run it.

同时启动4个节点
启动灵巧手驱动 (linker_hand.launch.py)。
启动我们的FSR传感器驱动 (fsr_publisher)。
启动AI识别节点 (digit_recognizer)。
启动手势控制器 (gesture_controller)。

$ ros2 launch linker_hand_ros2_sdk linker_hand.launch.py
$ ros2 run fsr_array_publisher fsr_publisher
$ ros2 run fsr_array_publisher digit_recognizer
$ ros2 run fsr_array_publisher gesture_controller

=》 ros2 launch fsr_array_publisher hardware_drivers.launch.py （驱动）
=》 ros2 launch fsr_array_publisher application_logic.launch.py （操作）

```



## 🗂️ Project Structure

```
hand_ws/
├── src/                    # ROS2 packages source code
├── dataset/               # Training data (not pushed to git)
├── requirements.txt       # Python dependencies  
├── setup.sh              # Automated setup script
├── fix_shebang.sh        # Fix virtual environment issues
├── train_model_pytorch.py # Neural network training script
└── README.md             # This file
```