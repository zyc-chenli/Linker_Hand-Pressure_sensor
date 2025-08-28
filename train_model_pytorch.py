import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt

# --- 1. 配置参数 ---
# 确认这个路径指向你采集的数据集
# 因为脚本和dataset文件夹都在工作区根目录，所以可以直接用相对路径
DATASET_PATH = 'dataset' 

# 图像和模型参数
IMG_HEIGHT = 5
IMG_WIDTH = 5
NUM_CLASSES = 10 # 数字 0-9

# 模型将保存在当前运行脚本的目录下
MODEL_SAVE_PATH = 'fsr_digit_classifier.pth' # PyTorch模型通常用 .pth 或 .pt 后缀

# --- 2. 创建自定义数据集类 ---
# 这是PyTorch加载数据的标准方式
class FSRDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 将Numpy数组转换为PyTorch张量
        # 增加一个通道维度: [H, W] -> [1, H, W]
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        label_tensor = torch.tensor(label).long()
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, label_tensor

def load_data(dataset_path):
    """从文件夹加载图像数据和标签"""
    images, labels = [], []
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return None, None
    class_names = sorted(os.listdir(dataset_path))
    print(f"正在从以下路径加载数据: {dataset_path}")
    print(f"发现以下类别 (文件夹): {class_names}")
    for class_name in class_names:
        if not class_name.isdigit(): continue
        label = int(class_name)
        class_path = os.path.join(dataset_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                # 归一化到 [0, 1]
                image = image.astype('float32') / 255.0
                images.append(image)
                labels.append(label)
    print(f"成功加载 {len(images)} 张图像。")
    return np.array(images), np.array(labels)

# --- 3. 定义PyTorch模型 ---
class SimpleCNN(nn.Module):
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
            # 经过conv(5x5), pool(3x3), conv(3x3) -> 32*3*3
            nn.Linear(32 * 3 * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

def main():
    # --- 数据加载 ---
    images, labels = load_data(DATASET_PATH)
    if images is None: return

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = FSRDataset(X_train, y_train)
    val_dataset = FSRDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # +++ 新增数据增强 +++
    train_transform = transforms.Compose([
        transforms.ToPILImage(), # 先转成PIL Image
        # 随机小角度旋转 (-10到10度)
        transforms.RandomRotation(10), 
        # 随机微小平移 (水平和垂直最多移动10%)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor() # 再转回Tensor
    ])
    # +++ 结束 +++

    # 创建数据集实例时，把 transform 传给训练集
    # 验证集不需要增强，我们要看模型在原始数据上的表现
    train_dataset = FSRDataset(X_train, y_train, transform=train_transform)
    val_dataset = FSRDataset(X_val, y_val) # 验证集不变


    # --- 模型、损失函数和优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = SimpleCNN().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 训练循环 ---
    epochs = 100
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print("\n--- 开始训练 ---")
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # --- 验证循环 ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)

        print(f'轮次 [{epoch+1}/{epochs}], 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%, '
              f'验证损失: {val_loss_avg:.4f}, 验证准确率: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 发现更优模型，已保存至 {MODEL_SAVE_PATH}")

    print("--- 训练结束 ---\n")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")

    # --- 可视化 ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot([acc/100 for acc in history['train_acc']], label='训练准确率')
    plt.plot([acc/100 for acc in history['val_acc']], label='验证准确率')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('准确率')
    plt.legend()
    plt.title('准确率变化曲线')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('损失')
    plt.legend()
    plt.title('损失变化曲线')
    
    plt.show()

if __name__ == '__main__':
    main()
