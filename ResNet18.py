import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 单通道
    transforms.Resize((224, 224)),  # ResNet标准输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 自定义数据集类 - 只处理图像
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 获取图片文件夹的所有子文件夹
        image_sub_folders = [f for f in os.listdir(image_folder) 
                           if os.path.isdir(os.path.join(image_folder, f))]
        
        if len(image_sub_folders) != 2:
            raise ValueError("图片文件夹下必须有且只有两个子文件夹")
            
        # 为每个子文件夹分配标签
        self.class_to_idx = {image_sub_folders[0]: 0, image_sub_folders[1]: 1}
        self.idx_to_class = {0: image_sub_folders[0], 1: image_sub_folders[1]}
        
        # 收集所有图像路径和标签
        for class_name in image_sub_folders:
            class_path = os.path.join(image_folder, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('L')  # 灰度模式
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 早停类
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# 标准ResNet18模型 - 预训练权重 + 单通道输入
class ResNet18Model(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18Model, self).__init__()
        
        # 使用预训练的ResNet18
        self.resnet = models.resnet18(pretrained=True)
        # 修改第一层卷积以适应单通道输入
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 获取原始fc层的输入特征维度
        num_features = self.resnet.fc.in_features
        
        # 替换最后的全连接层
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# 训练函数 - 修改指标计算方式
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=7):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='best_model.pth')
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 计算训练指标 - 准确率使用微平均，其他使用宏平均
        train_acc = accuracy_score(all_labels, all_preds)  # 微平均
        train_accuracies.append(train_acc)
        train_precision = precision_score(all_labels, all_preds, average='macro')  # 宏精确度
        train_recall = recall_score(all_labels, all_preds, average='macro')  # 宏召回率
        train_f1 = f1_score(all_labels, all_preds, average='macro')  # 宏F1
        train_f1_scores.append(train_f1)
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_all_preds = []
        val_all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        
        # 计算验证指标 - 准确率使用微平均，其他使用宏平均
        val_acc = accuracy_score(val_all_labels, val_all_preds)  # 微平均
        val_accuracies.append(val_acc)
        val_precision = precision_score(val_all_labels, val_all_preds, average='macro')  # 宏精确度
        val_recall = recall_score(val_all_labels, val_all_preds, average='macro')  # 宏召回率
        val_f1 = f1_score(val_all_labels, val_all_preds, average='macro')  # 宏F1
        val_f1_scores.append(val_f1)
        
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')
        print()
        
        # 早停检查
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores

# 测试函数 - 修改指标计算方式
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算测试指标 - 准确率使用微平均，其他使用宏平均
    test_acc = accuracy_score(all_labels, all_preds)  # 微平均
    test_precision = precision_score(all_labels, all_preds, average='macro')  # 宏精确度
    test_recall = recall_score(all_labels, all_preds, average='macro')  # 宏召回率
    test_f1 = f1_score(all_labels, all_preds, average='macro')  # 宏F1
    
    return {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1
    }, all_preds, all_labels

# 绘制训练过程图表
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores):
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 损失图
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率图
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1分数图
    ax3.plot(epochs, train_f1_scores, 'b-', label='Training F1 Score', linewidth=2)
    ax3.plot(epochs, val_f1_scores, 'r-', label='Validation F1 Score', linewidth=2)
    ax3.set_title('Training and Validation F1 Score')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies,
        'train_f1': train_f1_scores,
        'val_f1': val_f1_scores
    })
    history_df.to_csv('training_history.csv', index=False)
    print("Training history saved to 'training_history.csv'")

# 主函数
def main():
    # 参数设置
    image_dir = '/data/coding/DataSet-pheme/phemetxt-image'  # 图片文件夹
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001
    patience = 7
    
    # 创建数据集
    full_dataset = ImageDataset(image_dir, transform=transform)
    
    # 划分数据集
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset size: {len(full_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # 初始化标准ResNet18模型
    model = ResNet18Model(num_classes=2).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("Starting training...")
    model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, patience
    )
    
    # 测试模型
    print("Testing model...")
    test_metrics, test_preds, test_labels = test_model(model, test_loader)
    
    print(f"\nTest Results:")
    print(f"Accuracy (micro): {test_metrics['accuracy']:.4f}")
    print(f"Precision (macro): {test_metrics['precision']:.4f}")
    print(f"Recall (macro): {test_metrics['recall']:.4f}")
    print(f"F1-Score (macro): {test_metrics['f1']:.4f}")
    
    # 绘制训练过程图表
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores)
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'prediction': test_preds,
        'true_label': test_labels
    })
    results_df.to_csv('test_predictions.csv', index=False)
    print("Test predictions saved to 'test_predictions.csv'")

if __name__ == "__main__":
    main()