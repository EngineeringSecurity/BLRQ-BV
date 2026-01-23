# T_resnet.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random

# 导入向量合成函数
from Vector_composing import transform_vectors_numpy

# 从统一配置获取参数
from unified_config import get_config
config = get_config()

# 设置随机种子为42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号
sns.set_style("whitegrid")

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 修改为更适合灰度图的预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 保持单通道
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图归一化
])

# 自定义数据集类 - 支持原始类别结构
class MultiModalDataset(Dataset):
    def __init__(self, image_folder, npy_folder1, npy_folder2, transform=None):
        self.image_folder = image_folder
        self.npy_folder1 = npy_folder1
        self.npy_folder2 = npy_folder2
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 检查输入目录是否存在
        if not os.path.exists(image_folder):
            raise ValueError(f"图像文件夹不存在: {image_folder}")
        if not os.path.exists(npy_folder1):
            raise ValueError(f"特征文件夹1不存在: {npy_folder1}")
        if not os.path.exists(npy_folder2):
            raise ValueError(f"特征文件夹2不存在: {npy_folder2}")
        
        # 获取图片文件夹的所有子文件夹（应该是 non_rumor 和 rumor）
        image_sub_folders = [f for f in os.listdir(image_folder) 
                           if os.path.isdir(os.path.join(image_folder, f))]
        
        # 确保有 non_rumor 和 rumor 两个类别
        required_classes = ['non_rumor', 'rumor']
        for cls in required_classes:
            if cls not in image_sub_folders:
                raise ValueError(f"图片文件夹缺少类别: {cls}")
        
        # 为每个子文件夹分配标签
        self.class_to_idx = {'non_rumor': 0, 'rumor': 1}
        self.idx_to_class = {0: 'non_rumor', 1: 'rumor'}
        
        print(f"检测到类别: {self.class_to_idx}")
        
        # 收集所有图像路径和标签
        for class_name in required_classes:
            class_path = os.path.join(image_folder, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        base_name = os.path.splitext(img_name)[0]
                        
                        # 检查对应的.npy文件是否存在（特征文件夹也有相同的类别结构）
                        npy_path1 = os.path.join(npy_folder1, class_name, base_name + '.npy')
                        npy_path2 = os.path.join(npy_folder2, class_name, base_name + '.npy')
                        
                        # 检查文件是否存在
                        npy1_exists = os.path.exists(npy_path1)
                        npy2_exists = os.path.exists(npy_path2)
                        
                        if npy1_exists and npy2_exists:
                            self.image_paths.append({
                                'image': os.path.join(class_path, img_name),
                                'npy1': npy_path1,
                                'npy2': npy_path2,
                                'base_name': base_name,
                                'class': class_name
                            })
                            self.labels.append(self.class_to_idx[class_name])
                        else:
                            # 只输出前几个警告
                            if len(self.image_paths) < 10:
                                print(f"警告: 文件 {base_name} 的.npy文件不存在，跳过该样本。")
                                print(f"  npy1路径: {npy_path1} - {'存在' if npy1_exists else '不存在'}")
                                print(f"  npy2路径: {npy_path2} - {'存在' if npy2_exists else '不存在'}")
        
        print(f"总共找到 {len(self.image_paths)} 个有效样本")
        for cls_name, idx in self.class_to_idx.items():
            count = sum(1 for label in self.labels if label == idx)
            print(f"  {cls_name}: {count} 个样本")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]['image']
        npy_path1 = self.image_paths[idx]['npy1']
        npy_path2 = self.image_paths[idx]['npy2']
        label = self.labels[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('L')  # 以灰度模式读取
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"加载图像失败: {img_path}")
            print(f"错误: {e}")
            # 创建空的图像张量作为占位符
            image = torch.zeros(1, 256, 256)
            
        # 加载.npy文件
        try:
            npy_data1 = np.load(npy_path1)
            npy_data2 = np.load(npy_path2)
        except Exception as e:
            print(f"加载.npy文件失败: {npy_path1} 或 {npy_path2}")
            print(f"错误: {e}")
            # 创建空的numpy数组作为占位符
            npy_data1 = np.zeros(1024)  # BiLSTM特征维度
            npy_data2 = np.zeros(4)     # 量子特征维度
        
        return image, npy_data1, npy_data2, label
    
    def get_sample_info(self, idx):
        """获取样本信息"""
        return {
            'image_path': self.image_paths[idx]['image'],
            'base_name': self.image_paths[idx]['base_name'],
            'class': self.image_paths[idx]['class'],
            'class_idx': self.labels[idx]
        }

# 早停类 - 修改为基于最低验证损失
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
        score = -val_loss  # 负值因为我们要最小化损失
        
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

# 修改后的多模态ResNet18模型 - 从头训练 + 单通道输入
class MultiModalResNet18(nn.Module):
    def __init__(self, npy1_dim, npy2_dim, num_classes=2, feature_dim=512):
        super(MultiModalResNet18, self).__init__()
        
        # 修改：使用pretrained=False从头训练，并修改第一层为单通道输入
        self.resnet = models.resnet18(pretrained=False)  # 从头训练
        # 修改第一层卷积以适应单通道输入
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 移除最后的全连接层，保留特征提取部分
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 两个.npy文件的线性映射层
        self.npy1_fc = nn.Linear(npy1_dim, feature_dim)
        self.npy2_fc = nn.Linear(npy2_dim, feature_dim)
        
        # 特征合成后的分类层
        self.classifier = nn.Linear(feature_dim * 3, num_classes)  # 3倍因为合成后每个向量变成3个值
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, image, npy1, npy2):
        # 提取图像特征
        image_features = self.resnet(image)
        image_features = self.global_avg_pool(image_features)
        image_features = image_features.view(image_features.size(0), -1)  # [batch_size, 512]
        
        # 处理.npy特征
        npy1_features = self.npy1_fc(npy1.float())  # 确保数据类型为float
        npy2_features = self.npy2_fc(npy2.float())
        
        # 应用激活函数
        npy1_features = self.relu(npy1_features)
        npy2_features = self.relu(npy2_features)
        
        # 特征合成 - 使用向量合成方法
        batch_size = image_features.size(0)
        
        # 批量向量合成（优化版）
        img_np = image_features.detach().cpu().numpy()
        npy1_np = npy1_features.detach().cpu().numpy()
        npy2_np = npy2_features.detach().cpu().numpy()
        
        # 批量处理所有样本
        fused_list = []
        for i in range(batch_size):
            # 每个样本单独调用transform_vectors_numpy
            fused_vec = transform_vectors_numpy(
                img_np[i], npy2_np[i], npy1_np[i], use_degrees=False
            )#图像，序列，量子特征，对于x，y，z
            fused_list.append(fused_vec)
        
        # 转换为tensor
        fused_features = torch.tensor(fused_list, dtype=torch.float32).to(image.device)
        
        # 分类
        output = self.classifier(fused_features)
        
        return output, image_features, npy1_features, npy2_features, fused_features

# 训练函数 - 修改为保存最低验证损失的模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=7, class_names=None):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_metrics = []
    val_metrics = []
    
    # 记录最佳验证损失
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs in train_loader:
            images, npy1, npy2, labels = inputs
            images = images.to(device)
            npy1 = npy1.to(device)
            npy2 = npy2.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs, _, _, _, _ = model(images, npy1, npy2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 计算训练混淆矩阵
        train_cm = confusion_matrix(all_labels, all_preds)
        
        # 计算训练指标 - 统一使用加权平均
        train_acc = accuracy_score(all_labels, all_preds)
        train_accuracies.append(train_acc)
        train_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # 如果需要，也可以计算宏平均作为参考
        train_precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        train_recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        train_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        train_metrics.append({
            'accuracy': train_acc,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
            'precision_macro': train_precision_macro,
            'recall_macro': train_recall_macro,
            'f1_macro': train_f1_macro
        })
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_all_preds = []
        val_all_labels = []
        
        with torch.no_grad():
            for inputs in val_loader:
                images, npy1, npy2, labels = inputs
                images = images.to(device)
                npy1 = npy1.to(device)
                npy2 = npy2.to(device)
                labels = labels.to(device)
                
                outputs, _, _, _, _ = model(images, npy1, npy2)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        
        # 计算验证混淆矩阵
        val_cm = confusion_matrix(val_all_labels, val_all_preds)
        
        # 计算验证指标 - 统一使用加权平均
        val_acc = accuracy_score(val_all_labels, val_all_preds)
        val_accuracies.append(val_acc)
        val_precision = precision_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
        val_recall = recall_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
        
        # 如果需要，也可以计算宏平均作为参考
        val_precision_macro = precision_score(val_all_labels, val_all_preds, average='macro', zero_division=0)
        val_recall_macro = recall_score(val_all_labels, val_all_preds, average='macro', zero_division=0)
        val_f1_macro = f1_score(val_all_labels, val_all_preds, average='macro', zero_division=0)
        
        val_metrics.append({
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'precision_macro': val_precision_macro,
            'recall_macro': val_recall_macro,
            'f1_macro': val_f1_macro
        })
        
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.4f}')
        print(f'Train Precision (weighted): {train_precision:.4f} | Train Recall (weighted): {train_recall:.4f} | Train F1 (weighted): {train_f1:.4f}')
        print(f'Train Precision (macro): {train_precision_macro:.4f} | Train Recall (macro): {train_recall_macro:.4f} | Train F1 (macro): {train_f1_macro:.4f}')
        print(f'Val Precision (weighted): {val_precision:.4f} | Val Recall (weighted): {val_recall:.4f} | Val F1 (weighted): {val_f1:.4f}')
        print(f'Val Precision (macro): {val_precision_macro:.4f} | Val Recall (macro): {val_recall_macro:.4f} | Val F1 (macro): {val_f1_macro:.4f}')
        
        # 输出训练混淆矩阵
        print("\nTrain Confusion Matrix:")
        print_confusion_matrix(train_cm, class_names)
        
        # 输出验证混淆矩阵
        print("\nValidation Confusion Matrix:")
        print_confusion_matrix(val_cm, class_names)
        print()
        
        # 保存最佳模型 - 基于最低验证损失
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_f1': val_f1,  # 也保存F1分数供参考
                'train_loss': epoch_loss,
                'train_f1': train_f1,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, 'best_model.pth')
            print(f'保存最佳模型 (验证损失: {best_val_loss:.4f}, 验证F1: {val_f1:.4f})')
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies, train_metrics, val_metrics

# 测试函数
def test_model(model, test_loader, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for inputs in test_loader:
            images, npy1, npy2, labels = inputs
            images = images.to(device)
            npy1 = npy1.to(device)
            npy2 = npy2.to(device)
            labels = labels.to(device)
            
            outputs, img_feat, npy1_feat, npy2_feat, fused_feat = model(images, npy1, npy2)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 保存特征用于分析
            batch_features = {
                'image_features': img_feat.cpu().numpy(),
                'npy1_features': npy1_feat.cpu().numpy(),
                'npy2_features': npy2_feat.cpu().numpy(),
                'fused_features': fused_feat.cpu().numpy()
            }
            all_features.append(batch_features)
    
    # 计算测试混淆矩阵
    test_cm = confusion_matrix(all_labels, all_preds)
    
    # 计算测试指标 - 统一使用加权平均
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 如果需要，也可以计算宏平均作为参考
    test_precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'precision_macro': test_precision_macro,
        'recall_macro': test_recall_macro,
        'f1_macro': test_f1_macro
    }, test_cm, all_preds, all_labels, all_features

# 绘制训练过程图表
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_dir='.'):
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'multimodal_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    history_df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    })
    
    csv_path = os.path.join(save_dir, 'multimodal_training_history.csv')
    history_df.to_csv(csv_path, index=False)
    print(f"Training history saved to '{csv_path}'")

# 绘制混淆矩阵
def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path='confusion_matrix.png'):
    """
    绘制并保存混淆矩阵
    """
    plt.figure(figsize=(8, 6))
    
    # 创建热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Confusion matrix saved to '{save_path}'")

# 在控制台打印混淆矩阵
def print_confusion_matrix(cm, class_names):
    """
    在控制台打印格式化的混淆矩阵
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    # 计算每列的宽度
    max_class_name_len = max([len(name) for name in class_names])
    cell_width = max(8, max_class_name_len + 2)
    
    # 打印表头
    header = " " * (max_class_name_len + 2) + " | "
    for name in class_names:
        header += f"{name:^{cell_width}} | "
    print(header)
    print("-" * len(header))
    
    # 打印矩阵行
    for i, (row, true_class) in enumerate(zip(cm, class_names)):
        row_str = f"{true_class:>{max_class_name_len}} | "
        for value in row:
            row_str += f"{value:^{cell_width}} | "
        print(row_str)
    
    # 计算并打印评估指标
    print("\nConfusion Matrix Metrics:")
    print("-" * 30)
    
    # 对于二分类
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"True Positives (TP): {tp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall/Sensitivity: {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
    else:
        # 多分类的简单统计
        total = np.sum(cm)
        correct = np.trace(cm)
        accuracy = correct / total if total > 0 else 0
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Total Samples: {total}")
        print(f"Correct Predictions: {correct}")

# 获取.npy文件的维度
def get_npy_dimensions(npy_folder1, npy_folder2):
    # 获取第一个.npy文件的维度
    first_npy1 = None
    first_npy2 = None
    
    # 查找第一个.npy文件
    for root, dirs, files in os.walk(npy_folder1):
        for file in files:
            if file.endswith('.npy'):
                first_npy1 = np.load(os.path.join(root, file))
                break
        if first_npy1 is not None:
            break
    
    for root, dirs, files in os.walk(npy_folder2):
        for file in files:
            if file.endswith('.npy'):
                first_npy2 = np.load(os.path.join(root, file))
                break
        if first_npy2 is not None:
            break
    
    if first_npy1 is None:
        raise ValueError(f"无法在 {npy_folder1} 中找到.npy文件来获取维度")
    if first_npy2 is None:
        raise ValueError(f"无法在 {npy_folder2} 中找到.npy文件来获取维度")
    
    return first_npy1.shape[0], first_npy2.shape[0]

# 保存详细结果报告
def save_detailed_results(train_metrics, val_metrics, test_metrics, class_names, save_dir):
    """
    保存详细的评估结果到CSV文件
    """
    # 创建结果DataFrame
    results = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'Accuracy': [
            train_metrics[-1]['accuracy'] if train_metrics else 0,
            val_metrics[-1]['accuracy'] if val_metrics else 0,
            test_metrics['accuracy']
        ],
        'Precision (weighted)': [
            train_metrics[-1]['precision'] if train_metrics else 0,
            val_metrics[-1]['precision'] if val_metrics else 0,
            test_metrics['precision']
        ],
        'Recall (weighted)': [
            train_metrics[-1]['recall'] if train_metrics else 0,
            val_metrics[-1]['recall'] if val_metrics else 0,
            test_metrics['recall']
        ],
        'F1-Score (weighted)': [
            train_metrics[-1]['f1'] if train_metrics else 0,
            val_metrics[-1]['f1'] if val_metrics else 0,
            test_metrics['f1']
        ],
        'Precision (macro)': [
            train_metrics[-1]['precision_macro'] if train_metrics else 0,
            val_metrics[-1]['precision_macro'] if val_metrics else 0,
            test_metrics['precision_macro']
        ],
        'Recall (macro)': [
            train_metrics[-1]['recall_macro'] if train_metrics else 0,
            val_metrics[-1]['recall_macro'] if val_metrics else 0,
            test_metrics['recall_macro']
        ],
        'F1-Score (macro)': [
            train_metrics[-1]['f1_macro'] if train_metrics else 0,
            val_metrics[-1]['f1_macro'] if val_metrics else 0,
            test_metrics['f1_macro']
        ]
    }
    
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, 'multimodal_detailed_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to '{csv_path}'")
    
    # 打印总结
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\nClass Names: {class_names}")
    print("\nWeighted Average (recommended for imbalanced datasets):")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1']:.4f}")
    
    print("\nMacro Average (equal weight for each class):")
    print(f"Test Precision: {test_metrics['precision_macro']:.4f}")
    print(f"Test Recall: {test_metrics['recall_macro']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_macro']:.4f}")
    print("="*70)

# 主函数
def main():
    # 设置随机种子
    set_seed(42)
    
    # 从统一配置获取参数
    unified_config = get_config()
    
    # 参数设置
    image_dir = unified_config.image_dir
    npy_dir1 = unified_config.lil_qhn_feat_dir  # npy1是量子LIL_QHN特征
    npy_dir2 = os.path.join(unified_config.bilstm_save_dir, unified_config.bilstm_features_dir)  # npy2是BiLSTM特征
    batch_size = unified_config.t_resnet_batch_size
    num_epochs = unified_config.t_resnet_num_epochs
    learning_rate = unified_config.t_resnet_learning_rate
    patience = unified_config.t_resnet_patience
    output_dir = unified_config.t_resnet_output_dir
    
    print(f"T_resnet配置:")
    print(f"  随机种子: 42")
    print(f"  图像文件夹: {image_dir}")
    print(f"  LIL_QHN特征文件夹: {npy_dir1}")# npy1是量子LIL_QHN特征
    print(f"  BiLSTM特征文件夹: {npy_dir2}")# npy2是BiLSTM特征
    print(f"  批量大小: {batch_size}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  学习率: {learning_rate}")
    print(f"  早停耐心值: {patience}")
    print(f"  输出目录: {output_dir}")
    
    # 检查路径是否存在
    if not os.path.exists(image_dir):
        print(f"错误: 图像文件夹不存在: {image_dir}")
        return
    
    if not os.path.exists(npy_dir1):
        print(f"错误: LIL_QHN特征文件夹不存在: {npy_dir1}")
        print("请先运行LIL_QHN模型")
        return
    
    if not os.path.exists(npy_dir2):
        print(f"错误: BiLSTM特征文件夹不存在: {npy_dir2}")
        print("请先运行BiLSTM模型")
        return
    
    # 检查特征目录结构
    print(f"\n检查特征目录结构...")
    for npy_dir, name in [(npy_dir1, "LIL_QHN"), (npy_dir2, "BiLSTM")]:
        if os.path.exists(npy_dir):
            classes = [d for d in os.listdir(npy_dir) if os.path.isdir(os.path.join(npy_dir, d))]
            print(f"{name}特征目录中的类别: {classes}")
            
            # 检查每个类别的文件数量
            for cls in classes:
                cls_dir = os.path.join(npy_dir, cls)
                npy_files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
                print(f"  {cls}: {len(npy_files)} 个.npy文件")
        else:
            print(f"{name}特征目录不存在: {npy_dir}")
    
    # 获取.npy文件的维度
    try:
        npy1_dim, npy2_dim = get_npy_dimensions(npy_dir1, npy_dir2)
        print(f"npy1维度: {npy1_dim}, npy2维度: {npy2_dim}")
    except Exception as e:
        print(f"获取.npy文件维度失败: {e}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据集
    print(f"\n创建数据集...")
    try:
        full_dataset = MultiModalDataset(image_dir, npy_dir1, npy_dir2, transform=transform)
    except Exception as e:
        print(f"创建数据集失败: {e}")
        return
    
    # 获取类别名称
    class_names = [full_dataset.idx_to_class[0], full_dataset.idx_to_class[1]]
    print(f"类别名称: {class_names}")
    
    # 划分数据集
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)  # 固定划分随机种子
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # 初始化多模态模型 - 从头训练
    model = MultiModalResNet18(npy1_dim=npy1_dim, npy2_dim=npy2_dim, num_classes=2).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("Starting training from scratch...")
    model, train_losses, val_losses, train_accuracies, val_accuracies, train_metrics, val_metrics = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, class_names
    )
    
    # 测试模型
    print("\nTesting model...")
    print("-" * 50)
    
    # 加载最佳模型
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_cm, test_preds, test_labels, test_features = test_model(model, test_loader, class_names)
    
    # 输出测试混淆矩阵
    print("\nTest Confusion Matrix:")
    print_confusion_matrix(test_cm, class_names)
    
    # 绘制并保存测试混淆矩阵
    cm_path = os.path.join(output_dir, 'test_confusion_matrix.png')
    plot_confusion_matrix(test_cm, class_names, title='Test Confusion Matrix', save_path=cm_path)
    
    # 保存详细结果报告
    save_detailed_results(train_metrics, val_metrics, test_metrics, class_names, output_dir)
    
    # 绘制训练过程图表
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, output_dir)
    
    # 保存特征用于进一步分析
    features_path = os.path.join(output_dir, 'multimodal_features.npz')
    np.savez(features_path, 
             image_features=np.vstack([f['image_features'] for f in test_features]),
             npy1_features=np.vstack([f['npy1_features'] for f in test_features]),
             npy2_features=np.vstack([f['npy2_features'] for f in test_features]),
             fused_features=np.vstack([f['fused_features'] for f in test_features]),
             predictions=test_preds,
             labels=test_labels)
    print(f"Multimodal features saved to '{features_path}'")
    
    # 保存配置
    config_save_path = os.path.join(output_dir, "t_resnet_config.json")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'random_seed': 42,
            'image_dir': image_dir,
            'npy_dir1': npy_dir1,
            'npy_dir2': npy_dir2,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'patience': patience,
            'npy1_dim': npy1_dim,
            'npy2_dim': npy2_dim,
            'class_names': class_names,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'best_epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint['val_loss'],
            'best_val_f1': checkpoint['val_f1']
        }, f, indent=4, ensure_ascii=False)
    
    print(f"T_resnet配置已保存到: {config_save_path}")
    
    # 保存模型
    model_path = os.path.join(output_dir, 't_resnet_model.pth')
    torch.save({
        'epoch': checkpoint['epoch'],
        'model_state_dict': checkpoint['model_state_dict'],
        'val_loss': checkpoint['val_loss'],
        'val_f1': checkpoint['val_f1'],
        'train_loss': checkpoint['train_loss'],
        'train_f1': checkpoint['train_f1'],
        'train_acc': checkpoint['train_acc'],
        'val_acc': checkpoint['val_acc']
    }, model_path)
    print(f"模型已保存到: {model_path}")

if __name__ == "__main__":
    main()