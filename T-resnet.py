import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torchvision.datasets as datasets
import os
import pandas as pd
import numpy as np
from PIL import Image
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 导入向量合成函数
from Vector_composing import transform_vectors_numpy

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

# 自定义数据集类 - 修改为支持三个文件夹
class MultiModalDataset(Dataset):
    def __init__(self, image_folder, npy_folder1, npy_folder2, transform=None, mode='train'):
        self.image_folder = image_folder
        self.npy_folder1 = npy_folder1
        self.npy_folder2 = npy_folder2
        self.transform = transform
        self.mode = mode
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
                        base_name = os.path.splitext(img_name)[0]
                        
                        # 检查对应的.npy文件是否存在
                        npy_path1 = os.path.join(npy_folder1, class_name, base_name + '.npy')
                        npy_path2 = os.path.join(npy_folder2, class_name, base_name + '.npy')
                        
                        if os.path.exists(npy_path1) and os.path.exists(npy_path2):
                            self.image_paths.append({
                                'image': os.path.join(class_path, img_name),
                                'npy1': npy_path1,
                                'npy2': npy_path2
                            })
                            self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]['image']
        npy_path1 = self.image_paths[idx]['npy1']
        npy_path2 = self.image_paths[idx]['npy2']
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('L')  # 以灰度模式读取
        if self.transform:
            image = self.transform(image)
            
        # 加载.npy文件
        npy_data1 = np.load(npy_path1)
        npy_data2 = np.load(npy_path2)
        
        return image, npy_data1, npy_data2, label

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
        feature_dim = image_features.size(1)
        
        # 将三个特征向量进行合成
        fused_features = []
        for i in range(batch_size):
            # 获取单个样本的三个特征向量
            img_vec = image_features[i].detach().cpu().numpy()
            npy1_vec = npy1_features[i].detach().cpu().numpy()
            npy2_vec = npy2_features[i].detach().cpu().numpy()
            
            # 使用向量合成函数
            fused_vec = transform_vectors_numpy([img_vec], [npy1_vec], [npy2_vec], use_degrees=False)
            fused_features.append(fused_vec)
        
        # 转换为tensor
        fused_features = torch.tensor(fused_features, dtype=torch.float32).to(image.device)
        
        # 分类
        output = self.classifier(fused_features)
        
        return output, image_features, npy1_features, npy2_features, fused_features

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=7, class_names=None):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_metrics = []
    val_metrics = []
    train_cm_history = []  # 保存训练混淆矩阵历史
    val_cm_history = []    # 保存验证混淆矩阵历史
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='pheme_best_model.pth')
    
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
        train_cm_history.append(train_cm)
        
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
        val_cm_history.append(val_cm)
        
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
        
        # 早停检查
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_losses, val_losses, train_accuracies, val_accuracies, train_metrics, val_metrics, train_cm_history, val_cm_history

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
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, early_stop_epoch=None):
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    if early_stop_epoch:
        ax1.axvline(x=early_stop_epoch, color='gray', linestyle='--', 
                   label=f'Early Stop (Epoch {early_stop_epoch})')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    if early_stop_epoch:
        ax2.axvline(x=early_stop_epoch, color='gray', linestyle='--', 
                   label=f'Early Stop (Epoch {early_stop_epoch})')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multimodal_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    history_df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    })
    history_df.to_csv('multimodal_training_history.csv', index=False)
    print("Training history saved to 'multimodal_training_history.csv'")

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
    
    if first_npy1 is None or first_npy2 is None:
        raise ValueError("无法找到.npy文件来获取维度")
    
    return first_npy1.shape[0], first_npy2.shape[0]

# 保存详细结果报告
def save_detailed_results(train_metrics, val_metrics, test_metrics, class_names):
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
    results_df.to_csv('multimodal_detailed_results.csv', index=False)
    print("Detailed results saved to 'multimodal_detailed_results.csv'")
    
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
    # 参数设置
    image_dir = '/data/coding/DataSet-pheme/phemetxt-image'  # 图片文件夹
    npy_dir1 = '/data/coding/Qcnn_bert_pheme/bert_pheme_CNN1D_Quantum_Features'  # 第一个.npy文件夹
    npy_dir2 = '/data/coding/pheme_bilstm_features'  # 第二个.npy文件夹
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    patience = 7
    
    # 获取.npy文件的维度
    npy1_dim, npy2_dim = get_npy_dimensions(npy_dir1, npy_dir2)
    print(f"npy1维度: {npy1_dim}, npy2维度: {npy2_dim}")
    
    # 创建数据集
    full_dataset = MultiModalDataset(image_dir, npy_dir1, npy_dir2, transform=transform)
    
    # 获取类别名称
    class_names = [full_dataset.idx_to_class[0], full_dataset.idx_to_class[1]]
    print(f"类别名称: {class_names}")
    
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
    model, train_losses, val_losses, train_accuracies, val_accuracies, train_metrics, val_metrics, train_cm_history, val_cm_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, class_names
    )
    
    # 保存最后一个epoch的混淆矩阵
    if train_cm_history:
        np.save('train_confusion_matrices.npy', np.array(train_cm_history))
        np.save('val_confusion_matrices.npy', np.array(val_cm_history))
        print("Confusion matrices saved to 'train_confusion_matrices.npy' and 'val_confusion_matrices.npy'")
    
    # 测试模型
    print("\nTesting model...")
    print("-" * 50)
    test_metrics, test_cm, test_preds, test_labels, test_features = test_model(model, test_loader, class_names)
    
    # 输出测试混淆矩阵
    print("\nTest Confusion Matrix:")
    print_confusion_matrix(test_cm, class_names)
    
    # 绘制并保存测试混淆矩阵
    plot_confusion_matrix(test_cm, class_names, title='Test Confusion Matrix', save_path='test_confusion_matrix.png')
    
    # 保存详细结果报告
    save_detailed_results(train_metrics, val_metrics, test_metrics, class_names)
    
    # 绘制训练过程图表
    early_stop_epoch = len(train_losses) if len(train_losses) < num_epochs else None
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, early_stop_epoch)
    
    # 保存特征用于进一步分析
    np.savez('multimodal_features.npz', 
             image_features=np.vstack([f['image_features'] for f in test_features]),
             npy1_features=np.vstack([f['npy1_features'] for f in test_features]),
             npy2_features=np.vstack([f['npy2_features'] for f in test_features]),
             fused_features=np.vstack([f['fused_features'] for f in test_features]),
             predictions=test_preds,
             labels=test_labels)
    print("Multimodal features saved to 'multimodal_features.npz'")

if __name__ == "__main__":
    main()