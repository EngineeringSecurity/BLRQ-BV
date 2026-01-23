# BiLSTM.py
# ==========================================
# BiLSTM 谣言分类器 – 支持独立训练
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score,
                             f1_score, precision_score, recall_score)
import re, os, csv, shutil, datetime, json
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

# --------------------------------------------------
# 从统一配置获取参数
# --------------------------------------------------
from unified_config import get_config

# 创建全局配置对象
config = get_config()

# --------------------------------------------------
# 超参数配置（从统一配置获取）
# --------------------------------------------------
class Config:
    """BiLSTM配置类，从统一配置获取参数"""
    
    # 数据路径
    DATA_DIR = config.data_dir
    
    # 保存路径（所有输出将保存在这个目录下）
    SAVE_DIR = config.bilstm_save_dir
    
    # 训练参数
    BATCH_SIZE = config.bilstm_batch_size
    EMBEDDING_DIM = config.bilstm_embedding_dim
    HIDDEN_DIM = config.bilstm_hidden_dim
    OUTPUT_DIM = config.bilstm_output_dim
    N_LAYERS = config.bilstm_n_layers
    DROPOUT = config.bilstm_dropout
    MAX_LEN = config.bilstm_max_len
    EPOCHS = config.bilstm_epochs
    LR = config.bilstm_lr
    MIN_FREQ = config.bilstm_min_freq
    
    # 文件名配置
    MODEL_NAME = config.bilstm_model_name
    FEATURES_DIR = config.bilstm_features_dir
    RESULTS_CSV = config.bilstm_results_csv
    METRICS_CSV = config.bilstm_metrics_csv
    LOSS_PLOT = config.bilstm_loss_plot
    CONFIG_FILE = config.bilstm_config_file
    
    @classmethod
    def create_save_dir(cls):
        """创建保存目录"""
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.SAVE_DIR, cls.FEATURES_DIR), exist_ok=True)
        print(f"所有输出将保存到: {cls.SAVE_DIR}")
    
    @classmethod
    def save_config(cls):
        """保存配置到JSON文件"""
        config_dict = {
            'DATA_DIR': cls.DATA_DIR,
            'SAVE_DIR': cls.SAVE_DIR,
            'BATCH_SIZE': cls.BATCH_SIZE,
            'EMBEDDING_DIM': cls.EMBEDDING_DIM,
            'HIDDEN_DIM': cls.HIDDEN_DIM,
            'OUTPUT_DIM': cls.OUTPUT_DIM,
            'N_LAYERS': cls.N_LAYERS,
            'DROPOUT': cls.DROPOUT,
            'MAX_LEN': cls.MAX_LEN,
            'EPOCHS': cls.EPOCHS,
            'LR': cls.LR,
            'MIN_FREQ': cls.MIN_FREQ,
            'MODEL_NAME': cls.MODEL_NAME,
            'FEATURES_DIR': cls.FEATURES_DIR,
            'RESULTS_CSV': cls.RESULTS_CSV,
            'METRICS_CSV': cls.METRICS_CSV,
            'LOSS_PLOT': cls.LOSS_PLOT,
            'CONFIG_FILE': cls.CONFIG_FILE,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        config_path = os.path.join(cls.SAVE_DIR, cls.CONFIG_FILE)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        print(f"配置已保存到: {config_path}")

# --------------------------------------------------
# 1. 数据集类
# --------------------------------------------------
class TweetDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.texts, self.labels, self.word2idx, self.max_len = texts, labels, word2idx, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()[:self.max_len]
        indices = [self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens]
        if len(indices) < self.max_len:
            indices += [self.word2idx['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return {'text': torch.tensor(indices, dtype=torch.long),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

# --------------------------------------------------
# 2. 模型
# --------------------------------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, dropout, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.feature_dim = hidden_dim * 2  # 用于联合训练的特征维度

    def forward(self, text):
        embed = self.embedding(text)                      # [B, T, E]
        _, (hidden, _) = self.lstm(embed)                 # hidden: [2*layers, B, H]
        forward_h  = hidden[-2]                           # 前向顶层
        backward_h = hidden[-1]                           # 后向顶层
        concat = torch.cat((forward_h, backward_h), dim=1)
        features = self.dropout(concat)
        return self.fc(features), features

    def extract_features(self, text):
        with torch.no_grad():
            embed = self.embedding(text)
            _, (hidden, _) = self.lstm(embed)
            concat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return concat

# --------------------------------------------------
# 3. 文本处理器
# --------------------------------------------------
class TextProcessor:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2

    def build_vocab(self, texts, min_freq=1):
        cnt = Counter()
        for t in texts:
            cnt.update(t.split())
        for w, c in cnt.items():
            if c >= min_freq:
                self.word2idx[w] = self.vocab_size
                self.idx2word[self.vocab_size] = w
                self.vocab_size += 1
        print(f'词汇表大小: {self.vocab_size}')
        return self.word2idx

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

# --------------------------------------------------
# 4. 数据加载（修改为保留原始文件名）
# --------------------------------------------------
def load_data_from_folders(data_dir):
    texts, labels, filenames = [], [], []
    classes = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    class_to_label = {cls: idx for idx, cls in enumerate(sorted(classes))}
    print(f'发现 {len(classes)} 个类别: {class_to_label}')
    
    # 确保只处理 non_rumor 和 rumor 两个类别
    valid_classes = ['non_rumor', 'rumor']
    for cls in classes:
        if cls not in valid_classes:
            print(f"警告: 跳过未知类别 '{cls}'")
            continue
            
        cls_path = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_path):
            if not fname.endswith('.txt'):
                continue
            with open(os.path.join(cls_path, fname), encoding='utf-8', errors='ignore') as f:
                content = f.read()
            texts.append(content)
            labels.append(class_to_label[cls])
            filenames.append(fname)
    
    print(f'共加载 {len(texts)} 条样本')
    for cls, idx in class_to_label.items():
        if cls in valid_classes:
            print(f'  {cls}: {labels.count(idx)}')
    
    return texts, labels, filenames, class_to_label

# --------------------------------------------------
# 5. 文本提取
# --------------------------------------------------
def extract_tweet_texts_from_content(content):
    if '[Tweet' not in content:
        return content.strip()
    tweets, cur = [], ''
    for line in content.split('\n'):
        if line.startswith('[Tweet'):
            if cur:
                tweets.append(cur)
            cur = ''
        elif line.startswith('Text:'):
            cur += ' ' + line.replace('Text:', '').strip()
        elif line.strip() == '' and cur:
            tweets.append(cur)
            cur = ''
    if cur:
        tweets.append(cur)
    return ' '.join(tweets) if tweets else content.strip()

# --------------------------------------------------
# 6. 训练函数
# --------------------------------------------------
def calculate_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return acc, precision, recall, f1

def save_training_log(epoch, train_loss, train_acc, train_f1, 
                     val_loss, val_acc, val_f1, csv_path):
    """保存训练日志到CSV文件"""
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'train_f1',
                           'val_loss', 'val_acc', 'val_f1', 'timestamp'])
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([epoch, f'{train_loss:.4f}', f'{train_acc:.4f}', f'{train_f1:.4f}',
                        f'{val_loss:.4f}', f'{val_acc:.4f}', f'{val_f1:.4f}', timestamp])

def save_detailed_metrics(preds, labels, class_names, csv_path):
    """保存详细评估指标到CSV"""
    report = classification_report(labels, preds, 
                                   target_names=class_names,
                                   output_dict=True)
    
    # 将报告转换为DataFrame
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(csv_path, encoding='utf-8')
    print(f"详细评估指标已保存到: {csv_path}")

def plot_loss_curve(train_losses, val_losses, save_path):
    """绘制损失曲线并保存"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"损失曲线图已保存到: {save_path}")

def train_standalone_model(model, train_loader, val_loader, device, 
                          epochs=20, lr=1e-3, save_dir=None):
    """
    独立训练BiLSTM模型
    """
    if save_dir is None:
        save_dir = Config.SAVE_DIR
    
    model_path = os.path.join(save_dir, Config.MODEL_NAME)
    csv_path = os.path.join(save_dir, Config.RESULTS_CSV)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 改为跟踪最低验证损失
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, train_preds, train_labels = 0, [], []
        
        for batch in tqdm(train_loader, desc=f'独立训练Epoch {epoch+1}/{epochs}'):
            texts, labels = batch['text'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs, _ = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc, _, _, train_f1 = calculate_metrics(train_preds, train_labels)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                texts, labels = batch['text'].to(device), batch['label'].to(device)
                outputs, _ = model(texts)
                val_loss += criterion(outputs, labels).item()
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc, _, _, val_f1 = calculate_metrics(val_preds, val_labels)
        val_losses.append(val_loss)
        
        # 保存训练日志
        save_training_log(epoch+1, train_loss, train_acc, train_f1,
                         val_loss, val_acc, val_f1, csv_path)
        
        print(f'独立训练 Epoch {epoch+1}/{epochs}: '
              f'Train Loss={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f} | '
              f'Val Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}')
        
        # 修改保存条件：基于最低验证损失
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,  # 保存验证损失
                'val_f1': val_f1,      # 也保存F1分数供参考
                'epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, model_path)
            print(f'模型已保存到 {model_path} (最低验证损失: {best_val_loss:.4f})')
    
    # 绘制损失曲线
    plot_loss_curve(train_losses, val_losses, 
                    os.path.join(save_dir, Config.LOSS_PLOT))
    
    print(f'独立训练完成，最低验证损失: {best_val_loss:.4f}')
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device, class_names, save_dir=None):
    """评估模型并保存详细指标"""
    if save_dir is None:
        save_dir = Config.SAVE_DIR
    
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='测试'):
            texts = batch['text'].to(device)
            outputs, _ = model(texts)
            preds.extend(outputs.argmax(1).cpu().numpy())
            labels.extend(batch['label'].numpy())
    
    acc, precision, recall, f1 = calculate_metrics(preds, labels)
    print(f'测试集: 准确率={acc:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}')
    
    # 保存详细评估指标
    metrics_path = os.path.join(save_dir, Config.METRICS_CSV)
    save_detailed_metrics(preds, labels, class_names, metrics_path)
    
    # 保存总体结果
    results_summary = {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = os.path.join(save_dir, 'test_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)
    
    return acc, precision, recall, f1

# --------------------------------------------------
# 7. 特征提取（按原始类别结构保存）
# --------------------------------------------------
def extract_features_by_original_structure(model, device, data_dir, output_dir,
                                          word2idx, max_len, batch_size=32):
    """
    提取特征并保存为npy文件，按照原始类别结构保存
    """
    # 加载原始数据
    texts, labels, filenames, class_to_label = load_data_from_folders(data_dir)
    processor = TextProcessor()
    
    # 预处理文本并提取推文内容
    proc_texts = []
    for t in texts:
        # 提取推文文本并预处理
        tweet_text = extract_tweet_texts_from_content(t)
        processed = processor.preprocess_text(tweet_text)
        proc_texts.append(processed)
    
    # 创建数据集
    dataset = TweetDataset(proc_texts, labels, word2idx, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='提取特征'):
            texts_batch = batch['text'].to(device)
            features = model.extract_features(texts_batch)
            feats.append(features.cpu().numpy())
    
    feats = np.vstack(feats)
    
    # 清理输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建类别目录
    for cls_name in class_to_label.keys():
        if cls_name in ['non_rumor', 'rumor']:  # 只创建有效类别目录
            cls_dir = os.path.join(output_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
    
    # 按照原始类别和文件名保存特征
    label2name = {v: k for k, v in class_to_label.items()}
    
    print("保存特征文件...")
    for i, (fname, label) in enumerate(tqdm(zip(filenames, labels), total=len(filenames), desc='保存特征')):
        cls = label2name[label]
        if cls not in ['non_rumor', 'rumor']:
            continue
            
        # 去除文件后缀，保留原始文件名
        base_name = os.path.splitext(fname)[0]
        
        # 保存特征
        save_path = os.path.join(output_dir, cls, f"{base_name}.npy")
        np.save(save_path, feats[i])
    
    print(f'特征已保存到 {output_dir}')
    print(f'类别分布:')
    for cls_name in ['non_rumor', 'rumor']:
        if cls_name in class_to_label:
            idx = class_to_label[cls_name]
            count = labels.count(idx)
            print(f'  {cls_name}: {count} 个样本')
    
    # 保存特征统计信息
    feat_stats = {
        'total_samples': len(feats),
        'feature_dim': feats.shape[1],
        'classes_distribution': {cls: labels.count(idx) for cls, idx in class_to_label.items()},
        'features_dir': output_dir,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'saved_by': 'original_structure'
    }
    
    stats_path = os.path.join(Config.SAVE_DIR, 'features_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(feat_stats, f, indent=4, ensure_ascii=False)
    
    # 保存文件名映射
    file_mapping = []
    for i, (fname, label) in enumerate(zip(filenames, labels)):
        cls = label2name[label]
        if cls not in ['non_rumor', 'rumor']:
            continue
            
        base_name = os.path.splitext(fname)[0]
        file_mapping.append({
            'original_filename': fname,
            'base_name': base_name,
            'class': cls,
            'feature_file': f"{base_name}.npy",
            'feature_index': i
        })
    
    mapping_path = os.path.join(Config.SAVE_DIR, 'file_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(file_mapping, f, indent=4, ensure_ascii=False)
    
    print(f"文件映射已保存到: {mapping_path}")
    return feats.shape[1]  # 返回特征维度

# --------------------------------------------------
# 8. 主函数（独立训练）
# --------------------------------------------------
def main_standalone():
    # 创建保存目录
    Config.create_save_dir()
    Config.save_config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 1. 加载数据
    print('加载数据...')
    texts, labels, filenames, class_to_label = load_data_from_folders(Config.DATA_DIR)
    class_names = list(class_to_label.keys())
    
    # 2. 提取推文文本
    texts = [extract_tweet_texts_from_content(t) for t in texts]
    
    # 3. 预处理
    processor = TextProcessor()
    texts = [processor.preprocess_text(t) for t in texts]
    
    # 4. 构建词汇表
    word2idx = processor.build_vocab(texts, min_freq=Config.MIN_FREQ)
    
    # 保存词汇表
    vocab_path = os.path.join(Config.SAVE_DIR, 'vocabulary.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(word2idx, f, indent=4, ensure_ascii=False)
    print(f"词汇表已保存到: {vocab_path}")
    
    # 5. 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f'训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}')
    
    # 6. 创建数据加载器
    train_dataset = TweetDataset(X_train, y_train, word2idx, Config.MAX_LEN)
    val_dataset = TweetDataset(X_val, y_val, word2idx, Config.MAX_LEN)
    test_dataset = TweetDataset(X_test, y_test, word2idx, Config.MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 7. 创建模型
    model = BiLSTMClassifier(
        vocab_size=len(word2idx),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM,
        n_layers=Config.N_LAYERS,
        dropout=Config.DROPOUT
    ).to(device)
    
    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')
    print(f'可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    # 8. 独立训练
    print('开始独立训练...')
    model, train_losses, val_losses = train_standalone_model(
        model, train_loader, val_loader, device,
        epochs=Config.EPOCHS, lr=Config.LR, save_dir=Config.SAVE_DIR
    )
    
    # 9. 测试
    print('测试模型...')
    evaluate_model(model, test_loader, device, class_names, Config.SAVE_DIR)
    
    # 10. 提取特征（按原始类别结构保存）
    print('提取特征（按原始类别结构保存）...')
    feature_dir = os.path.join(Config.SAVE_DIR, Config.FEATURES_DIR)
    feature_dim = extract_features_by_original_structure(
        model, device, Config.DATA_DIR, feature_dir,
        word2idx, Config.MAX_LEN, Config.BATCH_SIZE
    )
    
    # 11. 保存最终总结
    summary = {
        'project': 'BiLSTM谣言检测器',
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'device': str(device),
        'save_directory': Config.SAVE_DIR,
        'model_file': Config.MODEL_NAME,
        'features_directory': Config.FEATURES_DIR,
        'vocabulary_size': len(word2idx),
        'feature_dimension': feature_dim,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'classes': class_to_label,
        'output_files': {
            'configuration': Config.CONFIG_FILE,
            'training_log': Config.RESULTS_CSV,
            'detailed_metrics': Config.METRICS_CSV,
            'loss_plot': Config.LOSS_PLOT,
            'vocabulary': 'vocabulary.json',
            'features_statistics': 'features_statistics.json',
            'file_mapping': 'file_mapping.json',
            'test_summary': 'test_summary.json'
        }
    }
    
    final_summary_path = os.path.join(Config.SAVE_DIR, 'final_summary.json')
    with open(final_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f'\n{"="*60}')
    print(f"BiLSTM独立训练完成！")
    print(f"所有输出文件已保存到: {Config.SAVE_DIR}")
    print(f"最终总结文件: {final_summary_path}")
    print(f"{'='*60}\n")
    
    # 显示保存的文件列表
    print("生成的文件列表:")
    for root, dirs, files in os.walk(Config.SAVE_DIR):
        level = root.replace(Config.SAVE_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")
    
    return model, feature_dim

if __name__ == '__main__':
    main_standalone()