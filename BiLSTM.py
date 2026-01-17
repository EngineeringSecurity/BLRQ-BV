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
import re, os, csv, shutil, datetime
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

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
# 4. 数据加载
# --------------------------------------------------
def load_data_from_folders(data_dir):
    texts, labels, filenames = [], [], []
    classes = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    class_to_label = {cls: idx for idx, cls in enumerate(sorted(classes))}
    print(f'发现 {len(classes)} 个类别: {class_to_label}')
    for cls in classes:
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

def train_standalone_model(model, train_loader, val_loader, device, 
                          epochs=20, lr=1e-3, save_path='bilstm_standalone.pth'):
    """
    独立训练BiLSTM模型
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_f1 = 0
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
        
        print(f'独立训练 Epoch {epoch+1}/{epochs}: '
              f'Train Loss={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f} | '
              f'Val Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'epoch': epoch
            }, save_path)
            print(f'模型已保存到 {save_path}')
    
    print(f'独立训练完成，最佳验证F1: {best_val_f1:.4f}')
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device):
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
    return acc, precision, recall, f1

# --------------------------------------------------
# 7. 特征提取
# --------------------------------------------------
def extract_features(model, device, data_dir, output_dir,
                     word2idx, max_len, batch_size=32):
    """
    提取特征并保存为npy文件
    """
    texts, labels, filenames, class_to_label = load_data_from_folders(data_dir)
    processor = TextProcessor()
    proc_texts = [processor.preprocess_text(extract_tweet_texts_from_content(t)) for t in texts]
    
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
    
    # 保存特征
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    label2name = {v: k for k, v in class_to_label.items()}
    for name in class_to_label.keys():
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    
    for i, fname in enumerate(filenames):
        cls = label2name[labels[i]]
        np.save(os.path.join(output_dir, cls, fname.replace('.txt', '.npy')), feats[i])
    
    print(f'特征已保存到 {output_dir}')
    return feats.shape[1]  # 返回特征维度

# --------------------------------------------------
# 8. 主函数（独立训练）
# --------------------------------------------------
def main_standalone():
    # 配置参数
    DATA_DIR = '/data/coding/DataSet-pheme/phemetxt'
    BATCH_SIZE = 32
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    OUTPUT_DIM = 2  # 二分类
    N_LAYERS = 2
    DROPOUT = 0.5
    MAX_LEN = 500
    EPOCHS = 20
    LR = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 1. 加载数据
    print('加载数据...')
    texts, labels, filenames, class_to_label = load_data_from_folders(DATA_DIR)
    
    # 2. 提取推文文本
    texts = [extract_tweet_texts_from_content(t) for t in texts]
    
    # 3. 预处理
    processor = TextProcessor()
    texts = [processor.preprocess_text(t) for t in texts]
    
    # 4. 构建词汇表
    word2idx = processor.build_vocab(texts, min_freq=2)
    
    # 5. 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f'训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}')
    
    # 6. 创建数据加载器
    train_dataset = TweetDataset(X_train, y_train, word2idx, MAX_LEN)
    val_dataset = TweetDataset(X_val, y_val, word2idx, MAX_LEN)
    test_dataset = TweetDataset(X_test, y_test, word2idx, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 7. 创建模型
    model = BiLSTMClassifier(
        vocab_size=len(word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # 8. 独立训练
    print('开始独立训练...')
    model, train_losses, val_losses = train_standalone_model(
        model, train_loader, val_loader, device,
        epochs=EPOCHS, lr=LR, save_path='bilstm_standalone.pth'
    )
    
    # 9. 测试
    print('测试模型...')
    evaluate_model(model, test_loader, device)
    
    # 10. 提取特征（用于联合训练）
    print('提取特征...')
    feature_dim = extract_features(
        model, device, DATA_DIR, 'pheme_bilstm_features',
        word2idx, MAX_LEN, BATCH_SIZE
    )
    
    print(f'BiLSTM独立训练完成，特征维度: {feature_dim}')
    return model, feature_dim

if __name__ == '__main__':
    main_standalone()