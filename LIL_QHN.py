# LIL_QHN.py
# -*- coding: utf-8 -*-
"""
1D-QuantumCNN + BERT逻辑回归整合版（完整版）
端到端流程：BERT提取特征 -> 逻辑回归分类 -> 概率向量 -> 1D-QC-CNN训练
包含：训练、测试、量子特征提取与保存（按原始类别结构）
"""

import os, csv, random, logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import json
import joblib
import shutil

# ========================== 从统一配置获取参数 ==========================
from unified_config import get_config
config = get_config()

# ========================== 配置日志 ==========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
np.set_printoptions(precision=4, suppress=True)

# ========================== 量子层 ==========================
import pennylane as qml

class BinaryQuantumClassifier(nn.Module):
    def __init__(self, input_dim, num_qubits=4):
        super().__init__()
        self.num_qubits = num_qubits
        self.classical_to_quantum = nn.Linear(input_dim, num_qubits * 2)
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.q_params = nn.Parameter(torch.randn(num_qubits * 3) * 0.1)
        self.classifier = nn.Linear(num_qubits, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.classical_to_quantum, self.classifier]:
            if hasattr(m, 'weight'): nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias'):   nn.init.zeros_(m.bias)
        nn.init.normal_(self.q_params, 0, 0.1)

    def _circuit(self):
        n = self.num_qubits
        @qml.qnode(self.dev, interface='torch')
        def circ(feat, p):
            for i in range(n): qml.Hadamard(wires=i)
            for i, v in enumerate(feat[:n]): qml.RY(v * np.pi, wires=i)
            idx = 0
            for i in range(n):
                qml.RX(p[idx], wires=i); idx += 1
                qml.RY(p[idx], wires=i); idx += 1
                qml.RZ(p[idx], wires=i); idx += 1
            for i in range(n - 1): qml.CNOT(wires=[i, i + 1])
            if n > 1: qml.CNOT(wires=[n - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]
        return circ

    def forward(self, x):
        batch_size = x.shape[0]
        compressed = self.classical_to_quantum(x).view(batch_size, 2, self.num_qubits)
        q_out = torch.zeros(batch_size, self.num_qubits, device=x.device)
        circ = self._circuit()
        for i in range(batch_size):
            seg = [torch.tensor(circ(compressed[i, reuse], self.q_params), dtype=torch.float32, device=x.device)
                   for reuse in range(2)]
            q_out[i] = torch.stack(seg).mean(0)
        return self.classifier(q_out)


class BinaryQuantumCNN(nn.Module):
    def __init__(self, in_channels=1, num_qubits=4):
        super().__init__()
        self.cnn_features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.quantum_classifier = BinaryQuantumClassifier(32, num_qubits)
        self.config = {}

    def forward(self, x):
        feat = self.cnn_features(x).flatten(1)
        return self.quantum_classifier(feat)

# ========================== BERT特征提取器 ==========================
class BERTFeatureExtractor:
    def __init__(self, model_path="bert-base-uncased", max_length=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"BERT设备: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path).to(self.device).eval()
        self.max_length = max_length

    def get_cls_vector(self, text: str) -> np.ndarray:
        """提取BERT CLS向量 (768维)"""
        if not text:
            return np.zeros(768)
        encoded = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**encoded, output_hidden_states=False)
        return out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

# ========================== 内存数据集 ==========================
class ProbabilityVectorDataset(Dataset):
    """直接使用概率向量作为输入的数据集"""
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: 概率向量数组 (n_samples, seq_len)
            y: 标签数组 (n_samples,)
            transform: 可选的归一化转换
        """
        self.X = torch.FloatTensor(X).unsqueeze(1)  # (n, 1, seq_len)
        self.y = torch.LongTensor(y)
        self.transform = transform
        self.id_to_label = None
        if isinstance(y, np.ndarray):
            self.id_to_label = {0: "class_0", 1: "class_1"}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]
    
    def get_sample_info(self, idx):
        """获取样本信息，用于特征保存"""
        label = self.y[idx].item()
        label_name = self.id_to_label.get(label, f"class_{label}")
        return label, label_name

class Normalize1D:
    """1D归一化"""
    def __init__(self, mean, std): 
        self.mean, self.std = mean, std
    def __call__(self, x): 
        return (x - self.mean) / self.std

# ========================== 工具函数 ==========================
def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calc_metrics(y_true, y_pred):
    """计算评估指标"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average='macro', zero_division=0),
        recall=recall_score(y_true, y_pred, average='macro', zero_division=0),
        f1=f1_score(y_true, y_pred, average='macro', zero_division=0),
        specificity=tn / (tn + fp) if (tn + fp) > 0 else 0.,
        cm=cm
    )

def extract_quantum_features_by_original_structure(model, device, text_data_dir, save_dir, config_dict):
    """
    按照原始类别结构提取和保存量子特征
    """
    model.eval()
    
    # 创建BERT特征提取器
    extractor = BERTFeatureExtractor(config_dict['bert_model'], max_length=config_dict['max_length'])
    
    # 加载逻辑回归模型（如果存在）
    lr_path = os.path.join(config_dict['save_dir'], 'logistic_regression_model.pkl')
    if os.path.exists(lr_path):
        lr_clf = joblib.load(lr_path)
        logger.info(f"加载逻辑回归模型: {lr_path}")
    else:
        logger.error(f"逻辑回归模型未找到: {lr_path}")
        return None
    
    # 加载标签映射
    label_mapping_path = os.path.join(config_dict['save_dir'], 'label_mapping.pkl')
    if os.path.exists(label_mapping_path):
        id2label = joblib.load(label_mapping_path)
        logger.info(f"标签映射: {id2label}")
    else:
        id2label = {0: "non_rumor", 1: "rumor"}
        logger.warning(f"使用默认标签映射: {id2label}")
    
    # 清理并创建保存目录
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取归一化参数
    mean = config_dict.get('normalize_mean', 0.0)
    std = config_dict.get('normalize_std', 1.0)
    
    # 遍历原始数据目录
    label2id = {v: k for k, v in id2label.items()}
    total_samples = 0
    
    for class_name in ['non_rumor', 'rumor']:
        class_dir = os.path.join(text_data_dir, class_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"类别目录不存在: {class_dir}")
            continue
        
        # 创建对应的保存目录
        save_class_dir = os.path.join(save_dir, class_name)
        os.makedirs(save_class_dir, exist_ok=True)
        
        # 获取该类别下的所有txt文件
        txt_files = [f for f in os.listdir(class_dir) if f.endswith('.txt')]
        logger.info(f"处理类别 {class_name}: {len(txt_files)} 个文件")
        
        # 分批处理，避免内存溢出
        batch_size = 32
        for i in tqdm(range(0, len(txt_files), batch_size), desc=f"提取 {class_name} 特征"):
            batch_files = txt_files[i:i+batch_size]
            batch_texts = []
            batch_filenames = []
            
            # 读取文本
            for txt_file in batch_files:
                txt_path = os.path.join(class_dir, txt_file)
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    batch_texts.append(text)
                    batch_filenames.append(txt_file)
                except Exception as e:
                    logger.warning(f"读取文件失败 {txt_path}: {e}")
            
            if not batch_texts:
                continue
            
            # 提取BERT特征
            batch_bert_features = []
            for text in batch_texts:
                bert_feat = extractor.get_cls_vector(text)
                batch_bert_features.append(bert_feat)
            
            batch_bert_features = np.array(batch_bert_features)
            
            # 使用逻辑回归得到概率向量
            batch_prob = lr_clf.predict_proba(batch_bert_features)
            
            # 转换为tensor并应用归一化
            batch_tensor = torch.FloatTensor(batch_prob).unsqueeze(1)
            if std > 0:
                batch_tensor = (batch_tensor - mean) / std
            
            batch_tensor = batch_tensor.to(device)
            
            with torch.no_grad():
                # 获取CNN特征
                cnn_feat = model.cnn_features(batch_tensor).flatten(1)
                
                # 获取量子特征
                compressed = model.quantum_classifier.classical_to_quantum(cnn_feat).view(
                    batch_tensor.size(0), 2, model.quantum_classifier.num_qubits
                )
                
                circ = model.quantum_classifier._circuit()
                qvec = torch.zeros(batch_tensor.size(0), model.quantum_classifier.num_qubits, device=device)
                
                for j in range(batch_tensor.size(0)):
                    seg = [torch.tensor(circ(compressed[j, reuse], 
                                          model.quantum_classifier.q_params), 
                                      dtype=torch.float32, 
                                      device=device)
                          for reuse in range(2)]
                    qvec[j] = torch.stack(seg).mean(0)
                
                # 保存量子特征
                for j, filename in enumerate(batch_filenames):
                    base_name = os.path.splitext(filename)[0]
                    save_path = os.path.join(save_class_dir, f"{base_name}.npy")
                    np.save(save_path, qvec[j].cpu().numpy())
                    total_samples += 1
    
    logger.info(f"量子特征已保存到: {save_dir}")
    logger.info(f"总共处理了 {total_samples} 个样本")
    
    # 保存文件映射
    file_mapping = []
    for class_name in ['non_rumor', 'rumor']:
        class_dir = os.path.join(save_dir, class_name)
        if os.path.isdir(class_dir):
            npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            for npy_file in npy_files:
                base_name = os.path.splitext(npy_file)[0]
                file_mapping.append({
                    'original_filename': f"{base_name}.txt",
                    'base_name': base_name,
                    'class': class_name,
                    'feature_file': npy_file
                })
    
    mapping_path = os.path.join(config_dict['save_dir'], 'quantum_file_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(file_mapping, f, indent=4, ensure_ascii=False)
    
    logger.info(f"文件映射已保存到: {mapping_path}")
    return save_dir

# ========================== 主流程 ==========================
def run_bert_qcnn_pipeline():
    """端到端流程：BERT -> 逻辑回归 -> 1D-QC-CNN -> 量子特征保存（按原始类别结构）"""
    
    # 从统一配置获取参数
    unified_config = get_config()
    
    # 构建配置字典
    CONFIG = {
        'bert_model': unified_config.lil_qhn_bert_model,
        'text_data_dir': unified_config.data_dir,
        'split': unified_config.lil_qhn_split,
        'seed': unified_config.lil_qhn_seed,
        'batch_size': unified_config.lil_qhn_batch_size,
        'epochs': unified_config.lil_qhn_epochs,
        'lr': unified_config.lil_qhn_lr,
        'num_qubits': unified_config.lil_qhn_num_qubits,
        'test_size': unified_config.lil_qhn_test_size,
        
        # 保存路径
        'save_dir': unified_config.lil_qhn_save_dir,
        'feat_dir': unified_config.lil_qhn_feat_dir,
        'max_length': unified_config.lil_qhn_max_length
    }
    
    set_seed(CONFIG['seed'])
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['feat_dir'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"训练设备: {device}")
    
    # ========== 阶段1: BERT特征提取 ==========
    logger.info("阶段1: 提取BERT特征...")
    extractor = BERTFeatureExtractor(CONFIG['bert_model'])
    
    # 读取文本数据并提取特征
    X_raw, y_raw, filenames = [], [], []
    label2id = {}
    
    for label in sorted(os.listdir(CONFIG['text_data_dir'])):
        label_dir = os.path.join(CONFIG['text_data_dir'], label)
        if not os.path.isdir(label_dir) or label.startswith('.') or label not in ['non_rumor', 'rumor']:
            continue
            
        if label not in label2id:
            label2id[label] = len(label2id)
            
        txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        for fn in tqdm(txt_files, desc=f"提取 {label}"):
            txt_path = os.path.join(label_dir, fn)
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                vec = extractor.get_cls_vector(text)
                X_raw.append(vec)
                y_raw.append(label2id[label])
                filenames.append(fn)
            except Exception as e:
                logger.warning(f"读取文件失败 {txt_path}: {e}")
    
    X_raw = np.array(X_raw)  # (n_samples, 768)
    y_raw = np.array(y_raw)
    
    logger.info(f"BERT特征提取完成: {X_raw.shape[0]} 个样本, {X_raw.shape[1]} 维特征")
    
    # ========== 阶段2: 逻辑回归训练 ==========
    logger.info("阶段2: 训练逻辑回归分类器...")
    from sklearn.linear_model import LogisticRegression
    
    # 划分训练集和测试集
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
        X_raw, y_raw, test_size=CONFIG['test_size'], random_state=CONFIG['seed'], stratify=y_raw
    )
    
    # 训练逻辑回归
    lr_clf = LogisticRegression(max_iter=1000, random_state=CONFIG['seed'])
    lr_clf.fit(X_train_lr, y_train_lr)
    
    # 评估逻辑回归
    y_pred_lr = lr_clf.predict(X_test_lr)
    lr_acc = accuracy_score(y_test_lr, y_pred_lr)
    lr_f1 = f1_score(y_test_lr, y_pred_lr, average='macro')
    logger.info(f"逻辑回归结果: 准确率={lr_acc:.4f}, F1={lr_f1:.4f}")
    
    # 保存逻辑回归模型和标签映射
    lr_path = os.path.join(CONFIG['save_dir'], 'logistic_regression_model.pkl')
    joblib.dump(lr_clf, lr_path)
    
    id2label = {v: k for k, v in label2id.items()}
    label_mapping_path = os.path.join(CONFIG['save_dir'], 'label_mapping.pkl')
    joblib.dump(id2label, label_mapping_path)
    
    # 保存文件名列表
    filename_path = os.path.join(CONFIG['save_dir'], 'filenames.pkl')
    joblib.dump(filenames, filename_path)
    
    logger.info(f"逻辑回归模型已保存到: {lr_path}")
    
    # 保存逻辑回归结果
    with open(os.path.join(CONFIG['save_dir'], 'logistic_regression_results.txt'), 'w') as f:
        f.write(f"逻辑回归测试结果\n")
        f.write(f"准确率: {lr_acc:.4f}\n")
        f.write(f"F1分数: {lr_f1:.4f}\n")
        f.write(f"混淆矩阵:\n{confusion_matrix(y_test_lr, y_pred_lr)}\n")
    
    # 获取所有样本的概率向量
    logger.info("生成概率向量...")
    X_prob = lr_clf.predict_proba(X_raw)  # (n_samples, 2)
    
    # ========== 阶段3: 1D-QC-CNN训练 ==========
    logger.info("阶段3: 训练1D-QC-CNN...")
    
    # 划分数据集（使用原始标签，但特征是概率向量）
    indices = np.arange(len(X_prob))
    train_idx, val_test_idx = train_test_split(
        indices, test_size=1-CONFIG['split'][0], random_state=CONFIG['seed'], stratify=y_raw
    )
    val_idx, test_idx = train_test_split(
        val_test_idx, 
        test_size=CONFIG['split'][2]/(CONFIG['split'][1]+CONFIG['split'][2]), 
        random_state=CONFIG['seed'], 
        stratify=y_raw[val_test_idx]
    )
    
    # 创建数据集
    train_set = ProbabilityVectorDataset(X_prob[train_idx], y_raw[train_idx])
    val_set = ProbabilityVectorDataset(X_prob[val_idx], y_raw[val_idx])
    test_set = ProbabilityVectorDataset(X_prob[test_idx], y_raw[test_idx])
    
    # 计算归一化参数（仅使用训练集）
    tmp_loader = DataLoader(train_set, batch_size=64, shuffle=False)
    mean, std = 0., 0.
    n = 0
    for x, _ in tmp_loader:
        x = x.float()
        mean += x.sum().item()
        std += (x ** 2).sum().item()
        n += x.numel()
    mean /= n
    std = (std / n - mean ** 2) ** 0.5
    
    # 保存归一化参数到配置
    CONFIG['normalize_mean'] = mean
    CONFIG['normalize_std'] = std
    
    # 应用归一化
    normalize = Normalize1D(mean, std)
    train_set.transform = normalize
    val_set.transform = normalize
    test_set.transform = normalize
    
    # 创建数据加载器
    train_loader = DataLoader(train_set, CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, CONFIG['batch_size'], shuffle=False)
    
    # 初始化模型
    model = BinaryQuantumCNN(1, CONFIG['num_qubits']).to(device)
    model.config = CONFIG
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # 训练日志
    csv_log = os.path.join(CONFIG['save_dir'], 'training_metrics.csv')
    with open(csv_log, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'train_loss', 'val_loss', 
            'train_acc', 'train_prec', 'train_rec', 'train_f1', 'train_spec',
            'val_acc', 'val_prec', 'val_rec', 'val_f1', 'val_spec', 'val_cm'
        ])
    
    # 训练循环 - 修改为保存最低验证损失的模型
    best_val_loss = float('inf')  # 初始化为无穷大
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss, train_pred, train_true = 0., [], []
        
        for x, y in tqdm(train_loader, desc=f'Epoch{epoch + 1}'):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pred.extend(out.argmax(1).cpu().tolist())
            train_true.extend(y.cpu().tolist())
        
        # 验证
        model.eval()
        val_loss, val_pred, val_true = 0., [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item()
                val_pred.extend(out.argmax(1).cpu().tolist())
                val_true.extend(y.cpu().tolist())
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 计算指标
        tr_met = calc_metrics(train_true, train_pred)
        val_met = calc_metrics(val_true, val_pred)
        scheduler.step(val_met['f1'])
        
        logger.info(
            f"Epoch{epoch + 1}: "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} "
            f"val_acc={val_met['accuracy']:.4f} "
            f"val_f1={val_met['f1']:.4f}"
        )
        
        # 保存最佳模型 - 基于最低验证损失
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,  # 保存验证损失
                'val_f1': val_met['f1'],    # 也保存F1分数供参考
                'config': CONFIG
            }, os.path.join(CONFIG['save_dir'], 'best_model.pth'))
            logger.info(f"  --> 保存最佳模型 (val_loss={best_val_loss:.4f}, val_f1={val_met['f1']:.4f})")
        
        # 记录到CSV
        with open(csv_log, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                avg_train_loss, avg_val_loss,
                tr_met['accuracy'], tr_met['precision'], tr_met['recall'], tr_met['f1'], tr_met['specificity'],
                val_met['accuracy'], val_met['precision'], val_met['recall'], val_met['f1'], val_met['specificity'],
                str(val_met['cm'].tolist())
            ])
    
    # ========== 阶段4: 测试模型 ==========
    logger.info("阶段4: 测试模型...")
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(CONFIG['save_dir'], 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_pred, test_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            test_pred.extend(model(x).argmax(1).cpu().tolist())
            test_true.extend(y.cpu().tolist())
    
    tst_met = calc_metrics(test_true, test_pred)
    
    logger.info("\n" + "="*50)
    logger.info("1D-QC-CNN 最终测试结果:")
    logger.info(f"最佳epoch: {checkpoint['epoch']}")
    logger.info(f"验证损失: {checkpoint['val_loss']:.4f}")
    logger.info(f"验证F1: {checkpoint['val_f1']:.4f}")
    logger.info(f"测试准确率: {tst_met['accuracy']:.4f}")
    logger.info(f"测试精确率: {tst_met['precision']:.4f}")
    logger.info(f"测试召回率: {tst_met['recall']:.4f}")
    logger.info(f"测试F1分数: {tst_met['f1']:.4f}")
    logger.info(f"测试特异性: {tst_met['specificity']:.4f}")
    logger.info(f"混淆矩阵:\n{tst_met['cm']}")
    logger.info("="*50)
    
    # 保存测试结果
    with open(os.path.join(CONFIG['save_dir'], 'test_results.txt'), 'w') as f:
        f.write(f"1D-QC-CNN 测试结果\n")
        f.write(f"最佳epoch: {checkpoint['epoch']}\n")
        f.write(f"验证损失: {checkpoint['val_loss']:.4f}\n")
        f.write(f"验证F1: {checkpoint['val_f1']:.4f}\n")
        f.write(f"测试准确率: {tst_met['accuracy']:.4f}\n")
        f.write(f"测试精确率: {tst_met['precision']:.4f}\n")
        f.write(f"测试召回率: {tst_met['recall']:.4f}\n")
        f.write(f"测试F1分数: {tst_met['f1']:.4f}\n")
        f.write(f"测试特异性: {tst_met['specificity']:.4f}\n")
        f.write(f"混淆矩阵:\n{tst_met['cm']}\n")
    
    # ========== 阶段5: 按照原始类别结构提取和保存量子特征 ==========
    logger.info("阶段5: 按照原始类别结构提取和保存量子特征...")
    
    # 使用训练好的模型，按照原始类别结构提取量子特征
    extract_quantum_features_by_original_structure(
        model, device, CONFIG['text_data_dir'], CONFIG['feat_dir'], CONFIG
    )
    
    # ========== 阶段6: 生成特征统计信息 ==========
    logger.info("阶段6: 生成特征统计信息...")
    
    def compute_feature_stats(feat_dir):
        """计算特征统计信息"""
        stats = {}
        
        # 遍历类别目录
        for cls_name in ['non_rumor', 'rumor']:
            cls_dir = os.path.join(feat_dir, cls_name)
            if os.path.exists(cls_dir) and os.path.isdir(cls_dir):
                all_features = []
                npy_files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
                
                for npy_file in npy_files:
                    feat = np.load(os.path.join(cls_dir, npy_file))
                    all_features.append(feat)
                
                if all_features:
                    all_features = np.array(all_features)
                    stats[cls_name] = {
                        'mean': all_features.mean(axis=0).tolist(),
                        'std': all_features.std(axis=0).tolist(),
                        'min': all_features.min(axis=0).tolist(),
                        'max': all_features.max(axis=0).tolist(),
                        'samples': len(all_features)
                    }
        return stats
    
    feat_stats = compute_feature_stats(CONFIG['feat_dir'])
    
    # 保存统计信息
    stats_path = os.path.join(CONFIG['save_dir'], 'feature_statistics_by_class.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(feat_stats, f, indent=4, ensure_ascii=False)
    
    logger.info("特征统计信息:")
    for cls_name, stat in feat_stats.items():
        logger.info(f"  {cls_name}: {stat['samples']} 个样本")
    
    # 保存配置
    config_save_path = os.path.join(CONFIG['save_dir'], 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    logger.info(f"配置文件已保存到: {config_save_path}")
    
    logger.info("="*50)
    logger.info("所有流程完成!")
    logger.info(f"模型和日志保存在: {CONFIG['save_dir']}")
    logger.info(f"量子特征（按原始类别）保存在: {CONFIG['feat_dir']}")
    logger.info("="*50)

if __name__ == '__main__':
    run_bert_qcnn_pipeline()