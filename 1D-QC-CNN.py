# -*- coding: utf-8 -*-
"""
1D-QuantumCNN  不截断、不补零版本
输入：原来 .npy 里多长就多长
量子：4 qubit
"""
import os, csv, random, joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

# ========================== 量子层 ==========================
import pennylane as qml

class BinaryQuantumClassifier(nn.Module):
    def __init__(self, input_dim, num_qubits=4):
        super().__init__()
        self.num_qubits = num_qubits
        self.classical_to_quantum = nn.Linear(input_dim, num_qubits * 2)  # 8
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
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # 保持长度
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)   # 输出 32×1
        )
        self.quantum_classifier = BinaryQuantumClassifier(32, num_qubits)

    def forward(self, x):
        feat = self.cnn_features(x).flatten(1)   # (B, 32)
        return self.quantum_classifier(feat)

# ========================== 数据集 ==========================
class BERTFeatureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for c in self.classes:
            for f in os.listdir(os.path.join(data_dir, c)):
                if f.endswith('.npy'):
                    self.samples.append((os.path.join(data_dir, c, f), self.class_to_idx[c]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = np.load(path)
        if feat.ndim == 2: feat = feat.flatten()
        feat = torch.from_numpy(feat).unsqueeze(0).float()  # (1, L)  原来多长就多长
        if self.transform: feat = self.transform(feat)
        return feat, label


class Normalize1D:
    def __init__(self, mean, std): self.mean, self.std = mean, std
    def __call__(self, x): return (x - self.mean) / self.std


# ========================== 工具 ==========================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def calc_metrics(y_true, y_pred):
    """计算指标：微平均准确率、宏精确度、宏召回率、宏F1"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    return dict(
        accuracy=accuracy_score(y_true, y_pred),  # 微平均准确率
        precision=precision_score(y_true, y_pred, average='macro', zero_division=0),  # 宏精确度
        recall=recall_score(y_true, y_pred, average='macro', zero_division=0),  # 宏召回率
        f1=f1_score(y_true, y_pred, average='macro', zero_division=0),  # 宏F1
        specificity=tn / (tn + fp) if (tn + fp) > 0 else 0.,  # 特异性（可选保留）
        cm=cm
    )


def extract_quantum_features(model, dataset, device, save_dir, batch_size=20):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    for c in dataset.classes:
        os.makedirs(os.path.join(save_dir, c), exist_ok=True)
    circ = model.quantum_classifier._circuit()
    with torch.no_grad():
        for b, (x, _) in enumerate(tqdm(loader, desc='Extract Q-features')):
            x = x.to(device)
            cnn_feat = model.cnn_features(x).flatten(1)
            compressed = model.quantum_classifier.classical_to_quantum(cnn_feat).view(x.size(0), 2, 4)
            qvec = torch.zeros(x.size(0), 4, device=device)
            for i in range(x.size(0)):
                qvec[i] = torch.stack([torch.tensor(circ(compressed[i, reuse], model.quantum_classifier.q_params),
                                                  dtype=torch.float32, device=x.device)
                                       for reuse in range(2)]).mean(0)
            for i in range(x.size(0)):
                idx = b * batch_size + i
                if idx >= len(dataset.samples): continue
                path, label = dataset.samples[idx]
                np.save(os.path.join(save_dir, dataset.classes[label], os.path.basename(path)),
                        qvec[i].cpu().numpy())


# ========================== 主流程 ==========================
CONFIG = {
    'data_dir': "/data/coding/bert_twitter16txt_npy",  # <-- 硬编码路径
    'split': [0.70, 0.15, 0.15],
    'seed': 42,
    'batch_size': 20,
    'epochs': 10,
    'lr': 1e-4,
    'num_qubits': 4,
    'save_dir': "Qcnn_twitter16txt_pheme/bert_twitter16txt_CNN1D_Quantum_Results",
    'feat_dir': "Qcnn_twitter16txt_pheme/bert_twitter16txt_CNN1D_Quantum_Features"
}


def run():
    set_seed(CONFIG['seed'])
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['feat_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    full_set = BERTFeatureDataset(CONFIG['data_dir'])
    assert len(full_set.classes) == 2, '需要二分类数据'
    cls_idx = {l: [] for l in range(len(full_set.classes))}
    for idx, (path, label) in enumerate(full_set.samples): cls_idx[label].append(idx)
    train_idx, val_idx, test_idx = [], [], []
    for indices in cls_idx.values():
        n = len(indices)
        a, b = int(CONFIG['split'][0] * n), int(sum(CONFIG['split'][:2]) * n)
        train_idx.extend(indices[:a])
        val_idx.extend(indices[a:b])
        test_idx.extend(indices[b:])
    train_set = Subset(full_set, train_idx)
    val_set   = Subset(full_set, val_idx)
    test_set  = Subset(full_set, test_idx)

    # 计算整个训练集的均值方差
    tmp_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=4)
    mean, std = 0., 0.; n = 0
    for x, _ in tmp_loader:
        x = x.float()
        mean += x.sum().item(); std += (x ** 2).sum().item(); n += x.numel()
    mean /= n; std = (std / n - mean ** 2) ** 0.5
    full_set.transform = Normalize1D(mean, std)

    train_loader = DataLoader(train_set, CONFIG['batch_size'], shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_set,   CONFIG['batch_size'], shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set,  CONFIG['batch_size'], shuffle=False, num_workers=4)

    model = BinaryQuantumCNN(1, CONFIG['num_qubits']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    csv_log = os.path.join(CONFIG['save_dir'], 'metrics.csv')
    with open(csv_log, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_loss', 'val_loss', 'train_acc', 'train_prec', 'train_rec', 'train_f1', 'train_spec',
             'val_acc', 'val_prec', 'val_rec', 'val_f1', 'val_spec', 'cm'])
    best_f1 = 0.
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss, train_pred, train_true = 0., [], []
        for x, y in tqdm(train_loader, desc=f'Epoch{epoch + 1}'):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            train_loss += loss.item()
            train_pred.extend(out.argmax(1).cpu().tolist())
            train_true.extend(y.cpu().tolist())
        model.eval()
        val_loss, val_pred, val_true = 0., [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item()
                val_pred.extend(out.argmax(1).cpu().tolist())
                val_true.extend(y.cpu().tolist())
        tr_met, val_met = calc_metrics(train_true, train_pred), calc_metrics(val_true, val_pred)
        scheduler.step(val_met['f1'])
        print(f"\nEpoch{epoch + 1}: train_loss={train_loss / len(train_loader):.4f} "
              f"val_loss={val_loss / len(val_loader):.4f}  val_acc={val_met['accuracy']:.4f}  val_f1={val_met['f1']:.4f}")
        if val_met['f1'] > best_f1:
            best_f1 = val_met['f1']
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best.pth'))
            print(f"  --> best model saved (f1={best_f1:.4f})")
        with open(csv_log, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1,
                                    train_loss / len(train_loader), val_loss / len(val_loader),
                                    tr_met['accuracy'], tr_met['precision'], tr_met['recall'], tr_met['f1'],
                                    tr_met['specificity'],
                                    val_met['accuracy'], val_met['precision'], val_met['recall'], val_met['f1'],
                                    val_met['specificity'],
                                    str(val_met['cm'].tolist())])

    model.load_state_dict(torch.load(os.path.join(CONFIG['save_dir'], 'best.pth'), map_location=device))
    model.eval()
    test_pred, test_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            test_pred.extend(model(x).argmax(1).cpu().tolist())
            test_true.extend(y.cpu().tolist())
    tst_met = calc_metrics(test_true, test_pred)
    print("\nTest metrics:")
    print(f"Accuracy (micro): {tst_met['accuracy']:.4f}")
    print(f"Precision (macro): {tst_met['precision']:.4f}")
    print(f"Recall (macro): {tst_met['recall']:.4f}")
    print(f"F1-Score (macro): {tst_met['f1']:.4f}")
    if 'specificity' in tst_met:
        print(f"Specificity: {tst_met['specificity']:.4f}")

    print("\nExtracting quantum features ...")
    extract_quantum_features(model, full_set, device, CONFIG['feat_dir'], CONFIG['batch_size'])
    print("All done!")


if __name__ == '__main__':
    run()