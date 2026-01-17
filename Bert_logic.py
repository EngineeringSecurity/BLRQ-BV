# rumor_prob_export.py
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
np.set_printoptions(precision=4, suppress=True)


# ---------- 1. 特征提取器 ----------
class BERTFeatureExtractor:
    def __init__(self, model_path: str, max_length: int = 512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path).to(self.device).eval()
        self.max_length = max_length

    def get_cls_vector(self, text: str) -> np.ndarray:
        if not text:
            return None
        encoded = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**encoded, output_hidden_states=False)
        return out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()  # (768,)


# ---------- 2. 分类器 ----------
class RumorClassifier:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, random_state=42)
        self.label2id = None
        self.id2label = None

    # 2.1 训练
    def fit_from_texts(self, extractor: BERTFeatureExtractor, input_root: str, test_size: float = 0.2):
        logger.info("读取文本并提取特征...")
        X, y = [], []
        self.label2id = {}
        for label in sorted(os.listdir(input_root)):
            label_dir = os.path.join(input_root, label)
            # 过滤非目录、隐藏目录
            if not os.path.isdir(label_dir) or label.startswith('.'):
                continue
            if label not in self.label2id:
                self.label2id[label] = len(self.label2id)
            txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
            for fn in tqdm(txt_files, desc=f"提取 {label}"):
                with open(os.path.join(label_dir, fn), 'r', encoding='utf-8') as f:
                    vec = extractor.get_cls_vector(f.read().strip())
                if vec is not None:
                    X.append(vec)
                    y.append(self.label2id[label])
        X, y = np.array(X), np.array(y)
        self.id2label = {i: l for l, i in self.label2id.items()}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        logger.info(f"训练集 {len(X_train)} 条，测试集 {len(X_test)} 条")
        logger.info("训练 LogisticRegression...")
        self.clf.fit(X_train, y_train)

        y_pred = self.clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        wf1 = f1_score(y_test, y_pred, average='weighted')

        logger.info("="*50)
        logger.info("测试集评估")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"Weighted-F1: {wf1:.4f}")

        # 只在出现过的类别上生成报告
        labels_present = sorted(set(y_test))
        report = classification_report(
            y_test, y_pred,
            labels=labels_present,
            target_names=[self.id2label[i] for i in labels_present],
            digits=4
        )
        logger.info("\n" + report)

    # 2.2 导出概率向量
    def export_prob_vectors(self, extractor: BERTFeatureExtractor,
                            input_root: str, output_root: str):
        logger.info("导出分类概率向量...")
        os.makedirs(output_root, exist_ok=True)
        for label in sorted(os.listdir(input_root)):
            label_dir = os.path.join(input_root, label)
            if not os.path.isdir(label_dir) or label.startswith('.'):
                continue
            out_dir = os.path.join(output_root, label)
            os.makedirs(out_dir, exist_ok=True)
            for fn in tqdm([f for f in os.listdir(label_dir) if f.endswith('.txt')],
                           desc=f"概率导出 {label}"):
                with open(os.path.join(label_dir, fn), 'r', encoding='utf-8') as f:
                    vec = extractor.get_cls_vector(f.read().strip())
                if vec is None:
                    continue
                prob = self.clf.predict_proba(vec.reshape(1, -1)).squeeze()
                np.save(os.path.join(out_dir, fn.replace('.txt', '.npy')), prob.astype(np.float32))
        logger.info("概率向量已保存至 {}".format(output_root))


# ---------- 3. 主入口 ----------
def main():
    MODEL_PATH = "bert-base-uncased"   # ← 改这里
    INPUT_DIR = "/data/coding/DataSet-pheme/phemetxt"             # ← 改这里
    PROB_OUTPUT_DIR = "bert_pheme_npy"

    extractor = BERTFeatureExtractor(MODEL_PATH)
    clf = RumorClassifier()
    clf.fit_from_texts(extractor, INPUT_DIR, test_size=0.3)
    clf.export_prob_vectors(extractor, INPUT_DIR, PROB_OUTPUT_DIR)


if __name__ == "__main__":
    main()