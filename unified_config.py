# unified_config.py
"""
统一配置管理模块
所有模型都从此模块获取配置
"""
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import datetime


@dataclass
class UnifiedConfig:
    """统一配置类，管理所有模块的配置"""
    
    # ========== 项目基本信息 ==========
    project_name: str = "BiLSTM_LIL_QHN_T_resnet_联合谣言检测系统"
    timestamp: str = ""
    output_root: str = "/data/coding/saving/combined_results"
    
    # ========== 数据路径配置 ==========
    data_dir: str = "/data/coding/DateSet-Twitter/twitter16txt"
    image_dir: str = "/data/coding/DateSet-Twitter/twitter16txt-image"
    
    # ========== BiLSTM 配置 ==========npy2，确认过了
    bilstm_batch_size: int = 32
    bilstm_embedding_dim: int = 300
    bilstm_hidden_dim: int = 512
    bilstm_output_dim: int = 2
    bilstm_n_layers: int = 2
    bilstm_dropout: float = 0.5
    bilstm_max_len: int = 500
    bilstm_epochs: int = 20
    bilstm_lr: float = 1e-3
    bilstm_min_freq: int = 2
    bilstm_save_dir: str = "/data/coding/saving/bilstm_twitter16txt"
    bilstm_model_name: str = "bilstm_standalone.pth"
    bilstm_features_dir: str = "twitter_bilstm_features"
    bilstm_results_csv: str = "training_results.csv"
    bilstm_metrics_csv: str = "detailed_metrics.csv"
    bilstm_loss_plot: str = "loss_curve.png"
    bilstm_config_file: str = "training_config.json"
    
    # ========== LIL_QHN 配置 ==========npy1
    lil_qhn_bert_model: str = "bert-base-uncased"
    lil_qhn_max_length: int = 512
    lil_qhn_split: List[float] = field(default_factory=lambda: [0.70, 0.15, 0.15])
    lil_qhn_batch_size: int = 20
    lil_qhn_epochs: int = 10
    lil_qhn_lr: float = 1e-4
    lil_qhn_num_qubits: int = 4
    lil_qhn_test_size: float = 0.3
    lil_qhn_seed: int = 42
    lil_qhn_save_dir: str = "/data/coding/saving/Qcnn_twitter16txt/bert_twitter16txt_CNN1D_Quantum_Results"
    lil_qhn_feat_dir: str = "/data/coding/saving/Qcnn_twitter16txt/bert_twitter16txt_CNN1D_Quantum_Features"
    
    # ========== T_resnet 配置 ==========
    t_resnet_batch_size: int = 32
    t_resnet_num_epochs: int = 20
    t_resnet_learning_rate: float = 0.001
    t_resnet_patience: int = 7
    t_resnet_feature_dim: int = 512
    t_resnet_output_dir: str = "/data/coding/saving/t_resnet_results"
    t_resnet_model_save_path: str = "best_model.pth"
    
    # ========== 执行配置 ==========
    run_bilstm: bool = True
    run_lil_qhn: bool = True
    run_t_resnet: bool = True
    sequential_execution: bool = True
    execution_order: List[str] = field(default_factory=lambda: ["bilstm", "lil_qhn", "t_resnet"])
    create_log: bool = True
    check_paths: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 确保输出目录存在
        os.makedirs(self.output_root, exist_ok=True)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"警告: 配置中未找到参数 '{key}'，跳过")
    
    def update_from_json(self, json_path: str):
        """从JSON文件更新配置"""
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            self.update_from_dict(config_dict)
            print(f"从 {json_path} 加载配置")
        else:
            print(f"警告: JSON配置文件不存在: {json_path}")
    
    def save_to_json(self, json_path: str = None):
        """保存配置到JSON文件"""
        if json_path is None:
            json_path = os.path.join(self.output_root, f"unified_config_{self.timestamp}.json")
        
        # 将dataclass转换为字典
        config_dict = self.to_dict()
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        
        print(f"统一配置已保存到: {json_path}")
        return json_path
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        import dataclasses
        return {field.name: getattr(self, field.name) 
                for field in dataclasses.fields(self)}
    
    def get_bilstm_save_dir(self):
        """获取BiLSTM保存目录"""
        return self.bilstm_save_dir
    
    def get_lil_qhn_save_dir(self):
        """获取LIL_QHN保存目录"""
        return self.lil_qhn_save_dir
    
    def get_t_resnet_save_dir(self):
        """获取T_resnet保存目录"""
        return self.t_resnet_output_dir
    
    def get_bilstm_features_dir(self):
        """获取BiLSTM特征目录"""
        return os.path.join(self.bilstm_save_dir, self.bilstm_features_dir)
    
    def get_lil_qhn_features_dir(self):
        """获取LIL_QHN特征目录"""
        return self.lil_qhn_feat_dir
    
    def check_all_paths(self):
        """检查所有必要的路径"""
        paths_to_check = [
            ("数据目录", self.data_dir),
            ("图像目录", self.image_dir),
            ("输出根目录", self.output_root),
        ]
        
        missing_paths = []
        for name, path in paths_to_check:
            if path and not os.path.exists(path):
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            print("警告: 以下路径不存在:")
            for path_info in missing_paths:
                print(f"  - {path_info}")
            
            # 询问是否创建缺失的目录
            response = input("是否创建缺失的目录？(y/n): ")
            if response.lower() == 'y':
                for _, path in paths_to_check:
                    if path:
                        os.makedirs(path, exist_ok=True)
                print("已创建缺失的目录")
                return True
            else:
                return False
        return True
    
    def print_summary(self):
        """打印配置摘要"""
        print("="*60)
        print("统一配置摘要")
        print("="*60)
        print(f"项目: {self.project_name}")
        print(f"时间戳: {self.timestamp}")
        print(f"输出根目录: {self.output_root}")
        print(f"数据目录: {self.data_dir}")
        print(f"图像目录: {self.image_dir}")
        print()
        print("执行配置:")
        print(f"  顺序: {' -> '.join(self.execution_order)}")
        print(f"  BiLSTM: {'启用' if self.run_bilstm else '禁用'}")
        print(f"  LIL_QHN: {'启用' if self.run_lil_qhn else '禁用'}")
        print(f"  T_resnet: {'启用' if self.run_t_resnet else '禁用'}")
        print("="*60)


# 创建全局配置实例
_config_instance = None


def get_config() -> UnifiedConfig:
    """获取全局配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = UnifiedConfig()
    return _config_instance


def update_config_from_file(config_file: str):
    """从配置文件更新全局配置"""
    config = get_config()
    config.update_from_json(config_file)


def save_current_config():
    """保存当前配置"""
    config = get_config()
    return config.save_to_json()


def create_default_config():
    """创建默认配置"""
    config = get_config()
    config.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config_path = config.save_to_json()
    return config_path