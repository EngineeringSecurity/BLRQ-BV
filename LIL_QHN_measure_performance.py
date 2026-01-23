# measure_lil_qhn_simple.py
"""
简化版LIL_QHN模型性能测量
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
import json

sys.path.append('.')

def measure_lil_qhn_simple():
    """简化版LIL_QHN模型性能测量"""
    
    model_path = '/data/coding/saving/Qcnn_twitter16txt/bert_twitter16txt_CNN1D_Quantum_Results/best_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    num_runs = 100
    
    print(f"使用设备: {device}")
    print(f"模型路径: {model_path}")
    
    try:
        # 1. 导入模型
        from LIL_QHN import BinaryQuantumCNN
        
        # 2. 加载检查点
        checkpoint = torch.load(model_path, map_location=device)
        
        # 3. 从检查点推断模型参数
        # 查看模型状态字典的键
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"状态字典键: {list(state_dict.keys())[:5]}...")  # 只显示前5个
            
            # 尝试推断参数
            # 查找cnn_features层
            cnn_keys = [k for k in state_dict.keys() if 'cnn_features' in k]
            if cnn_keys:
                # 获取第一个卷积层的权重形状
                for k in cnn_keys:
                    if 'weight' in k and 'cnn_features.0' in k:
                        weight_shape = state_dict[k].shape
                        print(f"CNN权重形状: {weight_shape}")
                        in_channels = weight_shape[1]  # 输入通道数
                        break
            else:
                in_channels = 1  # 默认
            
            # 查找量子分类器
            quantum_keys = [k for k in state_dict.keys() if 'quantum_classifier' in k]
            num_qubits = 4  # 默认
            for k in quantum_keys:
                if 'num_qubits' in k or 'q_params' in k:
                    # 根据参数形状推断量子比特数
                    if 'q_params' in k:
                        param_len = len(state_dict[k])
                        # q_params通常是 num_qubits * 3
                        if param_len % 3 == 0:
                            num_qubits = param_len // 3
                            break
        else:
            print("警告: 检查点中没有model_state_dict")
            in_channels = 1
            num_qubits = 4
        
        print(f"推断参数: in_channels={in_channels}, num_qubits={num_qubits}")
        
        # 4. 初始化模型
        model = BinaryQuantumCNN(
            in_channels=in_channels,
            num_qubits=num_qubits
        ).to(device)
        
        # 5. 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ 模型加载成功")
        else:
            # 尝试直接加载
            model.load_state_dict(checkpoint)
            print("✓ 模型加载成功（直接加载）")
        
        model.eval()
        
        # 6. 创建模拟输入
        seq_len = 2  # 二分类概率向量
        dummy_input = torch.randn(batch_size, in_channels, seq_len).to(device)
        dummy_input = torch.nn.functional.softmax(dummy_input, dim=2)
        
        # 7. 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数量: {total_params:,} ({total_params/1e6:.4f}M)")
        
        # 8. 测量推理时间
        print(f"\n测量推理时间 ({num_runs} 次)...")
        
        # 预热
        for _ in range(10):
            _ = model(dummy_input)
        
        # 正式测量
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = 1000 / avg_time
        
        print(f"平均推理时间: {avg_time:.4f}ms")
        print(f"时间标准差: {std_time:.4f}ms")
        print(f"吞吐量: {throughput:.2f} samples/s")
        
        # 9. 测量内存使用（GPU）
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            initial = torch.cuda.memory_allocated()
            with torch.no_grad():
                _ = model(dummy_input)
            peak = torch.cuda.max_memory_allocated()
            
            memory_used = (peak - initial) / 1e6  # MB
            print(f"推理内存使用: {memory_used:.2f}MB")
        
        # 10. 模型文件大小
        file_size = os.path.getsize(model_path) / 1e6
        print(f"模型文件大小: {file_size:.2f}MB")
        
        # 11. 保存结果
        results = {
            'Model Params (M)': total_params / 1e6,
            'Inference Time (ms) - mean': avg_time,
            'Inference Time (ms) - std': std_time,
            'Throughput (sam/s)': throughput,
            'Num Qubits': num_qubits,
            'Input Channels': in_channels,
            'Module Size (MB)': file_size,
        }
        
        if device == 'cuda':
            results['Inference Memory (MB)'] = memory_used
        
        # 保存到CSV
        import csv
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f'lil_qhn_simple_metrics_{timestamp}.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Unit'])
            for key, value in results.items():
                writer.writerow([key, value, ''])
        
        print(f"\n结果已保存到: {csv_path}")
        
        return results
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    measure_lil_qhn_simple()