# measure_bilstm_standalone.py
import torch
import torch.nn as nn
import numpy as np
import time
import os
from thop import profile
from thop import clever_format
import psutil
import json
from tqdm import tqdm
import sys
import csv

sys.path.append('.')

def measure_bilstm_smart():
    """智能测量BiLSTM模型性能，自动检测词汇表大小"""
    
    # 模型路径
    model_path = '/data/coding/saving/bilstm_twitter16txt/bilstm_standalone.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    num_runs = 100
    max_len = 500
    
    print(f"使用设备: {device}")
    print(f"模型路径: {model_path}")
    print(f"批处理大小: {batch_size}")
    print(f"测试次数: {num_runs}")
    print(f"最大序列长度: {max_len}")
    
    try:
        # 1. 导入BiLSTM模型类
        from BiLSTM import BiLSTMClassifier
        
        # 2. 加载模型检查点，获取实际参数
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' not in checkpoint:
            print("错误: 检查点中没有找到model_state_dict")
            return
        
        # 从状态字典获取embedding权重形状
        embedding_weight_shape = checkpoint['model_state_dict']['embedding.weight'].shape
        actual_vocab_size = embedding_weight_shape[0]
        actual_embedding_dim = embedding_weight_shape[1]
        
        # 从状态字典获取LSTM权重形状来推断隐藏维度
        lstm_weight_shape = checkpoint['model_state_dict']['lstm.weight_ih_l0'].shape
        # LSTM权重形状: [hidden_dim*4, input_dim]
        actual_hidden_dim = lstm_weight_shape[0] // 4
        
        print(f"从模型文件推断参数:")
        print(f"  词汇表大小: {actual_vocab_size}")
        print(f"  嵌入维度: {actual_embedding_dim}")
        print(f"  隐藏维度: {actual_hidden_dim}")
        
        # 3. 初始化模型
        model = BiLSTMClassifier(
            vocab_size=actual_vocab_size,
            embedding_dim=actual_embedding_dim,
            hidden_dim=actual_hidden_dim,
            output_dim=2,
            n_layers=2,  # 根据文件名推断
            dropout=0.5
        ).to(device)
        
        # 4. 加载预训练权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✓ 模型加载成功")
        
        # 打印模型信息
        if 'val_loss' in checkpoint:
            print(f"模型信息: val_loss={checkpoint.get('val_loss', 'N/A'):.4f}, val_f1={checkpoint.get('val_f1', 'N/A'):.4f}")
        if 'epoch' in checkpoint:
            print(f"最佳epoch: {checkpoint.get('epoch', 'N/A')}")
        
        # 5. 创建模拟输入
        dummy_text = torch.randint(0, actual_vocab_size, (batch_size, max_len)).to(device)
        
        results = {}
        
        # 6. 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results['Model Params (M)'] = total_params / 1e6
        results['Trainable Params (M)'] = trainable_params / 1e6
        
        print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        # 7. 计算FLOPs
        try:
            flops, params = profile(model, inputs=(dummy_text,), verbose=False)
            results['FLOPs (G)'] = flops / 1e9
            results['FLOPs_formatted'] = clever_format([flops, params], "%.3f")[0]
            print(f"FLOPs: {flops/1e9:.2f}G")
        except Exception as e:
            print(f"计算FLOPs时出错: {e}")
            results['FLOPs (G)'] = None
            results['FLOPs_formatted'] = "N/A"
        
        # 8. 测量推理时间
        print("\n测量推理时间...")
        warmup_runs = 10
        inference_times = []
        
        # 预热
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(dummy_text)
        
        # 正式测量
        for i in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(dummy_text)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        inference_times = np.array(inference_times)
        results['Inference Time (ms) - mean'] = np.mean(inference_times)
        results['Inference Time (ms) - std'] = np.std(inference_times)
        results['Inference Time (ms) - min'] = np.min(inference_times)
        results['Inference Time (ms) - max'] = np.max(inference_times)
        
        # 吞吐量 (samples/second)
        results['Throughput (sam/s)'] = 1000 / results['Inference Time (ms) - mean']
        
        print(f"平均推理时间: {results['Inference Time (ms) - mean']:.2f}ms")
        print(f"吞吐量: {results['Throughput (sam/s)']:.2f} samples/s")
        
        # 9. 测量内存使用
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            initial_memory = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                _ = model(dummy_text)
            
            peak_memory = torch.cuda.max_memory_allocated()
            
            results['Inference Memory (MB)'] = (peak_memory - initial_memory) / 1e6
            results['Peak Memory (MB)'] = peak_memory / 1e6
            
            print(f"推理内存: {results['Inference Memory (MB)']:.2f}MB")
            print(f"峰值内存: {results['Peak Memory (MB)']:.2f}MB")
        
        # 10. 测量模型文件大小
        model_size_bytes = os.path.getsize(model_path)
        results['Module Size (MB)'] = model_size_bytes / 1e6
        print(f"模型文件大小: {model_size_bytes/1e6:.2f}MB")
        
        # 11. 计算参数效率
        if results.get('FLOPs (G)') is not None:
            results['Params/FLOPs Ratio'] = results['Model Params (M)'] / results['FLOPs (G)']
        
        # 打印结果表格
        print("\n" + "="*80)
        print("BiLSTM 模型性能指标")
        print("="*80)
        
        print("\n{:<30} {:<20} {:<10}".format("指标", "数值", "单位"))
        print("-" * 65)
        
        metrics = [
            ("Model Params", f"{results.get('Model Params (M)', 'N/A'):.2f}", "M"),
            ("Trainable Params", f"{results.get('Trainable Params (M)', 'N/A'):.2f}", "M"),
            ("FLOPs", f"{results.get('FLOPs (G)', 'N/A'):.2f}", "G"),
            ("Inference Time (mean)", f"{results.get('Inference Time (ms) - mean', 'N/A'):.2f}", "ms"),
            ("Throughput", f"{results.get('Throughput (sam/s)', 'N/A'):.2f}", "sam/s"),
            ("Inference Memory", f"{results.get('Inference Memory (MB)', 'N/A'):.2f}", "MB"),
            ("Module Size", f"{results.get('Module Size (MB)', 'N/A'):.2f}", "MB"),
        ]
        
        for name, value, unit in metrics:
            print("{:<30} {:<20} {:<10}".format(name, value, unit))
        
        # 保存结果到CSV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f'bilstm_performance_{timestamp}.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Unit'])
            
            for key, value in results.items():
                if value is not None and 'formatted' not in key and 'ratio' not in key:
                    writer.writerow([key, value, ''])
        
        print(f"\n指标已保存到: {csv_path}")
        
        return results
        
    except Exception as e:
        print(f"测量BiLSTM模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    measure_bilstm_smart()