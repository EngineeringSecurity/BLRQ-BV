# 文件名: measure_performance.py
import torch
import torch.nn as nn
import numpy as np
import time
import os
from thop import profile
from thop import clever_format
import psutil
import gc
import json
from tqdm import tqdm

def measure_t_resnet_metrics(model_path, device='cuda', batch_size=1, num_runs=100):
    """
    测量T_resnet模型的性能指标
    
    Args:
        model_path: 预训练模型路径
        device: 设备 ('cuda' 或 'cpu')
        batch_size: 批处理大小
        num_runs: 推理次数
    """
    
    print(f"加载T_resnet模型: {model_path}")
    
    try:
        # 尝试从模型路径中提取配置信息
        config_path = None
        if os.path.exists(os.path.join(os.path.dirname(model_path), 't_resnet_config.json')):
            config_path = os.path.join(os.path.dirname(model_path), 't_resnet_config.json')
        elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(model_path)), 't_resnet_config.json')):
            config_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), 't_resnet_config.json')
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"从 {config_path} 加载配置")
        else:
            # 使用默认配置
            config = {
                'npy1_dim': 4,     # 量子特征维度
                'npy2_dim': 1024,  # BiLSTM特征维度
            }
            print("使用默认配置")
        
        # 1. 导入T_resnet模型类
        try:
            from T_resnet import MultiModalResNet18
        except ImportError:
            print("无法导入T_resnet模块，尝试从当前目录导入")
            import sys
            sys.path.append('.')
            from T_resnet import MultiModalResNet18
        
        # 2. 初始化模型
        npy1_dim = config.get('npy1_dim', 4)
        npy2_dim = config.get('npy2_dim', 1024)
        
        print(f"初始化模型: npy1_dim={npy1_dim}, npy2_dim={npy2_dim}")
        model = MultiModalResNet18(
            npy1_dim=npy1_dim,
            npy2_dim=npy2_dim,
            num_classes=2,
            feature_dim=512
        ).to(device)
        
        # 3. 加载预训练权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 尝试直接加载为模型状态字典
                model.load_state_dict(checkpoint)
            
            print("✓ 模型加载成功")
            
            # 打印模型信息
            if 'val_loss' in checkpoint:
                print(f"模型信息: val_loss={checkpoint.get('val_loss', 'N/A'):.4f}, val_f1={checkpoint.get('val_f1', 'N/A'):.4f}")
            if 'epoch' in checkpoint:
                print(f"最佳epoch: {checkpoint.get('epoch', 'N/A')}")
        else:
            print(f"✗ 模型文件不存在: {model_path}")
            return None
        
        model.eval()
        
        # 4. 创建模拟输入
        # T_resnet需要三个输入: image, npy1, npy2
        dummy_image = torch.randn(batch_size, 1, 256, 256).to(device)  # 灰度图像
        dummy_npy1 = torch.randn(batch_size, npy1_dim).to(device)      # 量子特征
        dummy_npy2 = torch.randn(batch_size, npy2_dim).to(device)      # BiLSTM特征
        
        results = {}
        
        # 5. 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results['Model Params (M)'] = total_params / 1e6
        results['Trainable Params (M)'] = trainable_params / 1e6
        
        print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        # 6. 计算FLOPs - 修复版本
        try:
            # 创建一个包装函数来适应thop的要求
            class ModelWrapper(nn.Module):
                def __init__(self, model):
                    super(ModelWrapper, self).__init__()
                    self.model = model
                
                def forward(self, x):
                    image, npy1, npy2 = x
                    return self.model(image, npy1, npy2)[0]  # 只返回输出
        
            wrapper = ModelWrapper(model).to(device)
            wrapper.eval()
            
            # 使用thop计算FLOPs
            flops, params = profile(wrapper, inputs=((dummy_image, dummy_npy1, dummy_npy2),), verbose=False)
            results['FLOPs (G)'] = flops / 1e9
            results['FLOPs_formatted'] = clever_format([flops, params], "%.3f")[0]
            print(f"FLOPs: {flops/1e9:.2f}G")
        except ImportError:
            print("未安装thop，无法计算FLOPs。使用: pip install thop")
            results['FLOPs (G)'] = None
            results['FLOPs_formatted'] = "N/A"
        except Exception as e:
            print(f"计算FLOPs时出错: {e}")
            results['FLOPs (G)'] = None
            results['FLOPs_formatted'] = "N/A"
        
        # 7. 测量推理时间
        print("\n测量推理时间...")
        warmup_runs = 10
        inference_times = []
        
        # 预热
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(dummy_image, dummy_npy1, dummy_npy2)
        
        # 正式测量
        for i in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(dummy_image, dummy_npy1, dummy_npy2)
            
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
        
        # 8. 测量内存使用
        if device == 'cuda':
            # 清除GPU缓存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 测量模型加载内存
            initial_memory = torch.cuda.memory_allocated()
            
            # 进行一次推理
            with torch.no_grad():
                _ = model(dummy_image, dummy_npy1, dummy_npy2)
            
            peak_memory = torch.cuda.max_memory_allocated()
            
            results['Inference Memory (MB)'] = (peak_memory - initial_memory) / 1e6
            results['Peak Memory (MB)'] = peak_memory / 1e6
            
            print(f"推理内存: {results['Inference Memory (MB)']:.2f}MB")
            print(f"峰值内存: {results['Peak Memory (MB)']:.2f}MB")
        else:
            # CPU内存测量
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            with torch.no_grad():
                _ = model(dummy_image, dummy_npy1, dummy_npy2)
            
            final_memory = process.memory_info().rss
            results['Inference Memory (MB)'] = (final_memory - initial_memory) / 1e6
            results['Peak Memory (MB)'] = None
        
        # 9. 测量模型文件大小
        if os.path.exists(model_path):
            model_size_bytes = os.path.getsize(model_path)
            results['Module Size (MB)'] = model_size_bytes / 1e6
            print(f"模型文件大小: {model_size_bytes/1e6:.2f}MB")
        
        # 10. 计算参数效率
        if results.get('FLOPs (G)') is not None:
            results['Params/FLOPs Ratio'] = results['Model Params (M)'] / results['FLOPs (G)']
        
        return results, model
    
    except Exception as e:
        print(f"测量T_resnet模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def print_metrics_table(results):
    """以表格形式打印指标"""
    print("\n" + "="*80)
    print("T_resnet 模型性能指标")
    print("="*80)
    
    # 创建表格
    print("\n{:<30} {:<20} {:<10}".format("指标", "数值", "单位"))
    print("-" * 65)
    
    # 安全地获取每个指标的值
    def safe_get(key, default='N/A'):
        value = results.get(key, default)
        if value is None:
            return default
        return value
    
    metrics = [
        ("Model Params", f"{safe_get('Model Params (M)')}", "M"),
        ("Trainable Params", f"{safe_get('Trainable Params (M)')}", "M"),
        ("FLOPs", f"{safe_get('FLOPs (G)')}", "G"),
        ("Inference Time (mean)", f"{safe_get('Inference Time (ms) - mean', 'N/A')}", "ms"),
        ("Inference Time (std)", f"{safe_get('Inference Time (ms) - std', 'N/A')}", "ms"),
        ("Inference Memory", f"{safe_get('Inference Memory (MB)', 'N/A')}", "MB"),
        ("Peak Memory", f"{safe_get('Peak Memory (MB)', 'N/A')}", "MB"),
        ("Module Size", f"{safe_get('Module Size (MB)', 'N/A')}", "MB"),
        ("Throughput", f"{safe_get('Throughput (sam/s)', 'N/A')}", "sam/s"),
    ]
    
    for name, value, unit in metrics:
        print("{:<30} {:<20} {:<10}".format(name, value, unit))
    
    # 打印额外信息
    if 'FLOPs_formatted' in results and results['FLOPs_formatted'] != "N/A":
        print(f"\n详细FLOPs: {results['FLOPs_formatted']}")
    
    if 'Inference Time (ms) - std' in results and results['Inference Time (ms) - std'] is not None:
        print(f"推理时间标准差: ±{results['Inference Time (ms) - std']:.2f} ms")
        print(f"推理时间范围: {results['Inference Time (ms) - min']:.2f} - {results['Inference Time (ms) - max']:.2f} ms")
    
    if 'Params/FLOPs Ratio' in results and results['Params/FLOPs Ratio'] is not None:
        print(f"参数效率 (Params/FLOPs): {results['Params/FLOPs Ratio']:.4f}")
    
    print("="*80)

def compare_with_other_batch_sizes(model_path, device='cuda'):
    """比较不同批处理大小下的性能"""
    print("\n比较不同批处理大小下的性能...")
    print("-" * 60)
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    for bs in batch_sizes:
        print(f"\n批处理大小: {bs}")
        results, _ = measure_t_resnet_metrics(
            model_path, device, batch_size=bs, num_runs=30
        )
        
        if results:
            print(f"  推理时间: {results.get('Inference Time (ms) - mean', 'N/A'):.2f} ms")
            print(f"  吞吐量: {results.get('Throughput (sam/s)', 'N/A'):.2f} sam/s")
            if 'Inference Memory (MB)' in results and results['Inference Memory (MB)'] is not None:
                print(f"  推理内存: {results['Inference Memory (MB)']:.2f} MB")
            if 'FLOPs (G)' in results and results['FLOPs (G)'] is not None:
                print(f"  FLOPs: {results['FLOPs (G)']:.2f}G")
        else:
            print(f"  测量失败")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测量模型性能指标')
    parser.add_argument('--model-path', type=str, default='/data/coding/saving/t_resnet_results/t_resnet_model.pth',
                       help='模型文件路径')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='批处理大小')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='推理次数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda 或 cpu)')
    parser.add_argument('--compare-batch', action='store_true',
                       help='比较不同批处理大小的性能')
    
    args = parser.parse_args()
    
    # 选择设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    print(f"使用设备: {device}")
    print(f"模型路径: {args.model_path}")
    print(f"批处理大小: {args.batch_size}")
    print(f"测试次数: {args.num_runs}")
    
    # 测量T_resnet模型
    results, model = measure_t_resnet_metrics(
        args.model_path, device, args.batch_size, args.num_runs
    )
    
    if results:
        # 打印指标表格
        print_metrics_table(results)
        
        # 比较不同批处理大小
        if args.compare_batch and device == 'cuda':
            compare_with_other_batch_sizes(args.model_path, device)
        
        # 保存结果到CSV
        import csv
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f't_resnet_metrics_{timestamp}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Unit'])
            
            # 写入主要指标
            metrics_to_save = [
                ('Model_Params_M', results.get('Model Params (M)'), 'M'),
                ('Trainable_Params_M', results.get('Trainable Params (M)'), 'M'),
                ('FLOPs_G', results.get('FLOPs (G)'), 'G'),
                ('Inference_Time_ms_mean', results.get('Inference Time (ms) - mean'), 'ms'),
                ('Inference_Time_ms_std', results.get('Inference Time (ms) - std'), 'ms'),
                ('Inference_Time_ms_min', results.get('Inference Time (ms) - min'), 'ms'),
                ('Inference_Time_ms_max', results.get('Inference Time (ms) - max'), 'ms'),
                ('Inference_Memory_MB', results.get('Inference Memory (MB)'), 'MB'),
                ('Peak_Memory_MB', results.get('Peak Memory (MB)'), 'MB'),
                ('Module_Size_MB', results.get('Module Size (MB)'), 'MB'),
                ('Throughput_sam_s', results.get('Throughput (sam/s)'), 'sam/s'),
                ('Params_FLOPs_Ratio', results.get('Params/FLOPs Ratio'), ''),
            ]
            
            for metric, value, unit in metrics_to_save:
                if value is not None:
                    writer.writerow([metric, value, unit])
        
        print(f"\n指标已保存到: {csv_path}")
        
        # 保存详细报告
        report_path = f't_resnet_performance_report_{timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"T_RESNET 模型性能报告\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"报告时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型文件: {args.model_path}\n")
            f.write(f"设备: {device}\n")
            f.write(f"批处理大小: {args.batch_size}\n")
            f.write(f"测试次数: {args.num_runs}\n\n")
            
            f.write("性能指标:\n")
            f.write("-" * 60 + "\n")
            
            for key, value in results.items():
                if value is not None and 'formatted' not in key and 'ratio' not in key:
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            if 'FLOPs_formatted' in results:
                f.write(f"FLOPs_formatted: {results['FLOPs_formatted']}\n")
            
            if 'Params/FLOPs Ratio' in results and results['Params/FLOPs Ratio'] is not None:
                f.write(f"Params/FLOPs Ratio: {results['Params/FLOPs Ratio']:.4f}\n")
        
        print(f"详细报告已保存到: {report_path}")
        
        # 打印总结
        print(f"\n{'='*60}")
        print("性能总结:")
        print(f"  模型参数量: {results.get('Model Params (M)', 'N/A'):.2f}M")
        print(f"  推理时间: {results.get('Inference Time (ms) - mean', 'N/A'):.2f}ms")
        print(f"  吞吐量: {results.get('Throughput (sam/s)', 'N/A'):.2f} samples/s")
        print(f"  推理内存: {results.get('Inference Memory (MB)', 'N/A'):.2f}MB")
        print(f"{'='*60}")
        
    else:
        print("未能获取性能指标")