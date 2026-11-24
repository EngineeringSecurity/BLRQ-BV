# main1.py
# ==========================================
# 主程序：BiLSTM + LIL_QHN + T_resnet 联合执行
# ==========================================
import os
import json
import argparse
import sys
from datetime import datetime

# 导入统一配置
from unified_config import get_config, update_config_from_file, save_current_config, create_default_config

def check_prerequisites():
    """检查前置条件"""
    print("检查前置条件...")
    
    config = get_config()
    
    # 检查路径
    if config.check_paths:
        if not config.check_all_paths():
            print("路径检查失败，请检查配置")
            return False
    
    # 检查依赖
    try:
        import torch
        import numpy as np
        print(f"PyTorch版本: {torch.__version__}")
        print(f"NumPy版本: {np.__version__}")
    except ImportError as e:
        print(f"缺少依赖: {e}")
        return False
    
    return True

def run_bilstm_model():
    """运行BiLSTM模型"""
    print("\n" + "="*60)
    print("开始执行 BiLSTM 模型")
    print("="*60)
    
    try:
        from BiLSTM import main_standalone
        model, feature_dim = main_standalone()
        print("BiLSTM 执行完成")
        return True, feature_dim
    except Exception as e:
        print(f"BiLSTM 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def run_lil_qhn_model():
    """运行LIL_QHN模型"""
    print("\n" + "="*60)
    print("开始执行 LIL_QHN 模型")
    print("="*60)
    
    try:
        from LIL_QHN import run_bert_qcnn_pipeline
        run_bert_qcnn_pipeline()
        print("LIL_QHN 执行完成")
        return True
    except Exception as e:
        print(f"LIL_QHN 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_t_resnet_model():
    """运行T_resnet模型"""
    print("\n" + "="*60)
    print("开始执行 T_resnet 模型")
    print("="*60)
    
    try:
        import T_resnet
        
        # 检查特征文件是否存在
        config = get_config()
        bilstm_features_dir = os.path.join(config.bilstm_save_dir, config.bilstm_features_dir)
        lil_qhn_features_dir = config.lil_qhn_feat_dir
        
        print(f"检查特征目录...")
        print(f"BiLSTM特征目录: {bilstm_features_dir}")
        print(f"LIL_QHN特征目录: {lil_qhn_features_dir}")
        
        # 检查目录是否存在
        if not os.path.exists(bilstm_features_dir):
            print(f"错误: BiLSTM特征目录不存在: {bilstm_features_dir}")
            print("请先运行BiLSTM模型")
            return False
        
        if not os.path.exists(lil_qhn_features_dir):
            print(f"错误: LIL_QHN特征目录不存在: {lil_qhn_features_dir}")
            print("请先运行LIL_QHN模型")
            return False
        
        # 检查目录结构
        print(f"\n检查目录结构...")
        
        # 检查BiLSTM特征目录结构
        bilstm_classes = [d for d in os.listdir(bilstm_features_dir) 
                         if os.path.isdir(os.path.join(bilstm_features_dir, d))]
        print(f"BiLSTM特征目录中的类别: {bilstm_classes}")
        
        # 检查LIL_QHN特征目录结构
        lil_qhn_classes = [d for d in os.listdir(lil_qhn_features_dir) 
                          if os.path.isdir(os.path.join(lil_qhn_features_dir, d))]
        print(f"LIL_QHN特征目录中的类别: {lil_qhn_classes}")
        
        # 检查.npy文件数量
        bilstm_npy_count = 0
        for cls in bilstm_classes:
            if cls in ['non_rumor', 'rumor']:
                cls_dir = os.path.join(bilstm_features_dir, cls)
                npy_files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
                bilstm_npy_count += len(npy_files)
                print(f"  BiLSTM {cls}: {len(npy_files)} 个.npy文件")
        
        lil_qhn_npy_count = 0
        for cls in lil_qhn_classes:
            if cls in ['non_rumor', 'rumor']:
                cls_dir = os.path.join(lil_qhn_features_dir, cls)
                npy_files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
                lil_qhn_npy_count += len(npy_files)
                print(f"  LIL_QHN {cls}: {len(npy_files)} 个.npy文件")
        
        # 执行T_resnet
        print(f"\n开始执行T_resnet...")
        T_resnet.main()
        print("T_resnet 执行完成")
        return True
        
    except Exception as e:
        print(f"T_resnet 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BiLSTM + LIL_QHN + T_resnet 联合谣言检测系统')
    parser.add_argument('--config', type=str, default='unified_config.json', 
                       help='统一配置文件路径 (默认: unified_config.json)')
    parser.add_argument('--create-config', action='store_true',
                       help='创建默认配置文件')
    parser.add_argument('--bilstm-only', action='store_true',
                       help='仅运行BiLSTM模型')
    parser.add_argument('--lil-qhn-only', action='store_true',
                       help='仅运行LIL_QHN模型')
    parser.add_argument('--t-resnet-only', action='store_true',
                       help='仅运行T_resnet模型')
    parser.add_argument('--skip-bilstm', action='store_true',
                       help='跳过BiLSTM模型')
    parser.add_argument('--skip-lil-qhn', action='store_true',
                       help='跳过LIL_QHN模型')
    parser.add_argument('--skip-t-resnet', action='store_true',
                       help='跳过T_resnet模型')
    parser.add_argument('--output-dir', type=str,
                       help='输出根目录 (覆盖配置文件中的设置)')
    
    args = parser.parse_args()
    
    # 处理配置文件
    if args.create_config or not os.path.exists(args.config):
        config_path = create_default_config()
        
        if args.create_config:
            print(f"配置文件已创建: {config_path}")
            print("请编辑配置文件后重新运行程序")
            return
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    update_config_from_file(args.config)
    config = get_config()
    
    # 覆盖输出目录（如果指定）
    if args.output_dir:
        config.output_root = args.output_dir
        os.makedirs(config.output_root, exist_ok=True)
    
    # 打印配置摘要
    config.print_summary()
    
    # 检查前置条件
    if not check_prerequisites():
        print("前置条件检查失败，退出")
        return
    
    # 创建运行日志
    log_file = os.path.join(config.output_root, f"execution_log_{config.timestamp}.txt")
    if config.create_log:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"BiLSTM + LIL_QHN + T_resnet 联合执行日志\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"配置文件: {args.config}\n")
            f.write(f"输出根目录: {config.output_root}\n")
            f.write("="*50 + "\n")
    
    # 根据参数调整执行配置
    if args.bilstm_only:
        config.run_bilstm = True
        config.run_lil_qhn = False
        config.run_t_resnet = False
        config.execution_order = ['bilstm']
    elif args.lil_qhn_only:
        config.run_bilstm = False
        config.run_lil_qhn = True
        config.run_t_resnet = False
        config.execution_order = ['lil_qhn']
    elif args.t_resnet_only:
        config.run_bilstm = False
        config.run_lil_qhn = False
        config.run_t_resnet = True
        config.execution_order = ['t_resnet']
    
    if args.skip_bilstm:
        config.run_bilstm = False
        config.execution_order = [m for m in config.execution_order if m != 'bilstm']
    if args.skip_lil_qhn:
        config.run_lil_qhn = False
        config.execution_order = [m for m in config.execution_order if m != 'lil_qhn']
    if args.skip_t_resnet:
        config.run_t_resnet = False
        config.execution_order = [m for m in config.execution_order if m != 't_resnet']
    
    # 执行模型
    results = {
        'bilstm_success': False,
        'lil_qhn_success': False,
        't_resnet_success': False,
        'bilstm_feature_dim': None,
        'execution_order': config.execution_order,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config_file': args.config,
        'output_root': config.output_root
    }
    
    try:
        print(f"\n执行顺序: {' -> '.join(config.execution_order)}")
        
        # 按顺序执行
        for model_name in config.execution_order:
            if model_name == 'bilstm' and config.run_bilstm:
                success, feature_dim = run_bilstm_model()
                results['bilstm_success'] = success
                results['bilstm_feature_dim'] = feature_dim
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\nBiLSTM执行: {'成功' if success else '失败'}\n")
                    
            elif model_name == 'lil_qhn' and config.run_lil_qhn:
                success = run_lil_qhn_model()
                results['lil_qhn_success'] = success
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\nLIL_QHN执行: {'成功' if success else '失败'}\n")
                    
            elif model_name == 't_resnet' and config.run_t_resnet:
                success = run_t_resnet_model()
                results['t_resnet_success'] = success
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\nT_resnet执行: {'成功' if success else '失败'}\n")
        
        results['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存执行结果
        results_file = os.path.join(config.output_root, f"execution_results_{config.timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"\n执行结果已保存到: {results_file}")
        
        # 生成最终报告
        generate_final_report(results, config)
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n执行错误: {str(e)}\n")
            f.write(traceback.format_exc())

def generate_final_report(results, config):
    """生成最终执行报告"""
    report_path = os.path.join(config.output_root, f"final_report_{config.timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("BI-LSTM + LIL-QHN + T-ResNet 联合谣言检测系统执行报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("执行概况:\n")
        f.write(f"  开始时间: {results['start_time']}\n")
        f.write(f"  结束时间: {results['end_time']}\n")
        f.write(f"  配置文件: {results['config_file']}\n")
        f.write(f"  输出根目录: {results['output_root']}\n")
        f.write(f"  执行顺序: {' -> '.join(results['execution_order'])}\n\n")
        
        f.write("各模块执行状态:\n")
        f.write(f"  BiLSTM: {'✓ 成功' if results['bilstm_success'] else '✗ 失败'}\n")
        if results['bilstm_feature_dim']:
            f.write(f"    特征维度: {results['bilstm_feature_dim']}\n")
        
        f.write(f"  LIL_QHN: {'✓ 成功' if results['lil_qhn_success'] else '✗ 失败'}\n")
        f.write(f"  T_resnet: {'✓ 成功' if results['t_resnet_success'] else '✗ 失败'}\n\n")
        
        f.write("生成的文件:\n")
        for root, dirs, files in os.walk(config.output_root):
            level = root.replace(config.output_root, '').count(os.sep)
            indent = ' ' * 2 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            sub_indent = ' ' * 2 * (level + 1)
            for file in files:
                f.write(f"{sub_indent}{file}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("报告生成完成!\n")
        f.write("="*70 + "\n")
    
    print(f"最终报告已生成: {report_path}")

if __name__ == '__main__':
    main()