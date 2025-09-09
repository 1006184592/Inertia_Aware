#!/usr/bin/env python3
"""
批量运行所有模型的鲁棒性对比实验
包括 adap_auto 和四个基线模型：DLinear, iTransformer, FEDformer, NBEATSx
"""

import subprocess
import sys
import argparse
import os
import time
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

def run_adap_auto_experiment(dataset, prediction_scale, args):
    """
    运行adap_auto模型实验
    """
    print("🏆 运行adap_auto模型实验...")
    
    cmd = [
        sys.executable, 'robustness_analysis_experiment.py',
        '--dataset', dataset,
        '--prediction_scale', prediction_scale,
        '--gpu', str(args.gpu),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--noise_levels'] + [str(level) for level in args.noise_levels] + [
        '--missing_ratio', str(args.missing_ratio),
        '--seasonal_mode', args.seasonal_mode,
        '--seed', str(args.seed)
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ adap_auto实验完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ adap_auto实验失败: {e}")
        return False

def run_baseline_model_experiment(model_name, dataset, prediction_scale, args):
    """
    运行单个基线模型实验
    """
    print(f"🔬 运行 {model_name} 模型实验...")
    
    cmd = [
        sys.executable, 'baseline_robustness_adapter.py',
        '--dataset', dataset,
        '--prediction_scale', prediction_scale,
        '--model', model_name,
        '--gpu', str(args.gpu),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--noise_levels'] + [str(level) for level in args.noise_levels] + [
        '--missing_ratio', str(args.missing_ratio),
        '--seed', str(args.seed)
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {model_name} 实验完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_name} 实验失败: {e}")
        return False

def collect_all_results(dataset, prediction_scale):
    """
    收集所有模型的实验结果
    """
    print("📊 收集所有实验结果...")
    
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    all_results = {}
    
    # 收集adap_auto结果
    adap_auto_file = script_dir / 'results' / 'robustness_analysis_experiment' / f'robustness_results_{dataset}_{prediction_scale}.pkl'
    if adap_auto_file.exists():
        with open(adap_auto_file, 'rb') as f:
            adap_auto_results = pickle.load(f)
        all_results['adap_auto'] = adap_auto_results
        print("✅ 已收集adap_auto结果")
    else:
        print("⚠️  未找到adap_auto结果")
    
    # 收集基线模型结果
    baseline_models = ['DLinear', 'iTransformer', 'FEDformer', 'NBEATSx']
    baseline_dir = script_dir / 'results' / 'baseline_robustness'
    
    for model_name in baseline_models:
        result_file = baseline_dir / f'robustness_results_{dataset}_{prediction_scale}_{model_name}.pkl'
        if result_file.exists():
            with open(result_file, 'rb') as f:
                model_results = pickle.load(f)
            all_results[model_name] = model_results
            print(f"✅ 已收集{model_name}结果")
        else:
            print(f"⚠️  未找到{model_name}结果")
    
    return all_results

def create_comprehensive_analysis(all_results, dataset, prediction_scale):
    """
    创建综合分析报告
    """
    print("📈 创建综合分析报告...")
    
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = script_dir / 'results' / 'comprehensive_analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 基线性能对比
    baseline_comparison = []
    
    for model_name, results in all_results.items():
        if 'baseline' in results:
            baseline_comparison.append({
                'Model': model_name,
                'MAE': results['baseline']['MAE'],
                'RMSE': results['baseline']['RMSE'],
                'MAPE': results['baseline']['MAPE'],
                'Training_Time': results.get('training_time', 0)
            })
        elif model_name == 'adap_auto' and results.get('seasonal_mode') == 'independent':
            # 处理独立季节模式
            seasonal_metrics = list(results['seasonal_performance'].values())
            if seasonal_metrics:
                avg_mae = np.mean([m['MAE'] for m in seasonal_metrics])
                avg_rmse = np.mean([m['RMSE'] for m in seasonal_metrics])
                avg_mape = np.mean([m['MAPE'] for m in seasonal_metrics])
                baseline_comparison.append({
                    'Model': f'{model_name} (Independent)',
                    'MAE': avg_mae,
                    'RMSE': avg_rmse,
                    'MAPE': avg_mape,
                    'Training_Time': results.get('training_time', 0)
                })
    
    # 保存基线性能对比
    if baseline_comparison:
        baseline_df = pd.DataFrame(baseline_comparison)
        baseline_df = baseline_df.sort_values('MAE')  # 按MAE排序
        baseline_file = analysis_dir / f'baseline_performance_{dataset}_{prediction_scale}.csv'
        baseline_df.to_csv(baseline_file, index=False)
        print(f"📋 基线性能对比已保存到: {baseline_file}")
    
    # 2. 鲁棒性对比分析
    robustness_data = []
    
    for model_name, results in all_results.items():
        # 噪声鲁棒性
        if 'noise_robustness' in results:
            for noise_level, metrics in results['noise_robustness'].items():
                robustness_data.append({
                    'Model': model_name,
                    'Test_Type': 'Noise',
                    'Test_Level': noise_level,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'MAE_Degradation': metrics['MAE_degradation'],
                    'RMSE_Degradation': metrics['RMSE_degradation'],
                    'MAPE_Degradation': metrics['MAPE_degradation']
                })
        
        # 缺失鲁棒性
        if 'missing_robustness' in results:
            for missing_level, metrics in results['missing_robustness'].items():
                robustness_data.append({
                    'Model': model_name,
                    'Test_Type': 'Missing',
                    'Test_Level': missing_level,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'MAE_Degradation': metrics['MAE_degradation'],
                    'RMSE_Degradation': metrics['RMSE_degradation'],
                    'MAPE_Degradation': metrics['MAPE_degradation']
                })
    
    # 保存鲁棒性对比
    if robustness_data:
        robustness_df = pd.DataFrame(robustness_data)
        robustness_file = analysis_dir / f'robustness_analysis_{dataset}_{prediction_scale}.csv'
        robustness_df.to_csv(robustness_file, index=False)
        print(f"🔊 鲁棒性分析已保存到: {robustness_file}")
    
    # 3. 季节性性能对比
    seasonal_data = []
    
    for model_name, results in all_results.items():
        if 'seasonal_performance' in results:
            for season, metrics in results['seasonal_performance'].items():
                seasonal_data.append({
                    'Model': model_name,
                    'Season': season,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'Sample_Count': metrics.get('sample_count', metrics.get('test_samples', 0))
                })
    
    # 保存季节性对比
    if seasonal_data:
        seasonal_df = pd.DataFrame(seasonal_data)
        seasonal_file = analysis_dir / f'seasonal_performance_{dataset}_{prediction_scale}.csv'
        seasonal_df.to_csv(seasonal_file, index=False)
        print(f"🌸 季节性性能对比已保存到: {seasonal_file}")
    
    return baseline_comparison, robustness_data, seasonal_data

def print_final_summary(baseline_comparison, robustness_data, seasonal_data):
    """
    打印最终实验总结
    """
    print(f"\n{'='*80}")
    print("🏆 完整实验结果总结")
    print(f"{'='*80}")
    
    # 1. 基线性能排名
    if baseline_comparison:
        print(f"\n🏅 基线性能排名 (按MAE排序):")
        for i, model in enumerate(baseline_comparison, 1):
            training_time = model.get('Training_Time', 0)
            time_str = f", 训练时间={training_time:.2f}s" if training_time > 0 else ""
            print(f"  {i}. {model['Model']}: MAE={model['MAE']:.4f}, RMSE={model['RMSE']:.4f}, MAPE={model['MAPE']:.2f}%{time_str}")
        
        # 最佳模型
        best_model = baseline_comparison[0]
        print(f"\n🥇 最佳模型: {best_model['Model']}")
        print(f"   性能: MAE={best_model['MAE']:.4f}, RMSE={best_model['RMSE']:.4f}, MAPE={best_model['MAPE']:.2f}%")
    
    # 2. 鲁棒性分析摘要
    if robustness_data:
        print(f"\n🔊 噪声鲁棒性排名 (按平均MAE退化排序):")
        noise_data = [r for r in robustness_data if r['Test_Type'] == 'Noise']
        if noise_data:
            # 计算每个模型的平均噪声退化
            model_noise_degradation = {}
            for item in noise_data:
                model = item['Model']
                if model not in model_noise_degradation:
                    model_noise_degradation[model] = []
                model_noise_degradation[model].append(item['MAE_Degradation'])
            
            # 计算平均值并排序
            avg_degradation = [(model, np.mean(degradations)) for model, degradations in model_noise_degradation.items()]
            avg_degradation.sort(key=lambda x: x[1])
            
            for i, (model, avg_deg) in enumerate(avg_degradation, 1):
                print(f"  {i}. {model}: 平均MAE退化={avg_deg:.2f}%")
        
        print(f"\n🕳️ 缺失鲁棒性排名 (按MAE退化排序):")
        missing_data = [r for r in robustness_data if r['Test_Type'] == 'Missing']
        if missing_data:
            missing_sorted = sorted(missing_data, key=lambda x: x['MAE_Degradation'])
            for i, item in enumerate(missing_sorted, 1):
                print(f"  {i}. {item['Model']}: MAE退化={item['MAE_Degradation']:.2f}%")
    
    # 3. 季节性性能摘要
    if seasonal_data:
        print(f"\n🌸 季节性性能分析:")
        # 计算每个季节的平均性能
        season_avg = {}
        for item in seasonal_data:
            season = item['Season']
            if season not in season_avg:
                season_avg[season] = []
            season_avg[season].append(item['MAE'])
        
        for season, maes in season_avg.items():
            avg_mae = np.mean(maes)
            print(f"  {season}: 平均MAE={avg_mae:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive robustness comparison for all models')
    
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='fujian', choices=['fujian', 'DSWE'], 
                        help='Dataset name')
    parser.add_argument('--prediction_scale', type=str, default='6-0_1', 
                        help='Prediction scale (e.g., 6-0_1, 24-1, etc.)')
    
    # 训练相关参数
    parser.add_argument('--gpu', type=int, default=1, help='GPU device id')
    parser.add_argument('--epochs', type=int, default=10, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 鲁棒性测试参数
    parser.add_argument('--noise_levels', nargs='+', type=float, default=[0.05, 0.1],
                        help='Noise levels for robustness testing')
    parser.add_argument('--missing_ratio', type=float, default=0.05,
                        help='Missing data ratio for robustness testing')
    
    # 实验控制参数
    parser.add_argument('--seasonal_mode', type=str, default='test_split', 
                        choices=['test_split', 'independent'],
                        help='Seasonal analysis mode for adap_auto')
    parser.add_argument('--baseline_models', nargs='+', type=str, 
                        default=['DLinear', 'iTransformer', 'FEDformer', 'NBEATSx'],
                        choices=['DLinear', 'iTransformer', 'FEDformer', 'NBEATSx'],
                        help='Baseline models to compare')
    parser.add_argument('--skip_adap_auto', action='store_true',
                        help='Skip adap_auto experiment')
    parser.add_argument('--skip_baselines', action='store_true',
                        help='Skip baseline experiments')
    
    args = parser.parse_args()
    
    print(f"\n🎯 开始完整鲁棒性对比实验")
    print(f"数据集: {args.dataset}")
    print(f"预测尺度: {args.prediction_scale}")
    print(f"基线模型: {args.baseline_models}")
    print(f"季节模式: {args.seasonal_mode}")
    print(f"训练轮数: {args.epochs}")
    
    # 切换到脚本目录
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(script_dir)
    
    successful_experiments = []
    failed_experiments = []
    
    start_time = time.time()
    
    # 1. 运行adap_auto实验
    if not args.skip_adap_auto:
        print(f"\n{'='*60}")
        print("🏆 步骤1: 运行adap_auto模型实验")
        print(f"{'='*60}")
        
        if run_adap_auto_experiment(args.dataset, args.prediction_scale, args):
            successful_experiments.append('adap_auto')
        else:
            failed_experiments.append('adap_auto')
    
    # 2. 运行基线模型实验
    if not args.skip_baselines:
        print(f"\n{'='*60}")
        print("🔬 步骤2: 运行基线模型实验")
        print(f"{'='*60}")
        
        for i, model_name in enumerate(args.baseline_models, 1):
            print(f"\n--- {i}/{len(args.baseline_models)}: {model_name} ---")
            
            if run_baseline_model_experiment(model_name, args.dataset, args.prediction_scale, args):
                successful_experiments.append(model_name)
            else:
                failed_experiments.append(model_name)
    
    # 3. 收集和分析结果
    if successful_experiments:
        print(f"\n{'='*60}")
        print("📊 步骤3: 收集和分析结果")
        print(f"{'='*60}")
        
        all_results = collect_all_results(args.dataset, args.prediction_scale)
        
        if all_results:
            baseline_comparison, robustness_data, seasonal_data = create_comprehensive_analysis(
                all_results, args.dataset, args.prediction_scale
            )
            
            print_final_summary(baseline_comparison, robustness_data, seasonal_data)
        else:
            print("❌ 未能收集到任何结果")
    
    # 4. 实验总结
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("📊 实验完成总结")
    print(f"{'='*80}")
    print(f"总用时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"成功实验: {len(successful_experiments)} 个")
    print(f"失败实验: {len(failed_experiments)} 个")
    
    if successful_experiments:
        print(f"✅ 成功的实验: {', '.join(successful_experiments)}")
    
    if failed_experiments:
        print(f"❌ 失败的实验: {', '.join(failed_experiments)}")
    
    # 显示结果文件位置
    results_base = script_dir / 'results'
    print(f"\n📁 结果文件位置:")
    print(f"  - adap_auto结果: {results_base / 'robustness_analysis_experiment'}")
    print(f"  - 基线模型结果: {results_base / 'baseline_robustness'}")
    print(f"  - 综合分析结果: {results_base / 'comprehensive_analysis'}")
    
    if len(successful_experiments) > 0:
        print(f"\n✅ 实验成功完成！")
    else:
        print(f"\n❌ 所有实验都失败了")

if __name__ == '__main__':
    main() 