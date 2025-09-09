#!/usr/bin/env python3
"""
性能基准测试脚本
比较传统方法与CPU Offloading方法的性能差异
测试模型预测上限的提升效果
"""

import os
import sys
import time
import psutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import subprocess
import argparse
from typing import Dict, List, Tuple, Optional

# GPU内存监控
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("⚠️  pynvml不可用，将无法监控GPU内存")


class PerformanceBenchmark:
    """
    性能基准测试类
    测试不同配置下的模型性能
    """
    
    def __init__(self, base_script_path: str):
        """
        初始化基准测试
        Args:
            base_script_path: 基础脚本路径（原始wind_1.py或优化后的wind_offloading.py）
        """
        self.base_script_path = base_script_path
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建结果目录
        self.results_dir = Path(f'benchmark_results_{self.timestamp}')
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"📊 性能基准测试初始化完成")
        print(f"📁 结果将保存到: {self.results_dir}")

    def get_system_info(self) -> Dict:
        """获取系统信息"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            
            if NVML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info['gpu_memory_total_gb'] = meminfo.total / (1024**3)
        
        return info

    def monitor_resources(self, duration: float = 1.0) -> Dict:
        """监控系统资源使用情况"""
        # CPU和内存监控
        cpu_percent = psutil.cpu_percent(interval=duration)
        memory = psutil.virtual_memory()
        
        result = {
            'cpu_percent': cpu_percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_percent': memory.percent,
        }
        
        # GPU监控 - 增强版本
        if torch.cuda.is_available():
            try:
                # 方法1：使用torch.cuda获取GPU内存信息
                gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                
                # 使用最大值作为GPU内存使用量
                gpu_memory_used = max(gpu_mem_allocated, gpu_mem_reserved)
                
                result.update({
                    'gpu_memory_used_gb': gpu_memory_used,
                    'gpu_memory_allocated_gb': gpu_mem_allocated,
                    'gpu_memory_reserved_gb': gpu_mem_reserved,
                    # 'gpu_memory_cached_gb': gpu_mem_cached,
                })
                
                # 方法2：使用nvidia-smi作为备用验证
                try:
                    import subprocess
                    nvidia_smi_output = subprocess.check_output([
                        'nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'
                    ], text=True, timeout=5)
                    
                    lines = nvidia_smi_output.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            parts = line.split(',')
                            if len(parts) >= 2:
                                gpu_memory_used_mb = float(parts[0].strip())
                                gpu_memory_total_mb = float(parts[1].strip())
                                nvidia_smi_gpu_mem = gpu_memory_used_mb / 1024  # 转换为GB
                                
                                # 使用nvidia-smi和torch.cuda的最大值
                                result['gpu_memory_used_gb'] = max(result['gpu_memory_used_gb'], nvidia_smi_gpu_mem)
                                result['gpu_memory_used_gb_nvidia_smi'] = nvidia_smi_gpu_mem
                                result['gpu_memory_total_gb'] = gpu_memory_total_mb / 1024
                                break
                                
                except Exception as e:
                    print(f"   nvidia-smi监控失败: {e}")
                    pass
                
                # 方法3：使用pynvml（如果可用）
                if NVML_AVAILABLE:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        
                        pynvml_gpu_mem = meminfo.used / (1024**3)
                        result['gpu_memory_used_gb'] = max(result['gpu_memory_used_gb'], pynvml_gpu_mem)
                        
                        result.update({
                            'gpu_memory_used_gb_pynvml': pynvml_gpu_mem,
                            'gpu_memory_percent': (meminfo.used / meminfo.total) * 100,
                            'gpu_utilization_percent': utilization.gpu,
                        })
                    except Exception as e:
                        print(f"   pynvml监控失败: {e}")
                        pass
                
                # 如果仍然为0，尝试强制刷新GPU状态
                if result['gpu_memory_used_gb'] == 0:
                    torch.cuda.synchronize()  # 强制同步GPU操作
                    torch.cuda.empty_cache()  # 清空缓存但不释放内存
                    
                    # 再次尝试获取内存信息
                    gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    result['gpu_memory_used_gb'] = max(gpu_mem_allocated, gpu_mem_reserved)
                
            except Exception as e:
                print(f"⚠️ GPU监控错误: {e}")
                # 即使出错，也尝试获取基本信息
                try:
                    gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    result['gpu_memory_used_gb'] = gpu_mem_allocated
                except Exception:
                    result['gpu_memory_used_gb'] = 0
        else:
            result['gpu_memory_used_gb'] = 0
        
        return result

    def run_single_test(self, 
                       test_config: Dict,
                       verbose_child: bool = False, 
                       timeout: Optional[int] = None) -> Dict:
        """
        运行单个测试配置
        Args:
            test_config: 测试配置字典
            timeout: 超时时间（秒），None表示不设置超时
        Returns:
            测试结果字典
        """
        import threading
        print(f"\n🔬 开始测试: {test_config['name']}")
        print(f"   配置: {test_config}")
        
        # 构建命令行参数
        cmd = [sys.executable, self.base_script_path]
        for key, value in test_config.get('args', {}).items():
            if isinstance(value, bool):
                if value: cmd.append(f'--{key}')
            else:
                cmd.extend([f'--{key}', str(value)])

        print(f"🚀 执行命令: {' '.join(cmd)}")
        start_time = time.time()

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace',
                cwd=os.path.dirname(os.path.abspath(self.base_script_path))
            )

            stdout_lines, stderr_lines = [], []
            def reader_thread(pipe, storage, pipe_name):
                for line in pipe:
                    if verbose_child: # <--- 根据verbose参数决定是否打印
                        print(f"  [{pipe_name}] > {line.strip()}")
                    storage.append(line)

            stdout_thread = threading.Thread(target=reader_thread, args=(process.stdout, stdout_lines, "STDOUT"))
            stderr_thread = threading.Thread(target=reader_thread, args=(process.stderr, stderr_lines, "STDERR"))
            stdout_thread.start()
            stderr_thread.start()
            
            peak_gpu_mem_gb = 0
            peak_ram_mem_gb = 0
            
            spinner = ['-', '\\', '|', '/']
            spin_idx = 0
            
            print("📊 开始并行监控...", end='', flush=True)
            while process.poll() is None:
                if timeout is not None and time.time() - start_time > timeout:
                    process.terminate()
                    raise subprocess.TimeoutExpired(cmd, timeout)

                current_resources = self.monitor_resources(duration=0.5)
                
                # 重新加入峰值内存的实时打印
                if 'gpu_memory_used_gb' in current_resources:
                    current_gpu_mem = current_resources.get('gpu_memory_used_gb', 0)
                    if current_gpu_mem > peak_gpu_mem_gb:
                        peak_gpu_mem_gb = current_gpu_mem
                        # \r 回到行首，打印新峰值，end=''不换行，flush立即显示
                        print(f"\r📊 监控中... 📈 新GPU峰值: {peak_gpu_mem_gb:.2f}GB", end='', flush=True)

                if 'memory_used_gb' in current_resources:
                    peak_ram_mem_gb = max(peak_ram_mem_gb, current_resources.get('memory_used_gb', 0))
                
                # 在安静模式下打印一个旋转的“等待”光标，表示程序仍在运行
                if not verbose_child:
                    print(f"\r📊 监控中... {spinner[spin_idx % len(spinner)]}", end='', flush=True)
                    spin_idx += 1

                time.sleep(0.5)

            # 清理最后一行监控输出
            print("\r" + " " * 50 + "\r", end='')

            stdout_thread.join()
            stderr_thread.join()
            process.wait()
            
            # (后续结果处理逻辑不变)
            # ...
            end_time = time.time()
            runtime = end_time - start_time
            return_code = process.returncode
            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)
            
            mse, mape = self._parse_performance_metrics(stdout + stderr)
            test_result = {
                'test_name': test_config['name'], 'config': test_config, 'success': return_code == 0,
                'runtime_seconds': runtime, 'max_memory_gb': peak_ram_mem_gb,
                'max_gpu_memory_gb': peak_gpu_mem_gb, 'mse': mse, 'mape': mape,
                'stdout': stdout, 'stderr': stderr, 'return_code': return_code,
            }
            if return_code == 0:
                print(f"✅ 测试完成: {runtime:.2f}秒, 峰值GPU内存: {peak_gpu_mem_gb:.2f}GB")
            else:
                print(f"❌ 测试失败: 返回码 {return_code}")

        except Exception as e:
            # ...
            test_result = {'test_name': test_config['name'], 'config': test_config, 'success': False, 'error': str(e)}

        return test_result

    def _parse_performance_metrics(self, output: str) -> Tuple[Optional[float], Optional[float]]:
        """从输出中解析性能指标"""
        import re
        mse, mape = None, None
        
        print("🔍 解析性能指标...")
        
        mse_pattern = re.compile(r"(?:MSE|最终MSE)(?:\s*\([^)]+\))?:\s*([0-9.]+)")
        mape_pattern = re.compile(r"(?:MAPE|最终MAPE)(?:\s*\([^)]+\))?:\s*([0-9.]+)")

        # 从后往前搜索，以获取最终的总结性指标
        for line in reversed(output.split('\n')):
            if mse is None:
                match = mse_pattern.search(line)
                if match:
                    try:
                        mse = float(match.group(1))
                    except ValueError:
                        pass
            
            if mape is None:
                match = mape_pattern.search(line)
                if match:
                    try:
                        mape = float(match.group(1))
                    except ValueError:
                        pass
            
            # 如果两个指标都找到了，就提前结束
            if mse is not None and mape is not None:
                break

        print(f"📊 解析结果: MSE={mse}, MAPE={mape}")
        return mse, mape

    def run_sequence_length_test(self, method_name: str, base_config: Dict) -> List[Dict]:
        """
        运行序列长度上限测试：逐步增加序列长度直到内存不足
        """
        print(f"\n🎯 开始{method_name}序列长度上限测试")
        print(f"🔍 测试策略：逐步增加序列长度，找到处理上限")
        
        # 定义不同序列长度的测试配置
        # 从较小的序列长度开始，逐步增加
        sequence_lengths = [1152, 1440, 1555, 1670, 1728, 2016]  # 基础序列长度
        
        # 如果是CPU Offloading版本，测试更大的序列长度
        if 'offloading' in method_name:
            sequence_lengths.extend([2304])  # 更大的序列长度
        
        scale_configs = []
        
        # 为每个序列长度创建测试配置
        for seq_len in sequence_lengths:
            # 根据序列长度调整其他参数以保持测试的合理性
            # 较长序列使用较小的batch_size以节省内存
            if seq_len <= 144:
                batch_size = 64
                hidden_size = 264
                n_head = 8
            elif seq_len <= 576:
                batch_size = 32
                hidden_size = 264
                n_head = 8
            elif seq_len <= 2304:
                batch_size = 16
                hidden_size = 264
                n_head = 8
            else:
                batch_size = 8
                hidden_size = 128
                n_head = 4
            
            config = {
                'name': f'{method_name}_seq_{seq_len}',
                'args': {
                    **base_config,
                    'seq_length': seq_len,
                    'c_out': seq_len,  # 预测长度等于序列长度
                    'hidden_size': hidden_size,
                    'n_head': n_head,
                    'batch_size': batch_size,
                    'epochs': 2,  # 使用较少epochs以节省时间
                    'max_gpu_memory': '1GiB',
                    'patience': 1,  # 快速停止
                }
            }
            scale_configs.append(config)
        
        print(f"📊 将测试 {len(scale_configs)} 种序列长度配置:")
        for config in scale_configs:
            args = config['args']
            print(f"   - 序列长度: {args['seq_length']}, 批次: {args['batch_size']}, 隐藏层: {args['hidden_size']}")
            
        print(f"\n⚡ 开始逐步测试，遇到内存不足将停止该方法的后续测试")
        
        scale_results = []
        max_successful_seq_length = 0
        
        for i, config in enumerate(scale_configs):
            seq_len = config['args']['seq_length']
            print(f"\n🧪 测试进度: [{i+1}/{len(scale_configs)}] 序列长度 {seq_len}")
            
            result = self.run_single_test(config, verbose_child=self.verbose_child)  # 无超时限制
            scale_results.append(result)
            self.results.append(result)
            
            # 保存中间结果
            self._save_intermediate_results()
            
            if result.get('success', False):
                max_successful_seq_length = seq_len
                print(f"✅ 序列长度 {seq_len} 测试成功!")
                print(f"   当前{method_name}最大成功序列长度: {max_successful_seq_length}")
            else:
                print(f"❌ 序列长度 {seq_len} 测试失败!")
                print(f"💡 {method_name}的序列长度上限为: {max_successful_seq_length}")
                
                # 如果是内存不足错误，停止后续更大序列长度的测试
                if 'out of memory' in str(result.get('error', '')).lower() or \
                   'cuda out of memory' in str(result.get('stdout', '')).lower():
                    print(f"🛑 检测到内存不足，停止{method_name}的后续测试")
                    break
                else:
                    print(f"⚠️  其他类型错误，继续测试下一个序列长度...")
        
        print(f"\n📊 {method_name}序列长度测试总结:")
        print(f"   最大成功序列长度: {max_successful_seq_length}")
        print(f"   成功测试数量: {len([r for r in scale_results if r.get('success', False)])}/{len(scale_configs)}")
        
        return scale_results

    def compare_methods(self, 
                       original_script: str, 
                       offloading_script: str,
                       base_config: Dict) -> Dict:
        """
        比较原始方法和CPU Offloading方法
        """
        print("\n" + "="*60)
        print("🥊 开始方法对比测试")
        print("="*60)
        
        results_comparison = {
            'system_info': self.get_system_info(),
            'timestamp': self.timestamp,
            'original_results': [],
            'offloading_results': [],
            'comparison_summary': {}
        }
        
        # 测试原始方法
        print("\n1️⃣  测试原始方法...")
        self.base_script_path = original_script
        original_results = self.run_sequence_length_test('original', base_config)
        results_comparison['original_results'] = original_results
        
        # 测试CPU Offloading方法
        print("\n2️⃣  测试CPU Offloading方法...")
        self.base_script_path = offloading_script
        offloading_results = self.run_sequence_length_test('offloading', base_config)
        results_comparison['offloading_results'] = offloading_results
        
        # 生成对比分析
        comparison_summary = self._analyze_comparison(original_results, offloading_results)
        results_comparison['comparison_summary'] = comparison_summary
        
        return results_comparison

    def _analyze_comparison(self, original_results: List[Dict], offloading_results: List[Dict]) -> Dict:
        """分析对比结果"""
        summary = {
            'successful_tests': {
                'original': len([r for r in original_results if r.get('success', False)]),
                'offloading': len([r for r in offloading_results if r.get('success', False)])
            },
            'max_scale_achieved': {
                'original': None,
                'offloading': None
            },
            'performance_comparison': {},
            'resource_usage': {},
        }
        
        # 找到成功运行的最大规模
        successful_original = [r for r in original_results if r.get('success', False)]
        successful_offloading = [r for r in offloading_results if r.get('success', False)]
        
        if successful_original:
            # 按seq_length排序找最大成功的配置
            max_original = max(successful_original, 
                             key=lambda x: x['config']['args'].get('seq_length', 0))
            summary['max_scale_achieved']['original'] = {
                'seq_length': max_original['config']['args'].get('seq_length'),
                'hidden_size': max_original['config']['args'].get('hidden_size'),
                'batch_size': max_original['config']['args'].get('batch_size'),
                'mse': max_original.get('mse'),
                'runtime': max_original.get('runtime_seconds')
            }
        
        if successful_offloading:
            max_offloading = max(successful_offloading,
                               key=lambda x: x['config']['args'].get('seq_length', 0))
            summary['max_scale_achieved']['offloading'] = {
                'seq_length': max_offloading['config']['args'].get('seq_length'),
                'hidden_size': max_offloading['config']['args'].get('hidden_size'),
                'batch_size': max_offloading['config']['args'].get('batch_size'),
                'mse': max_offloading.get('mse'),
                'runtime': max_offloading.get('runtime_seconds')
            }
        
        # 计算序列长度上限提升
        if summary['max_scale_achieved']['original'] and summary['max_scale_achieved']['offloading']:
            orig_seq_len = summary['max_scale_achieved']['original']['seq_length']
            off_seq_len = summary['max_scale_achieved']['offloading']['seq_length']
            
            summary['sequence_length_improvement'] = {
                'seq_length_ratio': off_seq_len / orig_seq_len if orig_seq_len > 0 else float('inf'),
                'improvement_percentage': ((off_seq_len - orig_seq_len) / orig_seq_len * 100) if orig_seq_len > 0 else float('inf'),
                'absolute_improvement': off_seq_len - orig_seq_len
            }
        
        return summary

    def _save_intermediate_results(self):
        """保存中间结果"""
        results_file = self.results_dir / 'intermediate_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

    def generate_report(self, comparison_results: Dict):
        """生成详细的性能报告"""
        print("\n📊 生成性能报告...")
        
        # 保存完整结果
        results_file = self.results_dir / 'full_results.json'
        with open(results_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        # 生成文本报告
        report_file = self.results_dir / 'performance_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("🚀 风电预测模型CPU Offloading性能基准测试报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"测试时间: {self.timestamp}\n")
            f.write(f"Python版本: {comparison_results['system_info']['python_version']}\n")
            f.write(f"PyTorch版本: {comparison_results['system_info']['pytorch_version']}\n")
            f.write(f"GPU可用: {comparison_results['system_info']['cuda_available']}\n")
            
            if comparison_results['system_info']['cuda_available']:
                f.write(f"GPU型号: {comparison_results['system_info']['gpu_names']}\n")
                gpu_mem_total = comparison_results['system_info'].get('gpu_memory_total_gb', 'N/A')
                if isinstance(gpu_mem_total, (int, float)):
                    f.write(f"GPU显存: {gpu_mem_total:.1f}GB\n")
                else:
                    f.write(f"GPU显存: {gpu_mem_total}\n")
                # f.write(f"GPU显存: {comparison_results['system_info'].get('gpu_memory_total_gb', 'N/A'):.1f}GB\n")
            
            memory_total = comparison_results['system_info'].get('memory_total_gb', 'N/A')
            if isinstance(memory_total, (int, float)):
                f.write(f"系统内存: {memory_total:.1f}GB\n")
            else:
                f.write(f"系统内存: {memory_total}\n")
            # f.write(f"系统内存: {comparison_results['system_info']['memory_total_gb']:.1f}GB\n")
            
            f.write("\n📈 测试结果摘要:\n")
            f.write("-" * 40 + "\n")
            
            summary = comparison_results['comparison_summary']
            
            # 成功测试数量
            f.write(f"原始方法成功测试: {summary['successful_tests']['original']}\n")
            f.write(f"CPU Offloading成功测试: {summary['successful_tests']['offloading']}\n")
            
            # 序列长度上限对比
            f.write("\n🎯 序列长度处理上限对比:\n")
            if summary['max_scale_achieved']['original']:
                orig = summary['max_scale_achieved']['original']
                f.write(f"原始方法最大序列长度:\n")
                f.write(f"  - 序列长度: {orig['seq_length']}\n")
                f.write(f"  - 隐藏层大小: {orig['hidden_size']}\n")
                f.write(f"  - 批次大小: {orig['batch_size']}\n")
                f.write(f"  - MSE: {orig['mse']:.6f}\n")
                f.write(f"  - 运行时间: {orig['runtime']:.2f}秒\n")
            
            if summary['max_scale_achieved']['offloading']:
                off = summary['max_scale_achieved']['offloading']
                f.write(f"CPU Offloading最大序列长度:\n")
                f.write(f"  - 序列长度: {off['seq_length']}\n")
                f.write(f"  - 隐藏层大小: {off['hidden_size']}\n")
                f.write(f"  - 批次大小: {off['batch_size']}\n")
                f.write(f"  - MSE: {off['mse']:.6f}\n")
                f.write(f"  - 运行时间: {off['runtime']:.2f}秒\n")
            
            # 序列长度上限提升
            if 'sequence_length_improvement' in summary:
                imp = summary['sequence_length_improvement']
                f.write(f"\n🎉 序列长度处理能力提升:\n")
                f.write(f"序列长度上限提升: {imp['improvement_percentage']:.1f}%\n")
                f.write(f"序列长度倍数: {imp['seq_length_ratio']:.2f}x\n")
                f.write(f"绝对提升: +{imp['absolute_improvement']} 个时间步长\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        # 生成可视化图表
        self._create_visualizations(comparison_results)
        
        print(f"✅ 报告已生成:")
        print(f"   📄 详细报告: {report_file}")
        print(f"   📊 完整数据: {results_file}")
        print(f"   📈 图表目录: {self.results_dir}")

    def _create_visualizations(self, comparison_results: Dict):
        """创建可视化图表"""
        try:
            # 性能对比图
            self._plot_performance_comparison(comparison_results)
            
            # 资源使用对比图
            self._plot_resource_usage(comparison_results)
            
            # 序列长度处理能力图
            self._plot_sequence_length_capability(comparison_results)
            
        except Exception as e:
            print(f"⚠️  图表生成失败: {e}")

    def _plot_performance_comparison(self, results: Dict):
        """绘制性能对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准备数据
        methods = []
        mse_values = []
        runtime_values = []
        
        for method, result_list in [('Original', results['original_results']), 
                                   ('CPU Offloading', results['offloading_results'])]:
            successful = [r for r in result_list if r.get('success', False) and r.get('mse') is not None]
            if successful:
                methods.append(method)
                mse_values.append([r['mse'] for r in successful])
                runtime_values.append([r['runtime_seconds'] for r in successful])
        
        # MSE对比
        if mse_values:
            ax1.boxplot(mse_values, labels=methods)
            ax1.set_title('MSE对比', fontsize=14, fontweight='bold')
            ax1.set_ylabel('MSE')
            ax1.grid(True, alpha=0.3)
        
        # 运行时间对比
        if runtime_values:
            ax2.boxplot(runtime_values, labels=methods)
            ax2.set_title('运行时间对比', fontsize=14, fontweight='bold')
            ax2.set_ylabel('时间 (秒)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_resource_usage(self, results: Dict):
        """绘制资源使用对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准备数据
        methods = []
        memory_usage = []
        gpu_memory_usage = []
        
        for method, result_list in [('Original', results['original_results']), 
                                   ('CPU Offloading', results['offloading_results'])]:
            successful = [r for r in result_list if r.get('success', False)]
            if successful:
                methods.append(method)
                memory_usage.append([r.get('max_memory_gb', 0) for r in successful])
                gpu_memory_usage.append([r.get('max_gpu_memory_gb', 0) for r in successful])
        
        # 内存使用对比
        if memory_usage:
            ax1.boxplot(memory_usage, labels=methods)
            ax1.set_title('系统内存使用对比', fontsize=14, fontweight='bold')
            ax1.set_ylabel('内存 (GB)')
            ax1.grid(True, alpha=0.3)
        
        # GPU内存使用对比
        if gpu_memory_usage:
            ax2.boxplot(gpu_memory_usage, labels=methods)
            ax2.set_title('GPU内存使用对比', fontsize=14, fontweight='bold')
            ax2.set_ylabel('GPU内存 (GB)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'resource_usage.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_sequence_length_capability(self, results: Dict):
        """绘制序列长度处理能力图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 图1：序列长度 vs MSE
        for method_name, result_list in [('Original', results['original_results']), 
                                        ('CPU Offloading', results['offloading_results'])]:
            successful = [r for r in result_list if r.get('success', False)]
            if successful:
                seq_lengths = [r['config']['args'].get('seq_length', 0) for r in successful]
                mse_values = [r.get('mse', float('inf')) for r in successful]
                
                # 过滤有效数据
                valid_data = [(s, m) for s, m in zip(seq_lengths, mse_values) if m != float('inf')]
                if valid_data:
                    seq_lengths, mse_values = zip(*valid_data)
                    ax1.plot(seq_lengths, mse_values, 'o-', label=method_name, linewidth=2, markersize=8)
        
        ax1.set_xlabel('序列长度', fontsize=12)
        ax1.set_ylabel('MSE', fontsize=12)
        ax1.set_title('序列长度 vs 预测精度', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')  # 使用对数坐标更好地显示不同量级的序列长度
        
        # 图2：成功的最大序列长度对比
        methods = []
        max_seq_lengths = []
        
        for method_name, result_list in [('Original', results['original_results']), 
                                        ('CPU Offloading', results['offloading_results'])]:
            successful = [r for r in result_list if r.get('success', False)]
            if successful:
                max_seq_len = max([r['config']['args'].get('seq_length', 0) for r in successful])
                methods.append(method_name)
                max_seq_lengths.append(max_seq_len)
        
        if max_seq_lengths:
            bars = ax2.bar(methods, max_seq_lengths, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
            ax2.set_ylabel('最大序列长度', fontsize=12)
            ax2.set_title('序列长度处理上限对比', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 在柱状图上添加数值标签
            for bar, value in zip(bars, max_seq_lengths):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(value)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sequence_length_capability.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='风电预测模型性能基准测试')
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument('--original_script', type=str, default=os.path.join(script_dir, '../new_hier/wind_1.py'), 
                       help='原始脚本路径')
    parser.add_argument('--offloading_script', type=str, default=os.path.join(script_dir, '../new_hier/wind_offloading.py'), 
                       help='CPU Offloading脚本路径')
    parser.add_argument('--quick_test', action='store_true', 
                       help='运行快速测试（较少epochs）')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='CSV数据文件路径（如果为None则自动查找）')
    parser.add_argument('--verbose_child', action='store_true', 
                    help='实时打印子进程的详细输出（如tqdm进度条）')
    args = parser.parse_args()
    
    print("🚀 风电预测模型性能基准测试")
    print("=" * 50)
    
    # 查找CSV文件路径
    if args.csv_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, '../data/fujian/Offshore Wind Farm Dataset3(WT1).csv')
        if not os.path.exists(csv_path):
            print(f"❌ 找不到CSV文件: {csv_path}")
            return
    else:
        csv_path = args.csv_path
    
    print(f"📁 使用CSV数据文件: {csv_path}")
    
    # 基础配置
    base_config = {
        'gpu': 0,
        'seed': 42,
        'patience': 3,
        'save_model': False,  # 测试时不保存模型
        'hyperparam_id': 'benchmark',  # 基准测试标识
    }
    
    if args.quick_test:
        print("⚡ 快速测试模式")
        base_config.update({
            'epochs': 2,
            'patience': 2,
        })
    
    # 创建基准测试实例
    benchmark = PerformanceBenchmark(args.original_script)
    benchmark.verbose_child = args.verbose_child
    try:
        # 运行对比测试
        comparison_results = benchmark.compare_methods(
            args.original_script,
            args.offloading_script,
            base_config
        )
        
        # 生成报告
        benchmark.generate_report(comparison_results)
        
        print("\n🎉 基准测试完成！")
        print(f"📊 查看结果目录: {benchmark.results_dir}")
        
        # 显示关键结论
        summary = comparison_results['comparison_summary']
        if 'sequence_length_improvement' in summary:
            improvement = summary['sequence_length_improvement']['improvement_percentage']
            seq_improvement = summary['sequence_length_improvement']['absolute_improvement']
            print(f"🏆 关键发现：CPU Offloading使序列长度处理能力提升了 {improvement:.1f}%")
            print(f"🎯 具体提升：序列长度上限增加了 {seq_improvement} 个时间步长")
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n💥 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 