#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Overall Complexity and Performance Summary
综合复杂性和性能总结实验

对比模型：
- adap_auto (您的模型)
- iTransformer  
- FEDformer
- Informer
- Reformer

评估指标：
- 测试集MSE(Fujian)
- 参数量
- 训练时长
- 推理时延
- 峰值显存
"""

import os
import sys
import time
import pickle
import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# 不再全局添加路径，而是在导入时动态添加

# 导入各个模型 - 使用完全隔离的导入环境
def safe_import_model_isolated(model_path, module_name, class_name):
    """从指定路径完全隔离地导入模型"""
    try:
        import importlib.util
        import sys
        import subprocess
        import tempfile
        
        # 构建模块文件的完整路径
        module_file_path = os.path.join(model_path, f"{module_name}.py")
        
        if not os.path.exists(module_file_path):
            raise FileNotFoundError(f"模块文件不存在: {module_file_path}")
        
        # 完全清理sys.path，只保留标准库路径和模型路径
        original_path = sys.path.copy()
        
        # 保留Python标准库路径
        std_paths = [path for path in sys.path if 'site-packages' in path or 'python' in path.lower()]
        std_paths.extend(['/usr/lib/python3', '/usr/local/lib/python3'])
        
        # 设置隔离的路径
        isolated_path = [model_path] + std_paths
        sys.path = isolated_path
        
        try:
            # 清除已导入的冲突模块
            modules_to_remove = []
            for mod_name in sys.modules:
                if any(conflict in mod_name for conflict in ['DataEmbedding', 'Encoder', 'Decoder', 'attention', 'seriesDecomp']):
                    if not mod_name.startswith('torch') and not mod_name.startswith('numpy'):
                        modules_to_remove.append(mod_name)
            
            for mod_name in modules_to_remove:
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
            
            # 创建模块规格
            spec = importlib.util.spec_from_file_location(f"isolated_{model_path.split('/')[-1]}_{module_name}", module_file_path)
            if spec is None:
                raise ImportError(f"无法创建模块规格: {module_file_path}")
            
            # 创建模块
            module = importlib.util.module_from_spec(spec)
            
            # 执行模块
            spec.loader.exec_module(module)
            model_class = getattr(module, class_name)
            
            print(f"✓ 成功导入 {class_name} 从 {model_path}")
            return model_class
            
        finally:
            # 恢复原始路径
            sys.path = original_path
            
    except Exception as e:
        print(f"警告: 无法从 {model_path} 导入 {class_name}: {str(e)}")
        return None

# 尝试导入所有模型
print("开始导入模型...")
adap_auto = safe_import_model_isolated('/home/forecasting/pts/adap_auto/new_hier', 'adap_auto', 'adap_auto')
FEDformer = safe_import_model_isolated('/home/forecasting/pts/FEDfomer', 'FEDformer', 'FEDformer')
Informer = safe_import_model_isolated('/home/forecasting/pts/informer', 'Informer', 'Informer')
iTransformer = safe_import_model_isolated('/home/forecasting/pts/iTransformer', 'iTransformer', 'iTransformer')
Reformer = safe_import_model_isolated('/home/forecasting/pts/Reformer', 'Reformer', 'Reformer')
print("模型导入完成")

# 导入评估函数 - 使用独立导入
def import_evaluate_functions():
    """导入评估函数"""
    try:
        import importlib.util
        import sys
        
        # 从adap_auto目录导入evaluate
        eval_path = '/home/forecasting/pts/adap_auto/new_hier/evaluate.py'
        if os.path.exists(eval_path):
            spec = importlib.util.spec_from_file_location("evaluate_adap", eval_path)
            eval_module = importlib.util.module_from_spec(spec)
            
            original_path = sys.path.copy()
            sys.path.insert(0, '/home/forecasting/pts/adap_auto/new_hier')
            
            try:
                spec.loader.exec_module(eval_module)
                return getattr(eval_module, 'MSE', None), getattr(eval_module, 'MAPE', None)
            finally:
                sys.path = original_path
        return None, None
    except Exception as e:
        print(f"警告: 无法导入评估函数: {str(e)}")
        return None, None

MSE_func, MAPE_func = import_evaluate_functions()

def seed_everything(seed=42):
    """设置随机种子确保实验可重复"""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_memory_usage():
    """获取当前内存使用量(MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory():
    """获取GPU显存使用量(MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_number(num):
    """格式化数字显示"""
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.0f}"

def inverse_transform_power(data, scaler, power_feature_idx=0):
    """对功率数据进行反标准化"""
    mean = scaler.mean_[power_feature_idx]
    scale = scaler.scale_[power_feature_idx]
    return data * scale + mean

def MSE(actual=0, forecast=0):
    return ((actual - forecast) ** 2).mean()

class ModelTrainer:
    """模型训练器基类"""
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        try:
            # 尝试加载标准化数据（如果存在）
            x_data = torch.tensor(np.load('/home/forecasting/pts/adap_auto/data/fujian/stdtrain_data6-1.npy')).to(dtype=torch.float32)
            y_data = torch.tensor(np.squeeze(np.load('/home/forecasting/pts/adap_auto/data/fujian/stdval_data6-1.npy')[:, :, 0:1], axis=2)).to(dtype=torch.float32)
            scaler_path = '/home/forecasting/pts/adap_auto/data/fujian/scaler.pkl'
            print("使用标准化数据")
        except:
            # 如果标准化数据不存在，使用原始数据
            try:
                x_data = torch.tensor(np.load('/home/forecasting/pts/adap_auto/data/train_data6-1.npy')).to(dtype=torch.float32)
                y_data = torch.tensor(np.squeeze(np.load('/home/forecasting/pts/adap_auto/data/val_data6-1.npy')[:, :, 0:1], axis=2)).to(dtype=torch.float32)
                scaler_path = None
                print("使用原始数据")
            except Exception as e:
                print(f"数据加载失败: {str(e)}")
                return None, None, None, None, None
        
        # 加载标准化器
        scaler = None
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                
        x_data, y_data = x_data.to(self.device), y_data.to(self.device)
        
        # 检查数据维度
        print(f"数据维度: x_data={x_data.shape}, y_data={y_data.shape}")
        
        # 如果数据维度不匹配，尝试调整
        if len(x_data.shape) == 3 and x_data.shape[2] != 8:
            print(f"警告: 输入特征数为 {x_data.shape[2]}，但模型期望 8")
            if x_data.shape[2] > 8:
                x_data = x_data[:, :, :8]  # 只取前8个特征
                print(f"截取前8个特征，新维度: {x_data.shape}")
        
        # 分割数据
        split_ratio = 0.9
        split_index = int(len(x_data) * split_ratio)
        X_train, X_test = x_data[0:split_index], x_data[split_index:]
        y_train, y_test = y_data[0:split_index], y_data[split_index:]
        
        return X_train, X_test, y_train, y_test, scaler
    
    def create_model(self, model_name):
        """创建指定模型"""
        if model_name == 'adap_auto':
            if adap_auto is None:
                raise Exception("adap_auto模型无法导入")
            
            # 尝试加载图结构数据，如果失败则创建简单的图结构
            edge_index = None
            try:
                with open('/home/forecasting/pts/adap_auto/new_data/fujian/adag_dict_train_data6-1_fused.pkl', 'rb') as f:
                    edge_index = pickle.load(f)
                print("成功加载图结构数据")
                
                # 验证图结构数据的有效性
                if isinstance(edge_index, list) and len(edge_index) > 0:
                    # 检查第一个图结构
                    first_edge = edge_index[0]
                    if hasattr(first_edge, 'edge_index'):
                        max_node_idx = first_edge.edge_index.max().item() if first_edge.edge_index.numel() > 0 else 0
                        print(f"图结构中最大节点索引: {max_node_idx}")
                        
                        # 如果节点索引超出合理范围，创建简单图结构
                        if max_node_idx >= 100:  # 假设合理的节点数量上限
                            print("图结构节点索引过大，将创建简单图结构")
                            edge_index = None
                    else:
                        print("图结构格式不正确，将创建简单图结构")
                        edge_index = None
                else:
                    print("图结构数据格式不正确，将创建简单图结构")
                    edge_index = None
                    
            except Exception as e:
                print(f"加载图结构数据失败: {str(e)}")
                print("将创建简单的图结构")
                edge_index = None
            
            # 如果没有有效的图结构，创建一个简单的全连接图
            if edge_index is None:
                print("创建简单的8节点全连接图结构")
                import torch
                
                # 创建8个节点的简单全连接图的邻接矩阵格式
                num_nodes = 8
                edge_list = []
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            edge_list.append([i, j])
                
                edge_tensor = torch.tensor(edge_list).t().contiguous().long()
                print(f"创建的图结构形状: {edge_tensor.shape}")
                
                # 为每个batch创建相同的图结构tensor
                edge_index = [edge_tensor] * 10000  # 创建足够多的图结构
            
            # 按照wind_6-1.py中的配置创建模型
            model = adap_auto(
                n_head=8,
                hidden_size=256,
                factor=2,
                dropout=0.05,
                conv_hidden_size=32,
                MovingAvg_window=3,
                activation="gelu",
                encoder_layers=1,
                decoder_layers=1,
                c_in=8,      # 按原配置
                seq_lenth=36, # 序列长度
                c_out=6,     # 输出长度
            )
            model.to(self.device)
            return model, edge_index
            
        elif model_name == 'fedformer':
            if FEDformer is None:
                raise Exception("FEDformer模型无法导入")
            model = FEDformer(
                n_head=8,
                hidden_size=256,
                input_size=-1,
                modes=64,
                dropout=0.05,
                conv_hidden_size=32,
                MovingAvg_window=3,
                activation="gelu",
                encoder_layers=2,
                decoder_layers=1,
                c_in=8,
                c_out=6
            )
            
        elif model_name == 'informer':
            if Informer is None:
                raise Exception("Informer模型无法导入")
            model = Informer(
                n_head=8,
                hidden_size=256,
                factor=2,
                dropout=0.05,
                conv_hidden_size=32,
                activation="gelu",
                encoder_layers=2,
                decoder_layers=1,
                c_in=8,
                c_out=6
            )
            
        elif model_name == 'itransformer':
            if iTransformer is None:
                raise Exception("iTransformer模型无法导入")
            model = iTransformer(
                n_heads=8,  # 你可以根据你的需求调整头的数量
                hidden_size=256,  # 你可以根据你的需求调整隐藏层的大小
                factor=2,
                dropout=0.05,
                e_layers=2,
                d_layers=1,
                input_size=36,
                c_out=6
            )
            
        elif model_name == 'reformer':
            if Reformer is None:
                raise Exception("Reformer模型无法导入")
            model = Reformer(
                n_heads=8,  # 你可以根据你的需求调整头的数量
                n_hashes=8,
                hidden_size=256,  # 你可以根据你的需求调整隐藏层的大小
                bucket_size=64,
                dropout=0.05,
                conv_hidden_size=32,
                activation="gelu",
                e_layers=2,
                c_in=8,
                c_out=6
            )
        else:
            raise Exception(f"未知模型: {model_name}")
        
        model.to(self.device)
        return model, None
    
    def train_model(self, model, X_train, y_train, model_name, edge_index=None, epochs=5, batch_size=64):
        """训练模型并记录训练时间和显存使用"""
        train_dataset = TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        
        # 记录训练开始时的显存
        torch.cuda.empty_cache()
        start_memory = get_gpu_memory()
        peak_memory = start_memory
        
        # 记录训练开始时间
        start_time = time.time()
        
        model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                # 根据不同模型调用不同的前向传播
                if model_name == 'adap_auto':
                    # 获取对应的图结构，添加安全检查
                    try:
                        batch_start = batch_idx * batch_size
                        batch_end = min((batch_idx + 1) * batch_size, len(X_train))
                        
                        # 确保不会超出edge_index的范围
                        if edge_index and len(edge_index) > batch_end:
                            batch_edge = [edge_index[i] for i in range(batch_start, batch_end)]
                        else:
                            # 如果edge_index不够长，重复使用第一个图结构
                            if edge_index and len(edge_index) > 0:
                                batch_edge = [edge_index[0]] * (batch_end - batch_start)
                            else:
                                # 如果没有图结构，传None
                                batch_edge = None
                        
                        outputs = model(batch_x, batch_edge)
                    except Exception as e:
                        print(f"adap_auto前向传播失败: {str(e)}")
                        # 尝试不使用图结构
                        try:
                            outputs = model(batch_x, None)
                        except:
                            # 如果还是失败，抛出异常
                            raise e
                # elif model_name in ['fedformer', 'informer']:
                #     # FEDformer和Informer需要decoder输入
                #     print(f"batch_x.shape: {batch_x.shape}")
                #     print(f"batch_y.shape: {batch_y.shape}")
                #     dec_inp = torch.zeros_like(batch_y[:, -6:, :]).float().to(self.device)
                #     dec_inp = torch.cat([batch_x[:, -12:, :], dec_inp], dim=1).float().to(self.device)
                #     outputs = model(batch_x, dec_inp)
                else:
                    outputs = model(batch_x)
                
                # 调试输出形状
                if batch_idx == 0 and epoch == 0:
                    print(f"模型输出形状: {outputs.shape}")
                    print(f"目标形状: {batch_y.shape}")
                
                # 如果形状不匹配，尝试调整
                if outputs.shape != batch_y.shape:
                    # print(f"形状不匹配: outputs={outputs.shape}, target={batch_y.shape}")
                    # 如果输出有多余的维度，去掉
                    if len(outputs.shape) == 3 and outputs.shape[2] == 1:
                        outputs = outputs.squeeze(-1)  # 去掉最后一个维度
                        # print(f"去掉多余维度后: outputs={outputs.shape}")
                    # # 如果输出的最后一个维度不匹配，可能需要调整
                    # elif len(outputs.shape) == 2 and len(batch_y.shape) == 2:
                    #     if outputs.shape[1] != batch_y.shape[1]:
                    #         # 截取或填充到匹配的维度
                    #         min_dim = min(outputs.shape[1], batch_y.shape[1])
                    #         outputs = outputs[:, :min_dim]
                    #         batch_y = batch_y[:, :min_dim]
                    #         print(f"调整后形状: outputs={outputs.shape}, target={batch_y.shape}")
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # 更新峰值显存
                current_memory = get_gpu_memory()
                peak_memory = max(peak_memory, current_memory)
            
            total_loss += epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_dataloader):.6f}")
        
        # 记录训练结束时间
        training_time = time.time() - start_time
        
        return training_time, peak_memory - start_memory
    
    def test_model(self, model, X_test, y_test, scaler, model_name, edge_index=None):
        """测试模型并记录推理时延"""
        model.eval()
        predictions = []
        inference_times = []
        
        with torch.no_grad():
            for i in range(len(X_test)):
                start_time = time.time()
                
                # 根据不同模型调用不同的前向传播
                if model_name == 'adap_auto':
                    try:
                        # 安全获取测试图结构
                        test_idx = len(edge_index) - len(X_test) + i
                        if edge_index and len(edge_index) > test_idx and test_idx >= 0:
                            test_edge = edge_index[test_idx]
                            pred = model(X_test[i:i+1], [test_edge])
                        elif edge_index and len(edge_index) > 0:
                            # 使用第一个图结构
                            test_edge = edge_index[0]
                            pred = model(X_test[i:i+1], [test_edge])
                        else:
                            # 不使用图结构
                            pred = model(X_test[i:i+1], None)
                    except Exception as e:
                        print(f"测试时adap_auto前向传播失败: {str(e)}")
                        # 尝试不使用图结构
                        pred = model(X_test[i:i+1], None)
                # elif model_name in ['fedformer', 'informer']:
                #     dec_inp = torch.zeros((1, 6, 8)).float().to(self.device)
                #     dec_inp = torch.cat([X_test[i:i+1, -12:, :], dec_inp], dim=1).float().to(self.device)
                #     pred = model(X_test[i:i+1], dec_inp)
                else:
                    pred = model(X_test[i:i+1])
                
                # 调整预测输出的形状
                if len(pred.shape) == 3 and pred.shape[2] == 1:
                    pred = pred.squeeze(-1)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time * 1000)  # 转换为毫秒
                
                predictions.append(pred.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        # 反标准化
        predictions_denorm = inverse_transform_power(predictions.reshape(-1, 6), scaler)
        y_test_denorm = inverse_transform_power(y_test.cpu().numpy().reshape(-1, 6), scaler)
        
        # 计算MSE
        mse = MSE(predictions_denorm, y_test_denorm)
        avg_inference_time = np.mean(inference_times)
        
        return mse, avg_inference_time
    
    def run_experiment(self, model_name, epochs=5):
        """运行单个模型的完整实验"""
        print(f"\n=== 开始测试 {model_name.upper()} ===")
        
        # 设置随机种子
        seed_everything(42)
        
        # 加载数据
        X_train, X_test, y_train, y_test, scaler = self.load_data()
        print(f"数据加载完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
        
        # 创建模型
        model, edge_index = self.create_model(model_name)
        param_count = count_parameters(model)
        print(f"模型参数量: {format_number(param_count)}")
        
        # 训练模型
        print("开始训练...")
        training_time, peak_memory = self.train_model(
            model, X_train, y_train, model_name, edge_index, epochs, batch_size=64
        )
        print(f"训练完成，耗时: {training_time:.2f}s, 峰值显存: {peak_memory:.2f}MB")
        
        # 测试模型
        print("开始测试...")
        mse, avg_inference_time = self.test_model(
            model, X_test, y_test, scaler, model_name, edge_index
        )
        print(f"测试完成，MSE: {mse:.6f}, 平均推理时延: {avg_inference_time:.2f}ms")
        
        # 记录结果
        self.results[model_name] = {
            'MSE': mse,
            'Parameters': param_count,
            'Training Time (s)': training_time,
            'Inference Time (ms)': avg_inference_time,
            'Peak Memory (MB)': peak_memory
        }
        
        # 清理显存
        del model
        torch.cuda.empty_cache()
        
        return self.results[model_name]

def create_comparison_table(results):
    """创建对比表格"""
    # 准备数据
    table_data = []
    model_names = {
        'adap_auto': 'adap_auto (Ours)',
        'itransformer': 'iTransformer',
        'fedformer': 'FEDformer', 
        'informer': 'Informer',
        'reformer': 'Reformer'
    }
    
    for model_key, model_name in model_names.items():
        if model_key in results:
            data = results[model_key]
            table_data.append([
                model_name,
                f"{data['MSE']:.6f}",
                format_number(data['Parameters']),
                f"{data['Training Time (s)']:.2f}s",
                f"{data['Inference Time (ms)']:.2f}ms",
                f"{data['Peak Memory (MB)']:.2f}MB"
            ])
    
    headers = ['Model', 'Test MSE (Fujian)', 'Parameters', 'Training Time', 'Inference Time', 'Peak Memory']
    
    return table_data, headers

def save_results_table(results, save_path='overall_performance_summary.csv'):
    """保存结果到CSV文件"""
    df_data = []
    model_names = {
        'adap_auto': 'adap_auto (Ours)',
        'itransformer': 'iTransformer',
        'fedformer': 'FEDformer',
        'informer': 'Informer', 
        'reformer': 'Reformer'
    }
    
    for model_key, model_name in model_names.items():
        if model_key in results:
            data = results[model_key]
            df_data.append({
                'Model': model_name,
                'Test_MSE_Fujian': data['MSE'],
                'Parameters': data['Parameters'],
                'Training_Time_s': data['Training Time (s)'],
                'Inference_Time_ms': data['Inference Time (ms)'],
                'Peak_Memory_MB': data['Peak Memory (MB)']
            })
    
    df = pd.DataFrame(df_data)
    df.to_csv(save_path, index=False)
    print(f"\n结果已保存到: {save_path}")
    
    return df

def create_visualization(results, save_path='performance_comparison.png'):
    """创建可视化图表"""
    if not results:
        print("没有结果数据，跳过可视化")
        return None
        
    model_names = {
        'adap_auto': 'adap_auto\n(Ours)',
        'itransformer': 'iTransformer',
        'fedformer': 'FEDformer',
        'informer': 'Informer',
        'reformer': 'Reformer'
    }
    
    # 准备数据
    models = []
    maes = []
    params = []
    train_times = []
    infer_times = []
    memories = []
    
    for model_key, model_name in model_names.items():
        if model_key in results:
            data = results[model_key]
            models.append(model_name)
            maes.append(data['MSE'])
            params.append(data['Parameters'] / 1e6)  # 转换为M
            train_times.append(data['Training Time (s)'])
            infer_times.append(data['Inference Time (ms)'])
            memories.append(data['Peak Memory (MB)'])
    
    if not models:
        print("没有有效的模型数据，跳过可视化")
        return None
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Overall Performance Comparison', fontsize=16, fontweight='bold')
    
    # MSE对比
    axes[0, 0].bar(models, maes, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Test MSE (Fujian)')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 参数量对比
    axes[0, 1].bar(models, params, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Model Parameters')
    axes[0, 1].set_ylabel('Parameters (M)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 训练时间对比
    axes[0, 2].bar(models, train_times, color='orange', alpha=0.7)
    axes[0, 2].set_title('Training Time')
    axes[0, 2].set_ylabel('Time (s)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 推理时间对比
    axes[1, 0].bar(models, infer_times, color='pink', alpha=0.7)
    axes[1, 0].set_title('Inference Time')
    axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 显存使用对比
    axes[1, 1].bar(models, memories, color='lightcoral', alpha=0.7)
    axes[1, 1].set_title('Peak Memory Usage')
    axes[1, 1].set_ylabel('Memory (MB)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 综合性能雷达图
    from math import pi
    categories = ['MSE\n(Lower Better)', 'Parameters\n(Lower Better)', 'Training Time\n(Lower Better)', 
                 'Inference Time\n(Lower Better)', 'Memory\n(Lower Better)']
    
    # 归一化数据 (越小越好的指标)
    max_mae = max(maes)
    max_param = max(params)
    max_train = max(train_times)
    max_infer = max(infer_times)
    max_mem = max(memories)
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    axes[1, 2].set_theta_offset(pi / 2)
    axes[1, 2].set_theta_direction(-1)
    axes[1, 2].set_thetagrids(np.degrees(angles[:-1]), categories)
    
    for i, (model_key, model_name) in enumerate(model_names.items()):
        if model_key in results:
            data = results[model_key]
            values = [
                1 - data['MSE'] / max_mae,  # 归一化并反转 (越小越好)
                1 - (data['Parameters'] / 1e6) / max_param,
                1 - data['Training Time (s)'] / max_train,
                1 - data['Inference Time (ms)'] / max_infer,
                1 - data['Peak Memory (MB)'] / max_mem
            ]
            values += values[:1]
            
            color = ['red', 'blue', 'green', 'orange', 'purple'][i]
            axes[1, 2].plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            axes[1, 2].fill(angles, values, alpha=0.1, color=color)
    
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title('Comprehensive Performance\n(Closer to edge = Better)')
    axes[1, 2].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到: {save_path}")
    
    return fig

def main():
    """主函数"""
    print("=== Overall Complexity and Performance Summary ===")
    print("开始综合性能对比实验...")
    
    # 初始化训练器
    trainer = ModelTrainer(device='cuda:0')
    
    # 检查哪些模型可以使用
    available_models = []
    model_classes = {
        'adap_auto': adap_auto,
        'itransformer': iTransformer,
        'fedformer': FEDformer,
        'informer': Informer,
        'reformer': Reformer
    }
    
    for model_name, model_class in model_classes.items():
        if model_class is not None:
            available_models.append(model_name)
            print(f"✓ {model_name} 可用")
        else:
            print(f"✗ {model_name} 不可用")
    
    if not available_models:
        print("❌ 没有可用的模型！")
        return {}, None, None
    
    print(f"\n将测试以下模型: {available_models}")
    
    # 运行实验
    all_results = {}
    for model_name in available_models:
        try:
            result = trainer.run_experiment(model_name, epochs=2)  # 使用较少的epochs进行快速测试
            all_results[model_name] = result
        except Exception as e:
            print(f"模型 {model_name} 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 创建对比表格
    table_data, headers = create_comparison_table(all_results)
    
    print("\n" + "="*100)
    print("OVERALL COMPLEXITY AND PERFORMANCE SUMMARY")
    print("="*100)
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print("="*100)
    
    # 保存结果
    results_dir = '/home/forecasting/pts/adap_auto/new_hier/ex-experiment/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存CSV文件
    csv_path = os.path.join(results_dir, 'overall_performance_summary.csv')
    df = save_results_table(all_results, csv_path)
    
    # 创建可视化
    viz_path = os.path.join(results_dir, 'overall_performance_comparison.png')
    fig = None
    if all_results:
        fig = create_visualization(all_results, viz_path)
    
    # 保存详细结果到pickle文件
    pickle_path = os.path.join(results_dir, 'overall_performance_results.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"详细结果已保存到: {pickle_path}")
    
    print(f"\n实验完成！所有结果已保存到: {results_dir}")
    
    return all_results, df, fig

if __name__ == "__main__":
    results, df, fig = main() 