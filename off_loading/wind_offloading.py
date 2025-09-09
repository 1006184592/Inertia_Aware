import os
# Set CUDA launch blocking for debugging
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pickle
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from adap_auto import adap_auto
from price.case_118 import price_case
from evaluate import MSE, MAPE
import torch.multiprocessing as mp
import time
import argparse
import random
from pathlib import Path
from typing import Tuple, Optional

# --- 核心修改：引入Accelerate库实现CPU Offloading ---
from accelerate import Accelerator, dispatch_model
from accelerate.utils import get_balanced_memory
from tqdm import tqdm
from dynamic_data_processor import create_dynamic_data


class WindPowerOffloadingPredictor:
    """
    集成CPU Offloading技术的风电功率预测系统
    突破GPU显存限制，支持超大规模序列预测
    """
    
    def __init__(self, args, csv_path: str):
        """
        初始化预测系统
        Args:
            args: 命令行参数
            csv_path: CSV数据文件路径
        """
        self.args = args
        self.csv_path = csv_path
        
        # --- 核心修改：使用Accelerator来管理设备和内存 ---
        print("🚀 初始化CPU Offloading系统...")
        # 由于FFT计算的限制，暂时禁用混合精度
        self.accelerator = Accelerator(
            mixed_precision='fp16',  # 禁用混合精度避免FFT问题
            gradient_accumulation_steps=1
        )
        
        print(f"📱 Accelerator设备: {self.accelerator.device}")
        print(f"🔧 进程数量: {self.accelerator.num_processes}")
        print(f"⚡ 混合精度模式: {self.accelerator.mixed_precision}")
        
        # 设置设备
        self.device = self.accelerator.device
        
        # 初始化数据相关属性
        self.scaler = None
        self.use_standardized = False
        self.feature_index = None
        
        # 模型相关
        self.model = None
        self.optimizer = None
        self.loss_function = nn.MSELoss()
        
        # 训练状态
        self.best_mse = float('inf')
        self.patience_counter = 0
        
        print("✅ 系统初始化完成")

    def setup_data(self):
        """设置和加载数据（使用动态数据处理器）"""
        print("📊 开始动态数据设置...")
        
        print(f"🔧 使用动态数据处理系统")
        print(f"   序列长度: {self.args.seq_length}")
        print(f"   预测长度: {self.args.c_out}")
        print(f"   数据来源: {self.csv_path}")
        
        # 创建动态数据
        try:
            model_data = create_dynamic_data(
                csv_path=self.csv_path,
                seq_length=self.args.seq_length,
                pred_length=self.args.c_out,  # 预测长度
                split_ratio=self.args.split_ratio,
                standardize=True,  # 总是进行标准化
                save_dir=None,  # 不保存中间文件
                verbose=True
            )
            
            # 提取数据（不立即移动到设备，让Accelerate处理）
            self.X_train = model_data['X_train']
            self.y_train = model_data['y_train']
            self.X_test = model_data['X_test']
            self.y_test = model_data['y_test']
            self.train_dict = model_data['train_edge_indices']
            self.test_dict = model_data['test_edge_indices']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.num_features = model_data['num_features']
            
            # 标记使用标准化数据
            self.use_standardized = True
            
            # 为了兼容性，创建feature_index
            self.feature_index = {feature: index for index, feature in enumerate(self.feature_names)}
            
            print(f"✅ 动态数据处理完成")
            print(f"   训练集: X{self.X_train.shape}, y{self.y_train.shape}")
            print(f"   测试集: X{self.X_test.shape}, y{self.y_test.shape}")
            print(f"   特征数量: {self.num_features}")
            print("✅ 数据设置完成")
            
        except Exception as e:
            print(f"❌ 动态数据处理失败: {e}")
            print("请检查CSV文件路径和参数设置")
            raise e

    def setup_model_with_offloading(self):
        """
        设置模型并应用CPU Offloading技术（恢复为全自动、高性能版本）
        """
        print("🔧 构建支持CPU Offloading的模型...")
        
        # 1. 在CPU上初始化模型
        model = adap_auto(
            n_head=self.args.n_head,
            hidden_size=self.args.hidden_size,
            factor=self.args.factor,
            dropout=self.args.dropout,
            conv_hidden_size=self.args.conv_hidden_size,
            MovingAvg_window=self.args.moving_avg_window,
            activation=self.args.activation,
            encoder_layers=self.args.encoder_layers,
            decoder_layers=self.args.decoder_layers,
            c_in=self.num_features,
            seq_lenth=self.args.seq_length,
            c_out=self.args.c_out,
            gruop_dec=self.args.group_dec
        )
        
        # 2. --- 恢复为全自动、高性能的Offloading流程 ---
        try:
            from accelerate.utils import infer_auto_device_map
            print("🗺️  生成智能设备映射...")
            print(f"   指定的GPU显存上限: {self.args.max_gpu_memory}")
            device_map = infer_auto_device_map(
                model,
                max_memory={0: self.args.max_gpu_memory},
                no_split_module_classes=getattr(model, '_no_split_modules', [])
            )
            print("📋 成功生成的设备映射:", device_map)
            print("🔄 应用CPU Offloading设备映射 (启用异步流水线)...")
            # self.model = dispatch_model(
            #     model, 
            #     device_map="auto",
            #     max_memory={0: self.args.max_gpu_memory} # 显存限制在这里生效
            # )
            self.model = dispatch_model(model, device_map=device_map)
            print("✅ 全自动CPU Offloading设置成功！")
            # 打印一下 Accelerate 自动生成的设备映射，看看它把层放到了哪里
            print("📋 Accelerate自动生成的设备映射:")
            print(self.model.hf_device_map)
            # 使用 get_balanced_memory 并传入从命令行获取的显存限制
            # device_map = get_balanced_memory(
            #     model,
            #     max_memory={0: self.args.max_gpu_memory}, # <--- 在这里使用命令行参数
            #     no_split_module_classes=getattr(model, '_no_split_modules', [])
            # )
            
            # print("📋 生成的设备映射:", device_map)
            # print("🔄 应用CPU Offloading设备映射 (启用异步流水线)...")
    
            # # 使用 dispatch_model 来启用所有性能优化
            # self.model = dispatch_model(model, device_map=device_map)
            
            # print("✅ 全自动CPU Offloading设置成功！")
            
        except Exception as e:
            print(f"⚠️ 自动Offloading失败: {e}。")
            # 如果自动分配失败，可以考虑回退到纯CPU模式
            print("📱 回退到纯CPU模式...")
            self.model = model.to('cpu')
    
        # 设置优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr, 
            weight_decay=self.args.weight_decay
        )
        
        # 使用Accelerate准备模型和优化器
        if hasattr(self.model, 'hf_device_map'):
            print("📋 模型已使用device_map，仅准备优化器和数据加载器。")
            self.optimizer = self.accelerator.prepare(self.optimizer)
        else:
            print("📋 标准模式，使用accelerator.prepare准备所有组件。")
            self.model, self.optimizer = self.accelerator.prepare(
                self.model, self.optimizer
            )

    def create_offloading_dataloader(self):
        """创建支持CPU Offloading的数据加载器"""
        print("📦 创建高效数据加载器...")
        
        # 创建训练数据集
        train_dataset = TensorDataset(self.X_train, self.y_train)
        
        # --- 使用较小的batch_size以配合CPU Offloading ---
        # CPU-GPU数据传输有开销，较小的batch可以减少单次传输延迟
        effective_batch_size = min(self.args.batch_size, 64)  # 限制最大batch_size
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=effective_batch_size,
            shuffle=True,
            pin_memory=True,  # 加速CPU-GPU传输
            num_workers=0     # 多进程数据加载
        )
        
        # 保存原始batch_size信息
        self.original_batch_size = effective_batch_size
        
        # 使用Accelerate准备数据加载器
        self.train_dataloader = self.accelerator.prepare(train_dataloader)
        
        print(f"📊 有效批次大小: {effective_batch_size}")
        print("✅ 数据加载器准备完成")

    def train_with_offloading(self):
        """
        使用CPU Offloading技术进行模型训练
        支持更大规模的模型和数据
        """
        print("🏋️  开始CPU Offloading训练...")
        print(f"🎯 目标epochs: {self.args.epochs}")
        print(f"⏰ 早停耐心值: {self.args.patience}")
        
        for epoch in range(self.args.epochs):
            start_time = time.time()
            
            # 训练阶段
            self.model.train()
            total_loss = 0
            batch_count = 0
            
            # 使用进度条显示训练进度
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.args.epochs}",
                leave=False
            )
            
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                # 获取对应的边字典
                start_idx = batch_idx * self.original_batch_size
                end_idx = start_idx + len(inputs)
                dicts = self.train_dict[start_idx:end_idx]
                
                # --- 核心：模型前向传播（Accelerate自动处理设备映射）---
                self.optimizer.zero_grad()
                
                # 这里是CPU Offloading的魔法时刻！
                # 当调用model时，异步流水线开始工作
                with self.accelerator.autocast():  # 混合精度计算
                    model_output = self.model(inputs, dicts).squeeze(-1)
                    loss = self.loss_function(model_output, targets)
                    
                    # L1正则化
                    l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                    loss = loss + self.args.l1_lambda * l1_norm
                
                # 反向传播（Accelerate自动处理梯度缩放）
                self.accelerator.backward(loss)
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg_Loss': f'{total_loss/batch_count:.4f}'
                })
                
                # 定期清理GPU缓存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            # 验证阶段
            val_mse = self._validate_with_offloading()
            
            # 早停检查
            if self._check_early_stopping(val_mse, epoch):
                break
            
            # 计算epoch时间
            end_time = time.time()
            epoch_time = end_time - start_time
            
            print(f"📊 Epoch {epoch + 1}/{self.args.epochs}")
            print(f"   损失: {total_loss / len(self.train_dataloader):.4f}")
            print(f"   验证MSE: {val_mse:.4f}")
            print(f"   用时: {epoch_time:.2f}秒")
            print(f"   最佳MSE: {self.best_mse:.4f}")
            
        print("🎉 训练完成！")

    def _validate_with_offloading(self) -> float:
        """使用CPU Offloading进行验证"""
        self.model.eval()
        X_test_on_device = self.X_test.to(self.accelerator.device)
        with torch.no_grad():
            # --- CPU Offloading在这里也发挥作用 ---
            # 大规模验证数据的处理变得可能
            prediction = self.model(X_test_on_device, self.test_dict).squeeze(-1)
            
            if self.use_standardized and self.scaler is not None:
                # 标准化数据的处理
                y_test_original = self._inverse_transform_power(
                    self.y_test.cpu().numpy(), self.scaler, power_feature_idx=0
                )
                prediction_original = self._inverse_transform_power(
                    prediction.cpu().numpy(), self.scaler, power_feature_idx=0
                )
                val_mse = MSE(y_test_original, prediction_original)
            else:
                # 原始数据的处理
                val_mse = MSE(self.y_test.cpu().numpy(), prediction.cpu().numpy())
        
        # 清理内存
        torch.cuda.empty_cache()
        return val_mse

    def predict_with_offloading(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        使用CPU Offloading进行最终预测
        Returns:
            (predictions, targets, mse, mape): 预测结果和评估指标
        """
        print("🔮 开始CPU Offloading预测...")
        
        # 加载最佳模型（如果存在）
        # model_path = self._get_model_path()
        # if model_path.exists():
        #     print(f"📥 加载最佳模型: {model_path}")
        #     # 注意：加载带有device_map的模型需要特殊处理
        #     try:
        #         self.model = torch.load(str(model_path), map_location='cpu')
        #         if not hasattr(self.model, 'hf_device_map'):
        #             self.model = self.model.to(self.device)
        #     except Exception as e:
        #         print(f"⚠️  模型加载警告: {e}")
        
        self.model.eval()
        X_test_on_device = self.X_test.to(self.accelerator.device)
        with torch.no_grad():
            # --- CPU Offloading让我们能处理更大的测试集 ---
            print("⚡ 执行大规模预测（CPU Offloading加速）...")
            prediction = self.model(X_test_on_device, self.test_dict).squeeze(-1)
        
        # 转换为numpy进行评估
        y_test_np = self.y_test.cpu().numpy()
        prediction_np = prediction.detach().cpu().numpy()
        
        if self.use_standardized and self.scaler is not None:
            # 反标准化处理
            print("🔄 执行反标准化...")
            y_test_original = self._inverse_transform_power(y_test_np, self.scaler, 0)
            prediction_original = self._inverse_transform_power(prediction_np, self.scaler, 0)
            
            mse_result = MSE(y_test_original, prediction_original)
            mape_result = MAPE(y_test_original, prediction_original)
            
            print(f"📈 原始尺度 - MSE: {mse_result:.6f}, MAPE: {mape_result:.6f}")
            return prediction_original, y_test_original, mse_result, mape_result
        else:
            # 原始数据处理
            mse_result = MSE(y_test_np, prediction_np)
            mape_result = MAPE(y_test_np, prediction_np)
            
            print(f"📈 MSE: {mse_result:.6f}, MAPE: {mape_result:.6f}")
            return prediction_np, y_test_np, mse_result, mape_result

    def save_results(self, predictions: np.ndarray, targets: np.ndarray, mse: float, mape: float):
        """保存预测结果"""
        print("💾 保存预测结果...")
        
        # 创建结果目录
        try:
            base_dir = Path('/home/forecasting/pts/results/fujian')
            predictions_dir = base_dir / 'forecasting_offloading_dynamic'
            predictions_dir.mkdir(parents=True, exist_ok=True)
            
            # 计算误差
            error = targets - predictions
            
            # 创建结果DataFrame
            data = {
                'adap_auto_offloading': predictions.flatten(),
                'real': targets.flatten(),
                'error': error.flatten()
            }
            df_results = pd.DataFrame(data)
            
            # 使用序列长度作为标识符
            model_identifier = f'seq{self.args.seq_length}_pred{self.args.c_out}_{self.args.hyperparam_id}'
            
            # 保存CSV文件
            predictions_path = predictions_dir / f'prediction_adap_auto_offloading_{model_identifier}.csv'
            df_results.to_csv(str(predictions_path), index=False)
            
            print(f"📊 结果已保存到: {predictions_path}")
        
            # 保存性能报告
            self._save_performance_report(predictions_path.parent, mse, mape)
        except Exception as e:
            print(f"❌ 保存结果时发生错误: {e}")
            # 即使保存失败，也不要让整个程序崩溃
            pass
    def _save_performance_report(self, save_dir: Path, mse: float, mape: float):
        """保存性能报告"""
        model_identifier = f'seq{self.args.seq_length}_pred{self.args.c_out}_{self.args.hyperparam_id}'
        report_path = save_dir / f'performance_report_offloading_{model_identifier}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("🚀 CPU Offloading风电预测性能报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"数据处理模式: 动态滑动窗口\n")
            f.write(f"序列长度: {self.args.seq_length}\n")
            f.write(f"预测长度: {self.args.c_out}\n")
            f.write(f"超参数ID: {self.args.hyperparam_id}\n")
            f.write(f"数据类型: {'标准化数据' if self.use_standardized else '原始数据'}\n")
            f.write(f"使用Scaler: {'是' if self.scaler is not None else '否'}\n")
            f.write(f"特征数量: {self.num_features}\n")
            f.write(f"训练集大小: {self.X_train.shape}\n")
            f.write(f"测试集大小: {self.X_test.shape}\n")
            f.write(f"\n📊 性能指标:\n")
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"MAPE: {mape:.6f}\n")
            f.write(f"最佳验证MSE: {self.best_mse:.6f}\n")
            f.write(f"\n🔧 模型配置:\n")
            f.write(f"隐藏层大小: {self.args.hidden_size}\n")
            f.write(f"注意力头数: {self.args.n_head}\n")
            f.write(f"编码器层数: {self.args.encoder_layers}\n")
            f.write(f"解码器层数: {self.args.decoder_layers}\n")
            f.write(f"学习率: {self.args.lr}\n")
            f.write(f"批次大小: {self.args.batch_size}\n")
            f.write(f"Dropout: {self.args.dropout}\n")
        
        print(f"📋 性能报告已保存到: {report_path}")

    # 辅助方法
    def _is_standardized_data(self, train_data_path: str, val_data_path: str) -> bool:
        """检测数据是否为标准化数据"""
        train_filename = os.path.basename(train_data_path)
        val_filename = os.path.basename(val_data_path)
        
        print(f"🔍 文件名检测: {train_filename}, {val_filename}")
        
        if train_filename.startswith('std') or val_filename.startswith('std'):
            print(f"✅ 检测结果: 标准化数据 (文件名以'std'开头)")
            return True
        
        if 'std' in train_filename.lower() or 'std' in val_filename.lower():
            print(f"✅ 检测结果: 标准化数据 (文件名包含'std')")
            return True
        
        standardized_keywords = ['standard', 'standardized', 'norm', 'normalized']
        for keyword in standardized_keywords:
            if keyword in train_filename.lower() or keyword in val_filename.lower():
                print(f"✅ 检测结果: 标准化数据 (文件名包含'{keyword}')")
                return True
        
        print(f"❌ 检测结果: 原始数据")
        return False

    def _load_scaler(self):
        """加载标准化器"""
        data_dir = os.path.dirname(self.data_paths['train_dir'])
        scaler_path = os.path.join(data_dir, 'scaler.pkl')
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ 已加载标准化器，特征数量: {len(self.scaler.mean_)}")
        else:
            print(f"⚠️  警告: 检测到标准化数据但未找到scaler.pkl文件")
            self.use_standardized = False

    def _inverse_transform_power(self, data: np.ndarray, scaler, power_feature_idx: int = 0) -> np.ndarray:
        """对功率数据进行反标准化"""
        mean = scaler.mean_[power_feature_idx]
        scale = scaler.scale_[power_feature_idx]
        return data * scale + mean

    def _check_early_stopping(self, val_mse: float, epoch: int) -> bool:
        """检查早停条件"""
        if val_mse < self.best_mse:
            self.best_mse = val_mse
            self.patience_counter = 0
            
            # 保存最佳模型
            if self.args.save_model:
                self._save_best_model()
            return False
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.args.patience:
            print(f'🛑 早停于epoch {epoch+1}')
            return True
        
        return False

    def _save_best_model(self):
        """保存最佳模型"""
        model_path = self._get_model_path()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 对于使用device_map的模型，需要特殊保存方式
        try:
            torch.save(self.model, str(model_path))
        except Exception as e:
            print(f"⚠️  模型保存警告: {e}")

    def _get_model_path(self) -> Path:
        """获取模型保存路径"""
        base_dir = Path('/home/forecasting/pts/results/adap_auto')
        models_dir = base_dir / 'models_offloading'
        model_identifier = f'seq{self.args.seq_length}_pred{self.args.c_out}_{self.args.hyperparam_id}'
        return models_dir / f'best_model_adap_auto_offloading_{model_identifier}.pth'


def seed_everything(seed=42):
    """设置所有随机种子以确保结果可重复"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """主函数：集成CPU Offloading的风电预测"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='风电功率预测 - CPU Offloading版本')
    parser.add_argument('--gpu', type=int, default=1, help='GPU设备ID')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    parser.add_argument('--l1_lambda', type=float, default=0.15, help='L1正则化系数')
    parser.add_argument('--weight_decay', type=float, default=0.15, help='L2权重衰减')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    parser.add_argument('--split_ratio', type=float, default=0.99, help='训练/测试划分比例')
    parser.add_argument('--seed', type=int, default=2, help='随机种子')
    parser.add_argument('--dataset_name', type=str, default='6-0_1', help='数据集名称')
    parser.add_argument('--hyperparam_id', type=str, default='offloading', help='超参数组合ID')
    parser.add_argument('--save_model', action='store_true', default=True, help='保存最佳模型')
    parser.add_argument('--max_gpu_memory', type=str, default='10GiB', help='GPU显存限制')
    
    # 模型架构参数
    parser.add_argument('--n_head', type=int, default=8, help='注意力头数')
    parser.add_argument('--hidden_size', type=int, default=264, help='隐藏层大小')
    parser.add_argument('--factor', type=int, default=2, help='注意力因子')
    parser.add_argument('--conv_hidden_size', type=int, default=32, help='卷积隐藏层大小')
    parser.add_argument('--moving_avg_window', type=int, default=3, help='移动平均窗口大小')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    parser.add_argument('--encoder_layers', type=int, default=1, help='编码器层数')
    parser.add_argument('--decoder_layers', type=int, default=1, help='解码器层数')
    parser.add_argument('--seq_length', type=int, default=36, help='序列长度')
    parser.add_argument('--c_out', type=int, default=1, help='输出通道数')
    parser.add_argument('--group_dec', action='store_true', default=True, help='使用组解码器')
    
    args = parser.parse_args()
    
    # 设置随机种子
    seed_everything(seed=args.seed)
    
    # 设置CSV数据路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, f'../data/fujian/Offshore Wind Farm Dataset3(WT1).csv')
    
    # 设置多进程
    mp.set_start_method('spawn', force=True)
    
    print("🌊 =" * 30)
    print("🌊 CPU Offloading风电预测系统启动")
    print("🌊 =" * 30)
    print(f"🎯 序列长度: {args.seq_length}")
    print(f"🎯 预测长度: {args.c_out}")
    print(f"🎯 GPU显存限制: {args.max_gpu_memory}")
    print(f"🎯 超参数ID: {args.hyperparam_id}")
    print(f"🎯 数据源: {csv_path}")
    
    try:
        # 初始化预测系统
        print("DEBUG: 步骤1 - 初始化Predictor类...")
        predictor = WindPowerOffloadingPredictor(args, csv_path)
        print("DEBUG: 步骤1 - 完成。")
        
        # 设置数据
        print("DEBUG: 步骤2 - 开始数据设置 (setup_data)...")
        predictor.setup_data()
        print("DEBUG: 步骤2 - 完成。")
        
        # 设置模型和CPU Offloading
        print("DEBUG: 步骤3 - 开始模型设置 (setup_model_with_offloading)...")
        predictor.setup_model_with_offloading()
        print("DEBUG: 步骤3 - 完成。")
        
        # 创建数据加载器
        print("DEBUG: 步骤4 - 开始创建数据加载器 (create_offloading_dataloader)...")
        predictor.create_offloading_dataloader()
        print("DEBUG: 步骤4 - 完成。")
        
        # 训练模型
        print("DEBUG: 步骤5 - 即将开始训练 (train_with_offloading)...")
        predictor.train_with_offloading()
        print("DEBUG: 步骤5 - 训练已结束。")
        
        # 预测和评估
        predictions, targets, mse, mape = predictor.predict_with_offloading()
        
        # 保存结果
        predictor.save_results(predictions, targets, mse, mape)
        
        # 输出最终总结
        print("\n" + "🎉" * 60)
        print("🎉 CPU Offloading风电预测完成!")
        print("🎉" * 60)
        print(f"✨ 数据类型: {'标准化数据' if predictor.use_standardized else '原始数据'}")
        print(f"✨ 最终MSE: {mse:.6f}")
        print(f"✨ 最终MAPE: {mape:.6f}")
        print(f"✨ 突破了传统GPU显存限制！")
        print("🎉" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        return 130
    except Exception as e:
        print(f"\n💥 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    exit_code = main()
    sys.exit(exit_code) 