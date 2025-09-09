#!/usr/bin/env python3
"""
基线模型鲁棒性分析适配器
直接使用 robustness_analysis_experiment.py 中的所有函数，只替换模型创建和训练部分
"""

import os
import sys
import warnings
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time
# 抑制警告
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 确保优先使用 adap_auto/new_hier 下的模块（如 DataEmbedding、adap_auto_x）
_ex_dir = os.path.dirname(os.path.abspath(__file__))            # .../adap_auto/new_hier/ex-experiment
_new_hier_dir = os.path.dirname(_ex_dir)                        # .../adap_auto/new_hier
if _new_hier_dir not in sys.path:
	sys.path.insert(0, _new_hier_dir)
# 清理可能的冲突缓存
for _m in ['DataEmbedding', 'adap_auto']:
	sys.modules.pop(_m, None)

# 导入所有鲁棒性分析函数
from robustness_analysis_experiment import (
	run_robustness_experiment,
	evaluate_model_robustness,
	load_window_data_directly,
	seed_everything,
	save_results,
	load_baseline_results,
	create_robustness_visualizations,
	split_data_by_season_independent,
	inverse_or_identity
)
# 确保可以导入到adap_auto/new_hier/evaluate.py
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(os.path.dirname(_current_dir))
if _parent_dir not in sys.path:
	sys.path.insert(0, _parent_dir)
from evaluate import MSE, MAPE

def create_baseline_model_adapter(model_name, seq_len, pred_len, enc_in, device):
	"""
	创建基线模型的适配器函数
	直接导入并使用现有的模型类
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	# 从 adap_auto/new_hier/ex-experiment 到项目根目录需要上3级
	project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
	
	# 临时保存原始sys.path
	original_path = sys.path.copy()
	
	if model_name == 'DLinear':
		# 添加DLinear路径并导入
		dlinear_path = os.path.join(project_root, 'DLinear')
		if dlinear_path not in sys.path:
			sys.path.insert(0, dlinear_path)
		
		try:
			from dlinear_adapted import DLinear  # 使用适配版本
			model = DLinear(
				input_size=seq_len,
				h=1,                     # 单特征预测（目标特征）
				c_in=enc_in,             # 输入特征数
				c_out=pred_len,          # 预测步长
				MovingAvg_window=25,
				dropout=0.05,
				individual=True
			)
		except ImportError as e:
			print(f"无法导入DLinear: {e}")
			raise
				
	elif model_name == 'iTransformer':
		# 添加iTransformer路径并导入
		itransformer_path = os.path.join(project_root, 'iTransformer')
		
		# 清除已加载的同名模块缓存（关键）
		for m in ['DataEmbedding', 'Encoder', 'attention', 'iTransformer', 'FEDformer']:
			sys.modules.pop(m, None)

		# 把目标模型目录插到 sys.path 最前
		sys.path.insert(0, itransformer_path)
		try:
			from iTransformer import iTransformer
			model = iTransformer(
				input_size=seq_len,      # 输入序列长度
				c_out=pred_len,          # 预测步长
				h=1,                     # 单特征预测
				hidden_size=256,         # 隐藏层大小
				n_heads=8,               # 注意力头数
				d_ff=512,                # 前馈网络维度
				factor=1,
				dropout=0.1,
				e_layers=2,              # 编码器层数
				d_layers=1,              # 解码器层数
				use_norm=True
			)
		except ImportError as e:
			print(f"无法导入iTransformer: {e}")
			raise
		finally:
			sys.path.pop(0)
				
	elif model_name == 'FEDformer':
		# 添加FEDformer路径并导入 (注意文件夹名是FEDfomer)
		fedformer_path = os.path.join(project_root, 'FEDfomer')
		
		# 清除已加载的同名模块缓存（关键）
		for m in ['DataEmbedding', 'Encoder', 'attention', 'iTransformer', 'FEDformer', 'seriesDecomp']:
			sys.modules.pop(m, None)

		# 把目标模型目录插到 sys.path 最前
		sys.path.insert(0, fedformer_path)
		try:
			from FEDformer import FEDformer
			model = FEDformer(
				input_size=seq_len,
				version="Fourier",
				modes=64,
				mode_select="ran",
				hidden_size=128,
				dropout=0.05,
				n_head=8,
				conv_hidden_size=32,
				activation="gelu",
				encoder_layers=2,
				decoder_layers=1,
				MovingAvg_window=25,
				c_in=enc_in,
				c_out=pred_len,
				h=1
			)
		except ImportError as e:
			print(f"无法导入FEDformer: {e}")
			raise
		finally:
			sys.path.pop(0)
				
	elif model_name == 'NBEATSx':
		# 添加NBEATSx路径并导入
		nbeatsx_path = os.path.join(project_root, 'nbeatsx')
		# 清除已加载模块缓存
		for m in ['NBEATSx']:
			sys.modules.pop(m, None)

		# 临时插入路径
		sys.path.insert(0, nbeatsx_path)
		try:
			from NBEATSx import NBEATSx
			model = NBEATSx(
				seq_len=seq_len,
				pred_len=pred_len,
				enc_in=1,  # NBEATSx只使用功率特征
				c_out=1,
				n_harmonics=2,
				n_polynomials=2,
				stack_types=["identity", "trend", "seasonality"],
				n_blocks=[1, 1, 1],
				mlp_units=[[512, 512], [512, 512], [512, 512]],
				dropout_prob_theta=0.05,
				activation="ReLU",
				shared_weights=False
			)
		except ImportError as e:
			print(f"无法导入NBEATSx: {e}")
			raise
		finally:
			sys.path.pop(0)
	else:
		raise ValueError(f"不支持的模型: {model_name}")
	
	# 恢复原始sys.path
	sys.path = original_path
	
	return model.to(device)

def run_baseline_robustness_experiment(dataset_name, prediction_scale, args, device, model_name):
	"""
	运行基线模型的鲁棒性分析实验
	直接复用 robustness_analysis_experiment.py 的逻辑，只替换模型部分
	"""
	print(f"\n🔬 开始 {model_name} 鲁棒性分析实验")
	print(f"数据集: {dataset_name}")
	print(f"预测尺度: {prediction_scale}")
	
	start_time = time.time()
	
	try:
		# 1. 数据准备 - 直接使用现有函数
		print("📊 准备数据...")
		model_data = load_window_data_directly(
			dataset_name=dataset_name,
			prediction_scale=prediction_scale,
			seq_length=args.seq_length,
			c_out=None,
			split_ratio=args.split_ratio,
			use_std=False if model_name in ['DLinear', 'NBEATSx'] else True
		)
		
		X_train = model_data['X_train'].to(device)
		y_train = model_data['y_train'].to(device)
		X_test = model_data['X_test'].to(device)
		y_test = model_data['y_test'].to(device)
		train_edge_indices = model_data['train_edge_indices']
		test_edge_indices = model_data['test_edge_indices']
		scaler = model_data['scaler']
		
		print(f"训练集形状: {X_train.shape}, {y_train.shape}")
		print(f"测试集形状: {X_test.shape}, {y_test.shape}")
		print(f"特征数量: {model_data['num_features']}")
		print(f"pred_len: {model_data['pred_length']}")
		
		# 2. 模型初始化 - 使用适配器创建基线模型
		print(f"🏗️ 初始化 {model_name} 模型...")
		model = create_baseline_model_adapter(
			model_name=model_name,
			seq_len=args.seq_length,
			pred_len=model_data['pred_length'],  # 不要写死 1
			enc_in=model_data['num_features'],
			device=device
		)
		
		# 2.1 训练前形状自检（小批次干跑）
		print("🧪 形状自检(dry-run)...")
		wrapper = BaselineModelWrapper(model, model_name)
		with torch.no_grad():
			bs = min(16, X_train.shape[0])
			out = wrapper(X_train[:bs])
			print(f"  dry-run output: {tuple(out.shape)}, target: {tuple(y_train[:bs].shape)}")
			if out.shape != y_train[:bs].shape:
				raise ValueError(f"模型输出形状{tuple(out.shape)}与标签{tuple(y_train[:bs].shape)}不一致，请检查pred_len/输入处理。")
		
		# 3. 训练并评估（支持季节模式）
		if args.seasonal_mode == 'independent':
			print("🌸 使用季节独立模式...")
			seasonal_data = split_data_by_season_independent({
				'X_train': X_train, 'y_train': y_train,
				'X_test': X_test, 'y_test': y_test,
				'train_edge_indices': train_edge_indices,
				'test_edge_indices': test_edge_indices,
				'scaler': scaler,
			}, device, season_split_ratio=args.season_split_ratio)

			seasonal_results = {}
			for season, data in seasonal_data.items():
				print(f"\n🔬 训练 {season} 季节的 {model_name} 模型...")
				season_model = create_baseline_model_adapter(
					model_name=model_name,
					seq_len=args.seq_length,
					pred_len=model_data['pred_length'],
					enc_in=model_data['num_features'],
					device=device
				)
				season_model = train_baseline_model_standard(
					season_model,
					data['X_train'], data['y_train'],
					data['X_test'], data['y_test'],
					args, device, model_name
				)
				# 评估
				season_model.eval()
				with torch.no_grad():
					season_wrapper = BaselineModelWrapper(season_model, model_name)
					preds_np = season_wrapper(data['X_test']).cpu().numpy()  # [batch, pred_len]
					y_true_np = data['y_test'].cpu().numpy()               # [batch, pred_len]
					y_true_denorm = inverse_or_identity(y_true_np, scaler, power_feature_idx=0)
					preds_denorm = inverse_or_identity(preds_np, scaler, power_feature_idx=0)
					mse = MSE(y_true_denorm, preds_denorm)
					mape = MAPE(y_true_denorm, preds_denorm)
				seasonal_results[season] = {
					'MSE': mse,
					'MAPE': mape,
					'train_samples': data.get('train_samples', len(data['X_train'])),
					'test_samples': data.get('test_samples', len(data['X_test'])),
					'total_samples': data.get('total_samples', len(data['X_train']) + len(data['X_test']))
				}
				print(f"  {season}: MSE={mse:.4f}, MAPE={mape:.2f}%")

			robustness_results = {
				'seasonal_performance': seasonal_results,
				'seasonal_mode': 'independent'
			}
		else:
			print(f"�� 开始训练 {model_name} 模型...")
			model = train_baseline_model_standard(
				model, X_train, y_train, X_test, y_test, args, device, model_name
			)
			print("🔍 开始鲁棒性评估...")
			robustness_results = evaluate_model_robustness(
				model=wrapper,
				X_test=X_test,
				y_test=y_test,
				test_edge_indices=test_edge_indices,
				scaler=scaler,
				device=device,
				noise_levels=args.noise_levels,
				missing_ratios=args.missing_ratio,
				dataset_name=dataset_name,
				include_seasonal_in_default=args.include_seasonal_eval
			)
		
		# 添加模型信息
		training_time = time.time() - start_time
		robustness_results['training_time'] = training_time
		robustness_results['model_name'] = model_name
		robustness_results['dataset'] = dataset_name
		robustness_results['prediction_scale'] = prediction_scale
		if 'seasonal_mode' not in robustness_results:
			robustness_results['seasonal_mode'] = 'test_split'
		
		return robustness_results
		
	except Exception as e:
		print(f"❌ {model_name} 实验失败: {str(e)}")
		import traceback
		traceback.print_exc()
		return None

class BaselineModelWrapper:
	"""
	基线模型包装器，使其接口与 adap_auto 兼容
	"""
	def __init__(self, model, model_name):
		self.model = model
		self.model_name = model_name
	
	def __call__(self, X, edge_indices=None):
		"""
		统一的调用接口
		edge_indices 对基线模型无用，忽略
		"""
		if self.model_name == 'NBEATSx':
			# NBEATSx 只需要功率特征，且期望三维输入 [B, L, 1]
			X_power = X[:, :, 0:1]  # [batch, seq_len, 1]
			output = self.model(X_power)
		else:
			output = self.model(X)
		
		# 统一输出形状为 [batch, pred_len]
		if output.ndim == 3 and output.shape[-1] == 1:
			output = output.squeeze(-1)
		# 不要将 [batch, pred_len] 误裁成 [batch, 1]
		return output
	
	def eval(self):
		return self.model.eval()
	
	def train(self):
		return self.model.train()

def train_baseline_model_standard(model, X_train, y_train, X_test, y_test, args, device, model_name):
	"""
	标准的基线模型训练函数
	"""
	from torch.utils.data import DataLoader, TensorDataset
	import time
	
	# 准备数据加载器
	train_dataset = TensorDataset(X_train, y_train)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	
	# 优化器和损失函数
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	criterion = nn.MSELoss()
	best_model_state = None
	
	# 训练循环
	best_val_loss = float('inf')
	patience_counter = 0
	
	for epoch in range(args.epochs):
		# 训练阶段
		model.train()
		train_loss = 0.0
		
		for batch_X, batch_y in train_loader:
			batch_X, batch_y = batch_X.to(device), batch_y.to(device)
			
			optimizer.zero_grad()
			
			# 根据模型类型调整输入
			if model_name == 'NBEATSx':
				batch_X_input = batch_X[:, :, 0:1]  # 只使用功率特征，保持 [B, L, 1]
			else:
				batch_X_input = batch_X
			
			outputs = model(batch_X_input)
			
			# 确保输出维度匹配
			if len(outputs.shape) > len(batch_y.shape):
				outputs = outputs.squeeze(-1)
			
			loss = criterion(outputs, batch_y)
			
			# L1正则化
			if hasattr(args, 'l1_lambda') and args.l1_lambda > 0:
				l1_reg = torch.tensor(0.).to(device)
				for param in model.parameters():
					l1_reg += torch.norm(param, 1)
				loss += args.l1_lambda * l1_reg
			
			loss.backward()
			optimizer.step()
			
			train_loss += loss.item()
		
		# 验证阶段
		model.eval()
		val_loss = 0.0
		
		with torch.no_grad():
			if model_name == 'NBEATSx':
				X_test_input = X_test[:, :, 0:1]
			else:
				X_test_input = X_test
				
			prediction = model(X_test_input)
			if len(prediction.shape) > len(y_test.shape):
				prediction = prediction.squeeze(-1)
			val_loss = criterion(prediction, y_test).item()
		
		train_loss /= len(train_loader)
		
		if epoch % 5 == 0:
			print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
		
		# 早停检查
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			patience_counter = 0
			best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
		else:
			patience_counter += 1
			if patience_counter >= args.patience:
				print(f"早停触发，在第 {epoch+1} 轮停止训练")
				break
	
	# 加载最佳模型
	if best_model_state is not None:
		model.load_state_dict(best_model_state)
	else:
		print("⚠️ 未捕获到更优模型参数，使用最后一轮参数")
	print(f"✅ {model_name} 模型训练完成")
	
	return model

def main():
	import argparse
	import time
	from pathlib import Path
	
	parser = argparse.ArgumentParser(description='Baseline Model Robustness Analysis')
	
	# 数据集相关参数
	parser.add_argument('--dataset', type=str, default='fujian', choices=['fujian', 'DSWE'], 
						help='Dataset name')
	parser.add_argument('--prediction_scale', type=str, default='6-0_1', 
						help='Prediction scale (e.g., 6-0_1, 24-1, etc.)')
	parser.add_argument('--model', type=str, required=False,
						choices=['DLinear', 'iTransformer', 'FEDformer', 'NBEATSx'],
						help='Baseline model to test')
	parser.add_argument('--models', nargs='+', type=str, required=False,
						choices=['DLinear', 'iTransformer', 'FEDformer', 'NBEATSx'],
						help='Run multiple baseline models in sequence')
	
	# 训练相关参数
	parser.add_argument('--gpu', type=int, default=1, help='GPU device id')
	parser.add_argument('--epochs', type=int, default=15, help='Maximum number of training epochs')
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
	parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
	parser.add_argument('--l1_lambda', type=float, default=0.01, help='L1 regularization coefficient')
	parser.add_argument('--weight_decay', type=float, default=0.05, help='L2 weight decay')
	parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
	parser.add_argument('--split_ratio', type=float, default=0.99, help='Train/test split ratio')
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--seq_length', type=int, default=36, help='Sequence length')
	
	# 鲁棒性测试参数
	parser.add_argument('--noise_levels', nargs='+', type=float, default=[0.05, 0.1],
						help='Noise levels for robustness testing')
	parser.add_argument('--missing_ratio', nargs='+', type=float, default=[0.05, 0.1],
						help='Missing data ratio(s) for robustness testing, e.g. --missing_ratio 0.05 0.1')
	parser.add_argument('--include_seasonal_eval', action='store_true', default=False,
						help='Include seasonal evaluation in default (test_split) mode')
	parser.add_argument('--seasonal_mode', type=str, default='independent', choices=['test_split', 'independent'],
						help='Seasonal analysis mode for baselines')
	parser.add_argument('--season_split_ratio', type=float, default=0.95,
						help='Train split ratio within each season in independent mode')
	
	args = parser.parse_args()
	
	# 设置随机种子
	seed_everything(seed=args.seed)
	
	# 设置设备
	torch.cuda.set_device(args.gpu)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	
	# 创建结果保存目录
	script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
	results_dir = script_dir / 'results' / 'baseline_robustness'
	results_dir.mkdir(parents=True, exist_ok=True)
	
	# 选择要运行的模型清单
	models_to_run = []
	if args.models and len(args.models) > 0:
		models_to_run = args.models
	elif args.model:
		models_to_run = [args.model]
	else:
		print("未指定 --model 或 --models，退出。")
		return

	print(f"\n🎯 将依次运行模型: {', '.join(models_to_run)}")
	print(f"数据集: {args.dataset}")
	print(f"预测尺度: {args.prediction_scale}")
	print(f"结果保存目录: {results_dir}")

	for model_name in models_to_run:
		print(f"\n--- 开始运行 {model_name} ---")
		results = run_baseline_robustness_experiment(
			args.dataset, args.prediction_scale, args, device, model_name
		)
		if results is not None:
			save_results(results, results_dir, args.dataset, f"{args.prediction_scale}_{model_name}")
			print(f"✅ {model_name} 完成并已保存结果")
			# 简要摘要
			if 'baseline' in results:
				baseline = results['baseline']
				print(f"  基线性能 - MSE: {baseline['MSE']:.4f}, MAPE: {baseline['MAPE']:.2f}%")
		else:
			print(f"⚠️ {model_name} 运行失败，继续下一个模型")

if __name__ == '__main__':
	main() 