import numpy as np
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from collections import deque
import torch
import pandas as pd
from scipy.stats import norm
import pickle 
import os  
import warnings

class GrangerCausalityNetwork:
    def __init__(self, data, target_feature, max_lag=1, threshold_p_value=0.05, rho=0.5, verbose=True):
        self.data = data
        self.target_feature = target_feature
        self.max_lag = max_lag
        self.threshold_p_value = threshold_p_value
        self.rho = rho 
        self.verbose = verbose 
    def _is_constant_feature(self, series, tolerance=1e-8):
        return series.var() < tolerance or series.std() < tolerance
    
    def _identify_constant_features(self, data):
        constant_features = []
        for col in data.columns:
            if self._is_constant_feature(data[col]):
                constant_features.append(col)
        return constant_features
    
    def _add_noise_to_constants(self, data, constant_features):
        processed_data = data.copy()
        
        for col in constant_features:
            noise = np.random.normal(0, 1e-8, len(processed_data))
            processed_data[col] = processed_data[col] + noise
            
        return processed_data

    def construct_macro_graph(self):
        constant_features = self._identify_constant_features(self.data)
        
        if constant_features and self.verbose:
            processed_data = self._add_noise_to_constants(self.data, constant_features)
        else:
            processed_data = self.data.copy()
        
        if len(processed_data.columns) < 2:
            if self.verbose:
            return np.ones((len(self.data.columns), len(self.data.columns)))
        
        if len(processed_data) <= self.max_lag + 10: 
            if self.verbose:
                print(f"警告：宏观数据长度 {len(processed_data)} 过短，建议至少 {self.max_lag + 50} 个样本点以获得可靠的因果关系")

        num_features = len(self.data.columns)
        A_Ma = np.ones((num_features, num_features))  
        failed_tests = 0
        total_tests = 0
        
        for i in processed_data.columns:
            for j in processed_data.columns:
                if i != j:
                    total_tests += 1
                    try:
                        if len(processed_data) <= self.max_lag + 1:
                            continue
                            
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            test_result = grangercausalitytests(processed_data[[i, j]], self.max_lag, verbose=False)
                        
                        min_p_value = min(test_result[lag][0]['ssr_chi2test'][1] for lag in range(1, self.max_lag + 1))
                        i_idx = self.data.columns.get_loc(i)
                        j_idx = self.data.columns.get_loc(j)
                        A_Ma[i_idx, j_idx] = min_p_value
                    except Exception as e:
                        failed_tests += 1
                        if self.verbose and failed_tests <= 5: 
                            print(f"格兰杰因果检验失败 ({i}->{j}): {str(e)[:100]}")
                        continue
        
        if self.verbose:
            success_rate = (total_tests - failed_tests) / total_tests if total_tests > 0 else 0
            print(f"宏观图构建完成：成功率 {success_rate:.2%} ({total_tests-failed_tests}/{total_tests})")
            
        return A_Ma

    def construct_micro_graph(self, window_data, macro_graph=None):
        constant_features = self._identify_constant_features(window_data)

        if len(window_data) <= self.max_lag + 1:
            print("警告：微观数据长度不足，使用宏观图")
            if macro_graph is not None:
                return macro_graph.copy()
            else:
                return np.ones((len(window_data.columns), len(window_data.columns)))

        num_features = len(window_data.columns)
        A_Mi = np.ones((num_features, num_features)) 

        if macro_graph is not None:
            A_Mi = macro_graph.copy()

        if constant_features:
            processed_window_data = self._add_noise_to_constants(window_data, constant_features)
        else:
            processed_window_data = window_data.copy()

        non_constant_features = [col for col in window_data.columns if col not in constant_features]
        
        if len(non_constant_features) >= 2:
            for i in non_constant_features:
                for j in non_constant_features:
                    if i != j:
                        try:
                            test_result = grangercausalitytests(processed_window_data[[i, j]], self.max_lag, verbose=False)
                            min_p_value = min(test_result[lag][0]['ssr_chi2test'][1] for lag in range(1, self.max_lag + 1))
                            i_idx = window_data.columns.get_loc(i)
                            j_idx = window_data.columns.get_loc(j)
                            A_Mi[i_idx, j_idx] = min_p_value
                        except Exception as e:
                            continue
        

        if constant_features and non_constant_features:
            for const_feat in constant_features:
                for non_const_feat in non_constant_features:

                    for i, j in [(const_feat, non_const_feat), (non_const_feat, const_feat)]:
                        try:
                            test_result = grangercausalitytests(processed_window_data[[i, j]], self.max_lag, verbose=False)
                            min_p_value = min(test_result[lag][0]['ssr_chi2test'][1] for lag in range(1, self.max_lag + 1))
                            i_idx = window_data.columns.get_loc(i)
                            j_idx = window_data.columns.get_loc(j)
                            A_Mi[i_idx, j_idx] = min_p_value
                        except Exception as e:

                            continue
                            
        return A_Mi

    def fuse_graphs(self, A_Ma, A_Mi):

        num_features = A_Ma.shape[0]
        A_fused = np.zeros((num_features, num_features))

        for i in range(num_features):
            for j in range(num_features):
                if i != j:
                    p_ma = A_Ma[i, j]
                    p_mi = A_Mi[i, j]

                    if p_ma > 0 and p_mi > 0 and p_ma < 1.0 and p_mi < 1.0:
                        try:
                            z_ma = norm.ppf(1 - max(min(p_ma, 0.9999), 0.0001))
                            z_mi = norm.ppf(1 - max(min(p_mi, 0.9999), 0.0001))

                            rho = self.rho

                            Z = (z_ma + z_mi) / np.sqrt(2 + 2 * rho)

                            p_fused = 1 - norm.cdf(Z)
                            A_fused[i, j] = max(min(p_fused, 1.0), 0.0)
                        except (ValueError, RuntimeWarning) as e:
                            A_fused[i, j] = min(p_ma, p_mi)
                    else:
                        if p_ma > 0 and p_ma < 1.0:
                            A_fused[i, j] = p_ma
                        elif p_mi > 0 and p_mi < 1.0:
                            A_fused[i, j] = p_mi
                        else:
                            A_fused[i, j] = 1.0
        return A_fused

    def add_edges_based_on_fused_graph(self, A_fused):

        G = nx.DiGraph()

        for column in self.data.columns:
            G.add_node(column)
        
        p_values = []
        indices = []
        for i in range(A_fused.shape[0]):
            for j in range(A_fused.shape[1]):
                if i != j:
                    p_values.append(A_fused[i, j])
                    indices.append((i, j))
        
        if not p_values:
            return G
            
        p_values = np.array(p_values)
        _, corrected_p_values, _, _ = multipletests(p_values, alpha=self.threshold_p_value, method='fdr_bh')

        for idx, (i, j) in enumerate(indices):
            p_corrected = corrected_p_values[idx]
            if p_corrected < self.threshold_p_value:
                if p_corrected <= 0:
                    p_corrected = 1e-10
                weight = -np.log(p_corrected)
                G.add_edge(self.data.columns[i], self.data.columns[j], weight=weight)
        return G

def construct_edge_index(G):
    edges = list(G.edges())

    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}

    if edges:
        edge_index = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in edges], dtype=torch.long).T
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return edge_index

class ADAGProcessor:
    def __init__(self):
        self.macro_cache = {}
        
    def clear_cache(self):
        self.macro_cache.clear()
        print("宏观图缓存已清除")
        
    def get_cache_info(self):
        if not self.macro_cache:
            print("没有缓存的宏观图")
            return None
        
        info = {}
        for key, value in self.macro_cache.items():
            A_Ma = value['macro_graph']
            feature_names = value['feature_names']
            threshold = value['gc_network'].threshold_p_value
            significant_edges = np.sum(A_Ma < threshold) - np.trace(A_Ma < threshold)
            total_possible_edges = A_Ma.shape[0] * (A_Ma.shape[1] - 1)
            
            info[key] = {
                'features': len(feature_names),
                'feature_names': list(feature_names.values()),
                'significant_edges': significant_edges,
                'total_possible_edges': total_possible_edges,
                'edge_density': significant_edges / total_possible_edges if total_possible_edges > 0 else 0,
                'threshold_p': threshold
            }
        
        return info

    def adag(self, input, fused=True, csv_path=None, min_macro_samples=1000, rho=0.5, force_refresh_macro=False, use_macro_only=False):
        windows = input

        num_features = windows.shape[2] if len(windows.shape) > 2 else 8
        feature_names = {}
        base_names = ['y', 'Pres_Pa','RH_pct','Cloud','WS10m','WD10m','Temp_K','Rad_Jm2','Precip_m','WS100m','WD100m']
        for i in range(num_features):
            if i < len(base_names):
                feature_names[i] = base_names[i]
            else:
                feature_names[i] = f'feature_{i}'

        cache_key = f"{csv_path}_{min_macro_samples}_{num_features}_{rho}"

        if cache_key not in self.macro_cache or force_refresh_macro:

            macro_df = load_macro_data(csv_path, min_macro_samples)

            input_feature_names = [feature_names[i] for i in range(num_features)]

            if len(macro_df.columns) != num_features:
                print(f"警告：宏观数据特征数 ({len(macro_df.columns)}) 与输入数据特征数 ({num_features}) 不匹配")

                available_features = list(macro_df.columns)
                missing_features = [name for name in input_feature_names if name not in available_features]
                extra_features = [name for name in available_features if name not in input_feature_names]
                
                if missing_features:
                    if len(macro_df.columns) >= num_features:
                        macro_df = macro_df.iloc[:, :num_features]
                        macro_df.columns = input_feature_names
                    else:
                        raise ValueError(f"宏观数据特征数不足：需要 {num_features}，实际 {len(macro_df.columns)}")
                else:
                    if extra_features:
                    macro_df = macro_df[input_feature_names]
            else:
                if list(macro_df.columns) != input_feature_names:
                    available_features = list(macro_df.columns)
                    if all(name in available_features for name in input_feature_names):
                        macro_df = macro_df[input_feature_names]
                    else:
                        macro_df.columns = input_feature_names
            
            gc_network = GrangerCausalityNetwork(macro_df, target_feature='y', rho=rho, verbose=True)
            A_Ma = gc_network.construct_macro_graph()

            self.macro_cache[cache_key] = {
                'macro_graph': A_Ma,
                'gc_network': gc_network,
                'feature_names': feature_names
            }
        else:
            cached_data = self.macro_cache[cache_key]
            A_Ma = cached_data['macro_graph']
            gc_network = cached_data['gc_network']
        if use_macro_only:
            G_macro = gc_network.add_edges_based_on_fused_graph(A_Ma)
            edge_index_macro = construct_edge_index(G_macro)
            return [edge_index_macro for _ in range(input.shape[0])]
        edge_results = []
        failed_windows = 0
        
        for i in range(windows.shape[0]):
            try:
                window_data = windows[i].numpy()
                window_df = pd.DataFrame(window_data, columns=[feature_names[j] for j in range(window_data.shape[1])])

                A_Mi = gc_network.construct_micro_graph(window_df, macro_graph=A_Ma)
                
                if fused:
                    A_fused = gc_network.fuse_graphs(A_Ma, A_Mi)
                else:
                    A_fused = A_Mi

                G = gc_network.add_edges_based_on_fused_graph(A_fused)
                edge_index = construct_edge_index(G)

                if edge_index.numel() == 0:
                    G = gc_network.add_edges_based_on_fused_graph(A_Ma)
                    edge_index = construct_edge_index(G)
                
                edge_results.append(edge_index)
                
            except Exception as e:
                failed_windows += 1
                if failed_windows <= 3:  
                    print(f"处理窗口 {i} 时出错: {str(e)[:100]}")

                G = gc_network.add_edges_based_on_fused_graph(A_Ma)
                edge_index = construct_edge_index(G)
                edge_results.append(edge_index)
        
        if failed_windows > 0:
            print(f"完成处理，{failed_windows}/{windows.shape[0]} 个窗口处理失败（已使用宏观图作为备用）")
        else:
            print(f"成功处理所有 {windows.shape[0]} 个时间窗口")

        return edge_results 

def load_macro_data(csv_path=None, min_samples=1000):
    
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, '..', 'data', 'fujian', 'Offshore Wind Farm Dataset3(WT1).csv'),
            os.path.join(current_dir, '..', 'Offshore Wind Farm Dataset3(WT1).csv'),
            os.path.join(current_dir, '..', '..', 'Offshore Wind Farm Dataset3(WT1).csv'),
            os.path.join(current_dir, '..', 'data', 'Offshore Wind Farm Dataset3(WT1).csv'),
            'data/fujian/Offshore Wind Farm Dataset3(WT1).csv',
            'data/Offshore Wind Farm Dataset3(WT1).csv'
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            raise FileNotFoundError("找不到 Offshore Wind Farm Dataset3(WT1).csv 文件")

    try:

        sample_data = pd.read_csv(csv_path, nrows=10)
        total_rows = sum(1 for line in open(csv_path)) - 1  

        actual_samples = min(max(min_samples, total_rows // 2), total_rows)
        
        print(f"从 {csv_path} 读取 {actual_samples}/{total_rows} 行数据构建宏观图")
        
        data = pd.read_csv(csv_path, nrows=actual_samples)
        
        columns_to_drop = ['Site_ID', 'Timestamp']
        df = data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1)
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"读取宏观数据时出错: {str(e)}")

_global_processor = ADAGProcessor()

def adag(input, fused=True, csv_path=None, min_macro_samples=1000, rho=0.5, force_refresh_macro=False):
    
    return _global_processor.adag(input, fused, csv_path, min_macro_samples, rho, force_refresh_macro)

def clear_macro_cache():
    _global_processor.clear_cache()

def get_macro_graph_info():
    return _global_processor.get_cache_info()

def example_usage():

    print("=== ADAG 改进版使用示例 ===")

    batch_size, time_steps, num_features = 10, 50, 8
    sample_data = torch.randn(batch_size, time_steps, num_features)
    
    print(f"输入数据形状: {sample_data.shape}")

    print("\n方法1：使用全局函数")
    try:
        edge_results = adag(
            input=sample_data,
            fused=True,  # 使用融合模式
            csv_path=None,  # 自动查找CSV文件
            min_macro_samples=1000,  # 使用1000个样本构建宏观图
            rho=0.5,  # 宏观-微观融合相关系数
            force_refresh_macro=False  # 不强制刷新缓存
        )
        
        print(f"成功生成 {len(edge_results)} 个时间窗口的边信息")

        macro_info = get_macro_graph_info()
        if macro_info:
            for key, info in macro_info.items():
                print(f"\n宏观图信息:")
                print(f"  特征数量: {info['features']}")
                print(f"  显著边数: {info['significant_edges']}/{info['total_possible_edges']}")
                print(f"  边密度: {info['edge_density']:.3f}")
                print(f"  p值阈值: {info['threshold_p']}")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保CSV文件路径正确")
        edge_results = None
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        edge_results = None

    try:
        processor = ADAGProcessor()
        
        edge_results2 = processor.adag(
            input=sample_data,
            fused=True,
            csv_path=None,
            min_macro_samples=1000,
            rho=0.5,
            force_refresh_macro=False
        )
        
        print(f"成功生成 {len(edge_results2)} 个时间窗口的边信息")
        
        print("\n第二次调用（使用缓存）:")
        edge_results3 = processor.adag(sample_data, fused=True, rho=0.3)

        cache_info = processor.get_cache_info()
        if cache_info:
            print(f"\n缓存中有 {len(cache_info)} 个宏观图")
        
        return edge_results2
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保CSV文件路径正确")
        return None
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return None

if __name__ == "__main__":
    # 运行示例
    results = example_usage()