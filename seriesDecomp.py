import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
class MovingAvg(nn.Module):

    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):

        padding_size = (self.kernel_size - 1) // 2
        if self.kernel_size % 2 == 0:
            padding = (padding_size, padding_size + 1)
        else:
            padding = (padding_size, padding_size)
        front = x[:, 0:1, :].repeat(1, padding[0], 1)
        end = x[:, -1:, :].repeat(1, padding[1], 1)
        x_padded = torch.cat([front, x, end], dim=1)

        x_avg = self.avg(x_padded.permute(0, 2, 1))
        x_avg = x_avg.permute(0, 2, 1)
        return x_avg

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size, hidden_size, alpha=0.5, num_iterations=2, return_initial_trend=False):

        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.return_initial_trend = return_initial_trend

    def forward(self, x, edge_index, num_nodes=8):

        B, T, hidden_size = x.shape
        d = hidden_size // num_nodes  
        x_nodes = x.view(B, T, num_nodes, d)


        x_nodes_flat = x_nodes.transpose(1, 2).reshape(B * num_nodes, T, d)
        trend_time_flat = self.moving_avg(x_nodes_flat)

        trend_time = trend_time_flat.reshape(B, num_nodes, T, d).transpose(1, 2)

        initial_trend_to_return = None
        if self.return_initial_trend:
            initial_trend_to_return = trend_time.clone()
        
        fused_trend_flat=[]
        for batch_idx in range(B):

            A = torch.zeros((num_nodes, num_nodes), device=x.device)
            A[edge_index[batch_idx][0], edge_index[batch_idx][1]] = 1.0  

            in_degree = A.sum(dim=0)
            in_degree[in_degree == 0] = 1.0 
            A_norm = A / in_degree

            H = trend_time[batch_idx]  # [B, T, num_nodes, d]
            for _ in range(self.num_iterations):
                H = (1 - self.alpha) * H + self.alpha * torch.einsum("ji,tjd->tid", A_norm, H)

            fused_trend = H  # [B, T, num_nodes, d]
            fused_trend_flat.append(fused_trend.contiguous().view(T, hidden_size))

        fused_trend_flat=torch.stack(fused_trend_flat, dim=0)
        residual = x - fused_trend_flat
        

        if self.return_initial_trend:

            initial_trend_flat = initial_trend_to_return.contiguous().view(B, T, hidden_size)
            return residual, fused_trend_flat, initial_trend_flat
        else:
            return residual, fused_trend_flat
