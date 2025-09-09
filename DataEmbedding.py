import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class TokenEmbedding(nn.Module):
    def __init__(self, hidden_size, c_in, kernel_size=3):
        super(TokenEmbedding, self).__init__()
        self.hidden_size = hidden_size
        padding = 1
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=hidden_size,
            # groups=c_in, 
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x):
        return self.embed(x)
class GCNWithEmbeddings(nn.Module):
    def __init__(self, exog_input_size, hidden_size, kernel_size=3, c_in=8, seq_lenth=6, dropout=0.1):
        super(GCNWithEmbeddings, self).__init__()
        # self.device = device
        # Token embedding for time series data
        self.value_embedding = TokenEmbedding(hidden_size=hidden_size, c_in=c_in, kernel_size=kernel_size)

        # Time feature embedding for exogenous time data
        if exog_input_size > 0:
            self.temporal_embedding = TimeFeatureEmbedding(
                input_size=exog_input_size, hidden_size=hidden_size
            )
        else:
            self.temporal_embedding = None

        # GCN for graph structure
        self.gcn1 = GCNConv(seq_lenth, hidden_size)
        self.gcn2 = GCNConv(hidden_size, seq_lenth) 

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index_list, x_mark=None):
        batch_size, sequence_length, num_features = x.shape
        outputs = []
        target_device = self.gcn1.lin.weight.device
        for batch_idx in range(batch_size):
            x_batch_slice = x[batch_idx]
            x_batch = x_batch_slice.T.to(target_device)
            edge_index = edge_index_list[batch_idx].to(target_device)
            x_t = self.gcn1(x_batch, edge_index)
            x_t = F.relu(x_t)
            x_t = self.gcn2(x_t, edge_index).T

            x_batch_processed = self.value_embedding(x_t.unsqueeze(0))
            x_batch_processed = self.dropout(x_batch_processed)

            outputs.append(x_batch_processed.to(x.device))

        return torch.cat(outputs, dim=0)
