import torch
import torch.nn as nn
from seriesDecomp import SeriesDecomp
from villanseriesDecomp import villanSeriesDecomp
import torch.nn.functional as F
class EncoderLayer(nn.Module):


    def __init__(
        self,
        attention,
        hidden_size,
        c_in,
        conv_hidden_size=None,
        MovingAvg=25,
        dropout=0.1,
        activation="relu",
        gruop_dec=True,
        return_initial_trend=False,
    ):
        super(EncoderLayer, self).__init__()
        conv_hidden_size = conv_hidden_size or 4 * hidden_size
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=conv_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_hidden_size,
            out_channels=hidden_size,
            kernel_size=1,
            bias=False,
        )
        if gruop_dec:
            self.decomp1 = SeriesDecomp(MovingAvg, hidden_size, return_initial_trend=return_initial_trend)
            self.decomp2 = SeriesDecomp(MovingAvg, hidden_size, return_initial_trend=return_initial_trend)
        else:
            self.decomp1 = villanSeriesDecomp(MovingAvg, hidden_size)
            self.decomp2 = villanSeriesDecomp(MovingAvg, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.c_in = c_in
        self.activation = F.relu if activation == "relu" else F.gelu
        self.return_initial_trend = return_initial_trend

    def forward(self, x, edge_index, attn_mask=None):

        all_aggregate = []
        all_attention = []
        for batch in range(x.shape[0]):

            each_batch_data = x[batch].unsqueeze(0)
            output, attn= self.attention(each_batch_data, each_batch_data, each_batch_data)
            each_batch_data = each_batch_data + self.dropout(output)

            all_aggregate.append(each_batch_data)
            all_attention.append(attn)

        x = torch.cat(all_aggregate, dim=0)
        if self.return_initial_trend:
            x, _, initial_trend1 = self.decomp1(x, edge_index, self.c_in)
        else:
            x, _ = self.decomp1(x, edge_index, self.c_in)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        if self.return_initial_trend:
            res, _, initial_trend2 = self.decomp2(x + y, edge_index, self.c_in)
            return res, all_attention, initial_trend1, initial_trend2
        else:
            res, _ = self.decomp2(x + y, edge_index, self.c_in)
            return res, all_attention

class Encoder(nn.Module):

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, edge_index, attn_mask=None):
        attns = []
        initial_trends = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                if hasattr(attn_layer, 'return_initial_trend') and attn_layer.return_initial_trend:
                    x, attn, trend1, trend2 = attn_layer(x, edge_index, attn_mask=attn_mask)
                    initial_trends.extend([trend1, trend2])
                else:
                    x, attn = attn_layer(x, edge_index, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            if hasattr(self.attn_layers[-1], 'return_initial_trend') and self.attn_layers[-1].return_initial_trend:
                x, attn, trend1, trend2 = self.attn_layers[-1](x, edge_index, attn_mask=attn_mask)
                initial_trends.extend([trend1, trend2])
            else:
                x, attn = self.attn_layers[-1](x, edge_index, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                if hasattr(attn_layer, 'return_initial_trend') and attn_layer.return_initial_trend:
                    x, attn, trend1, trend2 = attn_layer(x, edge_index, attn_mask=attn_mask)
                    initial_trends.extend([trend1, trend2])
                else:
                    x, attn = attn_layer(x, edge_index, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        
        if initial_trends:
            return x, attns, initial_trends
        else:
            return x, attns
