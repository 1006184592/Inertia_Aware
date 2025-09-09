import torch
import torch.nn as nn
from seriesDecomp import SeriesDecomp
from villanseriesDecomp import villanSeriesDecomp
import torch.nn.functional as F
from math import sqrt
class DecoderLayer(nn.Module):

    def __init__(
        self,
        self_attention,
        cross_attention,
        hidden_size,
        c_out,
        dec_embedding,
        h,
        c_in,
        seq_lenth,
        conv_hidden_size=None,
        MovingAvg=25,
        dropout=0.1,
        activation="relu",
        gruop_dec=True,

    ):
        super(DecoderLayer, self).__init__()
        conv_hidden_size = conv_hidden_size or 4 * hidden_size
        self.out = c_out
        self.c_in = c_in
        self.seq_lenth = seq_lenth
        self.dec_embedding = dec_embedding
        self.attention = self_attention
        self.cross_attention = cross_attention
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
            self.decomp0 = SeriesDecomp(MovingAvg,c_in)
            self.decomp1 = SeriesDecomp(MovingAvg,hidden_size)
            self.decomp2 = SeriesDecomp(MovingAvg,hidden_size)
            self.decomp3 = SeriesDecomp(MovingAvg,hidden_size)
        else:
            self.decomp0 = villanSeriesDecomp(MovingAvg, c_in)
            self.decomp1 = villanSeriesDecomp(MovingAvg, hidden_size)
            self.decomp2 = villanSeriesDecomp(MovingAvg, hidden_size)
            self.decomp3 = villanSeriesDecomp(MovingAvg, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=h,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, edge_index, x_mask=None, cross_mask=None):
        all_aggregate = []
        all_trend_init = []
        for batch in range(x.shape[0]):
            each_batch_data = x[batch].unsqueeze(0)
            x_dec = torch.zeros(size=(1, self.out, each_batch_data.shape[-1])).to(each_batch_data.device)
            x_dec = torch.cat([each_batch_data, x_dec], dim=1)

            if self.out==1:
                mean = torch.mean(each_batch_data[:, -(self.out+int(sqrt(self.seq_lenth))):, :], dim=1).unsqueeze(1).repeat(1, self.out, 1)
            else:
                mean = torch.mean(each_batch_data[:, -(self.out+int(self.seq_lenth/36)):, :], dim=1).unsqueeze(1).repeat(1, self.out, 1)
            zeros = torch.zeros(
                [x_dec.shape[0], self.out, x_dec.shape[2]], device=each_batch_data.device
            )

            seasonal_init, trend_init = self.decomp0(each_batch_data, edge_index, self.c_in)
            # decoder input
            trend_init = torch.cat([trend_init, mean], dim=1)
            trend_init = trend_init[:, :, 0:1]

            seasonal_init = torch.cat(
                [seasonal_init, zeros], dim=1
            )
            # dec
            dec_out = self.dec_embedding(seasonal_init, edge_index[batch:batch + 1])
            output, _ = self.attention(dec_out, dec_out, dec_out)
            dec_out = dec_out + self.dropout(output)

            all_aggregate.append(dec_out)
            all_trend_init.append(trend_init)

        x = torch.cat(all_aggregate, dim=0)
        trend_init = torch.cat(all_trend_init, dim=0)
        x, trend1 = self.decomp1(x, edge_index, self.c_in)
        x = x + self.dropout(self.cross_attention(x, cross, cross)[0])
        x, trend2 = self.decomp2(x, edge_index, self.c_in)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y, edge_index, self.c_in)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return  residual_trend+trend_init

class Decoder(nn.Module):


    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, edge_index, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x = layer(x, cross, edge_index, x_mask=x_mask, cross_mask=cross_mask)

        return x

