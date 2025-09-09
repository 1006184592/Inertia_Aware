import torch
import torch.nn as nn
from Encoder import Encoder,EncoderLayer
from Decoder import Decoder,DecoderLayer
from DataEmbedding import GCNWithEmbeddings
from seriesDecomp import SeriesDecomp
from attention import AutoCorrelation,AutoCorrelationLayer
class LayerNorm(nn.Module):


    def __init__(self, channels):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias

class adap_auto(nn.Module):

    # Class attributes
    SAMPLING_TYPE = "windows"

    def __init__(
        self,
        n_head: int,  
        hidden_size: int,  
        factor : int= 2,
        dropout : float= 0.05,
        conv_hidden_size : int= 32,
        MovingAvg_window : int= 3,
        activation : str="gelu",
        encoder_layers : int= 2,
        decoder_layers : int= 1,
        c_out : int= 1,
        h: int = 1,
        seq_lenth: int = 6,
        c_in: int = 1,
        gruop_dec: bool=True

    ):
        super(Hier_auto, self).__init__()

        if activation not in ["relu", "gelu"]:
            raise Exception(f"Check activation={activation}")

        self.c_out = c_out
        self.output_attention = False
        self.h = h
        # Decomposition
        self.enc_embedding = GCNWithEmbeddings(exog_input_size=0,hidden_size=hidden_size,
                                                c_in=c_in, seq_lenth=seq_lenth, dropout=dropout,)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=self.output_attention,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    c_in=c_in,
                    hidden_size=hidden_size,
                    conv_hidden_size=conv_hidden_size,
                    MovingAvg=MovingAvg_window,
                    dropout=dropout,
                    activation=activation,
                    gruop_dec=gruop_dec
                )
                for l in range(encoder_layers)
            ],
            norm_layer=LayerNorm(hidden_size),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size=hidden_size,
                    c_out=self.c_out,
                    dec_embedding=GCNWithEmbeddings(exog_input_size=0,hidden_size=hidden_size,
                                        c_in=c_in, seq_lenth=seq_lenth+c_out, dropout=dropout),
                    h=h,
                    c_in= c_in,
                    seq_lenth=seq_lenth,
                    conv_hidden_size=conv_hidden_size,
                    MovingAvg=MovingAvg_window,
                    dropout=dropout,
                    activation=activation,
                    gruop_dec=gruop_dec
                )
                for l in range(decoder_layers)
            ],
            norm_layer=LayerNorm(hidden_size),
            projection=nn.Linear(hidden_size, h, bias=True),
        )

    def forward(self, windows_batch, edge_index):
        enc_windows_batch = self.enc_embedding(windows_batch, edge_index)
        enc_out, _ = self.encoder(enc_windows_batch, edge_index)
        dec_out= self.decoder(windows_batch, enc_out, edge_index)
        forecast = dec_out[:, -self.c_out:]
        return forecast
