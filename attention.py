import math
import torch
import torch.nn as nn
class AutoCorrelation(nn.Module):

    def __init__(
        self,
        low_freq=0,
        high_freq=None,
        mask_flag=True,
        factor=1,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        band_pass=True
    ):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.band=band_pass
        self.dropout = nn.Dropout(attention_dropout)
        self.low_freq = low_freq
        self.high_freq = high_freq if high_freq is not None else 0.5

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values, dtype=torch.float, device=values.device)
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = (
            torch.arange(length, device=values.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
        )
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values, dtype=torch.float, device=values.device)
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                1
            ).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = (
            torch.arange(length, device=values.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
        )
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values, dtype=torch.float, device=values.device)
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(
                queries[:, : (L - S), :], dtype=torch.float, device=queries.device
            )
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        with torch.cuda.amp.autocast(enabled=False):
            queries_fp32 = queries.float()
            keys_fp32 = keys.float()
            # period-based dependencies
            q_fft = torch.fft.rfft(queries_fp32.permute(0, 2, 3, 1).contiguous(), dim=-1)
            k_fft = torch.fft.rfft(keys_fp32.permute(0, 2, 3, 1).contiguous(), dim=-1)
        if self.band:
            freq_length = q_fft.size(-1)
            freqs = torch.fft.rfftfreq(queries.size(1), d=1 / (2.0 * self.high_freq))[:freq_length]
            band_pass_filter = ((freqs > self.low_freq) & (freqs < self.high_freq)).float().to(queries.device)
            band_pass_filter = band_pass_filter.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            band_pass_filter = band_pass_filter.expand(q_fft.size(0), q_fft.size(1), 1, -1)
            q_fft = q_fft * band_pass_filter
            k_fft = k_fft * band_pass_filter
        with torch.cuda.amp.autocast(enabled=False):
            res = q_fft * torch.conj(k_fft)
            corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        V = self.time_delay_agg_training(
            values.permute(0, 2, 3, 1).contiguous(), corr
        ).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, hidden_size, n_head, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (hidden_size // n_head)
        d_values = d_values or (hidden_size // n_head)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(hidden_size, d_keys * n_head)
        self.key_projection = nn.Linear(hidden_size, d_keys * n_head)
        self.value_projection = nn.Linear(hidden_size, d_values * n_head)
        self.out_projection = nn.Linear(d_values * n_head, hidden_size)
        self.n_head = n_head

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_head
        queries = self.query_projection(queries).view(B, L, H, -1)

        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn
