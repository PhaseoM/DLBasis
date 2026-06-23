import torch
from torch import nn
import torch.nn.functional as nnF


class ScaledDotAttention(nn.Module):
    def __init__(self, masked=False):
        super().__init__()
        self.masked = masked

    def forward(self, Q_, K_, V_):
        seq_len = Q_.shape[-2]
        scale = Q_.shape[-1] ** 0.5
        att_score = torch.matmul(Q_, K_.transpose(-2, -1)) / scale
        if self.masked is True:
            causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=att_score.device).triu(diagonal=1)
            att_score = att_score.masked_fill(causal_mask, -torch.inf)
        att_score = nnF.softmax(att_score, dim=-1)
        att_score = torch.matmul(att_score, V_)
        return att_score


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, att_score, bias=False):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model can't be evenly divisible by num_heads")
        self.attention = att_score
        self.num_heads = num_heads
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.w_o = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)

    def forward(self, X):
        query = self._transpose(self.w_q(X))
        key = self._transpose(self.w_k(X))
        value = self._transpose(self.w_v(X))
        multi_att = self.attention(query, key, value)

        multi_att = self.w_o(self._transpose_inverse(multi_att))
        return multi_att

    def _transpose(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X

    def _transpose_inverse(self, X):
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(X.shape[0], X.shape[1], -1)
        return X
