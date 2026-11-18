import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, mask=None):
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        q = self.q_proj(q).view(q.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(kv.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(kv.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = q @ k.transpose(-2, -1) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(attn_output.size(0), -1, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)
        return (self.dropout(attn_output), attn_weights)
