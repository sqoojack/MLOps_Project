import torch
import torch.nn as nn
import math
import yaml

# 讀取參數
with open("params.yaml") as f:
    params = yaml.safe_load(f)

class GQA_Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # [CRITICAL FIX]: 使用 mask 為 True (1) 的位置填入 -inf
            # triu(diagonal=1) 產生上三角為 1 (未來)，正是我們要遮蔽的地方
            attn = attn.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class RecTransformer(nn.Module):
    def __init__(self, num_items):
        super().__init__()
        # 使用 params 
        embed_dim = params['model']['embed_dim']
        num_heads = params['model']['num_heads']
        num_layers = params['model']['num_layers']
        
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(params['model']['max_len'], embed_dim)
        
        self.layers = nn.ModuleList([
            GQA_Attention(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, num_items + 1)
        self.dropout = nn.Dropout(params['model']['dropout'])

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.item_embedding(x) + self.position_embedding(pos)
        x = self.dropout(x)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            x = x + layer(x, mask) # Residual connection 簡化版
        
        return self.fc(x)