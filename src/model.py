import torch
import torch.nn as nn
import math
import yaml

# 讀取參數
with open("params.yaml") as f:
    params = yaml.safe_load(f)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        
        self.num_rep = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def repeat_kv(self, x, num_rep):
        # x: [Batch, SeqLen, num_kv_heads, head_dim]
        # output: [Batch, SeqLen, num_heads, head_dim]
        if num_rep == 1:
            return x
        B, T, H, D = x.shape
        x = x[:, :, :, None, :].expand(B, T, H, num_rep, D)
        return x.reshape(B, T, H * num_rep, D)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Q: [Batch, SeqLen, Heads, Dim] -> [Batch, Heads, SeqLen, Dim]
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K, V: 保持 [Batch, SeqLen, KV_Heads, Dim] 格式，尚未轉置
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        
        # [FIX]: 在 transpose 之前執行 repeat_kv
        # 這樣輸入維度才是正確的 [B, T, H_kv, D]
        k = self.repeat_kv(k, self.num_rep) 
        v = self.repeat_kv(v, self.num_rep)
        
        # 擴充完後，現在 k, v 的 Head 數已經跟 q 一樣了
        # 再轉置成 [Batch, Heads, SeqLen, Dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Standard Attention Logic
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # mask 為 1 (True) 的位置填入 -inf
            attn = attn.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class RecTransformer(nn.Module):
    def __init__(self, num_items):
        super().__init__()
        embed_dim = params['model']['embed_dim']
        num_heads = params['model']['num_heads']
        num_kv_heads = params['model'].get('num_kv_heads', num_heads)
        num_layers = params['model']['num_layers']
        
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(params['model']['max_len'], embed_dim)
        
        self.layers = nn.ModuleList([
            GroupedQueryAttention(embed_dim, num_heads, num_kv_heads) 
            for _ in range(num_layers)
        ])
        
        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        self.fc = nn.Linear(embed_dim, num_items + 1)
        self.dropout = nn.Dropout(params['model']['dropout'])

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.item_embedding(x) + self.position_embedding(pos)
        x = self.dropout(x)

        # Causal Mask (上三角)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            # Pre-Norm or Post-Norm (這裡使用 Post-Norm 風格)
            # x = norm(x + layer(x))
            residual = x
            x = self.norm(residual + layer(x, mask))
        
        return self.fc(x)