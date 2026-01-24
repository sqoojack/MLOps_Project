# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA_Attention(nn.Module):
    def __init__(self, embed_dim, num_q_heads, num_kv_heads):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        self.kv_group_size = num_q_heads // num_kv_heads 

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None, kv_cache=None):
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # === KV Cache 邏輯 (為推理優化預留) ===
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)
        new_cache = (k, v)

        # GQA: Repeat K/V
        k = k.repeat_interleave(self.kv_group_size, dim=2)
        v = v.repeat_interleave(self.kv_group_size, dim=2)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            # 確保 mask 維度正確
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        output = (attn @ v).transpose(1, 2).reshape(batch, seq_len, -1)
        
        return self.o_proj(output), new_cache

class RecTransformer(nn.Module):
    def __init__(self, num_items, config):
        super().__init__()
        self.embedding = nn.Embedding(num_items, config["embed_dim"], padding_idx=0)
        self.pos_embedding = nn.Embedding(config["max_len"], config["embed_dim"])
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                GQA_Attention(config["embed_dim"], config["num_heads"], config["num_kv_heads"]),
                nn.LayerNorm(config["embed_dim"]),
                nn.Sequential(
                    nn.Linear(config["embed_dim"], config["embed_dim"] * 4),
                    nn.GELU(),
                    nn.Linear(config["embed_dim"] * 4, config["embed_dim"])
                ),
                nn.LayerNorm(config["embed_dim"])
            ])
            for _ in range(config["num_layers"])
        ])
        
        self.head = nn.Linear(config["embed_dim"], num_items)

    def forward(self, x, use_cache=False):
        batch, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)
        
        # Causal Mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # 簡單實作：目前不回傳 cache，僅確保架構正確
        for attn, ln1, ffn, ln2 in self.layers:
            h_attn, _ = attn(ln1(h), mask=mask)
            h = h + h_attn
            h_ffn = ffn(ln2(h))
            h = h + h_ffn
            
        return self.head(h)