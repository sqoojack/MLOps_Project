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
        # x: [Batch, Heads_KV, SeqLen, Head_Dim]
        if num_rep == 1:
            return x
        B, H, T, D = x.shape
        x = x[:, :, None, :, :].expand(B, H, num_rep, T, D)
        return x.reshape(B, H * num_rep, T, D)

    def forward(self, x, mask=None, past_key_value=None, use_cache=False):
        B, T, C = x.shape
        
        # Q: [Batch, SeqLen, Heads, Head_Dim] -> [Batch, Heads, SeqLen, Head_Dim]
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K, V: [Batch, SeqLen, KV_Heads, Head_Dim] -> [Batch, KV_Heads, SeqLen, Head_Dim]
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # KV Cache 處理邏輯
        if past_key_value is not None:
            # past_key_value[0] 是 k_cache, [1] 是 v_cache
            # 形狀: [Batch, KV_Heads, Past_SeqLen, Head_Dim]
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        current_key_value = (k, v) if use_cache else None

        # GQA: Repeat KV to match Q heads
        # k, v: [Batch, Heads_KV, Total_SeqLen, Head_Dim]
        k_rep = self.repeat_kv(k, self.num_rep)
        v_rep = self.repeat_kv(v, self.num_rep)
        
        # Attention Calculation
        # q: [B, H, T_new, D]
        # k_rep: [B, H, T_total, D]
        attn = (q @ k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # 確保 mask 形狀對齊 [B, 1, T_new, T_total]
            # 當使用 cache 時，T_new=1, T_total=cache_len+1，這時通常不需要 mask (因為只看過去)
            # 訓練時 mask 形狀為 [T, T]
            attn = attn.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v_rep) # [B, H, T_new, D]
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), current_key_value

class RecTransformer(nn.Module):
    def __init__(self, num_items):
        super().__init__()
        embed_dim = params['model']['embed_dim']
        num_heads = params['model']['num_heads']
        num_kv_heads = params['model'].get('num_kv_heads', num_heads)
        num_layers = params['model']['num_layers']
        self.max_len = params['model']['max_len']
        
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, embed_dim)
        
        self.layers = nn.ModuleList([
            GroupedQueryAttention(embed_dim, num_heads, num_kv_heads) 
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_items + 1)
        self.dropout = nn.Dropout(params['model']['dropout'])

    def forward(self, x, use_cache=False, past_key_values=None):
        """
        x: [Batch, SeqLen]
        past_key_values: List of (k, v) tuples from previous step
        """
        B, seq_len = x.shape
        device = x.device
        
        # 計算位置索引
        if past_key_values is not None:
            # 推論模式 (Inference with Cache)
            # 假設 x 只有最新的那個 token (seq_len=1)
            # 位置 = 過去 Cache 的長度
            past_len = past_key_values[0][0].shape[2]
            pos = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(0)
        else:
            # 訓練模式 (Training)
            pos = torch.arange(seq_len, device=device).unsqueeze(0)
            past_key_values = [None] * len(self.layers)

        # Embedding
        x = self.item_embedding(x) + self.position_embedding(pos)
        x = self.dropout(x)

        # Causal Mask (Training only)
        mask = None
        if not use_cache and seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            # 調整維度以適應 Multi-head attention [1, 1, Seq, Seq]
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Layers
        new_past_key_values = []
        for layer, past_kv in zip(self.layers, past_key_values):
            # Pre-Norm or Post-Norm (這裡是 Post-Norm)
            residual = x
            attn_out, new_kv = layer(x, mask=mask, past_key_value=past_kv, use_cache=use_cache)
            x = self.norm(residual + attn_out)
            
            if use_cache:
                new_past_key_values.append(new_kv)
        
        logits = self.fc(x)

        if use_cache:
            return logits, new_past_key_values
        else:
            return logits
        
class VanillaRecTransformer(nn.Module):
    def __init__(self, num_items):
        super().__init__()
        embed_dim = params['model']['embed_dim']
        num_heads = params['model']['num_heads']
        num_layers = params['model']['num_layers']
        self.max_len = params['model']['max_len']
        
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, embed_dim)
        
        # 使用 PyTorch 內建標準 MHA，不使用 GQA
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=params['model']['dropout'])
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_items + 1)
        self.dropout = nn.Dropout(params['model']['dropout'])

    def forward(self, x):
        # 原始版 Forward：不支援 KV Cache，每次都重算整個序列
        B, seq_len = x.shape
        device = x.device
        
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.item_embedding(x) + self.position_embedding(pos)
        x = self.dropout(x)

        # 標準 Causal Mask
        # nn.MultiheadAttention 需要的 mask 形狀是 [Batch*Heads, Seq, Seq] 或 [Seq, Seq]
        # 這裡用 [Seq, Seq] 並設為 -inf
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)

        for layer in self.layers:
            residual = x
            # 標準 MHA forward: query, key, value
            attn_out, _ = layer(x, x, x, attn_mask=mask, is_causal=True)
            x = self.norm(residual + attn_out)
        
        return self.fc(x)