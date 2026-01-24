# api/main.py
from fastapi import FastAPI, HTTPException
import torch
import pickle
import pandas as pd
import ast
from src.model import RecTransformer # 確保您已將模型架構拆分到 src/model.py

app = FastAPI()

# ====== 全域變數載入 ======
# 為了 API 回應速度，這些檔案會在啟動時載入記憶體
model = None
item_map = None
reverse_item_map = None
user_history = {}
popular_items = []
CONFIG = {
    "embed_dim": 64,
    "num_heads": 4, 
    "num_kv_heads": 2,
    "num_layers": 2,
    "max_len": 50,
    "device": "cpu" # API 推理通常用 CPU，除非有 GPU Server
}

@app.on_event("startup")
def load_artifacts():
    global model, item_map, reverse_item_map, user_history, popular_items
    
    # 1. 載入熱門商品 (Baseline)
    with open("models/popular_items.pkl", "rb") as f:
        popular_items = pickle.load(f)

    # 2. 載入 ID 映射表
    with open("models/item_map.pkl", "rb") as f:
        item_map = pickle.load(f)
        # 建立反向映射: ID -> 商品原始編號
        reverse_item_map = {v: k for k, v in item_map.items()}

    # 3. 載入 Transformer 模型
    # 注意: 這裡的 num_items 要 +1 (因為有 padding 0)
    model = RecTransformer(len(item_map) + 1, CONFIG)
    model.load_state_dict(torch.load("models/transformer_model.pth", map_location=CONFIG["device"]))
    model.eval()

    # 4. 載入使用者歷史資料 (模擬 Feature Store)
    # 實務上這會連接 Redis，這裡我們直接讀 CSV 模擬
    df = pd.read_csv("features/events_processed.csv")
    df["item_sequence"] = df["item_sequence"].apply(ast.literal_eval)
    # 建立 user_id -> [item_id_list] 的快速查找字典
    user_history = dict(zip(df["user_id"], df["item_sequence"]))
    
    print("[API] 所有模型與資料載入完成")

@app.get("/recommend")
def recommend(user_id: int):
    # === 策略 A: Transformer 個性化推薦 ===
    if user_id in user_history:
        # 取得使用者過去的行為序列
        history = user_history[user_id]
        
        # 資料前處理 (與訓練時相同)
        seq_idx = [item_map.get(item, 0) for item in history]
        seq_idx = seq_idx[-CONFIG["max_len"]:] # 截斷
        input_tensor = torch.tensor([seq_idx], dtype=torch.long).to(CONFIG["device"])
        
        with torch.no_grad():
            # 推理: 取得最後一個時間點的輸出
            output = model(input_tensor) # shape: (1, seq_len, vocab_size)
            last_token_logits = output[0, -1, :] # 取最後一個 token 的預測
            
            # 取 Top 5 預測
            topk_indices = torch.topk(last_token_logits, 5).indices.tolist()
            
        # 將模型輸出的 ID 轉回商品原始 ID，並過濾掉 0 (Padding)
        recs = [reverse_item_map.get(idx) for idx in topk_indices if idx in reverse_item_map]
        
        return {
            "user_id": user_id,
            "strategy": "Transformer (Personalized)",
            "input_history": history[-5:], # 顯示最近看過的5個供參考
            "recommended_items": recs
        }

    # === 策略 B: Cold Start (使用 Baseline) ===
    else:
        return {
            "user_id": user_id,
            "strategy": "Popularity (Baseline)",
            "reason": "User not found in history",
            "recommended_items": popular_items[:5]
        }