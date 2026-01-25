# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.model import RecTransformer

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RecSys API is running"}

# 載入設定與模型
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# 載入 Item Map (ID <-> Integer)
with open("item_map.json", "r") as f:
    item_map = json.load(f)
    # JSON key 是 str，轉回 int
    item_map = {int(k): v for k, v in item_map.items()}
    # 建立反向映射 (Model Output -> Item ID)
    rev_item_map = {v: k for k, v in item_map.items()}

# 載入模型
device = torch.device("cpu") # 推論通常用 CPU 或獨立 GPU 實例
num_items = len(item_map)
model = RecTransformer(num_items)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

class PredictionRequest(BaseModel):
    # 使用者過去瀏覽過的 Item IDs
    recent_interactions: list[int]

@app.post("/predict")
def predict(req: PredictionRequest):
    # 1. 資料前處理
    seq = [item_map.get(i) for i in req.recent_interactions if i in item_map]
    
    if not seq:
        return {"recommendations": []}
    
    # Padding / Truncating
    max_len = params['model']['max_len']
    if len(seq) > max_len:
        seq = seq[-max_len:]
    else:
        seq = [0] * (max_len - len(seq)) + seq
    
    # 2. 推論
    input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        logits = output[:, -1, :] # 取最後時間點
        
        # 取 Top 10
        _, top_indices = torch.topk(logits, 10, dim=-1)
        
    # 3. 解析結果 (轉回原始 Item ID)
    recs = [rev_item_map.get(idx.item()) for idx in top_indices[0] if idx.item() in rev_item_map]
    
    return {"recommendations": recs}