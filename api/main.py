from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import yaml
import sys
import os
import redis
import random  # [New] 用於隨機挑選瀏覽商品

# 將 src 加入路徑
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from src.model import RecTransformer
except ImportError:
    from model import RecTransformer

app = FastAPI()

# ---------------------------------------------------------
# 1. 初始化與載入設定
# ---------------------------------------------------------

with open("params.yaml") as f:
    params = yaml.safe_load(f)

# 載入 item_map
try:
    with open("item_map.json", "r") as f:
        item_map = json.load(f)
    num_items = len(item_map)
except FileNotFoundError:
    print("Warning: item_map.json not found. Model may fail to initialize.")
    item_map = {}
    num_items = 100 # Fallback

# 載入 Metadata
try:
    with open("items_metadata.json", "r") as f:
        metadata = json.load(f)
except FileNotFoundError:
    print("Warning: items_metadata.json not found.")
    metadata = {}

# 載入模型
device = torch.device("cpu")
model = RecTransformer(num_items)

if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth", map_location=device))
else:
    print("Warning: model.pth not found. Using untrained model.")
model.eval()

# 連線 Redis
try:
    redis_client = redis.Redis(
        host=params['redis']['host'], 
        port=params['redis']['port'], 
        db=params['redis']['db'],
        decode_responses=True
    )
    redis_client.ping()
    print("✅ Redis connected successfully!")
except Exception as e:
    print(f"⚠️ Redis connection failed: {e}")
    redis_client = None

# ---------------------------------------------------------
# 2. 定義資料結構
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    recent_interactions: list[int]

class RecRequest(BaseModel):
    user_id: str

class InteractionRequest(BaseModel):
    user_id: str
    item_idx: int

# ---------------------------------------------------------
# 3. 核心推論邏輯
# ---------------------------------------------------------

def _get_predictions(recent_interactions: list[int], top_k=10):
    seq = [i for i in recent_interactions if 0 < i <= num_items]
    
    if not seq:
        return []
    
    max_len = params['model']['max_len']
    if len(seq) > max_len:
        seq = seq[-max_len:]
    else:
        seq = [0] * (max_len - len(seq)) + seq
    
    input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        logits = output[:, -1, :] 
        _, top_indices = torch.topk(logits, top_k, dim=-1)
        
    return top_indices[0].tolist()

# ---------------------------------------------------------
# 4. API Endpoints
# ---------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RecSys API is running"}

@app.get("/browse")
def browse_items(limit: int = 20):
    """
    [New] 隨機回傳一些商品供使用者瀏覽
    """
    all_keys = list(metadata.keys())
    # 隨機挑選 limit 個商品
    if not all_keys:
        return []
        
    sample_keys = random.sample(all_keys, min(len(all_keys), limit))
    
    browse_list = []
    for k in sample_keys:
        item = metadata[k].copy()
        item['item_idx'] = int(k) # 重要：將 ID 放入回傳資料中
        browse_list.append(item)
        
    return browse_list

@app.post("/interact")
def interact(req: InteractionRequest):
    """
    [New] 使用者對某商品感興趣 (Like)，更新 Redis
    """
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    redis_key = f"user:{req.user_id}"
    history_str = redis_client.get(redis_key)
    
    if history_str:
        try:
            history = json.loads(history_str)
        except json.JSONDecodeError:
            history = []
    else:
        history = []
    
    # 避免重複連續點擊相同商品 (Optional)
    if not history or history[-1] != req.item_idx:
        history.append(req.item_idx)
    
    # 保持歷史長度限制 (例如只存最後 50 筆)
    if len(history) > 50:
        history = history[-50:]
        
    redis_client.set(redis_key, json.dumps(history))
    
    return {"status": "success", "message": f"Item {req.item_idx} added to history", "history_len": len(history)}

@app.post("/recommend")
def recommend(req: RecRequest):
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    redis_key = f"user:{req.user_id}"
    history_str = redis_client.get(redis_key)
    
    # 如果 Redis 沒資料，回傳空的推薦，或是給一些熱門預設值
    if not history_str:
        return {
            "user_id": req.user_id,
            "recommendations": [], 
            "source": "cold_start"
        }
    
    try:
        history_items = json.loads(history_str)
    except json.JSONDecodeError:
        return {"recommendations": [], "error": "Invalid history format"}

    if not history_items:
        return {"recommendations": [], "source": "empty_history"}

    # 執行推論
    recs = _get_predictions(history_items)
    
    # 組合詳細資訊
    detailed_recs = []
    for item_idx in recs:
        # 抓取 Metadata
        item_info = metadata.get(str(item_idx), {
            "name": f"Unknown Item ({item_idx})",
            "image": None,
            "asin": "N/A"
        }).copy()
        
        # [重要] 注入 item_idx 讓前端可以使用
        item_info['item_idx'] = item_idx
        detailed_recs.append(item_info)
    
    return {
        "user_id": req.user_id,
        "recommendations": detailed_recs,
        "source": "model_personalization_gqa"
    }

@app.post("/predict")
def predict(req: PredictionRequest):
    recs = _get_predictions(req.recent_interactions)
    return {"recommendations": recs, "source": "input_list"}