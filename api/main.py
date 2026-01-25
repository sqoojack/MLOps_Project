from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import yaml
import sys
import os
import redis

# 將 src 加入路徑以匯入模型
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from src.model import RecTransformer
except ImportError:
    # 這是為了防止在某些 docker 環境下找不到路徑的備用方案
    from model import RecTransformer

app = FastAPI()

# ---------------------------------------------------------
# 1. 初始化與載入設定
# ---------------------------------------------------------

# 載入設定
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# 載入 Item Map (ID <-> Integer)
with open("item_map.json", "r") as f:
    item_map = json.load(f)
    # JSON key 是 str，轉回 int
    item_map = {int(k): v for k, v in item_map.items()}
    # 建立反向映射 (Model Output Index -> Item ID)
    rev_item_map = {v: k for k, v in item_map.items()}

# 載入模型
device = torch.device("cpu")
num_items = len(item_map)
model = RecTransformer(num_items)

# 檢查模型檔案是否存在
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth", map_location=device))
else:
    print("Warning: model.pth not found. Using untrained model.")
model.eval()

# 連線 Redis
# 注意：如果在 Docker 內執行且 Redis 是另一個服務，host 可能需要改成 'redis' 而非 'localhost'
try:
    redis_client = redis.Redis(
        host=params['redis']['host'], 
        port=params['redis']['port'], 
        db=params['redis']['db'],
        decode_responses=True # 讓回傳結果直接是字串
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
    # 直接提供 Item ID 列表 (用於測試模型能力)
    recent_interactions: list[int]

class RecRequest(BaseModel):
    # 提供 User ID (用於實際應用場景)
    user_id: str

# ---------------------------------------------------------
# 3. 核心推論邏輯 (共用)
# ---------------------------------------------------------

def _get_predictions(recent_interactions: list[int], top_k=10):
    """
    輸入: 原始 Item ID 列表 (e.g., [101, 102])
    輸出: 推薦的 Item ID 列表
    """
    # 轉換為模型內部的 Index
    seq = [item_map.get(i) for i in recent_interactions if i in item_map]
    
    if not seq:
        return []
    
    # Padding / Truncating
    max_len = params['model']['max_len']
    if len(seq) > max_len:
        seq = seq[-max_len:]
    else:
        seq = [0] * (max_len - len(seq)) + seq
    
    # 推論
    input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        logits = output[:, -1, :] # 取最後時間點
        
        # 取 Top K
        _, top_indices = torch.topk(logits, top_k, dim=-1)
        
    # 解析結果 (轉回原始 Item ID)
    recs = [rev_item_map.get(idx.item()) for idx in top_indices[0] if idx.item() in rev_item_map]
    return recs

# ---------------------------------------------------------
# 4. API Endpoints
# ---------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RecSys API is running"}

@app.post("/predict")
def predict(req: PredictionRequest):
    """
    直接根據輸入的 item list 進行預測
    """
    recs = _get_predictions(req.recent_interactions)
    return {"recommendations": recs, "source": "input_list"}

@app.post("/recommend")
def recommend(req: RecRequest):
    """
    根據 User ID 從 Redis 撈取歷史紀錄後進行預測
    """
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    # 1. 從 Redis 獲取使用者歷史紀錄
    # key 格式需與 init_redis.py 一致 (e.g., "user:user_1")
    redis_key = f"user:{req.user_id}"
    history_str = redis_client.get(redis_key)
    
    if not history_str:
        # 如果找不到使用者，回傳空清單 (或可改為回傳熱門商品)
        return {
            "user_id": req.user_id,
            "recommendations": [], 
            "source": "unknown_user"
        }
    
    # 2. 解析歷史紀錄
    try:
        history_items = json.loads(history_str)
    except json.JSONDecodeError:
        return {"recommendations": [], "error": "Invalid history format"}

    # 3. 執行推論
    recs = _get_predictions(history_items)
    
    return {
        "user_id": req.user_id,
        "recommendations": recs, 
        "source": "model_personalization"
    }