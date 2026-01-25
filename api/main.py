from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import yaml
import sys
import os
import redis

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

# [FIX 1] 必須載入 item_map 才能知道 num_items
try:
    with open("item_map.json", "r") as f:
        item_map = json.load(f)
    num_items = len(item_map)
except FileNotFoundError:
    print("Warning: item_map.json not found. Model may fail to initialize.")
    item_map = {}
    num_items = 100 # Fallback

# 載入 Metadata (用於顯示圖片)
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

# ---------------------------------------------------------
# 3. 核心推論邏輯
# ---------------------------------------------------------

def _get_predictions(recent_interactions: list[int], top_k=10):
    """
    輸入: Item Index 列表 (對應 item_map 的 value)
    """
    # [FIX 2] 因為 Redis 存的已經是 Index (int)，不需要再查 item_map
    # 直接做邊界檢查即可
    seq = [i for i in recent_interactions if 0 < i <= num_items]
    
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
        logits = output[:, -1, :] 
        
        _, top_indices = torch.topk(logits, top_k, dim=-1)
        
    # 回傳推薦的 Item Indices
    return top_indices[0].tolist()

# ---------------------------------------------------------
# 4. API Endpoints
# ---------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RecSys API is running"}

@app.post("/predict")
def predict(req: PredictionRequest):
    recs = _get_predictions(req.recent_interactions)
    return {"recommendations": recs, "source": "input_list"}

@app.post("/recommend")
def recommend(req: RecRequest):
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    redis_key = f"user:{req.user_id}"
    history_str = redis_client.get(redis_key)
    
    if not history_str:
        return {
            "user_id": req.user_id,
            "recommendations": [], 
            "source": "unknown_user"
        }
    
    try:
        history_items = json.loads(history_str)
    except json.JSONDecodeError:
        return {"recommendations": [], "error": "Invalid history format"}

    # 執行推論
    recs = _get_predictions(history_items)
    
    # 組合詳細資訊 (圖片、名稱)
    detailed_recs = []
    for item_idx in recs:
        # metadata 的 key 是字串型態的 index (e.g. "105")
        item_info = metadata.get(str(item_idx), {
            "name": f"Unknown Item ({item_idx})",
            "image": None,
            "asin": "N/A"
        })
        detailed_recs.append(item_info)
    
    return {
        "user_id": req.user_id,
        "recommendations": detailed_recs,
        "source": "model_personalization_gqa"
    }