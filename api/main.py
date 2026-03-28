from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import yaml
import sys
import os
import redis
import random
import boto3

# 將 src 加入路徑
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from src.model import RecTransformer
except ImportError:
    from model import RecTransformer

app = FastAPI()

# ---------------------------------------------------------
# 1. 資料結構定義 (Pydantic Models) - 必須放在前面
# ---------------------------------------------------------
class PredictionRequest(BaseModel):
    recent_interactions: list[int]

class RecRequest(BaseModel):
    user_id: str

class InteractionRequest(BaseModel):
    user_id: str
    item_idx: int

# ---------------------------------------------------------
# 2. 初始化與載入邏輯
# ---------------------------------------------------------
with open("params.yaml") as f:
    params = yaml.safe_load(f)

item_map_path = params.get('data', {}).get('item_map_path', 'artifacts/item_map.json')
metadata_path = params.get('data', {}).get('metadata_path', 'artifacts/items_metadata.json')
model_path = params.get('data', {}).get('model_path', 'artifacts/model.pth')
device = torch.device("cpu")

def load_resources():
    """封裝載入邏輯，供初始化與熱更新使用"""
    global item_map, num_items, metadata, model
    
    # 載入 Item Map
    try:
        with open(item_map_path, "r") as f:
            item_map = json.load(f)
        num_items = len(item_map)
    except FileNotFoundError:
        item_map = {}
        num_items = 100
        print(f"Warning: {item_map_path} not found.")

    # 載入 Metadata
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {}

    # 初始化與載入模型權重
    model = RecTransformer(num_items)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")
    model.eval()

# 執行初始載入
load_resources()

# 初始化 SageMaker 與 Redis
sm_runtime = boto3.client('sagemaker-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))
ENDPOINT_NAME = os.getenv('SAGEMAKER_ENDPOINT_NAME', 'recsys-endpoint')

try:
    redis_client = redis.Redis(
        host=params['redis']['host'], 
        port=params['redis']['port'], 
        db=params['redis']['db'],
        decode_responses=True
    )
    redis_client.ping()
except Exception as e:
    print(f"⚠️ Redis connection failed: {e}")
    redis_client = None

# ---------------------------------------------------------
# 3. 核心推論邏輯 (修改點 1: 隨機採樣)
# ---------------------------------------------------------
def _get_predictions(recent_interactions: list[int], top_k=10):
    seq = [i for i in recent_interactions if 0 < i <= num_items]
    if not seq: return []
    
    max_len = params['model']['max_len']
    if len(seq) > max_len:
        seq = seq[-max_len:]
    else:
        seq = [0] * (max_len - len(seq)) + seq

    # 優先嘗試 SageMaker
    payload = json.dumps({"inputs": seq, "top_k": top_k})
    try:
        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME, ContentType='application/json', Body=payload
        )
        result = json.loads(response['Body'].read().decode())
        return result.get('recommendations', [])
    except Exception:
        pass # Fallback to local model
    
    # --- 本地模型隨機採樣邏輯 ---
    input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        logits = output[:, -1, :] 
        
        # 增加隨機性：透過 Temperature 調整機率分佈
        temperature = 1.2 
        probs = torch.softmax(logits / temperature, dim=-1)
        
        # 使用 multinomial 進行採樣（不再只是取最強的那幾個）
        top_indices = torch.multinomial(probs, num_samples=top_k)
        
    return top_indices[0].tolist()

# ---------------------------------------------------------
# 4. API Endpoints (已整合與去重)
# ---------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "RecSys API is running"}

@app.post("/reload")
def reload_model():
    """修改點 2: 熱更新端點"""
    try:
        load_resources()
        return {"status": "success", "message": "Resources reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

@app.post("/interact")
def interact(req: InteractionRequest):
    """修改點 3: 更新 Redis 並打印日誌"""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    redis_key = f"user:{req.user_id}"
    history_str = redis_client.get(redis_key)
    history = json.loads(history_str) if history_str else []
    
    if not history or history[-1] != req.item_idx:
        history.append(req.item_idx)
    
    if len(history) > 50: history = history[-50:]
    redis_client.set(redis_key, json.dumps(history))
    
    # 確認日誌輸出
    print(f"LOG: User {req.user_id} added item {req.item_idx}. Current history length: {len(history)}")
    
    return {"status": "success", "message": f"Item {req.item_idx} added", "history_len": len(history)}

@app.get("/browse")
def browse_items(limit: int = 20):
    all_keys = list(metadata.keys())
    if not all_keys: return []
    sample_keys = random.sample(all_keys, min(len(all_keys), limit))
    return [{"item_idx": int(k), **metadata[k]} for k in sample_keys]

@app.post("/recommend")
def recommend(req: RecRequest):
    if redis_client is None: raise HTTPException(status_code=503, detail="Redis unavailable")
    history_str = redis_client.get(f"user:{req.user_id}")
    
    if not history_str: return {"user_id": req.user_id, "recommendations": [], "source": "cold_start"}
    
    history_items = json.loads(history_str)
    recs = _get_predictions(history_items)
    
    detailed_recs = []
    for idx in recs:
        info = metadata.get(str(idx), {"name": f"Unknown ({idx})", "asin": "N/A"}).copy()
        info['item_idx'] = idx
        detailed_recs.append(info)
    
    return {"user_id": req.user_id, "recommendations": detailed_recs, "source": "model_sampling_gqa"}

@app.delete("/history")
def reset_history(user_id: str):
    if redis_client: redis_client.delete(f"user:{user_id}")
    return {"status": "success", "message": f"History for {user_id} reset."}