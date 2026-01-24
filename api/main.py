from fastapi import FastAPI, HTTPException
import torch
import redis
import json
import yaml
import mlflow.pytorch
from pydantic import BaseModel

app = FastAPI()

# 載入配置
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# 初始化 Redis
r = redis.Redis(
    host=params['redis']['host'], 
    port=params['redis']['port'], 
    db=params['redis']['db'],
    decode_responses=True
)

# [Robustness]: Try load model, fallback if fails
try:
    # 這裡假設使用 MLflow 下載，或者本地路徑
    # model = mlflow.pytorch.load_model(f"models:/{params['mlflow']['model_name']}/Production")
    # 開發環境簡單起見，我們假設有一個本地 saved_model.pth 或者透過 mlflow 最後一次 run 載入
    # 這裡示意載入空模型，實際請替換為正確載入邏輯
    from src.model import RecTransformer
    model = RecTransformer(num_items=1000) # 需與訓練一致
    # model.load_state_dict(...) 
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"WARNING: Model load failed ({e}). API running in degraded mode.")
    model = None

class RecRequest(BaseModel):
    user_id: str

@app.post("/recommend")
def recommend(req: RecRequest):
    # 1. 嘗試從 Redis 獲取使用者歷史
    user_history_json = r.get(f"user:{req.user_id}")
    
    if not user_history_json:
        # 冷啟動處理 (Cold Start)
        return {"recommendations": ["popular_item_1", "popular_item_2"], "source": "baseline"}
    
    history = json.loads(user_history_json)
    
    if model is None:
        return {"recommendations": ["popular_item_1"], "source": "fallback"}

    # 2. 模型推論
    try:
        input_seq = torch.tensor([history], dtype=torch.long)
        with torch.no_grad():
            output = model(input_seq)
            logits = output[:, -1, :]
            _, topk = torch.topk(logits, 10)
            
        return {
            "user_id": req.user_id,
            "recommendations": topk[0].tolist(),
            "source": "model"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))