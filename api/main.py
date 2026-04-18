import os
import sys
import json
import yaml
import redis
import random
import boto3
import torch
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.model import RecTransformer

app = FastAPI()

# --- Pydantic Models ---
class RecRequest(BaseModel):
    user_id: str

class InteractionRequest(BaseModel):
    user_id: str
    item_idx: int

# --- Resource Initialization ---
with open("params.yaml") as f:
    params = yaml.safe_load(f)

item_map_path = params['data'].get('item_map_path', 'artifacts/item_map.json')
metadata_path = params['data'].get('metadata_path', 'artifacts/items_metadata.json')
model_path = params['data'].get('model_path', 'artifacts/model.pth')
onnx_path = "artifacts/model.onnx"
device = torch.device("cpu")

def load_resources():
    global item_map, num_items, metadata, pt_model, ort_session
    
    # Load Item Map & Metadata
    with open(item_map_path, "r") as f:
        item_map = json.load(f)
    num_items = len(item_map)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # 1. Load ONNX Runtime v1.24.4 with TensorRT 10.x Support
    ort_session = None
    if os.path.exists(onnx_path):
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': 'artifacts/trt_cache'
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        try:
            ort_session = ort.InferenceSession(onnx_path, providers=providers)
            print(f"Inference backend: {ort_session.get_providers()[0]}")
        except Exception as e:
            print(f"ONNX/TRT Loading Error: {e}")

    # 2. PyTorch Fallback Model
    pt_model = RecTransformer(num_items)
    if os.path.exists(model_path):
        pt_model.load_state_dict(torch.load(model_path, map_location=device))
        pt_model.eval()
        print("PyTorch fallback model loaded.")

load_resources()

# --- External Services ---
sm_runtime = boto3.client('sagemaker-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))
ENDPOINT_NAME = os.getenv('SAGEMAKER_ENDPOINT_NAME', 'recsys-endpoint')

try:
    redis_client = redis.Redis(
        host=params['redis']['host'], 
        port=params['redis']['port'], 
        db=params['redis']['db'],
        decode_responses=True
    )
except Exception as e:
    print(f"Redis link failed: {e}")
    redis_client = None

# --- Core Inference Logic ---
def _get_predictions(recent_interactions: list[int], top_k=10):
    seq = [i for i in recent_interactions if 0 < i <= num_items]
    if not seq: return []
    
    max_len = params['model']['max_len']
    if len(seq) > max_len:
        seq = seq[-max_len:]
    else:
        seq = [0] * (max_len - len(seq)) + seq

    # Step 1: Attempt Optimized Inference (ONNX/TRT)
    if ort_session:
        input_name = ort_session.get_inputs()[0].name
        input_np = np.array([seq], dtype=np.int64)
        outputs = ort_session.run(None, {input_name: input_np})
        logits = torch.from_numpy(outputs[0])[:, -1, :]
    else:
        # Step 2: Fallback to PyTorch
        input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
        with torch.no_grad():
            output = pt_model(input_tensor)
            logits = output[:, -1, :]
    
    # Stochastic Sampling for Variety
    probs = torch.softmax(logits / 1.2, dim=-1)
    top_indices = torch.multinomial(probs, num_samples=top_k)
    return top_indices[0].tolist()

# --- API Endpoints ---
@app.get("/")
def health():
    return {"status": "ok", "engine": "TensorRT 10.16.1" if ort_session else "PyTorch"}

@app.post("/recommend")
def recommend(req: RecRequest):
    if not redis_client: raise HTTPException(status_code=503, detail="Redis down")
    history_str = redis_client.get(f"user:{req.user_id}")
    
    if not history_str: 
        return {"user_id": req.user_id, "recommendations": [], "source": "cold_start"}
    
    history_items = json.loads(history_str)
    recs = _get_predictions(history_items)
    
    detailed_recs = []
    for idx in recs:
        info = metadata.get(str(idx), {"name": f"Unknown ({idx})", "asin": "N/A"}).copy()
        info['item_idx'] = idx
        detailed_recs.append(info)
    
    return {
        "user_id": req.user_id, 
        "recommendations": detailed_recs, 
        "source": "onnx_trt" if ort_session else "pytorch"
    }

@app.post("/interact")
def interact(req: InteractionRequest):
    redis_key = f"user:{req.user_id}"
    history_str = redis_client.get(redis_key)
    history = json.loads(history_str) if history_str else []
    history.append(req.item_idx)
    if len(history) > 50: history = history[-50:]
    redis_client.set(redis_key, json.dumps(history))
    return {"status": "success"}

@app.get("/browse")
def browse(limit: int = 20):
    keys = list(metadata.keys())
    samples = random.sample(keys, min(len(keys), limit))
    return [{"item_idx": int(k), **metadata[k]} for k in samples]

@app.post("/reload")
def reload():
    load_resources()
    return {"status": "reloaded"}

@app.delete("/history")
def clear_history(user_id: str):
    if redis_client: redis_client.delete(f"user:{user_id}")
    return {"status": "cleared"}