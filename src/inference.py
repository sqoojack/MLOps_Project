# code/inference.py
import json
import torch
import os
from model import RecTransformer # 需將 model.py 一併打包進 code/ 目錄

# 1. 載入模型 (SageMaker 啟動端點時執行一次)
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 讀取訓練時一起存下來的設定檔
    with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        config = json.load(f)
    
    # 初始化模型架構
    model = RecTransformer(config['num_items'])
    
    # 載入權重
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

# 2. 處理請求並執行推論 (即你原本的程式碼邏輯)
def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 解析 FastAPI 傳來的 JSON
    data = json.loads(input_data)
    seq = data['inputs']
    top_k = data.get('top_k', 10)
    
    # 你原本的推論邏輯
    input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        logits = output[:, -1, :] 
        _, top_indices = torch.topk(logits, top_k, dim=-1)
        
    return top_indices[0].tolist()

# 3. 輸出格式化 (將結果轉回 JSON 傳給 FastAPI)
def output_fn(prediction, content_type):
    return json.dumps({"recommendations": prediction})