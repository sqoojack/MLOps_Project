import json
import torch
import os
from model import RecTransformer # 確保此檔案與 inference.py 一起打包在 code/ 目錄下

# 1. 載入模型
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        config = json.load(f)
    
    model = RecTransformer(config['num_items'])
    
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

# 2. 處理接收到的請求 (明確定義反序列化邏輯)
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        # request_body 在此階段是 raw bytes 或 string
        return json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# 3. 執行推論
def predict_fn(input_data, model):
    # 此時的 input_data 已經是 input_fn 解析好的 dict
    seq = input_data.get('inputs', [])
    top_k = input_data.get('top_k', 10)
    
    if not seq:
        return []

    # 直接使用模型所在的 device，避免設備不一致錯誤
    device = next(model.parameters()).device 
    input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        logits = output[:, -1, :] 
        _, top_indices = torch.topk(logits, top_k, dim=-1)
        
    return top_indices[0].tolist()

# 4. 輸出格式化
def output_fn(prediction, accept):
    if accept == 'application/json':
        return json.dumps({"recommendations": prediction}), accept
    raise ValueError(f"Unsupported accept type: {accept}")