import torch
import pandas as pd
import yaml
import json
import numpy as np
import mlflow
from torch.utils.data import DataLoader
from train import RecDataset, calculate_ndcg # 重用 train.py 的功能
from model import RecTransformer

def evaluate():
    # 載入參數
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # 準備測試資料
    test_dataset = RecDataset(params['data']['processed_test_path'], params['model']['max_len'])
    test_loader = DataLoader(test_dataset, batch_size=params['train']['batch_size'], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入模型 (這裡假設從 MLflow Registry 載入，或載入最新訓練的權重)
    # 為了 DVC 流程順暢，我們可以在 train.py 結束時存一個 local 的 'model_latest.pth'
    # 或者在這裡直接初始化新模型 (權重未訓練)，實際場景應該要 load_state_dict
    model = RecTransformer(test_dataset.num_items).to(device)
    
    # [模擬] 假設 train.py 存了一個 'model.pth'，這裡載入
    try:
        model.load_state_dict(torch.load("model.pth"))
    except:
        print("Warning: model.pth not found, evaluating initialized model (random weights)")

    model.eval()
    all_ndcg = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            input_seq = batch[:, :-1]
            target = batch[:, -1]
            
            output = model(input_seq)
            logits = output[:, -1, :]
            
            batch_ndcg = calculate_ndcg(logits, target, k=10)
            all_ndcg.append(batch_ndcg)
            
    avg_ndcg = np.mean(all_ndcg)
    print(f"Test NDCG@10: {avg_ndcg:.4f}")

    # 輸出 metrics 給 DVC
    metrics = {"ndcg_10": avg_ndcg}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    evaluate()