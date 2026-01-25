import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import yaml
import json
import numpy as np
from tqdm import tqdm
from model import RecTransformer
from train import RecDataset, calculate_ndcg  # 引用 train.py 中的 Dataset 和 Metric 函式

def evaluate():
    # 1. 載入參數
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 載入 Item Mapping (關鍵步驟)
    # 訓練時我們儲存了 item_map.json，評估時必須用同一個 map，
    # 否則 ID 會對不起來 (例如訓練時 ID 1 是商品A，測試時 ID 1 變成商品B)
    try:
        with open("item_map.json", "r") as f:
            item_map_str = json.load(f)
            # JSON 的 key 讀進來會變成字串，必須轉回 int
            item_map = {int(k): v for k, v in item_map_str.items()}
    except FileNotFoundError:
        print("Error: item_map.json not found. Please run 'dvc repro train' first.")
        return

    num_items = len(item_map)
    print(f"Loaded item map with {num_items} items.")

    # 3. 準備測試資料
    # RecDataset 現在預期接收 DataFrame，而不是檔案路徑
    print("Loading test data...")
    test_df = pd.read_csv(params['data']['processed_test_path'])
    
    test_dataset = RecDataset(test_df, item_map, params['model']['max_len'])
    test_loader = DataLoader(test_dataset, batch_size=params['train']['batch_size'], shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")

    # 4. 載入模型
    model = RecTransformer(num_items).to(device)
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
        print("Model loaded from model.pth")
    except FileNotFoundError:
        print("Error: model.pth not found. Please run training first.")
        return

    model.eval()

    # 5. 執行評估
    total_ndcg = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    
    # 為了避免 tqdm 顯示問題，也可以用簡單的迴圈
    with torch.no_grad():
        for input_seq, target in tqdm(test_loader, desc="Evaluating"):
            input_seq = input_seq.to(device)
            target = target.to(device)

            output = model(input_seq)
            logits = output[:, -1, :]
            
            # Loss
            loss = criterion(logits, target)
            total_loss += loss.item()

            # Metrics
            ndcg = calculate_ndcg(logits, target, k=10)
            total_ndcg += ndcg

    avg_loss = total_loss / len(test_loader)
    avg_ndcg = total_ndcg / len(test_loader)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test NDCG@10: {avg_ndcg:.4f}")

    # 6. 儲存指標供 DVC 追蹤
    metrics = {
        "test_loss": avg_loss,
        "test_ndcg_10": avg_ndcg
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to metrics.json")

if __name__ == "__main__":
    evaluate()