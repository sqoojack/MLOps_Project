import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import yaml
import json
import numpy as np
from tqdm import tqdm
from model import RecTransformer
from train import RecDataset, calculate_ndcg

def evaluate():
    # 1. 載入參數
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 載入 Item Mapping
    try:
        with open("item_map.json", "r") as f:
            item_map = json.load(f)
            # [FIX] 不需要將 key 轉為 int，因為 Amazon ID 是字串
            # 我們這裡讀進來主要是為了知道 num_items
    except FileNotFoundError:
        print("Error: item_map.json not found. Please run 'dvc repro train' first.")
        return

    num_items = len(item_map)
    print(f"Loaded item map with {num_items} items.")

    # 3. 準備測試資料
    print("Loading test data...")
    test_df = pd.read_csv(params['data']['processed_test_path'])
    
    # [FIX] 不再傳入 item_map，RecDataset 會直接用 CSV 裡的 item_idx
    test_dataset = RecDataset(test_df, params['model']['max_len'])
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