import pandas as pd
import json
import os
import yaml
from collections import Counter

def train_baseline():
    # 1. 載入參數
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    train_path = params['data']['processed_train_path']
    # 輸出路徑 (dvc.yaml 中定義的 models/baseline_top10.json)
    output_path = "models/baseline_top10.json" 
    top_n = 10

    print(f"Reading data from {train_path}...")
    
    # 2. 載入預處理後的資料
    # features.py 產生的是包含 item_idx 的資料
    df = pd.read_csv(train_path)

    # 確保我們統計的是模型使用的 item_idx，而不是原始 itemid
    if 'item_idx' not in df.columns:
        raise ValueError("Error: 'item_idx' column not found in processed data.")

    all_items = df['item_idx'].tolist()

    # 3. 計算熱門商品
    print("Calculating top items...")
    item_counts = Counter(all_items)
    
    # 過濾掉 padding (0) 如果有的話
    if 0 in item_counts:
        del item_counts[0]

    # 取出前 N 名的 item_idx
    popular_items = [item for item, count in item_counts.most_common(top_n)]

    # 4. 儲存為 JSON
    os.makedirs("models", exist_ok=True)
    
    # 轉為 Python int 以確保 JSON 序列化正常
    popular_items = [int(x) for x in popular_items]

    with open(output_path, "w") as f:
        json.dump(popular_items, f)

    print(f"[Baseline] Top {top_n} popular items (by idx): {popular_items}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    train_baseline()