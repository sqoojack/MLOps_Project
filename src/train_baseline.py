import pandas as pd
import json
import os
import yaml
from collections import Counter

def train_baseline():
    # ====== 1. 載入參數 ======
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    train_path = params['data']['processed_train_path']
    # 為了配合 dvc.yaml 的 outs 定義
    output_path = "models/baseline_top10.json" 
    top_n = 10

    print(f"Reading data from {train_path}...")
    
    # ====== 2. 載入資料 ======
    # 新的 features.py 產出的 csv 是長表格 (visitorid, itemid, timestamp)
    df = pd.read_csv(train_path)

    # 確保 itemid 是字串或整數一致
    # 這裡轉為 list 進行計數
    all_items = df['itemid'].tolist()

    # ====== 3. 計算熱門商品 ======
    item_counts = Counter(all_items)
    # most_common 回傳 [(item1, count1), (item2, count2)...]
    # 我們只需要 item id
    popular_items = [item for item, count in item_counts.most_common(top_n)]

    # ====== 4. 儲存為 JSON ======
    os.makedirs("models", exist_ok=True)
    
    # 為了相容性，將 numpy int64 轉為 python int (如果有的話)
    popular_items = [int(x) if isinstance(x, (int, float)) else str(x) for x in popular_items]

    with open(output_path, "w") as f:
        json.dump(popular_items, f)

    print(f"[train_baseline] 熱門推薦模型建立完成，已儲存至 {output_path}")
    print(f"Top {top_n} Items: {popular_items}")

if __name__ == "__main__":
    train_baseline()