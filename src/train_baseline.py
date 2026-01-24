# src/train_baseline.py
import pandas as pd
import pickle
import os
import ast

# ====== 設定 ======
# 修改 1: 讀取處理後的資料，確保與 Transformer 比較基準一致
DATA_PATH = "features/events_processed.csv"
MODEL_OUTPUT = "models/popular_items.pkl"
TOP_N = 10

# ====== 載入資料 ======
df = pd.read_csv(DATA_PATH)

df["item_sequence"] = df["item_sequence"].apply(ast.literal_eval)
all_items = [item for sublist in df["item_sequence"] for item in sublist]

# ====== 計算熱門商品 ======
from collections import Counter
item_counts = Counter(all_items)
popular_items = [item for item, count in item_counts.most_common(TOP_N)]

# ====== 儲存 ======
os.makedirs("models", exist_ok=True)
with open(MODEL_OUTPUT, "wb") as f:
    pickle.dump(popular_items, f)

print(f"[train_baseline] 熱門推薦模型建立完成，共推薦 {len(popular_items)} 個商品")