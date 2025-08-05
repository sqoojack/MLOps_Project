
# 評估推薦清單的命中率

import pandas as pd
import pickle
import mlflow
import os
import json

# ====== 路徑設定 ======
MODEL_PATH = "models/popular_items.pkl"
DATA_PATH = "data/raw/events.csv"
OUTPUT_METRICS = "metrics/metrics.json"

# ====== 載入熱門商品清單 ======
with open(MODEL_PATH, "rb") as f:
    popular_items = pickle.load(f)

# ====== 載入資料並評估 ======
df = pd.read_csv(DATA_PATH)

# 模擬每個使用者的實際點擊商品
user_clicks = df[df["event"] == "view"].groupby("visitorid")["itemid"].apply(set)

hit_count = 0
total_users = len(user_clicks)

for user_items in user_clicks:
    if any(item in user_items for item in popular_items):
        hit_count += 1

hit_rate = hit_count / total_users
precision = len(set(popular_items) & set.union(*user_clicks)) / len(popular_items)

# ====== 儲存與記錄指標 ======
metrics = {
    "hit_rate": round(hit_rate, 4),
    "precision": round(precision, 4)
}

os.makedirs("metrics", exist_ok=True)
with open(OUTPUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=4)

with mlflow.start_run(run_name="recommendation-eval"):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

print("[evaluate] 評估完成，Hit Rate:", round(hit_rate, 4), "Precision:", round(precision, 4))
