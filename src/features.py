#  src/features.py
import pandas as pd
import os

# 預號讀取 RetailRocket Dataset
os.makedirs("features", exist_ok=True)
df = pd.read_csv("data/raw/events.csv")

# 簡化純準化 (only view + item_id)
df = df[df["event"] == "view"]
df = df[["visitorid", "itemid", "timestamp"]]
df.columns = ["user_id", "item_id", "timestamp"]

# 儲存預號特徵
df.to_csv("features/events_processed.csv", index=False)
print("[features] 處理完成")