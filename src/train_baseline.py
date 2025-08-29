
# train.py
import pandas as pd
import pickle
import os


# ====== 設定 ======
DATA_PATH = "data/raw/events.csv"
MODEL_OUTPUT = "models/popular_items.pkl"
TOP_N = 10  # 可調整推薦商品數

# ====== 載入資料並計算熱門商品 ======
df = pd.read_csv(DATA_PATH)

# 選擇熱門商品的依據（這裡用點擊行為 'view'）
popular_items = (
    df[df["event"] == "view"]["itemid"]
    .value_counts()
    .head(TOP_N)
    .index
    .tolist()
)

# ====== 儲存熱門推薦清單 ======
os.makedirs("models", exist_ok=True)
with open(MODEL_OUTPUT, "wb") as f:
    pickle.dump(popular_items, f)

print(f"[train] 熱門推薦模型建立完成，共推薦 {TOP_N} 個商品")

