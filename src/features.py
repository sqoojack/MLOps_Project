# python src/features.py
import pandas as pd
import os

# ====== 設定 ======
RAW_DATA_PATH = "data/events.csv"
OUTPUT_PATH = "features/events_processed.csv"

def preprocess_to_sequences():
    # 建立輸出目錄
    os.makedirs("features", exist_ok=True)

    print(f"[features] start to read dataset: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    # use timestamp to sort
    df = df[df["event"] == "view"][["visitorid", "itemid", "timestamp"]]
    df.columns = ["user_id", "item_id", "timestamp"]

    # sort them -> let transformer to get position information
    df = df.sort_values(by=["user_id", "timestamp"])

    # 3. 轉換為序列：對每個 user_id 將其點擊過的 item_id 串接成列表
    # 結果會像這樣：user_1 -> [item_A, item_B, item_C]
    df_sequences = (
        df.groupby("user_id")["item_id"]    # groupby and apply() let one line 
        .apply(list)
        .reset_index()
    )
    df_sequences.columns = ["user_id", "item_sequence"]

    # 4. 儲存預處理後的序列特徵
    df_sequences.to_csv(OUTPUT_PATH, index=False)
    
    print(f"features is done, store to: {OUTPUT_PATH}")
    print(f"The amount of user sequence: {len(df_sequences)}")
    
    # 顯示前幾筆範例
    print("\nThe top example data:")
    print(df_sequences.head())

if __name__ == "__main__":
    preprocess_to_sequences()