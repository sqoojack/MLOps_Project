import pandas as pd
"""
The function `process_data` loads Amazon Reviews 2023 dataset, processes the data by creating item
maps, handling metadata including fixing prices, and then sorts and splits the data into training
and testing sets.
:return: The `process_data()` function processes Amazon reviews data, creates item mapping, loads
metadata, fixes prices for items with missing price info, sorts and splits the data into training
and testing sets, and saves the processed data into CSV files. The function does not return any
value explicitly.
"""
import yaml
import os
import json
import random  # 新增引用
from datasets import load_dataset
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)

def get_random_price():
    """生成隨機價格 (整合自 fix_prices.py)"""
    return f"${round(random.uniform(50, 200), 2)}"

def process_data():
    print("🚀 Loading Amazon Reviews 2023 from Hugging Face...")
    
    # 指定類別
    category = "All_Beauty" 
    
    # 1. 載入評論數據
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}", split="full", trust_remote_code=True)
    df = dataset.to_pandas()
    
    # 簡單過濾與重新命名
    df = df[['user_id', 'parent_asin', 'timestamp']]
    df.columns = ['visitorid', 'itemid', 'timestamp']
    
    # 2. 建立 Item Map
    unique_items = df['itemid'].unique()
    item_map = {asin: i+1 for i, asin in enumerate(unique_items)}
    
    # store item_map
    item_map_dir = os.path.dirname(params['data']['item_map_path'])
    if item_map_dir:  
        os.makedirs(item_map_dir, exist_ok=True)
        
    with open(params['data']['item_map_path'], 'w') as f:
        json.dump(item_map, f)
    
    df['item_idx'] = df['itemid'].map(item_map)
    
    # 新增點: 載入 Redis 產生的新互動資料並合併
    new_events_path = "feature/events_processed.csv"
    if os.path.exists(new_events_path):
        print(f"Loading new events from {new_events_path}...")
        new_events_df = pd.read_csv(new_events_path)
        
        # 檢查timestamp是否存在
        if 'timestamp' not in new_events_df.columns:
            print("⚠️ Warning: 'timestamp' column missing in events file. Skipping local events.")
            records = []
        else:
            # 根據先前的格式: user_id, item_sequence, timestamp 進行解析
            records = []
            for _, row in new_events_df.iterrows():
                uid = str(row['user_id'])
                ts = int(row['timestamp'])
                
                # 將字串 "1,2,5" 轉回獨立的事件紀錄
                if pd.notna(row['item_sequence']):
                    seq = str(row['item_sequence']).split(',')
                    for item_idx_str in seq:
                        if item_idx_str.strip():
                            records.append({
                                'visitorid': uid,
                                'itemid': 'LOCAL_NEW_ITEM', # 新增事件不依賴原始 asin
                                'timestamp': ts,
                                'item_idx': int(item_idx_str.strip())
                            })
            if records:
                local_df = pd.DataFrame(records)
                # 將新事件合併進主資料集
                df = pd.concat([df, local_df], ignore_index=True)
                print(f"Merged {len(records)} new interaction events.")
    else:
        print("⚠️ No local new events found. Using baseline dataset only.")
    
    # 3. 載入 Metadata 並同時處理價格 (Merge fix_prices logic)
    print("📦 Loading Metadata and Fixing Prices...")
    meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}", split="full", trust_remote_code=True)
    meta_df = meta_dataset.to_pandas()
    
    metadata_map = {}
    fixed_price_count = 0
    
    for _, row in meta_df.iterrows():
        asin = row['parent_asin']
        if asin in item_map:
            # 取得圖片
            img_url = row['images']['large'][0] if row['images'] and len(row['images']['large']) > 0 else None
            
            # [整合] 處理價格邏輯
            raw_price = row.get('price', None)
            if raw_price is None or str(raw_price).strip() in ["None", "N/A", ""]:
                final_price = get_random_price()
                fixed_price_count += 1
            else:
                final_price = raw_price

            metadata_map[str(item_map[asin])] = {
                "name": row['title'],
                "image": img_url,
                "asin": asin,
                "price": final_price
            }

    # 儲存 metadata
    metadata_dir = os.path.dirname(params['data']['metadata_path'])
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)
    with open(params['data']['metadata_path'], 'w') as f:
        json.dump(metadata_map, f)
        
    print(f"✅ Metadata processed. Fixed prices for {fixed_price_count} items missing price info.")

    # 4. 排序與分割
    df = df.sort_values(['visitorid', 'timestamp'])
    
    item_counts = df['item_idx'].value_counts()
    valid_items = item_counts[item_counts >= params['data']['min_item_count']].index
    df = df[df['item_idx'].isin(valid_items)]

    split_idx = int(len(df) * (1 - params['data']['test_size']))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv(params['data']['processed_train_path'], index=False)
    test_df.to_csv(params['data']['processed_test_path'], index=False)
    
    print(f"🎉 Data split done. Train: {len(train_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    process_data()