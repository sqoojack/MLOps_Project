import pandas as pd
import yaml
import os
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)

def process_data():
    print("🚀 Loading Amazon Reviews 2023 from Hugging Face...")
    
    # 指定類別，例如 "All_Beauty" (美妝), "Fashion" (時尚)
    # 完整列表可見: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
    category = "All_Beauty" 
    
    # 1. 載入評論數據 (User-Item Interactions)
    # trust_remote_code=True 是必須的，因為這是自定義 loading script
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}", split="full", trust_remote_code=True)
    
    # 轉為 Pandas DataFrame (為了方便後續處理，若資料量太大建議用 PyArrow)
    # 這裡示範取前 10 萬筆或是依照記憶體大小調整
    df = dataset.to_pandas()
    
    # 保留需要的欄位
    # 新版欄位名稱: rating, title, text, images, asin, parent_asin, user_id, timestamp
    df = df[['user_id', 'parent_asin', 'timestamp']]
    df.columns = ['visitorid', 'itemid', 'timestamp']
    
    # 2. 建立 Item Map
    unique_items = df['itemid'].unique()
    item_map = {asin: i+1 for i, asin in enumerate(unique_items)}
    
    with open(params['data']['item_map_path'], 'w') as f:
        json.dump(item_map, f)
    
    df['item_idx'] = df['itemid'].map(item_map)
    
    # 3. 載入 Metadata (商品資訊)
    print("📦 Loading Metadata...")
    meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}", split="full", trust_remote_code=True)
    meta_df = meta_dataset.to_pandas()
    
    metadata_map = {}
    # 建立查找表
    # 新版 Metadata 欄位: title, price, average_rating, main_category, images (list)
    for _, row in meta_df.iterrows():
        asin = row['parent_asin'] # 注意: 新版使用 parent_asin 作為主要 ID
        if asin in item_map:
            # 取得第一張圖 (大圖)
            img_url = row['images']['large'][0] if row['images'] and len(row['images']['large']) > 0 else None
            
            metadata_map[str(item_map[asin])] = {
                "name": row['title'],
                "image": img_url,
                "asin": asin,
                "price": row.get('price', 'N/A')
            }

    with open(params['data']['metadata_path'], 'w') as f:
        json.dump(metadata_map, f)
        
    print(f"✅ Metadata processed for {len(metadata_map)} items.")

    # 4. 排序與分割 (邏輯不變)
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