import pandas as pd
import yaml
import os
import redis
import json

with open("params.yaml") as f:
    params = yaml.safe_load(f)

def push_to_redis(df):
    """將最新的使用者歷史推送到 Redis (模擬 Feature Store)"""
    try:
        r = redis.Redis(
            host=params['redis']['host'], 
            port=params['redis']['port'], 
            db=params['redis']['db']
        )
        # 簡單測試連線，若失敗則跳過
        r.ping()
        print("Connected to Redis, syncing user history...")
        
        # 依照 User 分組，取最近的互動
        user_history = df.groupby('visitorid')['itemid'].apply(list).to_dict()
        
        pipe = r.pipeline()
        for user, items in user_history.items():
            # 只存最後 max_len 個
            recent_items = items[-params['model']['max_len']:]
            pipe.set(f"user:{user}", json.dumps(recent_items), ex=params['redis']['ttl'])
        pipe.execute()
        print("Redis sync complete.")
    except Exception as e:
        print(f"Skipping Redis sync (Service not available): {e}")

def process_data():
    df = pd.read_csv(params['data']['raw_path'])
    
    # 基礎清理
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['visitorid', 'timestamp'])
    
    # 過濾少於 N 次互動的 items
    item_counts = df['itemid'].value_counts()
    valid_items = item_counts[item_counts >= params['data']['min_item_count']].index
    df = df[df['itemid'].isin(valid_items)]

    # [Train/Test Split Strategy]: Time-based split
    # 取最後 20% 的時間點作為測試，或者依據最後一次互動
    # 這裡示範：每個使用者的最後一次互動放入 Test (Leave-One-Out) 或依全局時間切分
    # 為簡單起見，這裡採用全局時間切分 (Time Split)
    
    split_date = df['timestamp'].quantile(1 - params['data']['test_size'])
    
    train_df = df[df['timestamp'] <= split_date].copy()
    test_df = df[df['timestamp'] > split_date].copy()

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv(params['data']['processed_train_path'], index=False)
    test_df.to_csv(params['data']['processed_test_path'], index=False)
    
    print(f"Data split done. Train: {len(train_df)}, Test: {len(test_df)}")
    
    # 更新 Redis 供 API 使用
    push_to_redis(df)

if __name__ == "__main__":
    process_data()