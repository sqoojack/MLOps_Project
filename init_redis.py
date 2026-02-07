# python init_redis.py
import redis
import yaml
import json
import pandas as pd
import random

def init_redis():
    print("Connecting to Redis...")
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
        
    try:
        r = redis.Redis(host=params['redis']['host'], port=params['redis']['port'], db=0)
        r.ping()
        print("Redis connected!")
        
        # 讀取處理過後的 Train Data (確保 ID 是一致的)
        print("Loading processed data for seeding...")
        df = pd.read_csv(params['data']['processed_train_path'])
        
        # 隨機挑選 10 個真實使用者
        sample_users = df['visitorid'].unique()
        selected_users = random.sample(list(sample_users), 10)
        # 手動加入一個好記的測試帳號 (如果數據集裡剛好有，或者我們可以假造一個)
        
        print(f"Seeding {len(selected_users)} users...")
        
        for user_id in selected_users:
            # 取得該使用者的歷史 Item IDs (int format)
            user_history = df[df['visitorid'] == user_id]['item_idx'].tolist()
            
            # 只取最後 20 個
            if len(user_history) > 20:
                user_history = user_history[-20:]
            
            # 寫入 Redis
            r.set(f"user:{user_id}", json.dumps(user_history))
            print(f"User {user_id} seeded with {len(user_history)} items.")
            
        print("\nInitialization Complete!")
        print("Try these User IDs in Streamlit:")
        for u in selected_users[:5]:
            print(f"- {u}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    init_redis()