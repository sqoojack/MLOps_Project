# init_redis.py
import redis
import yaml
import json
import random

def init_redis():
    print("Connecting to Redis...")
    # 載入設定
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    # 載入 Item Map (為了確保塞入的 ID 是模型認得的)
    try:
        with open("item_map.json", "r") as f:
            item_map = json.load(f)
            # 取得所有合法的 item id (整數)
            valid_items = [int(k) for k in item_map.keys()]
    except FileNotFoundError:
        print("Error: item_map.json not found. Generating random IDs instead.")
        valid_items = list(range(1, 100))

    try:
        r = redis.Redis(
            host=params['redis']['host'], 
            port=params['redis']['port'], 
            db=params['redis']['db']
        )
        r.ping()
        print("Redis connected!")

        # 清空舊資料 (可選)
        # r.flushdb()

        # 建立 5 個測試使用者
        test_users = ["user_1", "user_2", "user_3", "vip_user", "new_guy"]
        
        print(f"Seeding data for {len(test_users)} users...")
        
        for user in test_users:
            # 隨機生成 5~10 個歷史互動
            history_len = random.randint(5, 10)
            history_items = random.sample(valid_items, history_len)
            
            # 寫入 Redis
            r.set(f"user:{user}", json.dumps(history_items))
            print(f"Set {user}: {history_items}")

        print("\nDone! You can now use these IDs in the UI:")
        print(test_users)

    except Exception as e:
        print(f"Redis connection failed: {e}")

if __name__ == "__main__":
    init_redis()