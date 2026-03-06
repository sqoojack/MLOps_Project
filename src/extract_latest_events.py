import redis
import pandas as pd
import os
import json
import yaml
from datetime import datetime

# 載入設定以確保 DB Index 一致
with open("params.yaml") as f:
    params = yaml.safe_load(f)

REDIS_HOST = os.getenv("REDIS_HOST", params['redis']['host'])
REDIS_PORT = int(os.getenv("REDIS_PORT", params['redis']['port']))
REDIS_DB = int(params['redis']['db'])

def extract_events():
    # 確保連線到正確的 DB
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    
    # 根據你的 api/main.py 與 init_redis.py，Key 格式為 user:{user_id}
    target_keys = r.keys("user:*")
    
    records = []
    for key in target_keys:
        try:
            # 排除非純數字 ID 的 Key (如果有)
            user_id = key.split(":")[1]
            
            # 你的 API 使用的是 SET + json.dumps，所以要用 get
            history_str = r.get(key)
            if history_str:
                history = json.loads(history_str) # 解析 JSON 字串為 List
                
                if history:
                    item_sequence = ",".join(map(str, history))
                    timestamp = int(datetime.now().timestamp() * 1000)
                    
                    records.append({
                        "user_id": user_id,
                        "item_sequence": item_sequence,
                        "timestamp": timestamp
                    })
        except Exception as e:
            print(f"解析 Key {key} 失敗: {e}")

    target_file = "feature/events_processed.csv"
    os.makedirs(os.path.dirname(target_file), exist_ok=True)

    if records:
        df_new = pd.DataFrame(records)
        df_new.to_csv(target_file, index=False)
        print(f"✅ 成功從 Redis 萃取 {len(records)} 位使用者的最新行為。")
    else:
        # 即使沒資料也產生標頭，避免 DVC 報錯
        pd.DataFrame(columns=["user_id", "item_sequence", "timestamp"]).to_csv(target_file, index=False)
        print("⚠️ Redis 中查無應用程式資料 (user:*)。已產生空白 CSV。")

if __name__ == "__main__":
    extract_events()