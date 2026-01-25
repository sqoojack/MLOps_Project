import streamlit as st
import requests
import os

# 確保路徑與 api/main.py 的 @app.post("/recommend") 一致
API_URL = os.getenv("API_URL", "http://localhost:8000/recommend")

st.set_page_config(page_title="MLOps Recommender", layout="wide")
st.title("🤖 推薦系統即時展示")

# 側邊欄輸入
st.sidebar.header("使用者查詢")
# 這裡讓使用者輸入 user_1, vip_user 等
user_id_input = st.sidebar.text_input("輸入 User ID (例如: user_1, vip_user)", "user_1")

if st.sidebar.button("取得推薦"):
    try:
        # 配合 api/main.py 的 RecRequest 格式 {"user_id": "..."}
        payload = {"user_id": user_id_input}
        
        with st.spinner(f"正在查詢 {user_id_input} 的個性化推薦..."):
            response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            recs = result.get("recommendations", [])
            source = result.get("source", "unknown")
            
            st.success(f"成功獲取推薦！ (來源: {source})")
            
            # 顯示結果
            if recs:
                st.write(f"### 為使用者 `{user_id_input}` 推薦的商品 ID：")
                st.table(recs)
            else:
                st.warning("該使用者暫無推薦結果。")
        else:
            st.error(f"API 錯誤: {response.text}")
            
    except Exception as e:
        st.error(f"連線失敗: {e}")

st.divider()
st.caption("這是一個 End-to-End MLOps 展示：從 Redis 讀取特徵，並透過 Transformer 模型進行推論。")