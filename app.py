# http://localhost:8501
import streamlit as st
import requests
import os

# 設定 API 網址
# 如果是在 Docker Compose 內執行，請使用 http://api:8000
# 如果是在本地直接跑 streamlit，請使用 http://localhost:8000
API_URL = os.getenv("API_URL", "http://localhost:8000/recommend")

st.set_page_config(page_title="Amazon Beauty 推薦系統", layout="wide")

st.title("🛍️ 個人化商品推薦系統")
st.subheader("基於 Transformer (GQA) 與 MLOps 架構")

# 使用者輸入區
user_id = st.text_input("輸入 User ID (例如: AF7EIDL62ECTXDFW2DNIIIN6LSKQ)", "")

if st.button("獲取推薦"):
    if user_id:
        try:
            # 發送請求給 FastAPI 後端
            response = requests.post(API_URL, json={"user_id": user_id})
            
            if response.status_code == 200:
                data = response.json()
                recs = data.get("recommendations", [])
                source = data.get("source", "unknown")

                if recs:
                    st.success(f"成功獲取推薦！ (來源: {source})")
                    st.divider() # 分隔線

                    # 迭代顯示推薦商品
                    for item in recs:
                        # 建立兩欄：左邊放圖，右邊放文字
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            if item['image'] and item['image'] != "None":
                                st.image(item['image'], use_container_width=True)
                            else:
                                # 如果沒有圖，顯示預設占位圖
                                st.image("https://via.placeholder.com/150?text=No+Image", use_container_width=True)
                        
                        with col2:
                            st.markdown(f"### {item['name']}")
                            st.write(f"**ASIN:** `{item['asin']}`")
                            if item.get('price'):
                                st.write(f"💰 **價格:** {item['price']}")
                            else:
                                st.write("💰 **價格:** 尚未提供")
                            
                        st.divider() # 商品間的分隔線
                else:
                    st.warning("該使用者暫無推薦結果，可能是歷史紀錄過少。")
            else:
                st.error(f"API 錯誤: {response.text}")
        except Exception as e:
            st.error(f"連線失敗: {str(e)}")
    else:
        st.info("請先輸入 User ID")

# 側邊欄資訊
with st.sidebar:
    st.write("## 系統資訊")
    st.info("""
    - **Dataset:** Amazon Beauty 2023
    - **Model:** Transformer w/ GQA
    - **Backend:** FastAPI + Redis
    - **Infrastructure:** Docker + DVC
    """)