# python -m streamlit run app.py
import streamlit as st
import requests
import os

# 設定 API 網址
API_BASE = os.getenv("API_URL", "http://localhost:8000")
URL_RECOMMEND = f"{API_BASE}/recommend"
URL_BROWSE = f"{API_BASE}/browse"
URL_INTERACT = f"{API_BASE}/interact"

st.set_page_config(page_title="Amazon 智能商城", layout="wide", page_icon="🛒")

# CSS 優化圖片顯示
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    div[data-testid="stImage"] > img {
        height: 150px;
        object-fit: contain;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛍️ AI-Powered Shopping Experience")
st.caption("Interacting with items updates your personalized profile in real-time using Redis & Transformers.")

# -----------------------------------------------------------------------------
# 1. User Identity Management
# -----------------------------------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = "AF7EIDL62ECTXDFW2DNIIIN6LSKQ" # 預設測試 ID

with st.sidebar:
    st.header("👤 使用者設定")
    user_id_input = st.text_input("User ID", value=st.session_state.user_id)
    if user_id_input != st.session_state.user_id:
        st.session_state.user_id = user_id_input
        st.success("User ID Updated!")
    
    st.info("在「逛商店」點擊喜歡後，切換到「專屬推薦」查看模型如何根據您的行為改變推薦結果。")
    if st.button("🗑️ 清除此 User 歷史 (模擬新客)"):
        # 這裡可以實作呼叫後端清除 Redis 的邏輯
        st.toast("功能尚未實作 (請參考 API 修改建議)", icon="⚠️")

# -----------------------------------------------------------------------------
# 2. Main Interface (Tabs)
# -----------------------------------------------------------------------------
tab_browse, tab_recs = st.tabs(["🛒 逛商店 (Browse)", "🎯 專屬推薦 (For You)"])

# === TAB 1: 瀏覽商品 (Browse) ===
with tab_browse:
    st.subheader("探索熱門商品")
    
    # 重新整理按鈕 (換一批商品)
    if st.button("🔄 換一批商品看看"):
        st.cache_data.clear() # 清除快取以獲取新隨機商品
        
    # 獲取隨機商品列表
    try:
        # 使用 session_state 避免每次點擊按鈕都重整整個頁面導致商品更換
        # 這裡簡單起見，直接呼叫
        response = requests.get(URL_BROWSE, params={"limit": 12})
        if response.status_code == 200:
            items = response.json()
            
            # 使用 Grid Layout 顯示商品
            cols = st.columns(4) # 4 欄位
            for idx, item in enumerate(items):
                col = cols[idx % 4]
                with col:
                    with st.container(border=True):
                        # 顯示圖片
                        img_url = item.get('image')
                        if img_url and img_url != "None":
                            st.image(img_url, use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/150?text=No+Image", use_container_width=True)
                        
                        # 顯示名稱 (截斷過長的名稱)
                        name = item.get('name', 'Unknown')
                        st.markdown(f"**{name[:40]}...**" if len(name) > 40 else f"**{name}**")
                        st.caption(f"${item.get('price', 'N/A')}")
                        
                        # 按鈕：我有興趣
                        # key 必須唯一
                        if st.button("❤️ 喜歡", key=f"like_{item['item_idx']}"):
                            # 呼叫後端 API
                            payload = {
                                "user_id": st.session_state.user_id,
                                "item_idx": item['item_idx']
                            }
                            try:
                                res = requests.post(URL_INTERACT, json=payload)
                                if res.status_code == 200:
                                    st.toast(f"已將「{name[:20]}」加入興趣清單！", icon="✅")
                                else:
                                    st.error("系統忙線中...")
                            except Exception as e:
                                st.error(f"連線錯誤: {e}")

        else:
            st.error("無法載入商品，請檢查後端 API。")
    except Exception as e:
        st.error(f"連線失敗: {e}")

# === TAB 2: 推薦結果 (Recommendations) ===
with tab_recs:
    st.subheader(f"為 {st.session_state.user_id[:8]}... 量身打造")
    
    if st.button("⚡ 刷新推薦結果", type="primary"):
        pass # 只是為了觸發 rerun
    
    try:
        response = requests.post(URL_RECOMMEND, json={"user_id": st.session_state.user_id})
        
        if response.status_code == 200:
            data = response.json()
            recs = data.get("recommendations", [])
            source = data.get("source", "unknown")
            
            if source == "cold_start":
                st.warning("👋 嗨！你看起來是新朋友。請先到「逛商店」頁面點選幾個喜歡的商品，我們才能為您推薦喔！")
            elif not recs:
                st.info("目前沒有相關推薦，請多與商品互動。")
            else:
                st.success(f"根據您最新的瀏覽紀錄分析 (Source: {source})")
                
                # 顯示推薦列表
                for item in recs:
                    with st.container():
                        c1, c2 = st.columns([1, 4])
                        with c1:
                            if item.get('image') and item['image'] != "None":
                                st.image(item['image'], use_container_width=True)
                            else:
                                st.image("https://via.placeholder.com/150?text=No+Image", use_container_width=True)
                        with c2:
                            st.markdown(f"### {item.get('name')}")
                            st.write(f"**ASIN:** `{item.get('asin')}` | **Price:** {item.get('price', 'N/A')}")
                            st.caption(f"Reason: Matched with your recent interests")
                        st.divider()
        else:
            st.error(f"API Error: {response.text}")
            
    except Exception as e:
        st.error(f"Backend Connection Failed: {e}")