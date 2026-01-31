import streamlit as st
import requests
import os
import datetime

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
API_BASE = os.getenv("API_URL", "http://localhost:8000")
URL_RECOMMEND = f"{API_BASE}/recommend"
URL_BROWSE = f"{API_BASE}/browse"
URL_INTERACT = f"{API_BASE}/interact"
URL_RESET = f"{API_BASE}/history"

st.set_page_config(page_title="AI Store", layout="wide", page_icon="🛒")

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
    }
    div[data-testid="stImage"] > img {
        height: 180px;
        object-fit: contain;
    }
    .price-tag {
        font-size: 1.2em;
        font-weight: bold;
        color: #B12704;
    }
    .cart-summary {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = "1"  # 簡化預設 ID

if "page" not in st.session_state:
    st.session_state.page = 1

if "cart" not in st.session_state:
    st.session_state.cart = []

if "show_checkout" not in st.session_state:
    st.session_state.show_checkout = False

if "browse_cache" not in st.session_state:
    st.session_state.browse_cache = {}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def add_to_cart(item):
    st.session_state.cart.append(item)
    st.toast(f"✅ Added '{item['name'][:20]}...' to Cart")

def like_item(item):
    payload = {"user_id": st.session_state.user_id, "item_idx": item['item_idx']}
    try:
        requests.post(URL_INTERACT, json=payload)
        st.toast(f"❤️ Liked '{item['name'][:20]}...'")
    except Exception as e:
        st.error(f"API Error: {e}")

def reset_history():
    try:
        res = requests.delete(URL_RESET, params={"user_id": st.session_state.user_id})
        if res.status_code == 200:
            st.success("History Cleared! You are now a new user.")
            st.session_state.browse_cache = {} # 清除快取
            st.session_state.cart = [] # 順便清空購物車
        else:
            st.error("Failed to reset history")
    except Exception as e:
        st.error(f"Connection Error: {e}")

# -----------------------------------------------------------------------------
# Checkout Modal (Dialog)
# -----------------------------------------------------------------------------
@st.dialog("💳 Secure Checkout")
def checkout_dialog():
    total_amount = sum([float(str(i.get('price', 0)).replace('$', '').replace(',', '')) for i in st.session_state.cart if i.get('price') != 'N/A'])
    
    st.write(f"**Items in Cart:** {len(st.session_state.cart)}")
    st.markdown(f"### Total: <span style='color:green'>${total_amount:.2f}</span>", unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Card Number", placeholder="XXXX-XXXX-XXXX-XXXX")
        st.text_input("Expiry Date", placeholder="MM/YY")
    with col2:
        st.text_input("CVV", placeholder="123", type="password")
        st.text_input("Cardholder Name")
        
    if st.button("💸 Pay Now", type="primary"):
        st.balloons()
        st.success("Payment Successful! Thank you for your purchase.")
        st.session_state.cart = [] # Clear cart
        st.session_state.show_checkout = False
        # st.rerun() # Dialogs auto-close on interaction usually, or wait for user to close

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛒 My Cart")
    
    if st.session_state.cart:
        total = sum([float(str(i.get('price', 0)).replace('$', '').replace(',', '')) for i in st.session_state.cart if i.get('price') != 'N/A'])
        st.markdown(f"""
        <div class="cart-summary">
            <h4>Total: ${total:.2f}</h4>
            <p>{len(st.session_state.cart)} Items</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Proceed to Checkout"):
            checkout_dialog()
            
        with st.expander("View Cart Details"):
            for i, item in enumerate(st.session_state.cart):
                st.caption(f"{i+1}. {item['name'][:20]}... - {item.get('price')}")
    else:
        st.info("Your cart is empty.")

    st.divider()
    st.header("👤 Settings")
    new_user = st.text_input("User ID", value=st.session_state.user_id)
    if new_user != st.session_state.user_id:
        st.session_state.user_id = new_user
        st.session_state.browse_cache = {} 
        st.rerun()

    if st.button("🗑️ Reset History"):
        reset_history()

# -----------------------------------------------------------------------------
# Main Tabs
# -----------------------------------------------------------------------------
st.title("🛍️ AI-Powered Store")

tab_browse, tab_recs = st.tabs(["🛒 Browse Shop", "🎯 For You (Recommendations)"])

# === TAB 1: BROWSE ===
with tab_browse:
    # 1. Page Control Logic
    MAX_PAGES = 4
    
    # Grid Layout
    current_page = st.session_state.page
    
    # Fetch Data if needed
    if current_page not in st.session_state.browse_cache:
        try:
            # 增加請求數量以填滿頁面
            response = requests.get(URL_BROWSE, params={"limit": 12})
            if response.status_code == 200:
                st.session_state.browse_cache[current_page] = response.json()
        except:
            st.error("Failed to connect to backend")

    items = st.session_state.browse_cache.get(current_page, [])

    # Display Grid (4x3)
    if items:
        cols = st.columns(4)
        for idx, item in enumerate(items):
            col = cols[idx % 4]
            with col:
                with st.container(border=True):
                    # Image
                    img = item.get('image')
                    st.image(img if img and img != "None" else "https://via.placeholder.com/150", use_container_width=True)
                    
                    # Info
                    st.markdown(f"**{item.get('name', 'Unknown')[:30]}...**")
                    st.markdown(f"<span class='price-tag'>{item.get('price', '$0.00')}</span>", unsafe_allow_html=True)
                    
                    # Action Buttons
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("❤️ Like", key=f"like_{item['item_idx']}"):
                            like_item(item)
                    with c2:
                        if st.button("➕ Add", key=f"add_{item['item_idx']}"):
                            add_to_cart(item)
                    
                    if st.button("⚡ Buy Now", key=f"buy_{item['item_idx']}", type="primary"):
                        add_to_cart(item)
                        checkout_dialog()

    # Pagination UI
    st.write("---")
    c_prev, c_display, c_next = st.columns([1, 2, 1])
    
    with c_prev:
        if st.session_state.page > 1:
            if st.button("⬅️ Previous"):
                st.session_state.page -= 1
                st.rerun()
        else:
            st.button("⬅️ Previous", disabled=True)

    with c_display:
        st.markdown(f"<h4 style='text-align: center;'>Page {st.session_state.page} / {MAX_PAGES}</h4>", unsafe_allow_html=True)

    with c_next:
        if st.session_state.page < MAX_PAGES:
            if st.button("Next ➡️"):
                st.session_state.page += 1
                st.rerun()
        else:
            st.button("Next ➡️", disabled=True)

# === TAB 2: RECOMMENDATIONS ===
with tab_recs:
    st.subheader(f"Recommendations for User: {st.session_state.user_id}")
    
    # 使用 container 來包裝刷新按鈕和狀態
    with st.container():
        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            # 這裡的 button 只是為了觸發 Rerun
            refresh = st.button("🔄 Refresh Recommendations", type="primary")
        with col_status:
            # 顯示最後更新時間，讓使用者知道系統真的有在運作
            st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")

    try:
        # 每次進入此頁面或點擊刷新，都會重新發送 POST 請求
        response = requests.post(URL_RECOMMEND, json={"user_id": st.session_state.user_id})
        
        if response.status_code == 200:
            data = response.json()
            recs = data.get("recommendations", [])
            source = data.get("source", "unknown")
            
            if source == "cold_start" or not recs:
                st.warning("🧐 Not enough data yet. Go to 'Browse Shop' and Like/Buy some items!")
            else:
                st.success(f"🎯 Personalized for you (Source: {source})")
                
                # 顯示推薦結果
                for item in recs:
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([1, 3, 1])
                        with c1:
                            img = item.get('image')
                            st.image(img if img and img != "None" else "https://via.placeholder.com/150", use_container_width=True)
                        with c2:
                            st.subheader(item.get('name'))
                            st.write(item.get('asin', 'N/A'))
                            st.markdown(f"**Price:** {item.get('price', 'N/A')}")
                        with c3:
                            st.write("") # Spacer
                            st.write("")
                            if st.button("View", key=f"rec_{item['item_idx']}"):
                                st.toast("Opening item details...")
                            if st.button("Add Rec", key=f"add_rec_{item['item_idx']}"):
                                add_to_cart(item)
        else:
            st.error(f"Backend Error: {response.text}")
            
    except Exception as e:
        st.error(f"Cannot connect to recommendation engine: {e}")