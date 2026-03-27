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

st.set_page_config(page_title="Simulated Amazon Marketplace", layout="wide", page_icon="🛍️")

# -----------------------------------------------------------------------------
# Advanced Custom CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Background */
    .stApp { background-color: #f8f9fa; }
    
    /* Hide default top red line */
    header {visibility: hidden;}
    
    /* Top Gradient Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        text-align: center;
    }
    .hero-banner h1 { font-weight: 800; margin-bottom: 0.5rem; }
    .hero-banner p { font-size: 1.1em; opacity: 0.9; }

    /* Enhanced Product Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 12px !important;
        border: 1px solid #e0e0e0 !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        padding: 10px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.1) !important;
        border-color: #2a5298 !important;
    }

    /* Unified Image Style */
    div[data-testid="stImage"] > img {
        height: 220px;
        object-fit: cover;
        border-radius: 8px;
        background-color: #ffffff;
    }

    /* Modern Price Tag */
    .price-tag {
        font-size: 1.4em;
        font-weight: 800;
        color: #e63946;
        margin: 10px 0;
        display: block;
    }
    
    /* Product Title Truncation */
    .product-title {
        font-size: 1.1em;
        font-weight: 600;
        color: #333;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 5px;
    }

    /* Cart Summary Block */
    .cart-summary {
        background: linear-gradient(135deg, #ffffff 0%, #f1f3f5 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #2a5298;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Unified Button Style */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover { opacity: 0.9; }
    
    /* Tabs Styling */
    div[data-testid="stTabs"] button {
        font-size: 1.1em !important;
        font-weight: 600 !important;
        padding-bottom: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = "1"
if "page" not in st.session_state:
    st.session_state.page = 1
if "cart" not in st.session_state:
    st.session_state.cart = []
if "browse_cache" not in st.session_state:
    st.session_state.browse_cache = {}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def add_to_cart(item):
    st.session_state.cart.append(item)
    st.toast(f"Added '{item['name'][:15]}...' to cart")

def remove_from_cart(index):
    if 0 <= index < len(st.session_state.cart):
        removed_item = st.session_state.cart.pop(index)
        st.toast(f"🗑️ Removed '{removed_item['name'][:15]}...'")
        st.rerun()

def like_item(item):
    payload = {"user_id": st.session_state.user_id, "item_idx": item['item_idx']}
    try:
        requests.post(URL_INTERACT, json=payload)
        st.toast(f"❤️ You liked '{item['name'][:15]}...'")
    except Exception as e:
        st.error(f"API Connection Error: {e}")

def reset_history():
    try:
        with st.spinner("Clearing history..."):
            res = requests.delete(URL_RESET, params={"user_id": st.session_state.user_id})
            if res.status_code == 200:
                st.success("History cleared! The recommendation system will relearn your preferences.")
                st.session_state.browse_cache = {}
                st.session_state.cart = []
            else:
                st.error("Failed to clear history")
    except Exception as e:
        st.error(f"Connection Error: {e}")

# -----------------------------------------------------------------------------
# Modals (Dialogs)
# -----------------------------------------------------------------------------
@st.dialog("🔍 Product Details")
def show_item_details(item):
    st.markdown(f"### {item.get('name', 'Unknown Product')}")
    
    img = item.get('image')
    st.image(img if img and img != "None" else "https://via.placeholder.com/400x300?text=No+Image", use_container_width=True)
    
    st.markdown(f"**Price:** <span class='price-tag' style='display:inline;'>{item.get('price', 'N/A')}</span>", unsafe_allow_html=True)
    
    st.info(f"**ASIN / Product ID:** {item.get('asin', 'N/A')}  \n**System Index (Item Index):** {item.get('item_idx', 'N/A')}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("❤️ Add to Wishlist", key=f"modal_like_{item['item_idx']}"):
            like_item(item)
    with col2:
        if st.button("🛒 Add to Cart", key=f"modal_add_{item['item_idx']}", type="primary"):
            add_to_cart(item)
            st.rerun()

@st.dialog("💳 Secure Checkout")
def checkout_dialog():
    total_amount = sum([float(str(i.get('price', 0)).replace('$', '').replace(',', '')) for i in st.session_state.cart if i.get('price') != 'N/A'])
    
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h3>Total Amount</h3>
        <h1 style='color: #2a5298;'>${total_amount:.2f}</h1>
        <p>{len(st.session_state.cart)} Items in Cart</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Credit Card Number", placeholder="XXXX-XXXX-XXXX-XXXX")
        st.text_input("Expiration Date", placeholder="MM/YY")
    with col2:
        st.text_input("Security Code (CVV)", placeholder="123", type="password")
        st.text_input("Cardholder Name")
        
    if st.button("💸 Confirm Payment", type="primary", use_container_width=True):
        st.balloons()
        st.success("Payment Successful! Thank you for your purchase, your order is being processed.")
        st.session_state.cart = []
        st.rerun()

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛒 My Cart")
    
    if st.session_state.cart:
        total = sum([float(str(i.get('price', 0)).replace('$', '').replace(',', '')) for i in st.session_state.cart if i.get('price') != 'N/A'])
        
        st.markdown(f"""
        <div class="cart-summary">
            <h3 style='margin:0; color:#333;'>Total: <span style='color:#e63946;'>${total:.2f}</span></h3>
            <span style='color:#666; font-size:0.9em;'>{len(st.session_state.cart)} Items</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Proceed to Checkout 💳", type="primary"):
            checkout_dialog()
            
        with st.expander("🛍️ View Cart Details", expanded=True):
            for i, item in enumerate(st.session_state.cart):
                c_info, c_remove = st.columns([5, 1])
                with c_info:
                    st.markdown(f"**{i+1}.** {item['name'][:12]}...  \n*{item.get('price')}*")
                with c_remove:
                    if st.button("🗑️", key=f"remove_{i}", help="Remove this item"):
                        remove_from_cart(i)
    else:
        st.info("Your cart is currently empty.")

    st.divider()
    
    st.header("⚙️ Account Settings")
    new_user = st.text_input("User ID (Switch Account)", value=st.session_state.user_id)
    if new_user != st.session_state.user_id:
        st.session_state.user_id = new_user
        st.session_state.browse_cache = {} 
        st.rerun()

    if st.button("🗑️ Reset Current User History"):
        reset_history()

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>🛍️ Simulated Amazon Marketplace</h1>
    <p>Powered by MLOps & AWS & Transformer system</p>
</div>
""", unsafe_allow_html=True)

tab_browse, tab_recs = st.tabs(["🛒 Browse All Products", "✨ Personalized Recommendations"])

# === TAB 1: BROWSE ===
with tab_browse:
    MAX_PAGES = 40
    current_page = st.session_state.page
    
    if current_page not in st.session_state.browse_cache:
        with st.spinner('Loading product catalog...'):
            try:
                response = requests.get(URL_BROWSE, params={"limit": 12})
                if response.status_code == 200:
                    st.session_state.browse_cache[current_page] = response.json()
            except:
                st.error("Cannot connect to backend server (API Offline)")

    items = st.session_state.browse_cache.get(current_page, [])

    if items:
        cols = st.columns(4)
        for idx, item in enumerate(items):
            col = cols[idx % 4]
            with col:
                with st.container(border=True):
                    # Product Image
                    img = item.get('image')
                    st.image(img if img and img != "None" else "https://via.placeholder.com/300x300?text=No+Image", use_container_width=True)
                    
                    # Product Info
                    st.markdown(f"<div class='product-title' title='{item.get('name', 'Unknown')}'>{item.get('name', 'Unknown')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='price-tag'>{item.get('price', '$0.00')}</div>", unsafe_allow_html=True)
                    
                    # Action Buttons
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("❤️ Like", key=f"like_{item['item_idx']}"):
                            like_item(item)
                    with c2:
                        if st.button("➕ Cart", key=f"add_{item['item_idx']}", type="primary"):
                            add_to_cart(item)
                            st.rerun()
                    
                    if st.button("🔍 View Details", key=f"view_{item['item_idx']}", use_container_width=True):
                        show_item_details(item)

    # Pagination UI
    st.markdown("<br>", unsafe_allow_html=True)
    c_prev, c_display, c_next = st.columns([1, 2, 1])
    
    with c_prev:
        if st.session_state.page > 1:
            if st.button("Previous", use_container_width=True):
                st.session_state.page -= 1
                st.rerun()
        else:
            st.button("Previous", disabled=True, use_container_width=True)

    with c_display:
        st.markdown(f"<h4 style='text-align: center; color: #666;'>Page {st.session_state.page} of {MAX_PAGES}</h4>", unsafe_allow_html=True)

    with c_next:
        if st.session_state.page < MAX_PAGES:
            if st.button("Next", use_container_width=True):
                st.session_state.page += 1
                st.rerun()
        else:
            st.button("Next", disabled=True, use_container_width=True)

# === TAB 2: RECOMMENDATIONS ===
with tab_recs:
    col_title, col_btn, col_status = st.columns([2, 1, 1])
    with col_title:
        st.subheader(f"Recommendations for User {st.session_state.user_id}")
    with col_btn:
        refresh = st.button("🔄 Refresh", type="primary", use_container_width=True)

    st.divider()

    try:
        with st.spinner('🤖 AI model is calculating your personalized recommendations...'):
            response = requests.post(URL_RECOMMEND, json={"user_id": st.session_state.user_id})
            
            if response.status_code == 200:
                data = response.json()
                recs = data.get("recommendations", [])
                source = data.get("source", "unknown")
                
                if source == "cold_start" or not recs:
                    st.info("💡 The system has not collected enough data yet. Please go to the 'Browse All Products' section to like or purchase some items to let the model learn your preferences!")
                else:
                    st.success(f"🎯 Recommendation calculation complete (Source: {source})")
                    
                    for item in recs:
                        with st.container(border=True):
                            c_img, c_info, c_action = st.columns([1.5, 3.5, 1])
                            with c_img:
                                img = item.get('image')
                                st.image(img if img and img != "None" else "https://via.placeholder.com/150", use_container_width=True)
                            with c_info:
                                st.markdown(f"### {item.get('name')}")
                                st.markdown(f"**ASIN:** `{item.get('asin', 'N/A')}`")
                                st.markdown(f"<div class='price-tag' style='margin:0;'>{item.get('price', 'N/A')}</div>", unsafe_allow_html=True)
                            with c_action:
                                st.markdown("<br>", unsafe_allow_html=True)
                                if st.button("🔍 Details", key=f"rec_view_{item['item_idx']}", use_container_width=True):
                                    show_item_details(item)
                                if st.button("➕ Add", key=f"rec_add_{item['item_idx']}", type="primary", use_container_width=True):
                                    add_to_cart(item)
                                    st.rerun()
            else:
                st.error(f"Backend service error: {response.text}")
                
    except Exception as e:
        st.error(f"Cannot connect to recommendation engine (API Offline): {e}")