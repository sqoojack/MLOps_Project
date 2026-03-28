import streamlit as st
import requests
import os
import hashlib

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
    .stApp { background-color: #f8f9fa; }
    header {visibility: hidden;}
    
    .hero-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        text-align: center;
    }
    .hero-banner h1 { font-weight: 800; margin-bottom: 0.5rem; }
    .hero-banner p { font-size: 1.1em; opacity: 0.9; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 12px !important;
        border: 1px solid #e0e0e0 !important;
        transition: all 0.3s ease;
        padding: 10px;
        position: relative;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.1) !important;
        border-color: #2a5298 !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #EE4D2D !important; 
        color: white !important;
        border: 1px solid #EE4D2D !important;
        border-radius: 4px !important;      
        font-weight: 500 !important;
        padding: 4px 16px !important;       
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"]:hover { background-color: #ff5722 !important; }
    
    .price-tag { font-size: 1.4em; font-weight: 800; color: #e63946; margin: 5px 0; display: block; }
    .product-title { font-size: 1.1em; font-weight: 600; color: #333; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 5px; }
    
    .cart-summary { background: linear-gradient(135deg, #ffffff 0%, #f1f3f5 100%); padding: 20px; border-radius: 12px; border-left: 6px solid #2a5298; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
    
    /* Ratings & Badges */
    .rating-stars { color: #FFD700; font-size: 1.1em; margin-bottom: 5px; }
    .badge {
        position: absolute; top: 10px; right: 10px; padding: 4px 8px; border-radius: 4px; 
        font-size: 0.8em; font-weight: bold; color: white; z-index: 10;
    }
    .badge-hot { background-color: #e63946; }
    .badge-discount { background-color: #2a9d8f; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Mock Data Generators
# -----------------------------------------------------------------------------
def enrich_item_data(item):
    idx_str = str(item.get('item_idx', '0'))
    hash_val = int(hashlib.md5(idx_str.encode()).hexdigest(), 16)
    
    if 'rating' not in item: 
        item['rating'] = (hash_val % 30 + 20) / 10  # 2.0 to 4.9
        
    if 'badge' not in item:
        badge_val = hash_val % 5
        item['badge'] = "Hot🔥" if badge_val == 0 else "20% OFF" if badge_val == 1 else None
        
    try:
        item['parsed_price'] = float(str(item.get('price', '0')).replace('$', '').replace(',', ''))
    except:
        item['parsed_price'] = 0.0
    return item

# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------
if "user_id" not in st.session_state: st.session_state.user_id = "1"
if "page" not in st.session_state: st.session_state.page = 1
if "cart" not in st.session_state: st.session_state.cart = {}
if "browse_cache" not in st.session_state: st.session_state.browse_cache = {}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def add_to_cart(item):
    idx = str(item['item_idx'])
    if idx in st.session_state.cart:
        st.session_state.cart[idx]['qty'] += 1
    else:
        st.session_state.cart[idx] = {'item': item, 'qty': 1}
    st.toast(f"🛒 Added '{item['name'][:15]}...' (Qty: {st.session_state.cart[idx]['qty']})")

def update_cart_qty(idx, delta):
    idx = str(idx)
    if idx in st.session_state.cart:
        st.session_state.cart[idx]['qty'] += delta
        if st.session_state.cart[idx]['qty'] <= 0:
            removed_name = st.session_state.cart[idx]['item']['name']
            del st.session_state.cart[idx]
            st.toast(f"🗑️ Removed '{removed_name[:15]}...'")

def get_cart_totals():
    subtotal = sum(details['item']['parsed_price'] * details['qty'] for details in st.session_state.cart.values())
    total_items = sum(details['qty'] for details in st.session_state.cart.values())
    return subtotal, total_items

def like_item(item):
    try:
        requests.post(URL_INTERACT, json={"user_id": st.session_state.user_id, "item_idx": item['item_idx']})
        st.toast(f"❤️ You liked '{item['name'][:15]}...'")
    except:
        st.toast("⚠️ Backend connection failed, but action recorded locally.")

# -----------------------------------------------------------------------------
# Modals (Dialogs)
# -----------------------------------------------------------------------------
@st.dialog("🔍 Product Details")
def show_item_details(item):
    st.markdown(f"### {item.get('name', 'Unknown Product')}")
    st.markdown(f"<div class='rating-stars'>{'★' * int(item['rating'])}{'☆' * (5-int(item['rating']))} ({item['rating']})</div>", unsafe_allow_html=True)
    
    img = item.get('image')
    st.image(img if img and img != "None" else "https://via.placeholder.com/400x300?text=No+Image", use_container_width=True)
    st.markdown(f"**Price:** <span class='price-tag' style='display:inline;'>${item['parsed_price']:.2f}</span>", unsafe_allow_html=True)
    st.info(f"**ASIN:** {item.get('asin', 'N/A')}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("❤️ Add to Wishlist", key=f"modal_like_{item['item_idx']}", use_container_width=True):
            like_item(item)
    with col2:
        if st.button("🛒 Add to Cart", key=f"modal_add_{item['item_idx']}", type="primary", use_container_width=True):
            add_to_cart(item)
            st.rerun()

@st.dialog("💳 Secure Checkout")
def checkout_dialog():
    subtotal, total_items = get_cart_totals()
    shipping = 0 if subtotal >= 500 or subtotal == 0 else 50
    
    coupon = st.text_input("Coupon Code", placeholder="e.g. SAVE20")
    discount = 20 if coupon.upper() == "SAVE20" else 0
    
    final_total = subtotal + shipping - discount
    final_total = max(0, final_total)
    
    st.markdown(f"""
    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
        <div style='display: flex; justify-content: space-between;'><span>Subtotal ({total_items} items):</span> <span>${subtotal:.2f}</span></div>
        <div style='display: flex; justify-content: space-between;'><span>Shipping (Free over $500):</span> <span>${shipping:.2f}</span></div>
        <div style='display: flex; justify-content: space-between; color: #2a9d8f;'><span>Discount:</span> <span>-${discount:.2f}</span></div>
        <hr>
        <div style='display: flex; justify-content: space-between; font-size: 1.5em; font-weight: bold; color: #e63946;'>
            <span>Total:</span> <span>${final_total:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Credit Card", placeholder="XXXX-XXXX-XXXX-XXXX")
    with col2:
        st.text_input("CVV", placeholder="123", type="password")
        
    if st.button("💸 Confirm Payment", type="primary", use_container_width=True):
        st.success("Payment Successful! Your order is being processed.")
        st.session_state.cart = {}
        st.rerun()

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛒 My Cart")
    subtotal, total_items = get_cart_totals()
    
    if total_items > 0:
        st.markdown(f"""
        <div class="cart-summary">
            <h3 style='margin:0; color:#333;'>Total: <span style='color:#e63946;'>${subtotal:.2f}</span></h3>
            <span style='color:#666; font-size:0.9em;'>{total_items} Items</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Proceed to Checkout 💳", type="primary", use_container_width=True):
            checkout_dialog()
            
        st.markdown("### Cart Details")
        for sidebar_idx, (idx, details) in enumerate(list(st.session_state.cart.items())):
            item = details['item']
            qty = details['qty']
            
            with st.container(border=True):
                col_img, col_info = st.columns([1, 2], vertical_alignment="center")
                with col_img:
                    img = item.get('image')
                    st.image(img if img and img != "None" else "https://via.placeholder.com/50", use_container_width=True)
                with col_info:
                    st.markdown(f"**{item['name'][:15]}**...")
                    st.markdown(f"**${item['parsed_price']:.2f}**")
                
                c_minus, c_qty, c_plus = st.columns([1, 1, 1], vertical_alignment="center")
                with c_minus:
                    if st.button("➖", key=f"sub_{idx}_{sidebar_idx}", use_container_width=True):
                        update_cart_qty(idx, -1)
                        st.rerun()
                with c_qty:
                    st.markdown(f"<div style='text-align:center; font-weight:bold;'>{qty}</div>", unsafe_allow_html=True)
                with c_plus:
                    if st.button("➕", key=f"add_{idx}_{sidebar_idx}", use_container_width=True):
                        update_cart_qty(idx, 1)
                        st.rerun()
    else:
        st.info("Your cart is currently empty.")

    st.divider()
    st.header("⚙️ Settings")
    
    new_user = st.text_input("User ID", value=st.session_state.user_id)
    if new_user != st.session_state.user_id:
        st.session_state.user_id = new_user
        st.session_state.browse_cache = {} 
        st.rerun()

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>🛍️ Simulated Amazon Marketplace</h1>
    <p>Powered by MLOps & AWS & Transformer system</p>
</div>
""", unsafe_allow_html=True)

search_query = st.text_input("🔍 Search for products...", placeholder="Type a product name and press Enter...")

tab_browse, tab_recs = st.tabs(["🛒 Browse All Products", "✨ Personalized Recommendations"])

# === TAB 1: BROWSE ===
with tab_browse:
    col_sort, _ = st.columns([1, 3])
    with col_sort:
        sort_option = st.selectbox("Sort by", ["Relevance", "Price: Low to High", "Price: High to Low", "Newest"])

    current_page = st.session_state.page
    if current_page not in st.session_state.browse_cache:
        with st.spinner('Loading product catalog...'):
            try:
                response = requests.get(URL_BROWSE, params={"limit": 60})
                if response.status_code == 200:
                    raw_items = [enrich_item_data(i) for i in response.json()]
                    st.session_state.browse_cache[current_page] = raw_items
            except:
                st.error("Cannot connect to backend server (API Offline)")

    items = st.session_state.browse_cache.get(current_page, [])

    filtered_items = items
    if search_query:
        filtered_items = [i for i in filtered_items if search_query.lower() in i.get('name', '').lower()]

    if sort_option == "Price: Low to High":
        filtered_items.sort(key=lambda x: x['parsed_price'])
    elif sort_option == "Price: High to Low":
        filtered_items.sort(key=lambda x: x['parsed_price'], reverse=True)
    elif sort_option == "Newest":
        filtered_items.sort(key=lambda x: int(x.get('item_idx', 0)), reverse=True)

    if filtered_items:
        cols = st.columns(4)
        for idx, item in enumerate(filtered_items[:12]):
            col = cols[idx % 4]
            with col:
                with st.container(border=True):
                    if item['badge']:
                        badge_class = "badge-hot" if "Hot" in item['badge'] else "badge-discount"
                        st.markdown(f"<div class='badge {badge_class}'>{item['badge']}</div>", unsafe_allow_html=True)
                    
                    img = item.get('image')
                    st.image(img if img and img != "None" else "https://via.placeholder.com/300", use_container_width=True)
                    
                    st.markdown(f"<div class='product-title' title='{item.get('name', '')}'>{item.get('name', 'Unknown')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rating-stars'>{'★' * int(item['rating'])} {item['rating']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='price-tag'>${item['parsed_price']:.2f}</div>", unsafe_allow_html=True)
                    
                    # 重新加入 Like 按鈕，並調整排版
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("❤️ Like", key=f"br_like_{item['item_idx']}_{idx}", use_container_width=True):
                            like_item(item)
                    with c2:
                        if st.button("🔍 Info", key=f"br_view_{item['item_idx']}_{idx}", use_container_width=True):
                            show_item_details(item)
                            
                    if st.button("➕ Cart", key=f"br_add_{item['item_idx']}_{idx}", type="primary", use_container_width=True):
                        add_to_cart(item)
                        st.rerun()
    else:
        st.warning("No products match your search criteria.")

    st.markdown("<br>", unsafe_allow_html=True)
    c_prev, c_display, c_next = st.columns([1, 2, 1])
    with c_prev:
        if st.button("Previous Page", disabled=(st.session_state.page <= 1), use_container_width=True):
            st.session_state.page -= 1
            st.rerun()
    with c_display:
        st.markdown(f"<h4 style='text-align: center; color: #666;'>Page {st.session_state.page}</h4>", unsafe_allow_html=True)
    with c_next:
        if st.button("Next Page", use_container_width=True):
            st.session_state.page += 1
            st.rerun()

# === TAB 2: RECOMMENDATIONS ===
with tab_recs:
    col_title, col_btn, _ = st.columns([0.35, 0.15, 0.5], gap="small")
    with col_title: st.subheader("Recommended for You")
    with col_btn: st.button("🔄 Refresh", type="primary")

    try:
        with st.spinner('Calculating recommendations...'):
            response = requests.post(URL_RECOMMEND, json={"user_id": st.session_state.user_id})
            if response.status_code == 200:
                data = response.json()
                recs = [enrich_item_data(i) for i in data.get("recommendations", [])]
                
                if not recs:
                    st.info("💡 The system has not collected enough data yet.")
                else:
                    for idx, item in enumerate(recs):
                        with st.container(border=True):
                            c_img, c_info, c_action = st.columns([1.5, 3.5, 1.5], vertical_alignment="center")
                            with c_img:
                                img = item.get('image')
                                st.image(img if img and img != "None" else "https://via.placeholder.com/150", use_container_width=True)
                            with c_info:
                                st.markdown(f"### {item.get('name')}")
                                st.markdown(f"<div class='rating-stars'>{'★' * int(item['rating'])}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='price-tag' style='margin:0;'>${item['parsed_price']:.2f}</div>", unsafe_allow_html=True)
                            with c_action:
                                # 推薦列表加入 Like 按鈕
                                if st.button("❤️ Like", key=f"rec_like_{item['item_idx']}_{idx}", use_container_width=True):
                                    like_item(item)
                                if st.button("🔍 Details", key=f"rec_view_{item['item_idx']}_{idx}", use_container_width=True):
                                    show_item_details(item)
                                if st.button("➕ Add", key=f"rec_add_{item['item_idx']}_{idx}", type="primary", use_container_width=True):
                                    add_to_cart(item)
                                    st.rerun()
    except Exception as e:
        st.error(f"Recommendation API Offline: {e}")