# python features/fix_price.py
import json
import random

def preprocess_prices(file_path="items_metadata.json"):
    print(f"Reading {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        fixed_count = 0
        for item_id in metadata:
            price = metadata[item_id].get("price")
            if price is None or str(price).strip() in ["None", "N/A", ""]:
                random_price = round(random.uniform(50, 200), 2)
                metadata[item_id]["price"] = f"${random_price}"
                fixed_count += 1
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        print(f"✅ Success! Fixed {fixed_count} items with random prices ($50-$200).")
    except FileNotFoundError:
        print("❌ Error: items_metadata.json not found.")

if __name__ == "__main__":
    preprocess_prices()