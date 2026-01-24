# src/evaluate.py
import pandas as pd
import torch
import pickle
import mlflow
import os
import ast
import numpy as np
from tqdm import tqdm
from src.model import RecTransformer  # 確保 src/model.py 存在

# ====== 設定 ======
CONFIG = {
    "data_path": "features/events_processed.csv",
    "baseline_path": "models/popular_items.pkl",
    "model_path": "models/transformer_model.pth",
    "map_path": "models/item_map.pkl",
    "metrics_path": "metrics/metrics.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "top_k": 10,  # 我們只看前 10 名推薦準不準
    # 模型參數需與 training 一致
    "model_config": {
        "embed_dim": 64,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 2,
        "max_len": 50
    }
}

def evaluate():
    print("[Evaluate] Loading artifacts...")
    
    # 1. 載入資料
    df = pd.read_csv(CONFIG["data_path"])
    df["item_sequence"] = df["item_sequence"].apply(ast.literal_eval)
    
    # 2. 載入 Baseline (熱門商品)
    with open(CONFIG["baseline_path"], "rb") as f:
        popular_items = pickle.load(f)
        
    # 3. 載入 Item Map
    with open(CONFIG["map_path"], "rb") as f:
        item_map = pickle.load(f)
    
    # 4. 載入 Transformer 模型
    num_items = len(item_map) + 1
    model = RecTransformer(num_items, CONFIG["model_config"]).to(CONFIG["device"])
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
    model.eval()

    print(f"[Evaluate] Start evaluating on {len(df)} users...")
    
    # 指標累加器
    metrics = {
        "baseline_hits": 0,
        "transformer_hits": 0,
        "total_count": 0
    }

    # 5. 開始迴圈評估
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            seq = row["item_sequence"]
            
            # 至少要有 2 個行為才能做 "已知前 N-1 個，預測第 N 個"
            if len(seq) < 2:
                continue
                
            # === 準備測試資料 ===
            ground_truth = seq[-1]      # 真實的最後一個商品
            input_seq = seq[:-1]        # 輸入序列 (除了最後一個)
            
            # === A. 評估 Baseline ===
            # Baseline 策略：永遠推薦固定的熱門商品
            if ground_truth in popular_items[:CONFIG["top_k"]]:
                metrics["baseline_hits"] += 1
            
            # === B. 評估 Transformer ===
            # 1. 轉 ID
            seq_idx = [item_map.get(item, 0) for item in input_seq]
            # 2. Padding / Truncation
            max_len = CONFIG["model_config"]["max_len"]
            seq_idx = seq_idx[-max_len:]
            
            input_tensor = torch.tensor([seq_idx], dtype=torch.long).to(CONFIG["device"])
            
            # 3. 推理
            output = model(input_tensor)  # (1, seq_len, vocab_size)
            last_logits = output[0, -1, :]  # 取最後一步的預測
            
            # 4. 取 Top K 預測
            top_k_indices = torch.topk(last_logits, CONFIG["top_k"]).indices.tolist()
            
            # 5. 轉回 Item ID
            # 建立反向表 (ID -> Item) 放在這有點慢，建議移到外面，但為了範例簡單先這樣
            reverse_map = {v: k for k, v in item_map.items()}
            rec_items = [reverse_map.get(idx) for idx in top_k_indices]
            
            if ground_truth in rec_items:
                metrics["transformer_hits"] += 1
            
            metrics["total_count"] += 1

    # 6. 計算最終分數 (Hit Rate @ 10)
    final_metrics = {
        "baseline_hr_10": round(metrics["baseline_hits"] / metrics["total_count"], 4),
        "transformer_hr_10": round(metrics["transformer_hits"] / metrics["total_count"], 4),
        "improvement": round((metrics["transformer_hits"] - metrics["baseline_hits"]) / metrics["total_count"], 4)
    }

    print("\n====== Evaluation Results ======")
    print(f"Total Users Tested: {metrics['total_count']}")
    print(f"Baseline Hit Rate@10:    {final_metrics['baseline_hr_10']}")
    print(f"Transformer Hit Rate@10: {final_metrics['transformer_hr_10']}")
    print("================================")

    # 7. 記錄到 MLflow 與檔案
    # 確保 metrics 資料夾存在
    os.makedirs(os.path.dirname(CONFIG["metrics_path"]), exist_ok=True)
    import json
    with open(CONFIG["metrics_path"], "w") as f:
        json.dump(final_metrics, f)

    # 這裡我們使用一個新的 run 或者延續之前的 run (通常 evaluate 是獨立步驟，建議開新 run)
    mlflow.set_experiment("Transformer_RecSys")
    with mlflow.start_run(run_name="evaluate_stage"):
        mlflow.log_metrics(final_metrics)

if __name__ == "__main__":
    evaluate()