import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import yaml
import json
import numpy as np
from tqdm import tqdm
from model import RecTransformer, VanillaRecTransformer
from train import RecDataset
import os
from datetime import datetime

# 定義一個通用的指標計算函數，避免重複排序
def calculate_metrics(logits, target, k=10):
    if logits.dim() == 2 and logits.size(1) > k:
        _, topk_indices = torch.topk(logits, k, dim=-1)
    else:
        topk_indices = logits

    batch_size = target.size(0)
    
    ndcg_sum = 0
    recall_sum = 0
    mrr_sum = 0
    
    # 轉回 CPU 處理列表運算 (對於小 Batch 來說通常夠快且易於除錯)
    topk_indices = topk_indices.tolist()
    targets = target.tolist()
    
    for i in range(batch_size):
        true_id = targets[i]
        recs = topk_indices[i]
        
        if true_id in recs:
            # 命中!
            rank = recs.index(true_id) # 0-based index
            
            # Recall (Hit Rate)
            recall_sum += 1
            
            # MRR (Mean Reciprocal Rank) - 1-based rank
            mrr_sum += 1.0 / (rank + 1)
            
            # NDCG
            ndcg_sum += 1.0 / np.log2(rank + 2)
            
    return {
        "ndcg": ndcg_sum,
        "recall": recall_sum,
        "mrr": mrr_sum
    }

def evaluate():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        with open("item_map.json", "r") as f:
            item_map = json.load(f)
    except FileNotFoundError:
        print("Error: item_map.json not found.")
        return

    num_items = len(item_map)
    
    # 載入 Baseline
    baseline_items = []
    try:
        with open("models/baseline_top10.json", "r") as f:
            baseline_items = json.load(f)
    except FileNotFoundError:
        pass

    # 準備測試資料
    test_df = pd.read_csv(params['data']['processed_test_path'])
    test_dataset = RecDataset(test_df, params['model']['max_len'])
    test_loader = DataLoader(test_dataset, batch_size=params['train']['batch_size'], shuffle=False)
    
    # [核心修改]: 動態決定要實例化哪一個模型
    model_type = params['model'].get('type', 'gqa')
    print(f"Loading model type: {model_type}")
    
    if model_type == "vanilla":
        model = VanillaRecTransformer(num_items).to(device)
    else:
        model = RecTransformer(num_items).to(device)

    # 載入權重
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # 執行評估 (此部分邏輯保持不變...)
    metrics_model = {"ndcg": 0, "recall": 0, "mrr": 0}
    metrics_baseline = {"ndcg": 0, "recall": 0, "mrr": 0}
    total_loss = 0
    num_samples = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    if baseline_items:
        baseline_tensor = torch.tensor(baseline_items, device=device).unsqueeze(0)

    with torch.no_grad():
        for input_seq, target in tqdm(test_loader, desc="Evaluating"):
            batch_size = input_seq.size(0)
            num_samples += batch_size
            input_seq, target = input_seq.to(device), target.to(device)

            output = model(input_seq)
            logits = output[:, -1, :]
            
            total_loss += criterion(logits, target).item() * batch_size
            batch_metrics = calculate_metrics(logits, target, k=10)
            for k in metrics_model: metrics_model[k] += batch_metrics[k]

            if baseline_items:
                batch_baseline = baseline_tensor.expand(batch_size, -1)
                base_metrics = calculate_metrics(batch_baseline, target, k=10)
                for k in metrics_baseline: metrics_baseline[k] += base_metrics[k]

    # 7. 計算平均值
    avg_loss = total_loss / num_samples
    
    final_metrics = {
        "test_loss": avg_loss,
        "model_ndcg_10": metrics_model["ndcg"] / num_samples,
        "model_recall_10": metrics_model["recall"] / num_samples,
        "model_mrr_10": metrics_model["mrr"] / num_samples,
        
        "baseline_ndcg_10": metrics_baseline["ndcg"] / num_samples,
        "baseline_recall_10": metrics_baseline["recall"] / num_samples,
        "baseline_mrr_10": metrics_baseline["mrr"] / num_samples,
    }

    print("\n" + "="*40)
    print(f"Final Test Results (N={num_samples}):")
    print(f"Loss: {final_metrics['test_loss']:.4f}")
    print("-" * 20)
    print(f"Metric      | Model   | Baseline")
    print(f"------------|---------|---------")
    print(f"NDCG@10     | {final_metrics['model_ndcg_10']:.4f}  | {final_metrics['baseline_ndcg_10']:.4f}")
    print(f"Recall@10   | {final_metrics['model_recall_10']:.4f}  | {final_metrics['baseline_recall_10']:.4f}")
    print(f"MRR@10      | {final_metrics['model_mrr_10']:.4f}  | {final_metrics['baseline_mrr_10']:.4f}")
    print("="*40)

    metrics_file = "metrics.json"
    final_metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_metrics["model_type"] = params['model'].get('type', 'gqa') # 從 params 讀取當前模型類型

    history = []
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                history = json.load(f)
                if isinstance(history, dict):
                    history = [history]
        except (json.JSONDecodeError, FileNotFoundError):
            history = []
    history.append(final_metrics)
    with open(metrics_file, "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"Metrics appended to {metrics_file}")

if __name__ == "__main__":
    evaluate()