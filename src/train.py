import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import yaml
import mlflow
import mlflow.pytorch
import numpy as np
from tqdm import tqdm
import json
from model import RecTransformer, VanillaRecTransformer

# 載入參數
with open("params.yaml") as f:
    params = yaml.safe_load(f)

class RecDataset(Dataset):
    def __init__(self, df, max_len=20):
        self.samples = []
        user_groups = df.groupby('visitorid')['item_idx'].apply(list)
        
        print(f"Processing {len(user_groups)} users for sliding window sequences...")
        
        for seq in tqdm(user_groups, desc="Building Sequences"):
            # seq 已經是 int list，直接使用
            if len(seq) < 2:
                continue
                
            # Sliding Window 策略
            for i in range(1, len(seq)):
                input_seq = seq[:i]
                if len(input_seq) > max_len:
                    input_seq = input_seq[-max_len:]
                
                pad_len = max_len - len(input_seq)
                input_seq = [0] * pad_len + input_seq
                
                target = seq[i]
                self.samples.append((input_seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def calculate_ndcg(pred_scores, target_item, k=10):
    _, topk_indices = torch.topk(pred_scores, k, dim=-1)
    ndcg_sum = 0
    batch_size = target_item.size(0)
    
    for i in range(batch_size):
        true_id = target_item[i].item()
        recs = topk_indices[i].tolist()
        if true_id in recs:
            rank = recs.index(true_id)
            ndcg_sum += 1.0 / np.log2(rank + 2)
            
    return ndcg_sum / batch_size

def train():
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    print("Loading data...")
    train_df = pd.read_csv(params['data']['processed_train_path'])
    test_df = pd.read_csv(params['data']['processed_test_path'])
    
    # [FIX] 直接讀取 features.py 產生的 item_map.json 來知道有多少商品
    # 不需要再自己生成 map，也不會發生 int() 轉換錯誤
    try:
        with open(params['data']['item_map_path'], 'r') as f:
            item_map = json.load(f)
        num_items = len(item_map)
    except FileNotFoundError:
        # Fallback (不建議，但以防萬一)
        print("Warning: item_map.json not found. Estimating from data...")
        num_items = max(train_df['item_idx'].max(), test_df['item_idx'].max())

    print(f"Total unique items: {num_items}")
    
    # [FIX] Dataset 不再需要傳入 item_map
    train_dataset = RecDataset(train_df, params['model']['max_len'])
    test_dataset = RecDataset(test_df, params['model']['max_len'])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=params['train']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['train']['batch_size'], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_type = params['model'].get('type', 'gqa') # 預設為 gqa
    print(f"Initializing model type: {model_type}")
    
    if model_type == "vanilla":
        model = VanillaRecTransformer(num_items).to(device)
    else:
        model = RecTransformer(num_items).to(device)
        
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=params['train']['lr'], weight_decay=0.05)

    with mlflow.start_run():
        mlflow.log_params(params['model'])
        mlflow.log_params(params['train'])
        mlflow.log_param("num_items", num_items)
        mlflow.log_param("train_samples", len(train_dataset))

        best_loss = float('inf')

        for epoch in range(params['train']['epochs']):
            model.train()
            total_loss = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['train']['epochs']} [Train]")
            
            for input_seq, target in train_pbar:
                input_seq = input_seq.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model(input_seq)
                logits = output[:, -1, :] 
                loss = criterion(logits, target)
                
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                total_loss += current_loss
                train_pbar.set_postfix({"loss": f"{current_loss:.4f}"})

            avg_train_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            if (epoch + 1) % params['train']['eval_interval'] == 0:
                model.eval()
                val_loss = 0
                val_ndcg = 0
                
                val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{params['train']['epochs']} [Val]", leave=False)
                
                with torch.no_grad():
                    for input_seq, target in val_pbar:
                        input_seq = input_seq.to(device)
                        target = target.to(device)
                        
                        output = model(input_seq)
                        logits = output[:, -1, :]
                        
                        batch_loss = criterion(logits, target).item()
                        val_loss += batch_loss
                        batch_ndcg = calculate_ndcg(logits, target, k=10)
                        val_ndcg += batch_ndcg
                        val_pbar.set_postfix({"val_loss": f"{batch_loss:.4f}"})
                
                avg_val_loss = val_loss / len(test_loader)
                avg_ndcg = val_ndcg / len(test_loader)
                
                tqdm.write(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | NDCG@10: {avg_ndcg:.4f}")
                
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("ndcg_10", avg_ndcg, step=epoch)
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    mlflow.pytorch.log_model(model, "model", registered_model_name=params['mlflow']['model_name'])
                    # 不需要再儲存 item_map，因為 features.py 已經存好了

    torch.save(model.state_dict(), "model.pth")
    print("Training complete. Model saved to model.pth")

if __name__ == "__main__":
    train()