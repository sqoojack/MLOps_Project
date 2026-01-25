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
from model import RecTransformer

# 載入參數
with open("params.yaml") as f:
    params = yaml.safe_load(f)

class RecDataset(Dataset):
    def __init__(self, df, item_map, max_len=20):
        self.df = df
        self.item_map = item_map
        self.max_len = max_len
        self.samples = []
        
        # 預處理：產生 Sliding Window 樣本
        # 優化：直接過濾並轉換，避免多次迴圈
        user_groups = self.df.groupby('visitorid')['itemid'].apply(list)
        
        print(f"Processing {len(user_groups)} users for sliding window sequences...")
        
        for user_items in tqdm(user_groups, desc="Building Sequences"):
            # 將 itemid 轉為整數 ID
            # 注意：這裡確保 item 為 Python int，雖然 map 已經轉了，但多一層防護
            seq = [self.item_map[i] for i in user_items if i in self.item_map]
            
            if len(seq) < 2:
                continue
                
            # Sliding Window 策略
            for i in range(1, len(seq)):
                input_seq = seq[:i]
                if len(input_seq) > self.max_len:
                    input_seq = input_seq[-self.max_len:]
                
                pad_len = self.max_len - len(input_seq)
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
    
    # [FIX]: 解決 json int64 錯誤的關鍵點
    # 1. 取得所有 unique items
    all_items = set(train_df['itemid'].unique()) | set(test_df['itemid'].unique())
    
    # 2. 強制轉型：使用 int(item) 確保 Key 是 Python 原生 int，而不是 numpy.int64
    item_map = {int(item): i+1 for i, item in enumerate(all_items)}
    
    num_items = len(item_map)
    print(f"Total unique items: {num_items}")
    
    train_dataset = RecDataset(train_df, item_map, params['model']['max_len'])
    test_dataset = RecDataset(test_df, item_map, params['model']['max_len'])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=params['train']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['train']['batch_size'], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RecTransformer(num_items).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=params['train']['lr'])

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
                    
                    # 儲存 Item Map 供 API 使用 (使用修正後的 map)
                    with open("item_map.json", "w") as f:
                        json.dump(item_map, f)

    torch.save(model.state_dict(), "model.pth")
    print("Training complete. Model saved to model.pth")

if __name__ == "__main__":
    train()