import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import yaml
import mlflow
import mlflow.pytorch
import numpy as np
from tqdm import tqdm  # 引入 tqdm
from model import RecTransformer

# 載入參數
with open("params.yaml") as f:
    params = yaml.safe_load(f)

class RecDataset(Dataset):
    def __init__(self, csv_path, max_len=20):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        # 簡單編碼 itemid -> int
        self.item_map = {item: i+1 for i, item in enumerate(self.df['itemid'].unique())}
        self.num_items = len(self.item_map)
        
        # Group by user
        self.user_data = self.df.groupby('visitorid')['itemid'].apply(
            lambda x: [self.item_map[i] for i in x]
        ).tolist()

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        seq = self.user_data[idx]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        else:
            seq = [0] * (self.max_len - len(seq)) + seq
        return torch.tensor(seq, dtype=torch.long)

def calculate_ndcg(pred_scores, target_item, k=10):
    """計算單個 Batch 的 NDCG"""
    _, topk_indices = torch.topk(pred_scores, k, dim=-1)
    
    ndcg_sum = 0
    for i in range(len(target_item)):
        true_id = target_item[i].item()
        recs = topk_indices[i].tolist()
        
        if true_id in recs:
            rank = recs.index(true_id)
            ndcg_sum += 1.0 / np.log2(rank + 2)
            
    return ndcg_sum / len(target_item)

def train():
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    # 準備數據
    train_dataset = RecDataset(params['data']['processed_train_path'], params['model']['max_len'])
    test_dataset = RecDataset(params['data']['processed_test_path'], params['model']['max_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=params['train']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['train']['batch_size'], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecTransformer(train_dataset.num_items).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=params['train']['lr'])

    with mlflow.start_run():
        mlflow.log_params(params['model'])
        mlflow.log_params(params['train'])

        best_loss = float('inf')

        for epoch in range(params['train']['epochs']):
            model.train()
            total_loss = 0
            
            # 使用 tqdm 包裝 train_loader
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['train']['epochs']} [Train]")
            
            for batch in train_pbar:
                batch = batch.to(device)
                input_seq = batch[:, :-1]
                target = batch[:, -1]

                optimizer.zero_grad()
                output = model(input_seq)
                
                logits = output[:, -1, :] 
                loss = criterion(logits, target)
                
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                total_loss += current_loss
                
                # 更新進度條右側的資訊
                train_pbar.set_postfix({"loss": f"{current_loss:.4f}"})

            avg_train_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            # [Validation Step]
            if (epoch + 1) % params['train']['eval_interval'] == 0:
                model.eval()
                val_loss = 0
                val_ndcg = 0
                
                # 使用 tqdm 包裝 test_loader
                val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{params['train']['epochs']} [Val]", leave=False)
                
                with torch.no_grad():
                    for batch in val_pbar:
                        batch = batch.to(device)
                        input_seq = batch[:, :-1]
                        target = batch[:, -1]
                        
                        output = model(input_seq)
                        logits = output[:, -1, :]
                        
                        batch_loss = criterion(logits, target).item()
                        val_loss += batch_loss
                        
                        batch_ndcg = calculate_ndcg(logits, target, k=10)
                        val_ndcg += batch_ndcg
                        
                        val_pbar.set_postfix({"val_loss": f"{batch_loss:.4f}"})
                
                avg_val_loss = val_loss / len(test_loader)
                avg_ndcg = val_ndcg / len(test_loader)
                
                # 使用 tqdm.write 以免破壞進度條顯示
                tqdm.write(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | NDCG@10: {avg_ndcg:.4f}")
                
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("ndcg_10", avg_ndcg, step=epoch)
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    mlflow.pytorch.log_model(
                        model, 
                        "model",
                        registered_model_name=params['mlflow']['model_name']
                    )
    
    # 儲存最後的模型權重供 evaluate.py 使用
    torch.save(model.state_dict(), "model.pth")
    print("Training complete. Model saved to model.pth")

if __name__ == "__main__":
    train()