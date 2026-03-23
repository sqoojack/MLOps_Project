import os
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

with open("params.yaml") as f:
    params = yaml.safe_load(f)

class RecDataset(Dataset):
    def __init__(self, df, max_len=20):
        self.samples = []
        user_groups = df.groupby('visitorid')['item_idx'].apply(list)
        for seq in tqdm(user_groups, desc="Building Sequences"):
            if len(seq) < 2:
                continue
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
        return torch.tensor(self.samples[idx][0], dtype=torch.long), torch.tensor(self.samples[idx][1], dtype=torch.long)

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
    
    # 擷取 SageMaker 環境變數
    sm_train_dir = os.environ.get('SM_CHANNEL_TRAIN')
    sm_test_dir = os.environ.get('SM_CHANNEL_TEST')
    sm_item_map_dir = os.environ.get('SM_CHANNEL_ITEM_MAP')
    sm_model_dir = os.environ.get('SM_MODEL_DIR', params.get('data', {}).get('model_path', 'artifacts/'))
    
    # 動態決定資料路徑 (SageMaker 掛載的目錄下，檔名即為上傳時的原始檔名)
    train_path = os.path.join(sm_train_dir, os.path.basename(params['data']['processed_train_path'])) if sm_train_dir else params['data']['processed_train_path']
    test_path = os.path.join(sm_test_dir, os.path.basename(params['data']['processed_test_path'])) if sm_test_dir else params['data']['processed_test_path']
    item_map_path = os.path.join(sm_item_map_dir, os.path.basename(params['data']['item_map_path'])) if sm_item_map_dir else params['data']['item_map_path']

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    try:
        with open(item_map_path, 'r') as f:
            item_map = json.load(f)
        num_items = len(item_map)
    except FileNotFoundError:
        num_items = max(train_df['item_idx'].max(), test_df['item_idx'].max())

    train_dataset = RecDataset(train_df, params['model']['max_len'])
    test_dataset = RecDataset(test_df, params['model']['max_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=params['train']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['train']['batch_size'], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = params['model'].get('type', 'gqa')
    
    if model_type == "vanilla":
        model = VanillaRecTransformer(num_items).to(device)
    else:
        model = RecTransformer(num_items).to(device)
        
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=params['train']['lr'], weight_decay=0.25)

    with mlflow.start_run():
        mlflow.log_params(params['model'])
        mlflow.log_params(params['train'])
        mlflow.log_param("loss_function", "CrossEntropy")

        best_ndcg = -float('inf')

        for epoch in range(params['train']['epochs']):
            model.train()
            total_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['train']['epochs']} [Train]")
            
            for input_seq, target in train_pbar:
                input_seq, target = input_seq.to(device), target.to(device)
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

            if (epoch + 1) % params['train']['eval_interval'] == 0:
                model.eval()
                val_loss, val_ndcg = 0, 0
                val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
                
                with torch.no_grad():
                    for input_seq, target in val_pbar:
                        input_seq, target = input_seq.to(device), target.to(device)
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
                
                mlflow.log_metrics({"val_loss": avg_val_loss, "ndcg_10": avg_ndcg}, step=epoch)
                
                # [修改點] 嚴格以 NDCG 創新高來決定是否更新 best model
                if avg_ndcg > best_ndcg:
                    best_ndcg = avg_ndcg
                    mlflow.pytorch.log_model(model, "model", registered_model_name=params['mlflow']['model_name'])
                    os.makedirs(sm_model_dir, exist_ok=True)
                    model_save_path = os.path.join(sm_model_dir, 'model.pth')
                    torch.save(model.state_dict(), model_save_path)
                    with open(os.path.join(sm_model_dir, 'model_config.json'), 'w') as f:
                        json.dump({"num_items": num_items, "params": params['model']}, f)
                        
                    tqdm.write("🌟 New Best NDCG! Model Saved to both MLflow and SageMaker.")

    print(f"Training complete. Best NDCG: {best_ndcg:.4f}")

if __name__ == "__main__":
    train()