# python src/train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import mlflow
import os
import pickle
from tqdm import tqdm
from model import RecTransformer

# ====== 全域設定 ======
CONFIG = {
    "data_path": "features/events_processed.csv",
    "model_path": "models/transformer_model.pth",
    "mapping_path": "models/item_map.pkl",
    "max_len": 50,
    "embed_dim": 64,
    "num_heads": 4,
    "num_kv_heads": 2,
    "num_layers": 2,
    "batch_size": 32,
    "epochs": 5,
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ====== Data Loader ======
class SequenceDataset(Dataset):
    def __init__(self, csv_path, max_len=50, item_map=None):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.df["item_sequence"] = self.df["item_sequence"].apply(ast.literal_eval)
        
        all_items = set(x for seq in self.df["item_sequence"] for x in seq)
        if item_map is None:
            self.item_map = {item: i+1 for i, item in enumerate(all_items)}
        else:
            self.item_map = item_map
        self.num_items = len(self.item_map) + 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx]["item_sequence"]
        seq_idx = [self.item_map.get(item, 0) for item in seq]
        seq_idx = seq_idx[-self.max_len:]
        pad_len = self.max_len - len(seq_idx)
        input_seq = [0] * pad_len + seq_idx
        
        x = torch.tensor(input_seq[:-1], dtype=torch.long)
        y = torch.tensor(input_seq[1:], dtype=torch.long)
        return x, y

# ====== 訓練主流程 ======
def train():
    os.makedirs("models", exist_ok=True)
    mlflow.set_experiment("Transformer_RecSys")

    print("[Train] Loading Data...")
    dataset = SequenceDataset(CONFIG["data_path"], max_len=CONFIG["max_len"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    with open(CONFIG["mapping_path"], "wb") as f:
        pickle.dump(dataset.item_map, f)
    
    model = RecTransformer(dataset.num_items, CONFIG).to(CONFIG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"[Train] Start training on {CONFIG['device']}...")
    
    with mlflow.start_run():
        mlflow.log_params(CONFIG)
        
        for epoch in range(CONFIG["epochs"]):
            total_loss = 0
            model.train()

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
            
            for x, y in pbar:
                x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
                
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits.view(-1, dataset.num_items), y.view(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(dataloader)
            
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
            
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        torch.save(model.state_dict(), CONFIG["model_path"])
        mlflow.log_artifact(CONFIG["model_path"])
        print(f"[Train] Model saved to {CONFIG['model_path']}")

if __name__ == "__main__":
    train()