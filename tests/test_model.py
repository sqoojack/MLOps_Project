# tests/test_model.py
import sys
import os
import pytest
import torch
import yaml

# [FIX]: 將專案根目錄加入 path，這樣才能 import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import RecTransformer

def test_model_output_shape():
    # 讀取 params.yaml (確保路徑正確)
    config_path = os.path.join(os.path.dirname(__file__), '..', 'params.yaml')
    with open(config_path) as f:
        params = yaml.safe_load(f)
    
    model = RecTransformer(num_items=100)
    batch_size = 2
    # 確保 max_len 來自參數
    seq_len = params['model']['max_len']
    
    x = torch.randint(0, 100, (batch_size, seq_len))
    output = model(x)
    
    # 檢查輸出: [Batch, Seq_Len, Num_Items + 1]
    assert output.shape == (batch_size, seq_len, 101)

def test_masking_logic():
    model = RecTransformer(num_items=50)
    x = torch.randint(1, 50, (1, 10))
    output = model(x)
    assert not torch.isnan(output).any(), "Model output contains NaN"