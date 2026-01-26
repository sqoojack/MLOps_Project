# python src/benchmark.py
import torch
import yaml
import time
import json
from model import RecTransformer, VanillaRecTransformer

def run_benchmark():
    # 1. 設定模擬參數
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
        
    num_items = 5000 # 假設商品數
    batch_size = 1   # 模擬單一使用者即時推薦 (Latency 敏感)
    seq_len = params['model']['max_len']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=== Efficiency Benchmark ===")
    print(f"Device: {device}, Batch: {batch_size}, SeqLen: {seq_len}")

    # ==========================================
    # 選手 A: Vanilla Model (無 Cache, 無 GQA)
    # ==========================================
    vanilla_model = VanillaRecTransformer(num_items).to(device)
    vanilla_model.eval()
    
    # 模擬輸入：隨著時間 t=1 到 t=20，序列越來越長
    print("\n[Running Vanilla Model (No Cache)]...")
    start_time = time.time()
    
    with torch.no_grad():
        # 模擬生成 20 個步驟 (Autoregressive)
        # 雖然推薦通常只推一次，但這裡模擬連續互動的情況來放大差異
        current_seq = torch.randint(1, num_items, (batch_size, 1)).to(device)
        
        for _ in range(seq_len):
            # Vanilla 必須輸入當前累積的「完整序列」
            output = vanilla_model(current_seq)
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
            current_seq = torch.cat([current_seq, next_token], dim=1)
            
    vanilla_time = (time.time() - start_time) * 1000 # ms
    print(f"Vanilla Total Time: {vanilla_time:.2f} ms")


    # ==========================================
    # 選手 B: GQA Model (有 Cache, 有 GQA)
    # ==========================================
    gqa_model = RecTransformer(num_items).to(device)
    gqa_model.eval()
    
    print("\n[Running GQA Model (With KV Cache)]...")
    start_time = time.time()
    
    with torch.no_grad():
        # 1. 初始輸入 (First Token)
        input_token = torch.randint(1, num_items, (batch_size, 1)).to(device)
        past_key_values = None
        
        for _ in range(seq_len):
            # GQA 只要輸入「最新的一個 token」，並傳入過去的 Cache
            # (注意：RecTransformer.forward 需要回傳 cache)
            logits, past_key_values = gqa_model(input_token, use_cache=True, past_key_values=past_key_values)
            
            # 決定下一個 token (模擬)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            input_token = next_token # 下一步只輸入這個新的
            
    gqa_time = (time.time() - start_time) * 1000 # ms
    print(f"GQA + Cache Total Time: {gqa_time:.2f} ms")

    # ==========================================
    # 總結比較
    # ==========================================
    speedup = vanilla_time / gqa_time
    print(f"\nResult: GQA+Cache is {speedup:.2f}x faster than Vanilla.")
    
    # 儲存結果供 DVC 追蹤 (可選)
    metrics = {
        "vanilla_latency_ms": vanilla_time,
        "gqa_latency_ms": gqa_time,
        "speedup_ratio": speedup
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    run_benchmark()