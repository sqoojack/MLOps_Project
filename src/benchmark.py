# src/benchmark.py
import torch
import yaml
import time
import json
import numpy as np
import onnxruntime as ort
import os
from model import RecTransformer, VanillaRecTransformer

def run_benchmark():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
        
    num_items = 5000 
    batch_size = 1   
    seq_len = params['model']['max_len']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== Efficiency Benchmark ===")
    print(f"Device: {device}, Batch: {batch_size}, SeqLen: {seq_len}")

    # ==========================================
    # A. Vanilla Model (No Cache, No GQA)
    # ==========================================
    vanilla_model = VanillaRecTransformer(num_items).to(device)
    vanilla_model.eval()
    
    print("\n[Running Vanilla Model (No Cache)]...")
    start_time = time.time()
    with torch.no_grad():
        current_seq = torch.randint(1, num_items, (batch_size, 1)).to(device)
        for _ in range(seq_len):
            output = vanilla_model(current_seq)
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
            current_seq = torch.cat([current_seq, next_token], dim=1)
    vanilla_time = (time.time() - start_time) * 1000 
    print(f"Vanilla Total Time: {vanilla_time:.2f} ms")

    # ==========================================
    # B. GQA Model (PyTorch KV Cache)
    # ==========================================
    gqa_model = RecTransformer(num_items).to(device)
    gqa_model.eval()
    
    print("\n[Running GQA Model (With KV Cache)]...")
    start_time = time.time()
    with torch.no_grad():
        input_token = torch.randint(1, num_items, (batch_size, 1)).to(device)
        past_key_values = None
        for _ in range(seq_len):
            logits, past_key_values = gqa_model(input_token, use_cache=True, past_key_values=past_key_values)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            input_token = next_token 
    gqa_time = (time.time() - start_time) * 1000 
    print(f"GQA + Cache Total Time: {gqa_time:.2f} ms")

    # ==========================================
    # C. ONNX/TRT (GQA + KV Cache)
    # ==========================================
    print("\n[Running ONNX/TensorRT Model (GQA + KV Cache)]...")
    onnx_path = "artifacts/model_kv.onnx"
    trt_time = float('inf')
    
    if os.path.exists(onnx_path):
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': 'artifacts/trt_cache'
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        try:
            ort_session = ort.InferenceSession(onnx_path, providers=providers)
            
            num_kv_heads = params['model'].get('num_kv_heads', params['model']['num_heads'])
            head_dim = params['model']['embed_dim'] // params['model']['num_heads']
            
            k_shape = (batch_size, num_kv_heads, 1, head_dim)
            k0 = np.zeros(k_shape, dtype=np.float32)
            v0 = np.zeros(k_shape, dtype=np.float32)
            k1 = np.zeros(k_shape, dtype=np.float32)
            v1 = np.zeros(k_shape, dtype=np.float32)
            
            input_id = np.random.randint(1, num_items, (batch_size, 1)).astype(np.int64)
            
            _ = ort_session.run(None, {'input_ids': input_id, 'k0_in': k0, 'v0_in': v0, 'k1_in': k1, 'v1_in': v1})
            
            start_time = time.time()
            for _ in range(seq_len):
                inputs = {
                    'input_ids': input_id,
                    'k0_in': k0, 'v0_in': v0,
                    'k1_in': k1, 'v1_in': v1
                }
                outputs = ort_session.run(None, inputs)
                
                logits = outputs[0]
                k0, v0, k1, v1 = outputs[1], outputs[2], outputs[3], outputs[4]
                
                next_token = np.argmax(logits[:, -1, :])
                input_id = np.array([[next_token]], dtype=np.int64)
                
            trt_time = (time.time() - start_time) * 1000
            print(f"ONNX/TRT + Cache Total Time: {trt_time:.2f} ms")
        except Exception as e:
            print(f"ONNX/TRT Execution Error: {e}")
    else:
        print(f"Missing file: {onnx_path}. Please run export_kv.py first.")

    # ==========================================
    # Metrics & Export
    # ==========================================
    speedup_gqa = vanilla_time / gqa_time if gqa_time > 0 else 0
    speedup_trt = vanilla_time / trt_time if trt_time != float('inf') and trt_time > 0 else 0
    
    print(f"\nSpeedup: PyTorch GQA is {speedup_gqa:.2f}x faster than Vanilla.")
    if speedup_trt > 0:
        print(f"Speedup: ONNX/TRT GQA is {speedup_trt:.2f}x faster than Vanilla.")
    
    metrics = {
        "vanilla_latency_ms": vanilla_time,
        "gqa_latency_ms": gqa_time,
        "trt_latency_ms": trt_time if trt_time != float('inf') else None,
        "speedup_gqa_ratio": speedup_gqa,
        "speedup_trt_ratio": speedup_trt if speedup_trt > 0 else None
    }
    
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/benchmark_results.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
if __name__ == "__main__":
    run_benchmark()