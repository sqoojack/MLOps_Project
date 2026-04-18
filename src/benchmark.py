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

    print("\n[Running ONNX/TensorRT Model with KV Cache]...")
    onnx_path = "artifacts/model.onnx"
    trt_time = float('inf')
    
    if os.path.exists(onnx_path):
        num_layers = params['model']['num_layers']
        num_kv_heads = params['model'].get('num_kv_heads', params['model']['num_heads'])
        head_dim = params['model']['embed_dim'] // params['model']['num_heads']

        # Generate dynamic shape profiles to prevent TRT engine recompilation
        min_shapes = ["input:1x1"]
        opt_shapes = ["input:1x1"]
        max_shapes = ["input:1x1"]
        
        for i in range(num_layers):
            min_shapes.append(f"past_k_{i}:1x{num_kv_heads}x0x{head_dim}")
            min_shapes.append(f"past_v_{i}:1x{num_kv_heads}x0x{head_dim}")
            
            opt_shapes.append(f"past_k_{i}:1x{num_kv_heads}x{seq_len//2}x{head_dim}")
            opt_shapes.append(f"past_v_{i}:1x{num_kv_heads}x{seq_len//2}x{head_dim}")
            
            max_shapes.append(f"past_k_{i}:1x{num_kv_heads}x{seq_len}x{head_dim}")
            max_shapes.append(f"past_v_{i}:1x{num_kv_heads}x{seq_len}x{head_dim}")

        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': 'artifacts/trt_cache',
                'trt_profile_min_shapes': ",".join(min_shapes),
                'trt_profile_max_shapes': ",".join(max_shapes),
                'trt_profile_opt_shapes': ",".join(opt_shapes)
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        try:
            ort_session = ort.InferenceSession(onnx_path, providers=providers)
            
            # --- Warm-up phase to build the engine before timing ---
            print("Warming up TRT Engine...")
            ort_inputs_warmup = {'input': np.random.randint(1, num_items, (batch_size, 1), dtype=np.int64)}
            for i in range(num_layers):
                ort_inputs_warmup[f'past_k_{i}'] = np.zeros((batch_size, num_kv_heads, 0, head_dim), dtype=np.float32)
                ort_inputs_warmup[f'past_v_{i}'] = np.zeros((batch_size, num_kv_heads, 0, head_dim), dtype=np.float32)
            
            for _ in range(seq_len):
                outputs = ort_session.run(None, ort_inputs_warmup)
                next_token = np.argmax(outputs[0][:, -1, :], axis=-1).reshape(batch_size, 1)
                
                ort_inputs_warmup['input'] = next_token
                for i in range(num_layers):
                    ort_inputs_warmup[f'past_k_{i}'] = outputs[1 + i]         
                    ort_inputs_warmup[f'past_v_{i}'] = outputs[1 + num_layers + i]

            # --- Actual timed benchmark ---
            ort_inputs = {'input': np.random.randint(1, num_items, (batch_size, 1), dtype=np.int64)}
            for i in range(num_layers):
                ort_inputs[f'past_k_{i}'] = np.zeros((batch_size, num_kv_heads, 0, head_dim), dtype=np.float32)
                ort_inputs[f'past_v_{i}'] = np.zeros((batch_size, num_kv_heads, 0, head_dim), dtype=np.float32)
            
            start_time = time.time()
            for _ in range(seq_len):
                outputs = ort_session.run(None, ort_inputs)
                next_token = np.argmax(outputs[0][:, -1, :], axis=-1).reshape(batch_size, 1)
                
                ort_inputs['input'] = next_token
                for i in range(num_layers):
                    ort_inputs[f'past_k_{i}'] = outputs[1 + i]         
                    ort_inputs[f'past_v_{i}'] = outputs[1 + num_layers + i]
                    
            trt_time = (time.time() - start_time) * 1000
            print(f"ONNX/TRT Total Time: {trt_time:.2f} ms")
        except Exception as e:
            print(f"ONNX/TRT Error: {e}")

    speedup_gqa = vanilla_time / gqa_time if gqa_time > 0 else 0
    speedup_trt = vanilla_time / trt_time if trt_time != float('inf') and trt_time > 0 else 0
    
    print(f"\nSpeedup: GQA is {speedup_gqa:.2f}x faster than Vanilla.")
    if speedup_trt > 0:
        print(f"Speedup: ONNX/TRT is {speedup_trt:.2f}x faster than Vanilla.")
    
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
        
    print("Saved to metrics/benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()