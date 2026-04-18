import os
import json
import yaml
import torch
import torch.nn as nn
import onnx
import tensorrt as trt
from model import RecTransformer

class DynamicONNXWrapper(nn.Module):
    def __init__(self, model, n_layers):
        super().__init__()
        self.model = model
        self.n_layers = n_layers
        
    def forward(self, *args):
        # args[0]: input_ids, args[1:]: flattened k0, k1..., v0, v1...
        input_ids = args[0]
        past_k = args[1:1+self.n_layers]
        past_v = args[1+self.n_layers:]
        
        past_kv = []
        for k, v in zip(past_k, past_v):
            past_kv.append((k, v))
            
        logits, new_kv = self.model(input_ids, use_cache=True, past_key_values=past_kv)
        
        # Flatten outputs
        flat_present = []
        for k, v in new_kv:
            flat_present.append(k)
        for k, v in new_kv:
            flat_present.append(v)
            
        return (logits, *flat_present)

def export_pipeline():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    model_path = params['data'].get('model_path', 'artifacts/model.pth')
    item_map_path = params['data'].get('item_map_path', 'artifacts/item_map.json')
    onnx_path = "artifacts/model.onnx"
    trt_path = "artifacts/model.engine"
    
    with open(item_map_path, "r") as f:
        item_map = json.load(f)
    num_items = len(item_map)

    print("Load PyTorch model.")
    base_model = RecTransformer(num_items)
    base_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    base_model.eval()

    num_layers = params['model']['num_layers']
    model = DynamicONNXWrapper(base_model, num_layers)
    model.eval()

    print("Export ONNX with flattened KV cache nodes.")
    seq_len = params['model']['max_len']
    num_kv_heads = params['model'].get('num_kv_heads', params['model']['num_heads'])
    head_dim = params['model']['embed_dim'] // params['model']['num_heads']
    
    dummy_input = torch.randint(1, num_items, (1, 1), dtype=torch.long)
    dummy_past_k = [torch.randn(1, num_kv_heads, 1, head_dim) for _ in range(num_layers)]
    dummy_past_v = [torch.randn(1, num_kv_heads, 1, head_dim) for _ in range(num_layers)]
    all_inputs = (dummy_input, *dummy_past_k, *dummy_past_v)

    input_names = ["input"] + [f"past_k_{i}" for i in range(num_layers)] + [f"past_v_{i}" for i in range(num_layers)]
    output_names = ["output"] + [f"present_k_{i}" for i in range(num_layers)] + [f"present_v_{i}" for i in range(num_layers)]
    
    dynamic_axes = {name: {0: 'batch_size', 2: 'seq_len'} for name in input_names + output_names}
    dynamic_axes["input"] = {0: 'batch_size', 1: 'seq_len'}

    torch.onnx.export(
        model,
        all_inputs,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    print(f"ONNX saved: {onnx_path}")

    print("Build TensorRT engine.")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(onnx_path):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 1), (1, seq_len), (128, seq_len))
    for i in range(num_layers):
        profile.set_shape(f"past_k_{i}", (1, num_kv_heads, 0, head_dim), (1, num_kv_heads, seq_len, head_dim), (128, num_kv_heads, seq_len, head_dim))
        profile.set_shape(f"past_v_{i}", (1, num_kv_heads, 0, head_dim), (1, num_kv_heads, seq_len, head_dim), (128, num_kv_heads, seq_len, head_dim))
    config.add_optimization_profile(profile)
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Engine build failed.")
        return

    with open(trt_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"TensorRT engine saved: {trt_path}")

if __name__ == "__main__":
    export_pipeline()