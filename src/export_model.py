import os
import json
import yaml
import torch
import torch.nn as nn
import onnx
import tensorrt as trt
from model import RecTransformer

class KVCacheWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, k0, v0, k1, v1):
        past_key_values = [(k0, v0), (k1, v1)]
        logits, new_past = self.model(x, use_cache=True, past_key_values=past_key_values)
        return logits, new_past[0][0], new_past[0][1], new_past[1][0], new_past[1][1]

def export_kv_pipeline():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    model_path = params['data'].get('model_path', 'artifacts/model.pth')
    with open(params['data']['item_map_path'], "r") as f:
        num_items = len(json.load(f))

    model = RecTransformer(num_items)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    wrapper = KVCacheWrapper(model).eval()

    num_kv_heads = params['model'].get('num_kv_heads', params['model']['num_heads'])
    head_dim = params['model']['embed_dim'] // params['model']['num_heads']
    max_len = params['model']['max_len']

    x = torch.randint(1, num_items, (1, 1), dtype=torch.long)
    k_dummy = torch.zeros(1, num_kv_heads, 1, head_dim)
    v_dummy = torch.zeros(1, num_kv_heads, 1, head_dim)

    onnx_path = "artifacts/model_kv.onnx"
    trt_path = "artifacts/model_kv.engine"

    print("Exporting ONNX with KV Cache...")
    torch.onnx.export(
        wrapper,
        (x, k_dummy, v_dummy, k_dummy, v_dummy),
        onnx_path,
        opset_version=17,
        input_names=['input_ids', 'k0_in', 'v0_in', 'k1_in', 'v1_in'],
        output_names=['logits', 'k0_out', 'v0_out', 'k1_out', 'v1_out'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'k0_in': {0: 'batch_size', 2: 'past_seq_len'},
            'v0_in': {0: 'batch_size', 2: 'past_seq_len'},
            'k1_in': {0: 'batch_size', 2: 'past_seq_len'},
            'v1_in': {0: 'batch_size', 2: 'past_seq_len'},
            'logits': {0: 'batch_size'},
            'k0_out': {0: 'batch_size', 2: 'seq_len'},
            'v0_out': {0: 'batch_size', 2: 'seq_len'},
            'k1_out': {0: 'batch_size', 2: 'seq_len'},
            'v1_out': {0: 'batch_size', 2: 'seq_len'},
        }
    )
    print(f"ONNX saved: {onnx_path}")

    print("Building TensorRT 10.16.1 engine...")
    logger = trt.Logger(trt.Logger.WARNING)
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
    profile.set_shape('input_ids', (1, 1), (1, 1), (128, 1))
    
    min_k = (1, num_kv_heads, 1, head_dim)
    opt_k = (1, num_kv_heads, max_len // 2, head_dim)
    max_k = (128, num_kv_heads, max_len, head_dim)

    for name in ['k0_in', 'v0_in', 'k1_in', 'v1_in']:
        profile.set_shape(name, min_k, opt_k, max_k)
        
    config.add_optimization_profile(profile)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled.")

    engine = builder.build_serialized_network(network, config)
    with open(trt_path, "wb") as f:
        f.write(engine)
    print(f"TensorRT Engine saved: {trt_path}")

if __name__ == "__main__":
    export_kv_pipeline()