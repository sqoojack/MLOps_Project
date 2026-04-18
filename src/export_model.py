import os
import json
import yaml
import torch
import onnx
import tensorrt as trt
from model import RecTransformer

def export_pipeline():
    # Load config
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Path setup
    model_path = params['data'].get('model_path', 'artifacts/model.pth')
    item_map_path = params['data'].get('item_map_path', 'artifacts/item_map.json')
    onnx_path = "artifacts/model.onnx"
    trt_path = "artifacts/model.engine"
    
    # Load Item Map for input dimensions
    with open(item_map_path, "r") as f:
        item_map = json.load(f)
    num_items = len(item_map)

    # 1. Initialize & Load PyTorch Model
    print("Loading PyTorch model weights...")
    model = RecTransformer(num_items)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 2. Export to ONNX (Opset 17 for 2026 standards)
    print(f"Exporting to ONNX v1.21.0...")
    dummy_input = torch.randint(1, num_items, (1, params['model']['max_len']), dtype=torch.long)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Verify ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX model saved: {onnx_path}")

    # 3. Build TensorRT 10.16 Engine
    print("Building TensorRT 10.16.1 engine...")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # Enable Explicit Batch (required for ONNX)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    # TensorRT 10.x workspace limit (2GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    
    # FP16 Optimization
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 optimization enabled.")

    # Build and serialize engine
    serialized_engine = builder.build_serialized_network(network, config)
    with open(trt_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"✅ TensorRT engine saved: {trt_path}")

if __name__ == "__main__":
    export_pipeline()