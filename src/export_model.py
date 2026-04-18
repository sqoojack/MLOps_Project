import os
import json
import yaml
import torch
import onnx
import tensorrt as trt
from model import RecTransformer

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
    model = RecTransformer(num_items)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("Export ONNX.")
    seq_len = params['model']['max_len']
    dummy_input = torch.randint(1, num_items, (1, seq_len), dtype=torch.long)
    
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
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX saved: {onnx_path}")

    print("Build TensorRT 10.16.1 engine.")
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
    
    # Define optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (1, seq_len), (1, seq_len), (128, seq_len))
    config.add_optimization_profile(profile)
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled.")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Engine build failed.")
        return

    with open(trt_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"TensorRT engine saved: {trt_path}")

if __name__ == "__main__":
    export_pipeline()