#!/usr/bin/env python
"""
Script to convert superpoint model from pytorch to onnx.
Adapted from https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""
import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch

from torch import nn
from torch.export import Dim  # <- correct import

import network
from superpoint_frontend import reduce_l2


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Script to convert superpoint model from pytorch to onnx')
    parser.add_argument('pth_file', help="Pytorch weights file (.pth)")
    parser.add_argument('height', type=int, help="Height in pixels of input image")
    parser.add_argument('width', type=int, help="Width in pixels of input image")
    parser.add_argument('--output_dir', default="output", help="Output directory")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="batch size of input")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weights_path = args.pth_file
    batch_size = args.batch_size
    h = args.height
    w = args.width

    # Load model.
    pt_model = network.SuperPointNet()
    pytorch_total_params = sum(p.numel() for p in pt_model.parameters())
    print('Total number of params: ', pytorch_total_params)

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    pt_model.load_state_dict(torch.load(weights_path, map_location=map_location, weights_only=True))
    pt_model.eval()
    pt_model.cpu()

    # Create input to the model for onnx trace.
    x = torch.randn(batch_size, 1, h, w, requires_grad=False)
    torch_out = pt_model(x)
    onnx_filename = os.path.join(output_dir, "superpoint.onnx")

     # Legacy dynamic axes (stable with onnxruntime)
    dynamic_axes = {
        'input': {2: 'height', 3: 'width'},
        'semi': {2: 'height_out', 3: 'width_out'},
        'desc': {2: 'height_out', 3: 'width_out'}
    }

    # Export the model (new exporter: requires onnxscript)
    torch.onnx.export(
        pt_model,
        x,
        onnx_filename,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['semi', 'desc'],
        dynamic_axes=dynamic_axes,
    )

    # Check onnx conversion.
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_filename, providers=["CPUExecutionProvider"])

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(reduce_l2(to_numpy(torch_out[1])), reduce_l2(ort_outs[1]), rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # Generate config for movidius blob
    json_filename = os.path.join(output_dir, f"superpoint_{h}x{w}.json")
    with open(json_filename, 'w') as f:
        f.write(
            json.dumps(
            {
            "tensors":
            [
                {       
                    "output_tensor_name": "semi",
                    "output_dimensions": torch_out[0].shape,
                    "output_entry_iteration_index": 0,
                    "output_properties_dimensions": [0],
                    "property_key_mapping": [],
                    "output_properties_type": "f16"
                },
                {       
                    "output_tensor_name": "desc",
                    "output_dimensions": torch_out[1].shape,
                    "output_entry_iteration_index": 0,
                    "output_properties_dimensions": [0],
                    "property_key_mapping": [],
                    "output_properties_type": "f16"
                },          
            ]
            },
            separators=(',', ': '), indent=2))

    # Quantize the model
    quantized_filename = os.path.join(output_dir, "superpoint_quantized.onnx")
    print("Quantizing the model...")
    quantize_dynamic(
        model_input=onnx_filename,
        model_output=quantized_filename,
        weight_type=QuantType.QUInt8  # 8-bit unsigned integers
    )
    print(f"Quantized model saved to {quantized_filename}")

if __name__ == '__main__':
    main()