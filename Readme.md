# superpoint_infer_engine

## Purpose

This repo exists as an exercise to prepare a pretrained model (superpoint) to be used on different hardware platforms. Specifically, we convert superpoint's pytorch implementation to onnx, intel targets (using openVINO), and finally the movidius inference accelerator. Future work may be conversion for Coral inference accelerator. To use openVINO, we have re-implemented parts of superpoint in numpy for layers and other parts of execution that are not available out-of-the-box.

## Generating onnx representation

To get onnx representation, we have a script to produce the onnx version of superpoint.

```shell
### Usage
python3 convert_onnx.py <path_to_superpoint_weights_file> <height_of_input_image> <width_of_input_image>

### Example
python3 convert_onnx.py weights/superpoint_v1.pth 100 100
 Output will default to output/ directory.
```

`convert_onnx.py` also quantizes the exported model, writing `output/superpoint_quantized.onnx`
alongside the float32 `output/superpoint.onnx`, via:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(model_input=onnx_filename, model_output=quantized_filename, weight_type=QuantType.QUInt8)
```

This is **dynamic** quantization: weights are quantized ahead of time, but activations are
quantized on the fly at inference using a runtime-computed range - no calibration dataset is
needed. Inspecting the produced graph confirms this is genuine integer computation rather than
a QDQ float-simulated pattern: every conv layer (`conv1a/b`, `conv2a/b`, `conv3a/b`, `conv4a/b`,
`convDa/b`, `convPa/b`) is replaced with a `ConvInteger` op, paired with `DynamicQuantizeLinear`
nodes that quantize each layer's input activations at runtime. There is no `QLinearConv` and no
`DequantizeLinear` in the graph - this is the "QOperator" quantization format, not QDQ.

Note that this only affects backends that actually run the ONNX graph as-is (e.g. ONNX Runtime
CPU/WASM). The OpenVINO and Movidius conversion steps below consume `output/superpoint.onnx`
(the float32 export), so this quantization step doesn't currently affect those targets.

## Generating openVINO intermediate representation

We use openVINO tools to build. This is done expecting the openVINO toolkit is already installed. If you don't have it, go [here](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) If you want to build to movidius, use openVINO 2020.1 or 2020.2 and follow the directions below. Example of usage can be seen in run_openvino.py

```shell
### Usage
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_shape [1,1,<height_of_input_image>,<width_of_input_image>] --input_model <path_to_onnx_file> --output_dir output --data_type FP16

### Example
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_shape [1,1,100,100] --input_model output/superpoint_100x100.onnx --output_dir output --data_type FP16
# Output .bin and .xml files will be in output/
```

## Generating movidius blob representation using myriad compiler

We generate this representation using openVINO, and use it using depthai, which can be installed [here](https://docs.luxonis.com/api/). Example of usge can be seen in run_oak.py. It expects to be running a MegaAI/OAK-1 module.

```shell

### Usage
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/myriad_compile -m <path_to_xml_file> -o <path_to_blob_file> -ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4

### Example
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/myriad_compile -m output/superpoint_100x100.xml -o output/superpoint_100x100.blob -ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4
```
