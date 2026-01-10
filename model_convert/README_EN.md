# Model Convert

## Create virtual environment

```
conda create -n whisper python=3.10
conda activate whisper
```

## Install requirements

```
git clone https://github.com/ml-inory/whisper.axera.git
cd model_convert
pip install -r requirements.txt
```

## Export model(PyTorch -> ONNX)

tiny
```
python export_onnx.py --model tiny
```

small
```
python export_onnx.py --model small
```

After successful export, two model files (with prefixes tiny-* or small-*) will be generated

## Export model(ONNX -> Axera)

This process requires the Pulsar2 toolchain, available from [Axera's HuggingFace](https://huggingface.co/AXERA-TECH/Pulsar2).

Use the Pulsar2 model conversion tool to convert ONNX models into .axmodel format suitable for Axera NPU execution. The conversion typically involves two steps:

1. Generate PTQ (Post-Training Quantization) calibration dataset

2. Perform model conversion (PTQ quantization and compilation) using pulsar2 build commands

For detailed instructions, refer to the [AXera Pulsar2 Toolchain User Manual](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)


### Generate Calibration Dataset

#### tiny

```
python generate_data.py --model tiny
```

#### small

```
python generate_data.py --model small
```

After running, calibration set and config file used for model conversion will be generated.

### Model Conversion

#### Pulsar2 build

Reference commands:

**encoder**

```
pulsar2 build --input small-encoder.onnx --config config_whisper_small_encoder.json --output_dir small_encoder --output_name small-encoder.axmodel --target_hardware AX650 --compiler.check 0
```

**decoder**

```
pulsar2 build --input small-decoder.onnx --config config_whisper_small_decoder.json --output_dir small_decoder --output_name small-decoder.axmodel --target_hardware AX650 --compiler.check 0
```
