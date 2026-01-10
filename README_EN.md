# whisper.axera

<div align="center">
  <a href="README_EN.md">English</a> | <a href="README.md">中文</a>
</div>

OpenAI Whisper on Axera Platform

## Overview

This project provides an optimized implementation of OpenAI's Whisper speech recognition model for Axera AI processors (AX650N/AX630C). It supports both C++ and Python interfaces for efficient on-device speech-to-text conversion.

## Features

- **Dual Language Support**: Both C++ and Python APIs available
- **Multiple Model Sizes**: Support for tiny, base, small, and turbo model variants
- **Multi-language Recognition**: Tested with English, Chinese, Japanese, and Korean
- **Optimized Performance**: Specially optimized for Axera NPU acceleration
- **Easy Deployment**: Pre-built packages and cross-compilation support

## Supported Platforms

- ✅ AX650N
- ✅ AX630C

## Pre-trained Models

Download pre-compiled models from:
- [Baidu Cloud](https://pan.baidu.com/s/1tOHVMZCin0A68T5HmKRJyg?pwd=axyz)
- [Huggingface](https://huggingface.co/AXERA-TECH/Whisper)

For custom model conversion, please refer to [Model Conversion Guide](./model_convert/README_EN.md).

## Model Conversion

Currently supported model scales:
- tiny
- base  
- small
- turbo

Tested languages:
- English
- Chinese
- Japanese
- Korean

For other languages or custom model sizes, please refer to the [Model Conversion Guide](./model_convert/README_EN.md).

## Deployment on Target Devices

### Prerequisites
- AX650N/AX630C devices with Ubuntu 22.04 pre-installed
- Internet connection for `apt install` and `pip install`
- Verified hardware platforms:
  - [MaixIV M4nDock (AX650N)](https://wiki.sipeed.com/hardware/zh/maixIV/m4ndock/m4ndock.html)
  - [M.2 Accelerator Card (AX650N)](https://axcl-docs.readthedocs.io/zh-cn/latest/doc_guide_hardware.html)
  - [Axera Pi 2 (AX630C)](https://axera-pi-2-docs-cn.readthedocs.io/zh-cn/latest/index.html)
  - [Module-LLM (AX630C)](https://docs.m5stack.com/zh_CN/module/Module-LLM)
  - [LLM630 Compute Kit (AX630C)](https://docs.m5stack.com/zh_CN/core/LLM630%20Compute%20Kit)

## Programming Language Support

### Python

Tested with Python 3.12. We recommend using [Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh) for environment management.

#### Installation

```bash
cd python
pip3 install -r requirements.txt
```

####  pyaxenigne

Install NPU Python API from: https://github.com/AXERA-TECH/pyaxengine

#### Usage

##### Command Line Interface

```
cd python  
python3 whisper_cli.py -t tiny --model_path ../models-ax650 -w ../demo.wav --language zh
```

Example Output:  

```
(whisper) root@ax650:/mnt/data/Github/whisper.axera/python# python whisper_cli.py -t tiny -w ../demo.wav
[INFO] Available providers:  ['AxEngineExecutionProvider']
{'wav': '../demo.wav', 'model_type': 'tiny', 'model_path': '../models-ax650', 'language': 'zh', 'task': 'transcribe'}
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Chip type: ChipType.MC50
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.12.0s
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 5.0 76f70fdc
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 5.0 76f70fdc
ASR result:
擅职出现交易几乎停止的情况
RTF: 0.11406774537746188

```

Command line arguments:
| Argument | Description | Default |
| --- | --- | --- |
| --wav | Input audio file | - |
| --model_type/-t | Model type: tiny/base/small | - |
| --model_path/-p | Model directory | ../models |
| --language/-l | Recognition language | zh |


##### Server Mode

```
(whisper) root@ax650:/mnt/data/Github/whisper.axera/python# python whisper_svr.py
[INFO] Available providers:  ['AxEngineExecutionProvider']
Server started at http://0.0.0.0:8000

```

Test the server:
```
python test_svr.py
```


<h3 id="CPP">CPP</h3>

#### Cross-compilation (on PC)

Tested on Ubuntu 22.04.

Setup Development Environment:
```
sudo apt update
sudo apt install build-essential cmake
```

Download Cross-compilation Toolchain
Download from: [AARCH64 Toolchain](https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz)  

Add the toolchain to your PATH.

Build   
```
cd cpp
./download_bsp.sh

# AX650
./build_ax650.sh

# AX630C
./build_ax630c.sh
```

#### Usage on Target Device

AX650N

```
./install/ax650/whisper_cli -w ../demo.wav
```

or with specific parameters:  

```
./install/ax650/whisper_cli --model_type small --model_path ../models-ax650 -w ../demo.wav
```

Example output:  

```
root@ax650:/mnt/qtang/whisper.axera/cpp# ./install/ax650/whisper_cli --wav ../demo.wav --model_type small --model_path ../models/ --language zh
wav_file: ../demo.wav
model_path: ../models-ax650
model_type: tiny
language: zh
Init whisper success, take 0.4360seconds
Result: 甚至出现交易几乎停止的情况
RTF: 0.1166

```

### Server Mode

```
./install/ax650/whisper_srv --model_type tiny --model_path ../models-ax650 --language zh --port 8080

port: 8080
model_path: ../models-ax650
model_type: tiny
language: zh
[I][                            main][  60]: Initializing server...
[I][                            main][  65]: Init server success
[I][                           start][  32]: Start server at port 8080, POST binary stream to IP:8080/asr

```

### Client test using curl:

```
ffmpeg -i demo.wav -f f32le -c:a pcm_f32le - 2>/dev/null | \
curl -X POST 10.126.33.192:8080/asr \
  -H "Content-Type: application/octet-stream" \
  --data-binary @-
```

## Performance Benchmarks

### Latency

RTF: Real-Time Factor

CPP:

| Models        | AX650N | AX630C |
| ------------- | ------ | ------ |
| Whisper-Tiny  | 0.08   |        |
| Whisper-Base  | 0.11   | 0.35   |
| Whisper-Small | 0.24   |        |
| Whisper-Turbo | 0.48   |        |

Python:  

| Models        | AX650N | AX630C |
| ------------- | ------ | ------ |
| Whisper-Tiny  | 0.12   |        |
| Whisper-Base  | 0.16   | 0.35   |
| Whisper-Small | 0.50   |        |
| Whisper-Turbo | 0.60   |        |

### Word Error Rate(Test on AIShell dataset)

| Models        | AX650N | AX630C |
| ------------- | ------ | ------ |
| Whisper-Tiny  |  0.24  |        |
| Whisper-Base  |  0.18  |        |
| Whisper-Small |  0.11  |        |
| Whisper-Turbo |  0.06  |        |

To reproduce WER test results:  

Download dataset:  
```
cd model_convert
bash download_dataset.sh
```

Run test script:  
```
cd python
conda activate whisper
python test_wer.py -d aishell --gt_path ../model_convert/datasets/ground_truth.txt --model_type tiny

```

### MEM Usage

* CMM Stands for Physical memory used by Axera modules like VDEC(Video decoder), VENC(Video encoder), NPU, etc.

Python:  

| Models        | CMM(MB)| OS(MB) |
| ------------- | ------ | ------ |
| Whisper-Tiny  |  332   |  512   |
| Whisper-Base  |  533   |  644   |
| Whisper-Small |  1106  |  906   |
| Whisper-Turbo |  2065  |  2084  |

C++:  

| Models        | CMM(MB)| OS(MB) |
| ------------- | ------ | ------ |
| Whisper-Tiny  |  332   |  31    |
| Whisper-Base  |  533   |  54    |
| Whisper-Small |  1106  |  146   |
| Whisper-Turbo |  2065  |  86    |


## Technical Discussion

- Github issues
- Tencent QQ Group: 139953715
