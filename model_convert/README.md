# 模型转换

## 创建虚拟环境

```
conda create -n whisper python=3.10
conda activate whisper
```

## 安装依赖

```
git clone https://github.com/ml-inory/whisper.axera.git
cd model_convert
pip install -r requirements.txt
```

## 导出模型（PyTorch -> ONNX）

导出 tiny 模型
```
python export_onnx.py --model tiny
```

导出 small 模型
```
python export_onnx.py --model small
```

导出成功后会生成以 `tiny-*` 或 `small-*` 开头的两个模型（`xxx-encoder.onnx`, `xxx-decoder.onnx`）和必要的文件

## 转换模型（ONNX -> Axera）

依赖Pulsar2工具链，可在[Axera的HuggingFace](https://huggingface.co/AXERA-TECH/Pulsar2)获取

使用模型转换工具 `Pulsar2` 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 `.axmodel`，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

### 下载数据集
```
bash download_dataset.sh
unzip dataset.zip
```

### 生成量化数据集

#### tiny 模型

```
python generate_data.py --model tiny
```

#### small 模型

```
python generate_data.py --model small
```

### 模型转换

#### 修改配置文件

修改以 config_whisper 开头的 json 文件中的所有 calibration_dataset 字段为 **生成量化数据集** 步骤中的 `tar.gz` 文件路径

#### Pulsar2 build

参考命令如下：

**encoder**

```
pulsar2 build --input small/small-encoder.onnx --config config_whisper_encoder_u16.json --output_dir small_encoder --output_name small-encoder.axmodel --target_hardware AX650 --compiler.check 0
```

**decoder_main**

```
pulsar2 build --input small/small-decoder-main.onnx --config config_whisper_decoder_main_u16.json --output_dir small_decoder_main --output_name small-decoder-main.axmodel --target_hardware AX650 --compiler.check 0
```

**decoder_loop**

```
pulsar2 build --input small/small-decoder-loop.onnx --config config_whisper_decoder_loop_u16.json --output_dir small_decoder_loop --output_name small-decoder-loop.axmodel --target_hardware AX650 --compiler.check 0
```
