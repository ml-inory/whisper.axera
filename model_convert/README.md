# 模型转换

## Requirements
（下面的步骤建议在Python虚拟环境下进行，可跳过）  
```
conda create -n whisper python=3.9
conda activate whisper
```

```
cd model_convert
pip install -r requirements.txt
```

## 导出ONNX
目前只支持导出tiny或small的模型，请根据需要选择

导出tiny模型
```
python export_onnx_ax_loop.py --model tiny
```

导出small模型
```
python export_onnx_ax_loop.py --model small
```

导出成功后会生成以tiny-或small-开头的三个模型（xxx-encoder.onnx xxx-decoder-main.onnx xxx-decoder-loop.onnx)和必要的文件

## 下载数据集
```
bash download_dataset.sh
unzip dataset.zip
```

## 生成量化数据集
tiny模型
```
python test_ax_loop.py --model tiny
```
small模型
```
python test_ax_loop.py --model small
```

## 修改配置文件
修改以config_whisper开头的json文件中的所有calibration_dataset字段为 生成量化数据集 步骤中的tar.gz文件路径

## 生成axmodel

参考命令如下：

encoder
```
pulsar2 build --input small-encoder.onnx --config config_whisper_encoder_u16.json --output_dir small_encoder --output_name small-encoder.axmodel --target_hardware AX650 --compiler.check 2
```

decoder_main
```
pulsar2 build --input small-decoder-main.onnx --config config_whisper_decoder_main_u16.json --output_dir small_decoder_main --output_name small-decoder-main.axmodel --target_hardware AX650 --compiler.check 2
```

decoder_loop
```
pulsar2 build --input small-decoder-loop.onnx --config config_whisper_decoder_loop_u16.json --output_dir small_decoder_loop --output_name small-decoder-loop.axmodel --target_hardware AX650 --compiler.check 2
```