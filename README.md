# whisper.axera

OpenAI Whisper on Axera

- 目前支持 C++ 和 Python 两种语言
- 预编译模型下载
  - [Baidu](https://pan.baidu.com/s/1tOHVMZCin0A68T5HmKRJyg?pwd=axyz)
  - [Huggingface](https://huggingface.co/AXERA-TECH/Whisper)

- 如需自行转换请参考[模型转换](/model_convert/README.md)

## 支持平台

- [x] AX650N
- [x] AX630C

## 模型转换

[模型转换](./model_convert/README.md)

## 上板部署

- 基于 AX650N、AX630C 的设备已预装 Ubuntu22.04
- 链接互联网，确保设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备：
  - [爱芯派Pro(AX650N)](https://wiki.sipeed.com/hardware/zh/maixIV/m4ndock/m4ndock.html)
  - [M.2 Accelerator card(AX650N)](https://axcl-docs.readthedocs.io/zh-cn/latest/doc_guide_hardware.html)
  - [爱芯派2(AX630C)](https://axera-pi-2-docs-cn.readthedocs.io/zh-cn/latest/index.html)
  - [Module-LLM(AX630C)](https://docs.m5stack.com/zh_CN/module/Module-LLM)
  - [LLM630 Compute Kit(AX630C)](https://docs.m5stack.com/zh_CN/core/LLM630%20Compute%20Kit)


### Python API 运行

#### Requirements

```
cd python
pip3 install -r requirements.txt
```

####  pyaxenigne

参考 https://github.com/AXERA-TECH/pyaxengine 安装 NPU Python API

#### 运行

登陆开发板后

输入命令

```
cd python  
python3 whisper.py --model_type small --model_path ../models --wav ../demo.wav --language zh
```

输出结果

```
root@ax650:/mnt/qtang/whisper.axera/python# python3 whisper.py --wav ../demo.wav --model_type small --model_path ../models/ --language zh
[INFO] Available providers:  ['AxEngineExecutionProvider']
wav: ../demo.wav
model_type: small
model_path: ../models/
language: zh
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Chip type: ChipType.MC50
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.10.1s
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 3.2-patch1 117f5fd4
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 3.2-patch1 117f5fd4
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 3.2-patch1 117f5fd4
Load models take 2322.563409805298ms
Preprocess wav take 6971.68493270874ms
Run encoder take 211.52877807617188ms
Run decoder_main take 79.00094985961914ms
First token: 17556
Run decoder_loop take 101.91774368286133ms
Iter 0   Token: 20844
Run decoder_loop take 60.30416488647461ms
Iter 1   Token: 7781
Run decoder_loop take 60.22000312805176ms
Iter 2   Token: 20204
Run decoder_loop take 60.23716926574707ms
Iter 3   Token: 28455
Run decoder_loop take 60.214996337890625ms
Iter 4   Token: 31962
Run decoder_loop take 60.17565727233887ms
Iter 5   Token: 6336
Run decoder_loop take 60.94002723693848ms
Iter 6   Token: 254
Run decoder_loop take 60.71639060974121ms
Iter 7   Token: 2930
Run decoder_loop take 60.225725173950195ms
Iter 8   Token: 236
Run decoder_loop take 60.167789459228516ms
Iter 9   Token: 36135
Run decoder_loop take 60.29987335205078ms
Iter 10          Token: 15868
Run decoder_loop take 61.163902282714844ms
Iter 11          Token: 252
Run decoder_loop take 60.273170471191406ms
Iter 12          Token: 1546
Run decoder_loop take 60.23144721984863ms
Iter 13          Token: 46514
Run decoder_loop take 60.31966209411621ms
Iter 14          Token: 50257
Result: 甚至出现交易几乎停滞的情况
```

运行参数说明:  
| 参数名称 | 说明 | 默认值 |
| --- | --- | --- |
| --wav | 输入音频文件 | |
| --model_type/-t | 模型类型, tiny/base/small | |
| --model_path/-p | 模型所在目录 | ../models |
| --language/-l | 识别语言 | zh |

### 示例

### CPP API 运行

#### 交叉编译

在 PC 上完成

```
cd cpp
./download_bsp.sh
./build.sh
```

#### 运行

在 AX650N 设备上执行

```
./install/whisper -w ../demo.wav
```

或  

```
./install/whisper --model_type small --model_path ../models -w ../demo.wav
```

输出结果

```
root@ax650:/mnt/qtang/whisper.axera/cpp# ./install/whisper --wav ../demo.wav --model_type small --model_path ../models/ --language zh
wav_file: ../demo.wav
model_path: ../models/
model_type: small
language: zh
Encoder run take 188.30 ms
First token: 17556       take 81.88ms
Next Token: 20844        take 29.64ms
Next Token: 7781         take 29.70ms
Next Token: 20204        take 29.64ms
Next Token: 28455        take 29.65ms
Next Token: 31962        take 29.61ms
Next Token: 6336         take 29.67ms
Next Token: 254          take 29.63ms
Next Token: 2930         take 29.61ms
Next Token: 236          take 29.56ms
Next Token: 36135        take 29.64ms
Next Token: 15868        take 29.71ms
Next Token: 252          take 29.51ms
Next Token: 1546         take 29.63ms
Next Token: 46514        take 29.51ms
Next Token: 50257        take 29.69ms
All take 801.13 ms
Result: 甚至出现交易几乎停滞的情况
```

### Latency

RTF: Real-Time Factor

| Models        | AX650N | AX630C |
| ------------- | ------ | ------ |
| Whisper-Tiny  | 0.05   |        |
| Whisper-Base  |        | 0.35   |
| Whisper-Small | 0.20   |        |

### Word Error Rate

| Models        | AX650N | AX630C |
| ------------- | ------ | ------ |
| Whisper-Tiny  |        |        |
| Whisper-Base  |        |        |
| Whisper-Small |  0.11  |        |

## 技术讨论

- Github issues
- QQ 群: 139953715
