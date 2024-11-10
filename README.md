# whisper.axera

OpenAI Whisper on Axera

- 目前支持 C++ 和 Python 两种语言
- 预编译模型下载([Baidu](https://pan.baidu.com/s/1tOHVMZCin0A68T5HmKRJyg?pwd=axyz))，如需自行转换请参考[模型转换](/model_convert/README.md)

TBD：

- [ ] 支持 AX630C

## 模型转换

[模型转换](./model_convert/README.md)

## 上板部署

- AX650N 的设备已预装 Ubuntu22.04
- 以 root 权限登陆 AX650N 的板卡设备
- 链接互联网，确保 AX650N 的设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备：AX650N DEMO Board、爱芯派Pro

### Python API 运行

#### Requirements

```
apt-get install libsndfile1-dev
mkdir /opt/site-packages
pip3 install -r requirements.txt --prefix=/opt/site-packages
``` 

#### 添加环境变量

将以下两行添加到 `/root/.bashrc`(实际添加的路径需要自行检查)后，重新连接终端或者执行 `source ~/.bashrc`

```
export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages  
export PATH=$PATH:/opt/site-packages/local/bin
``` 

#### 运行

登陆开发板后

输入命令

```
cd python  
python3 whisper.py --model_path ../models --model_type small--wav ../demo.wav
```  

输出结果

```

```

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
./install/whisper --model_type small -e ../small-encoder.axmodel -m ../small-decoder-main.axmodel -l ../small-decoder-loop.axmodel -w ../demo.wav  -p ../small-positional_embedding.bin -t ../small-tokens.txt
```

输出结果

```
root@ax650:/mnt/qtang/whisper.axera-main/cpp# ./install/whisper -w ../demo.wav
encoder: ../models/small-encoder.axmodel
decoder_main: ../models/small-decoder-main.axmodel
decoder_loop: ../models/small-decoder-loop.axmodel
wav_file: ../demo.wav
Read positional_embedding
First token: 17556
Next Token: 20844
Next Token: 7781
Next Token: 20204
Next Token: 28455
Next Token: 31962
Next Token: 6336
Next Token: 254
Next Token: 2930
Next Token: 236
Next Token: 36135
Next Token: 15868
Next Token: 252
Next Token: 1546
Next Token: 46514
Next Token: 50257
Result: 甚至出现交易几乎停滞的情况
```

### Latency

#### AX650N

| model | latency(ms) |
|---|---|
|tiny-encoder|20.41|
|tiny-decoder-main|3.49|
|tiny-decoder-loop|3.83|
|small-encoder|187.77|
|small-decoder-main|17.18|
|small-decoder-loop|19.56|

#### AX630C

(TODO)

## 技术讨论

- Github issues
- QQ 群: 139953715
