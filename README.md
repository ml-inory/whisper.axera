# whisper.axera
OpenAI Whisper demo on Axera  
目前支持C++和Python两种语言

## 转换好的模型
[Baidu](https://pan.baidu.com/s/1tOHVMZCin0A68T5HmKRJyg?pwd=axyz)

### [自行转换模型指导](/model_convert/README.md)

## Python

## C++

### 编译

```./download_bsp.sh```  
```./build.sh```

### 运行

```./install/whisper --model_type small -e ../small-encoder.axmodel -m ../small-decoder-main.axmodel -l ../small-decoder-loop.axmodel -w ../BAC009S0764W0124.wav  -p ../small-positional_embedding.bin -t ../small-tokens.txt```

### Latency
| model | latency(ms) |
|---|---|
|tiny-encoder|20.41|
|tiny-decoder-main|3.49|
|tiny-decoder-loop|3.83|
|small-encoder|187.77|
|small-decoder-main|17.18|
|small-decoder-loop|19.56|
