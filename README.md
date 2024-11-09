# whisper.axera
OpenAI Whisper demo on Axera  
目前支持C++和Python两种语言

## 转换好的模型
[Baidu](https://pan.baidu.com/s/1tOHVMZCin0A68T5HmKRJyg?pwd=axyz)

### [自行转换模型指导](/model_convert/README.md)

## Python
### Requirements
```apt-get install libsndfile1-dev```  
```mkdir /opt/site-packages```  
```pip3 install -r requirements.txt --prefix=/opt/site-packages```  

将这两行放到/root/.bashrc  
(实际添加的路径需要自行检查)
```export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages```  
```export PATH=$PATH:/opt/site-packages/local/bin```  
 重新连接终端

### 运行
```cd python```  
```python3 whisper.py --wav demo.wav```  

### 示例


## C++

### 编译

```
cd cpp
./download_bsp.sh
./build.sh
```

### 运行

```./install/whisper -w ../demo.wav```  
或  
```./install/whisper --model_type small -e ../small-encoder.axmodel -m ../small-decoder-main.axmodel -l ../small-decoder-loop.axmodel -w ../demo.wav  -p ../small-positional_embedding.bin -t ../small-tokens.txt```  

### Latency
| model | latency(ms) |
|---|---|
|tiny-encoder|20.41|
|tiny-decoder-main|3.49|
|tiny-decoder-loop|3.83|
|small-encoder|187.77|
|small-decoder-main|17.18|
|small-decoder-loop|19.56|
