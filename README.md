# whisper.axera
OpenAI Whisper demo on Axera

## Build

```./download_bsp.sh```  
```./build.sh```

## Run

```./whisper --model_type small -e ../small-encoder.axmodel -m ../small-decoder-main.axmodel -l ../small-decoder-loop.axmodel -w ../BAC009S0764W0124.wav  -p ../small-positional_embedding.bin -t ../small-tokens.txt```


## Latency
| model | latency(ms) |
|---|---|
|tiny-encoder|20.41|
|tiny-decoder-main|3.49|
|tiny-decoder-loop|3.83|
|small-encoder|187.77|
|small-decoder-main|17.18|
|small-decoder-loop|19.56|


## Converted Model
[Baidu](https://pan.baidu.com/s/1tOHVMZCin0A68T5HmKRJyg?pwd=axyz)


## [How to convert model](/doc/convert.md)
