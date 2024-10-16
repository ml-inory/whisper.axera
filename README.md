# whisper.axera
OpenAI Whisper demo on Axera

## Build

```./download_bsp.sh```  
```./build.sh```

## Run

```./whisper --model_type tiny -e ../tiny-encoder.axmodel -m ../tiny-decoder-main.axmodel -l ../tiny-decoder-loop.axmodel -w ../aishell_S0764/BAC009S0764W0124.wav  -p ../tiny-positional_embedding.bin -t ../tiny-tokens.txt```


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