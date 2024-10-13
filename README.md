# whisper.axera
OpenAI Whisper demo on Axera

## Build

```./download_bsp.sh```  
```./build.sh```

## Run

```./whisper -e ../tiny-encoder.axmodel -m ../tiny-decoder-main.axmodel -l ../tiny-decoder-loop.axmodel -w ../aishell_S0764/BAC009S0764W0124.wav  -p ../positional_embedding.bin -t ../tiny-tokens.txt```