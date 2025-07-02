import argparse
import axengine as axe 
import numpy as np
import librosa
import os
from typing import Tuple
import soundfile as sf
import base64
import zhconv
import time
# from languages import WHISPER_LANGUAGES
from whisper_tokenizer import *
import json


WHISPER_N_MELS      = 80
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_FFT       = 480
WHISPER_HOP_LENGTH  = 160

WHISPER_SOT           = 50258
WHISPER_EOT           = 50257
WHISPER_BLANK         = 220
WHISPER_NO_TIMESTAMPS = 50363
WHISPER_NO_SPEECH     = 50362
WHISPER_TRANSLATE     = 50358
WHISPER_TRANSCRIBE    = 50359
WHISPER_VOCAB_SIZE    = 51865
WHISPER_N_TEXT_CTX    = 448

NEG_INF = float("-inf")
SOT_SEQUENCE = np.array([0,0,0,0], dtype=np.int32)
WHISPER_N_TEXT_STATE_MAP = {
    "tiny": 384,
    "base": 512,
    "small": 768,
    "large": 1280,
    "large-v3": 1280,
    "turbo": 1280
}


def get_args():
    parser = argparse.ArgumentParser(
        prog="whisper",
        description="Run Whisper on input audio file"
    )
    parser.add_argument("--wav", "-w", type=str, required=True, help="Input audio file")
    parser.add_argument("--model_type", "-t", type=str, choices=["tiny", "base", "small", "large", "large-v3", "turbo"], required=True, help="model type, only support tiny, base and small currently")
    parser.add_argument("--model_path", "-p", type=str, required=False, default="../models", help="model path for *.axmodel, tokens.txt, positional_embedding.bin")
    parser.add_argument("--language", "-l", type=str, required=False, default="zh", help="Target language, support en, zh, ja, and others. See languages.py for more options.")
    parser.add_argument("--task", type=str, required=False, choices=["translate", "transcribe"], default="transcribe")
    return parser.parse_args()


def print_args(args):
    print(f"wav: {args.wav}")
    print(f"model_type: {args.model_type}")
    print(f"model_path: {args.model_path}")
    print(f"language: {args.language}")
    print(f"task: {args.task}")


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=WHISPER_SAMPLE_RATE)
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def load_models(model_path, model_type, language, task):
    encoder_path = f"{model_type}-encoder.axmodel"
    decoder_main_path = f"{model_type}-decoder-main.axmodel"
    decoder_loop_path = f"{model_type}-decoder-loop.axmodel"
    pe_path = f"{model_type}-positional_embedding.bin"
    model_config_file = f"{model_type}_config.json"

    required_files = [os.path.join(model_path, i) for i in (encoder_path, decoder_main_path, decoder_loop_path, pe_path, model_config_file)]
    # Check file existence
    for i, file_path in enumerate(required_files):
        assert os.path.exists(file_path), f"{file_path} NOT exist"

    # Load encoder
    encoder = axe.InferenceSession(required_files[0])
    # Load decoder main
    decoder_main = axe.InferenceSession(required_files[1])
    # Load decoder loop
    decoder_loop = axe.InferenceSession(required_files[2])
    # Load position embedding
    pe = np.fromfile(required_files[3], dtype=np.float32)
    # Load tokens
    model_config = json.load(open(required_files[4], "r"))
    model_config["all_language_tokens"] = [int(i) for i in model_config["all_language_tokens"].split(",")]
    model_config["all_language_codes"] = [i for i in model_config["all_language_codes"].split(",")]
    tokenizer = get_tokenizer(
        model_config["is_multilingual"],
        num_languages=len(model_config["all_language_codes"]),
        language=language,
        task=task,
    )
    # tokens = []
    # with open(required_files[4], "r") as f:
    #     for line in f:
    #         line = line.strip()
    #         tokens.append(line.split(" ")[0])

    return encoder, decoder_main, decoder_loop, pe, tokenizer, model_config


def compute_feature(wav_path, n_mels = WHISPER_N_MELS, padding = 480000):
    audio, sr = load_audio(wav_path)

    audio = np.concatenate((audio, np.zeros((padding,), dtype=np.float32)), axis=-1)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=WHISPER_N_FFT, hop_length=WHISPER_HOP_LENGTH, window="hann", center=True, pad_mode="reflect", power=2.0, n_mels=n_mels)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0

    # We pad 1500 frames at the end so that it is able to detect eot
    # You can use another value instead of 1500.
    # mel = np.concatenate((mel, np.zeros((n_mels, 1500), dtype=np.float32)), axis=-1)

    target = 3000
    if mel.shape[1] > target:
        # -50 so that there are some zero tail paddings.
        mel = mel[:, : target]
        mel[:, -50:] = 0

    # We don't need to pad it to 30 seconds now!
    if mel.shape[1] < target:
        mel = np.concatenate((mel, np.zeros((n_mels, target - mel.shape[1]), dtype=np.float32)), axis=-1)

    return mel


def supress_tokens(logits, is_initial):
    if is_initial:
        logits[WHISPER_EOT] = NEG_INF
        logits[WHISPER_BLANK] = NEG_INF

    logits[WHISPER_NO_TIMESTAMPS] = NEG_INF
    logits[WHISPER_SOT] = NEG_INF
    logits[WHISPER_NO_SPEECH] = NEG_INF
    logits[WHISPER_TRANSLATE] = NEG_INF
    return logits


def build_sot_sequence(lang, task, model_config):
    WHISPER_SOT = model_config["sot"]

    if lang not in LANGUAGES.keys():
        raise Exception(f"Unknown language: {lang}. Check whisper_tokenizer.py::LANGUAGES for correct options.")
    
    try:
        lang_ind = model_config["all_language_codes"].index(lang)
        lang_token = model_config["all_language_tokens"][lang_ind]
    except:
        raise ValueError(f"Unknown language: {lang}. Check whisper_tokenizer.py::LANGUAGES for correct options.")
    
    WHISPER_TRANSLATE = model_config["translate"]
    WHISPER_TRANSCRIBE = model_config["transcribe"]
    task_token = model_config[task]

    WHISPER_NO_TIMESTAMPS = model_config["no_timestamps"]

    SOT_SEQUENCE = np.array([WHISPER_SOT, lang_token, task_token, WHISPER_NO_TIMESTAMPS], dtype=np.int32)
    return SOT_SEQUENCE


def main():
    args = get_args()
    print_args(args)

    # Check wav existence
    wav_path = args.wav
    assert os.path.exists(wav_path), f"{wav_path} NOT exist"

    # Load models and other stuff
    start = time.time()
    encoder, decoder_main, decoder_loop, pe, tokenizer, model_config = load_models(args.model_path, args.model_type, args.language, args.task)
    print(f"Load models take {(time.time() - start) * 1000}ms")
    WHISPER_N_TEXT_STATE = model_config["n_text_state"]
    WHISPER_N_TEXT_CTX = model_config["n_text_ctx"]
    WHISPER_EOT = model_config["eot"]
    WHISPER_BLANK = model_config["blank_id"]
    WHISPER_NO_TIMESTAMPS = model_config["no_timestamps"]
    WHISPER_NO_SPEECH = model_config["no_speech"]

    # sot sequence
    build_sot_sequence(args.language, args.task, model_config)

    # Preprocess
    start = time.time()
    n_mels = model_config["n_mels"]
    mel = compute_feature(wav_path, n_mels=n_mels)
    print(f"Preprocess wav take {(time.time() - start) * 1000}ms")
    # mel.tofile("mel.bin")

    # Run encoder
    start = time.time()
    x = encoder.run(None, input_feed={"mel": mel[None, ...]})
    n_layer_cross_k, n_layer_cross_v = x
    print(f"Run encoder take {(time.time() - start) * 1000}ms")

    # n_layer_cross_k.tofile("n_layer_cross_k.bin")
    # n_layer_cross_v.tofile("n_layer_cross_v.bin")

    # Run decoder_main
    start = time.time()
    x = decoder_main.run(None, input_feed={
        "tokens": SOT_SEQUENCE[None, ...],
        "n_layer_cross_k": n_layer_cross_k,
        "n_layer_cross_v": n_layer_cross_v
    })
    logits, n_layer_self_k_cache, n_layer_self_v_cache = x
    print(f"Run decoder_main take {(time.time() - start) * 1000}ms")

    # Decode token
    logits = logits[0, -1, :]
    logits = supress_tokens(logits, is_initial=True)
    # logits.tofile("logits.bin")
    max_token_id = np.argmax(logits)
    output_tokens = []
    print(f"First token: {max_token_id}")

    # Position embedding offset
    offset = SOT_SEQUENCE.shape[0]

    # Autoregressively run decoder until token meets EOT
    for i in range(WHISPER_N_TEXT_CTX - SOT_SEQUENCE.shape[0]):
        if max_token_id >= WHISPER_EOT:
            break

        output_tokens.append(max_token_id)

        mask = np.zeros((WHISPER_N_TEXT_CTX,), dtype=np.float32)
        mask[: WHISPER_N_TEXT_CTX - offset - 1] = NEG_INF

        # Run decoder_loop
        start = time.time()
        x = decoder_loop.run(None, input_feed={
            "tokens": np.array([[output_tokens[-1]]], dtype=np.int32),
            "in_n_layer_self_k_cache": n_layer_self_k_cache,
            "in_n_layer_self_v_cache": n_layer_self_v_cache,
            "n_layer_cross_k": n_layer_cross_k,
            "n_layer_cross_v": n_layer_cross_v,
            "positional_embedding": pe[offset * WHISPER_N_TEXT_STATE : (offset + 1) * WHISPER_N_TEXT_STATE][None, ...],
            "mask": mask
        })
        logits, n_layer_self_k_cache, n_layer_self_v_cache = x
        print(f"Run decoder_loop take {(time.time() - start) * 1000}ms")

        # Decode token
        offset += 1
        logits = supress_tokens(logits.flatten(), is_initial=False)
        max_token_id = np.argmax(logits)

        print(f"Iter {i} \t Token: {max_token_id}")
    
    # s = b""
    # for i in output_tokens:
    #     s += base64.b64decode(token_table[i])
    # # print(s.decode().strip())
    # pd = s.decode().strip()
    # if args.language == "zh":
    #     pd = zhconv.convert(pd, 'zh-hans')
    pd = tokenizer.decode(output_tokens)
    print(f"Result: {pd}")


if __name__ == "__main__":
    main()
