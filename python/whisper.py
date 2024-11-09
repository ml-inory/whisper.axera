import argparse
from axengine import InferenceSession
import numpy as np
import librosa
import os
from typing import Tuple
import soundfile as sf
import base64
import zhconv
import time


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
SOT_SEQUENCE = np.array([WHISPER_SOT,50260,WHISPER_TRANSCRIBE,WHISPER_NO_TIMESTAMPS], dtype=np.int32)
WHISPER_N_TEXT_STATE_MAP = {
    "tiny": 384,
    "small": 768
}


def get_args():
    parser = argparse.ArgumentParser(
        prog="whisper",
        description="Run Whisper on input audio file"
    )
    parser.add_argument("--wav", "-w", type=str, required=True, help="Input audio file")
    parser.add_argument("--model_type", "-t", type=str, choices=["tiny", "small"], required=True, help="model type, only support tiny or small currently")
    parser.add_argument("--model_path", "-p", type=str, required=False, default="../models", help="model path for *.axmodel, tokens.txt, positional_embedding.bin")
    return parser.parse_args()


def print_args(args):
    print(f"wav: {args.wav}")
    print(f"model_type: {args.model_type}")
    print(f"model_path: {args.model_path}")


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def load_models(model_path, model_type):
    encoder_path = f"{model_type}-encoder.axmodel"
    decoder_main_path = f"{model_type}-decoder-main.axmodel"
    decoder_loop_path = f"{model_type}-decoder-loop.axmodel"
    pe_path = f"{model_type}-positional_embedding.bin"
    token_path = f"{model_type}-tokens.txt"

    required_files = [os.path.join(model_path, i) for i in (encoder_path, decoder_main_path, decoder_loop_path, pe_path, token_path)]
    # Check file existence
    for i, file_path in enumerate(required_files):
        assert os.path.exists(file_path), f"{file_path} NOT exist"

    # Load encoder
    encoder = InferenceSession.load_from_model(required_files[0])
    # Load decoder main
    decoder_main = InferenceSession.load_from_model(required_files[1])
    # Load decoder loop
    decoder_loop = InferenceSession.load_from_model(required_files[2])
    # Load position embedding
    pe = np.fromfile(required_files[3], dtype=np.float32)
    # Load tokens
    tokens = []
    with open(required_files[4], "r") as f:
        for line in f:
            tokens.append(line.split(" ")[0])

    return encoder, decoder_main, decoder_loop, pe, tokens


def compute_feature(wav_path, n_mels = WHISPER_N_MELS):
    sample, _ = load_audio(wav_path)
    mel = librosa.feature.melspectrogram(y=sample, sr=WHISPER_SAMPLE_RATE, n_fft=WHISPER_N_FFT, hop_length=WHISPER_HOP_LENGTH, window="hann", center=True, pad_mode="reflect", power=2.0, n_mels=n_mels)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0

    # We pad 1500 frames at the end so that it is able to detect eot
    # You can use another value instead of 1500.
    mel = np.concatenate((mel, np.zeros((n_mels, 1500), dtype=np.float32)), axis=-1)

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


def main():
    args = get_args()
    print_args(args)

    # Check wav existence
    wav_path = args.wav
    assert os.path.exists(wav_path), f"{wav_path} NOT exist"

    # Load models and other stuff
    encoder, decoder_main, decoder_loop, pe, token_table = load_models(args.model_path, args.model_type)
    WHISPER_N_TEXT_STATE = WHISPER_N_TEXT_STATE_MAP[args.model_type]

    # Preprocess
    mel = compute_feature(wav_path, n_mels=WHISPER_N_MELS)

    # Run encoder
    start = time.time()
    x = encoder.run(input_feed={"mel": mel})
    n_layer_cross_k, n_layer_cross_v = x["n_layer_cross_k"], x["n_layer_cross_v"]
    print(f"Run encoder take {(time.time() - start) * 1000}ms")

    # Run decoder_main
    start = time.time()
    x = decoder_main.run(input_feed={
        "tokens": SOT_SEQUENCE,
        "n_layer_cross_k": n_layer_cross_k,
        "n_layer_cross_v": n_layer_cross_v
    })
    logits, n_layer_self_k_cache, n_layer_self_v_cache = x["logits"], x["out_n_layer_self_k_cache"], x["out_n_layer_self_v_cache"]
    print(f"Run decoder_main take {(time.time() - start) * 1000}ms")

    # Decode token
    logits = logits.flatten()[-WHISPER_VOCAB_SIZE:]
    logits = supress_tokens(logits, is_initial=True)
    max_token_id = np.argmax(logits)
    output_tokens = []
    print(f"First token: {max_token_id}")

    # Position embedding offset
    offset = SOT_SEQUENCE.shape[0]

    # Autoregressively run decoder until token meets EOT
    for i in range(WHISPER_N_TEXT_CTX - SOT_SEQUENCE.shape[0]):
        if max_token_id == WHISPER_EOT:
            break

        output_tokens.append(max_token_id)

        mask = np.zeros((WHISPER_N_TEXT_CTX,), dtype=np.float32)
        mask[: WHISPER_N_TEXT_CTX - offset - 1] = NEG_INF

        # Run decoder_loop
        start = time.time()
        x = decoder_loop.run(input_feed={
            "tokens": np.array([output_tokens[-1]], dtype=np.int32),
            "in_n_layer_self_k_cache": n_layer_self_k_cache,
            "in_n_layer_self_v_cache": n_layer_self_v_cache,
            "n_layer_cross_k": n_layer_cross_k,
            "n_layer_cross_v": n_layer_cross_v,
            "positional_embedding": pe[offset * WHISPER_N_TEXT_STATE : (offset + 1) * WHISPER_N_TEXT_STATE],
            "mask": mask
        })
        logits, n_layer_self_k_cache, n_layer_self_v_cache = x["logits"], x["out_n_layer_self_k_cache"], x["out_n_layer_self_v_cache"]
        print(f"Run decoder_loop take {(time.time() - start) * 1000}ms")

        # Decode token
        offset += 1
        logits = supress_tokens(logits.flatten(), is_initial=False)
        max_token_id = np.argmax(logits)

        print(f"Iter {i} \t Token: {max_token_id}")
    
    s = b""
    for i in output_tokens:
        if i in token_table:
            s += base64.b64decode(token_table[i])
    # print(s.decode().strip())
    pd = zhconv.convert(s.decode().strip(), 'zh-hans')
    print(f"Result: {pd}")


if __name__ == "__main__":
    main()