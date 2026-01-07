#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
"""
Please first run ./export-onnx_ax_loop.py
before you run this script
"""
import os
import argparse
import base64
import csv
import random
import re
import zhconv
from typing import Tuple, List

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import tarfile
import glob
from tqdm import tqdm
import librosa
import soundfile as sf
import torch
import whisper
from export_onnx import get_args, causal_mask_1d


class OnnxModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.init_encoder(encoder)
        self.init_decoder(decoder)

    def init_encoder(self, encoder: str):
        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.encoder_input_names = []
        self.encoder_output_names = []

        print(f"-----{encoder}-----")
        print(f"----input----")
        for i in self.encoder.get_inputs():
            print(i)
            self.encoder_input_names.append(i.name)

        print("-----output-----")

        for i in self.encoder.get_outputs():
            print(i)
            self.encoder_output_names.append(i.name)

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.n_text_layer = int(meta["n_text_layer"])
        self.n_text_ctx = int(meta["n_text_ctx"])
        self.n_text_state = int(meta["n_text_state"])
        self.n_mels = int(meta["n_mels"])
        self.eot = int(meta["eot"])
        self.no_timestamps = int(meta["no_timestamps"])
        self.sot_sequence = list(map(int, meta["sot_sequence"].split(",")))
        self.sot_sequence.append(self.no_timestamps)

        self.all_language_tokens = list(
            map(int, meta["all_language_tokens"].split(","))
        )
        self.all_language_codes = meta["all_language_codes"].split(",")
        self.lang2id = dict(zip(self.all_language_codes, self.all_language_tokens))
        self.id2lang = dict(zip(self.all_language_tokens, self.all_language_codes))

    def init_decoder(self, decoder: str):
        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.decoder_input_names = []
        self.decoder_output_names = []

        print(f"-----{decoder}-----")
        print(f"----input----")
        for i in self.decoder.get_inputs():
            print(i)
            self.decoder_input_names.append(i.name)

        print("-----output-----")

        for i in self.decoder.get_outputs():
            print(i)
            self.decoder_output_names.append(i.name)

    def run_encoder(
        self,
        mel: np.ndarray,
    ) -> List[np.ndarray]:
        cross_kv = self.encoder.run(
            self.encoder_output_names,
            {
                self.encoder.get_inputs()[0].name: mel,
            },
        )
        return cross_kv

    def run_decoder(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        feed = {
            self.decoder.get_inputs()[i].name: inputs[i] for i in range(len(inputs))
        }

        out = self.decoder.run(
            self.decoder_output_names,
            feed,
        )
        return out

    def get_self_cache(self) -> List[np.ndarray]:
        self_cache = []
        batch_size = 1
        for i in range(self.n_text_layer):
            k = np.zeros(
                (batch_size, self.n_text_ctx, self.n_text_state), dtype=np.float32
            )
            v = np.zeros(
                (batch_size, self.n_text_ctx, self.n_text_state), dtype=np.float32
            )
            self_cache.extend([k, v])
        return self_cache


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_feat(filename: str, n_mels: int):
    wave, sample_rate = load_audio(filename)
    if sample_rate != 16000:
        import librosa

        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    audio = whisper.pad_or_trim(wave)
    assert audio.shape == (16000 * 30,), audio.shape

    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).unsqueeze(0)
    assert mel.shape == (1, n_mels, 3000), mel.shape

    return mel


def forward(model_type: str, model: OnnxModel, sound_file: str, lang: str, tokenizer):
    name = os.path.splitext(os.path.basename(sound_file))[0]

    model.sot_sequence[1] = model.lang2id[lang]

    mel = compute_feat(sound_file, n_mels=model.n_mels).numpy()

    os.makedirs(f"calibrations_{model_type}/encoder/mel", exist_ok=True)
    np.save(f"./calibrations_{model_type}/encoder/mel/{name}.npy", mel)

    cross_kv = model.run_encoder(mel)
    
    self_kv = model.get_self_cache()

    offset = np.array([0], dtype=np.int32)
    for t in model.sot_sequence:
        token = np.array([[t]], dtype=np.int32)  # sot
        mask = causal_mask_1d(offset.item(), model.n_text_ctx).numpy()

        os.makedirs(f"calibrations_{model_type}/decoder/tokens", exist_ok=True)
        os.makedirs(f"calibrations_{model_type}/decoder/offset", exist_ok=True)
        os.makedirs(f"calibrations_{model_type}/decoder/mask", exist_ok=True)

        np.save(f"calibrations_{model_type}/decoder/tokens/{name}_{offset.item()}.npy", token)
        np.save(f"calibrations_{model_type}/decoder/offset/{name}_{offset.item()}.npy", offset)
        np.save(f"calibrations_{model_type}/decoder/mask/{name}_{offset.item()}.npy", mask)

        for i in range(0, len(self_kv), 2):
            os.makedirs(f"calibrations_{model_type}/decoder/self_k_{i // 2}", exist_ok=True)
            os.makedirs(f"calibrations_{model_type}/decoder/self_v_{i // 2}", exist_ok=True)

            np.save(f"calibrations_{model_type}/decoder/self_k_{i // 2}/{name}_{offset.item()}.npy", self_kv[i])
            np.save(f"calibrations_{model_type}/decoder/self_v_{i // 2}/{name}_{offset.item()}.npy", self_kv[i + 1])

        for i in range(0, len(cross_kv), 2):
            os.makedirs(f"calibrations_{model_type}/decoder/cross_k_{i // 2}", exist_ok=True)
            os.makedirs(f"calibrations_{model_type}/decoder/cross_v_{i // 2}", exist_ok=True)

            np.save(f"calibrations_{model_type}/decoder/cross_k_{i // 2}/{name}_{offset.item()}.npy", cross_kv[i])
            np.save(f"calibrations_{model_type}/decoder/cross_v_{i // 2}/{name}_{offset.item()}.npy", cross_kv[i + 1])

        out = model.run_decoder([token] + self_kv + cross_kv + [offset, mask])

        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]

        offset += 1

    idx = out[0][0, 0].argmax()

    ans = []

    while idx != model.eot and offset.item() < 200:
        ans.append(idx)
        token = np.array([[idx]], dtype=np.int32)  # no_timestamps
        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]

        mask = causal_mask_1d(offset.item(), model.n_text_ctx).numpy()

        np.save(f"calibrations_{model_type}/decoder/tokens/{name}_{offset.item()}.npy", token)
        np.save(f"calibrations_{model_type}/decoder/offset/{name}_{offset.item()}.npy", offset)
        np.save(f"calibrations_{model_type}/decoder/mask/{name}_{offset.item()}.npy", mask)

        for i in range(0, len(self_kv), 2):
            np.save(f"calibrations_{model_type}/decoder/self_k_{i // 2}/{name}_{offset.item()}.npy", self_kv[i])
            np.save(f"calibrations_{model_type}/decoder/self_v_{i // 2}/{name}_{offset.item()}.npy", self_kv[i + 1])

        for i in range(0, len(cross_kv), 2):
            np.save(f"calibrations_{model_type}/decoder/cross_k_{i // 2}/{name}_{offset.item()}.npy", cross_kv[i])
            np.save(f"calibrations_{model_type}/decoder/cross_v_{i // 2}/{name}_{offset.item()}.npy", cross_kv[i + 1])

        out = model.run_decoder([token] + self_kv + cross_kv + [offset, mask])
        idx = out[0][0, 0].argmax()

        offset += 1

    print(ans)
    text = "".join(tokenizer.decode(ans)).strip()
    print(text)


def main():
    args = get_args()
    print(args)
    model_type = args.model

    args = get_args()
    print(vars(args))

    torch_model = whisper.load_model(args.model)
    tokenizer = whisper.tokenizer.get_tokenizer(
        torch_model.is_multilingual, num_languages=torch_model.num_languages
    )

    model = OnnxModel(f"./{args.model}-encoder.onnx", f"./{args.model}-decoder.onnx")

    # [sot, lang, task, notimestamps]
    model.sot_sequence[1] = model.lang2id["en"]

    # tiny.en: [50257, 50362]
    # tiny: [50258, 50259, 50359, 50363]
    print("sot sequence", model.sot_sequence)
    print(f"model.n_text_layer: {model.n_text_layer}")

    dataset = {
        'en': ['example/en.mp3'],
        'ja': ['example/ja.mp3'],
        'ko': ['example/ko.mp3'],
        'zh': ['example/zh.mp3']
    }

    dataset_num = sum(len(v) for v in dataset.values())
    assert dataset_num > 0, 'dataset is empty'
    print(f"Generating data, total {len(dataset)}")
    gen_num = 0
    for lang in dataset.keys():
        for sound_path in dataset[lang]:
            forward(model_type, model, sound_path, lang, tokenizer)

            gen_num += 1

    tar_dirs = [f"calibrations_{model_type}/encoder/mel", 
                f"calibrations_{model_type}/decoder/tokens",
                f"calibrations_{model_type}/decoder/mask",
                f"calibrations_{model_type}/decoder/offset", 
    ]
    for i in range(model.n_text_layer):
        tar_dirs.append(f"calibrations_{model_type}/decoder/self_k_{i}")
        tar_dirs.append(f"calibrations_{model_type}/decoder/self_v_{i}")

        tar_dirs.append(f"calibrations_{model_type}/decoder/cross_k_{i}")
        tar_dirs.append(f"calibrations_{model_type}/decoder/cross_v_{i}")
    
    for td in tar_dirs:
        tar_filename = os.path.join(td, "..", os.path.basename(td) + ".tar.gz")
        tar = tarfile.open(tar_filename, "w:gz")
        for f in glob.glob(td + "/*.npy"):
            tar.add(f)
        tar.close()
        print(f"Save {tar_filename}")

    # save ax config
    import json
    ax_config = json.load(open("config_whisper_encoder_u16.json"))
    ax_config["quant"]["input_configs"] = []
    for inp in model.encoder.get_inputs():
        name = inp.name
        ax_config["quant"]["input_configs"].append(
            {
                "tensor_name": name,
                "calibration_dataset": f"./calibrations_{model_type}/encoder/{name}.tar.gz",
                "calibration_size": -1,
                "calibration_format": "Numpy"
            }
        )

    with open(f"config_whisper_{model_type}_encoder.json", "w") as f:
        json.dump(ax_config, f, indent=4)
    print(f"Dump config to config_whisper_{model_type}_encoder.json")

    ax_config = json.load(open("config_whisper_decoder_u16.json"))
    ax_config["quant"]["input_configs"] = []
    for inp in model.decoder.get_inputs():
        name = inp.name
        ax_config["quant"]["input_configs"].append(
            {
                "tensor_name": name,
                "calibration_dataset": f"./calibrations_{model_type}/decoder/{name}.tar.gz",
                "calibration_size": -1,
                "calibration_format": "Numpy"
            }
        )
        
    with open(f"config_whisper_{model_type}_decoder.json", "w") as f:
        json.dump(ax_config, f, indent=4)
    print(f"Dump config to config_whisper_{model_type}_decoder.json")
    

if __name__ == "__main__":
    main()

'''
Usage:
python3 ./generate_data.py --model tiny
'''