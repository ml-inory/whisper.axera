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
from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import tarfile
import glob
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        # fmt: off
        choices=[
            "tiny", "tiny.en", "base", "base.en",
            "small", "small.en", "medium", "medium.en",
            "large", "large-v1", "large-v2", "large-v3",
            "distil-medium.en", "distil-small.en", "distil-large-v2",
            # "distil-large-v3", # distil-large-v3 is not supported!
            # for fine-tuned models from icefall
            "medium-aishell", "turbo"
            ],
        # fmt: on
    )

    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="""The actual spoken language in the audio.
        Example values, en, de, zh, jp, fr.
        If None, we will detect the language using the first 30s of the
        input audio
        """,
    )

    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        type=str,
        default="transcribe",
        help="Valid values are: transcribe, translate",
    )

    parser.add_argument(
        "--gt_file",
        type=str,
        default="./datasets/ground_truth.txt",
        help="Path to the test wave",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./datasets/aishell_S0764",
        help="Path to the test wave",
    )

    parser.add_argument(
        "--max_num",
        type=int,
        default=-1,
        help="Maximum num of data",
    )

    parser.add_argument(
        "--save_report",
        action="store_true"
    )
    return parser.parse_args()


class OnnxModel:
    def __init__(
        self,
        encoder: str,
        decoder_dynamic: str,
        decoder_static: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.init_encoder(encoder)
        self.init_decoder(decoder_dynamic, dynamic=True)
        self.init_decoder(decoder_static, dynamic=False)

    def init_encoder(self, encoder: str):
        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.n_text_layer = int(meta["n_text_layer"])
        self.n_text_ctx = int(meta["n_text_ctx"])
        self.n_text_state = int(meta["n_text_state"])
        self.n_mels = int(meta["n_mels"])
        self.sot = int(meta["sot"])
        self.eot = int(meta["eot"])
        self.translate = int(meta["translate"])
        self.transcribe = int(meta["transcribe"])
        self.no_timestamps = int(meta["no_timestamps"])
        self.no_speech = int(meta["no_speech"])
        self.blank = int(meta["blank_id"])

        self.sot_sequence = list(map(int, meta["sot_sequence"].split(",")))
        self.sot_sequence.append(self.no_timestamps)

        self.all_language_tokens = list(
            map(int, meta["all_language_tokens"].split(","))
        )
        self.all_language_codes = meta["all_language_codes"].split(",")
        self.lang2id = dict(zip(self.all_language_codes, self.all_language_tokens))
        self.id2lang = dict(zip(self.all_language_tokens, self.all_language_codes))

        self.is_multilingual = int(meta["is_multilingual"]) == 1

    def init_decoder(self, decoder: str, dynamic: bool):
        if dynamic:
            self.decoder_dynamic = ort.InferenceSession(
                decoder,
                sess_options=self.session_opts,
                providers=["CPUExecutionProvider"],
            )
        else:
            self.decoder_static = ort.InferenceSession(
                decoder,
                sess_options=self.session_opts,
                providers=["CPUExecutionProvider"],
            )

    def run_encoder(
        self,
        mel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_layer_cross_k, n_layer_cross_v = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
            ],
            {
                self.encoder.get_inputs()[0].name: mel.numpy(),
            },
        )
        return torch.from_numpy(n_layer_cross_k), torch.from_numpy(n_layer_cross_v)

    def run_decoder(
        self,
        tokens: torch.Tensor,
        n_layer_self_k_cache: torch.Tensor,
        n_layer_self_v_cache: torch.Tensor,
        n_layer_cross_k: torch.Tensor,
        n_layer_cross_v: torch.Tensor,
        positional_embedding: torch.Tensor,
        mask: torch.Tensor,
        dynamic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dynamic:
            decoder = self.decoder_dynamic
            logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache = decoder.run(
                [
                    decoder.get_outputs()[0].name,
                    decoder.get_outputs()[1].name,
                    decoder.get_outputs()[2].name,
                ],
                {
                    decoder.get_inputs()[0].name: tokens.numpy(),
                    decoder.get_inputs()[1].name: n_layer_cross_k.numpy(),
                    decoder.get_inputs()[2].name: n_layer_cross_v.numpy(),
                },
            )
            return (
                torch.from_numpy(logits),
                torch.from_numpy(out_n_layer_self_k_cache),
                torch.from_numpy(out_n_layer_self_v_cache),
            )
        else:
            decoder = self.decoder_static
            logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache = decoder.run(
                [
                    decoder.get_outputs()[0].name,
                    decoder.get_outputs()[1].name,
                    decoder.get_outputs()[2].name,
                ],
                {
                    decoder.get_inputs()[0].name: tokens.numpy(),
                    decoder.get_inputs()[1].name: n_layer_self_k_cache.numpy(),
                    decoder.get_inputs()[2].name: n_layer_self_v_cache.numpy(),
                    decoder.get_inputs()[3].name: n_layer_cross_k.numpy(),
                    decoder.get_inputs()[4].name: n_layer_cross_v.numpy(),
                    decoder.get_inputs()[5].name: positional_embedding.numpy(),
                    decoder.get_inputs()[6].name: mask.numpy(),
                },
            )
            return (
                torch.from_numpy(logits),
                torch.from_numpy(out_n_layer_self_k_cache),
                torch.from_numpy(out_n_layer_self_v_cache),
            )

    def get_self_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = 1
        n_layer_self_k_cache = torch.zeros(
            self.n_text_layer,
            batch_size,
            self.n_text_ctx,
            self.n_text_state,
        )
        n_layer_self_v_cache = torch.zeros(
            self.n_text_layer,
            batch_size,
            self.n_text_ctx,
            self.n_text_state,
        )
        return n_layer_self_k_cache, n_layer_self_v_cache

    def suppress_tokens(self, logits, is_initial: bool) -> None:
        # suppress blank
        if is_initial:
            logits[self.eot] = float("-inf")
            logits[self.blank] = float("-inf")

        # suppress <|notimestamps|>
        logits[self.no_timestamps] = float("-inf")

        logits[self.sot] = float("-inf")
        logits[self.no_speech] = float("-inf")

        # logits is changed in-place
        logits[self.translate] = float("-inf")

    def detect_language(
        self, n_layer_cross_k: torch.Tensor, n_layer_cross_v: torch.Tensor
    ) -> int:
        tokens = torch.tensor([[self.sot]], dtype=torch.int64)
        offset = torch.zeros(1, dtype=torch.int64)
        n_layer_self_k_cache, n_layer_self_v_cache = self.get_self_cache()

        logits, n_layer_self_k_cache, n_layer_self_v_cache = self.run_decoder(
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            offset=offset,
        )
        logits = logits.reshape(-1)
        mask = torch.ones(logits.shape[0], dtype=torch.int64)
        mask[self.all_language_tokens] = 0
        logits[mask != 0] = float("-inf")
        lang_id = logits.argmax().item()
        print("detected language: ", self.id2lang[lang_id])
        return lang_id


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


def compute_features(filename: str, dim: int = 80) -> torch.Tensor:
    """
    Args:
      filename:
        Path to an audio file.
    Returns:
      Return a 1-D float32 tensor of shape (1, 80, 3000) containing the features.
    """
    wave, sample_rate = load_audio(filename)
    # if sample_rate != 16000:
    #     import librosa

    #     wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000)
    #     sample_rate = 16000

    features = []
    opts = knf.WhisperFeatureOptions()
    opts.dim = dim
    online_whisper_fbank = knf.OnlineWhisperFbank(opts)
    online_whisper_fbank.accept_waveform(16000, wave)
    online_whisper_fbank.input_finished()
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        f = torch.from_numpy(f)
        features.append(f)

    features = torch.stack(features)

    log_spec = torch.clamp(features, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    # mel (T, 80)

    # We pad 1500 frames at the end so that it is able to detect eot
    # You can use another value instead of 1500.
    mel = torch.nn.functional.pad(mel, (0, 0, 0, 1500), "constant", 0)
    # Note that if it throws for a multilingual model,
    # please use a larger value, say 300

    target = 3000
    if mel.shape[0] > target:
        # -50 so that there are some zero tail paddings.
        mel = mel[: target - 50]
        mel = torch.nn.functional.pad(mel, (0, 0, 0, 50), "constant", 0)

    # We don't need to pad it to 30 seconds now!
    mel = torch.nn.functional.pad(mel, (0, 0, 0, target - mel.shape[0]), "constant", 0)

    mel = mel.t().unsqueeze(0)

    return mel

def forward(model_type: str, model: OnnxModel, sound_file: str, positional_embedding: np.ndarray, save_data: bool = False):
    name = sound_file.split("/")[-1][:-4]

    n_mels = model.n_mels
    mel = compute_features(sound_file, dim=n_mels)
    if save_data:
        os.makedirs(f"calibrations_{model_type}/encoder/mel", exist_ok=True)
        np.save(f"./calibrations_{model_type}/encoder/mel/{name}.npy", mel)

    n_layer_cross_k, n_layer_cross_v = model.run_encoder(mel)
    
    n_layer_self_k_cache, n_layer_self_v_cache = model.get_self_cache()

    tokens = torch.tensor([model.sot_sequence], dtype=torch.int64)
    offset = torch.zeros(1, dtype=torch.int64)

    if save_data:
        os.makedirs(f"calibrations_{model_type}/decoder_main/tokens", exist_ok=True)
        os.makedirs(f"calibrations_{model_type}/decoder_main/n_layer_cross_k", exist_ok=True)
        os.makedirs(f"calibrations_{model_type}/decoder_main/n_layer_cross_v", exist_ok=True)

        np.save(f"./calibrations_{model_type}/decoder_main/tokens/{name}.npy", tokens)
        np.save(f"./calibrations_{model_type}/decoder_main/n_layer_cross_k/{name}.npy", n_layer_cross_k)
        np.save(f"./calibrations_{model_type}/decoder_main/n_layer_cross_v/{name}.npy", n_layer_cross_v)

    logits, n_layer_self_k_cache, n_layer_self_v_cache = model.run_decoder(
        tokens=tokens,
        n_layer_self_k_cache=n_layer_self_k_cache,
        n_layer_self_v_cache=n_layer_self_v_cache,
        n_layer_cross_k=n_layer_cross_k, # torch.from_numpy(np.fromfile("n_layer_cross_k.bin", dtype=np.float32).reshape(n_layer_cross_k.shape)),
        n_layer_cross_v=n_layer_cross_v, # torch.from_numpy(np.fromfile("n_layer_cross_v.bin", dtype=np.float32).reshape(n_layer_cross_v.shape)),
        positional_embedding=None,
        mask=None,
        dynamic=True,
    )
    offset += len(model.sot_sequence)
    # logits.shape (batch_size, tokens.shape[1], vocab_size)
    logits = logits[0, -1]
    model.suppress_tokens(logits, is_initial=True)
    #  logits = logits.softmax(dim=-1)
    # for greedy search, we don't need to compute softmax or log_softmax
    max_token_id = logits.argmax(dim=-1)
    results = []
    is_broke = False
    for i in range(model.n_text_ctx - len(model.sot_sequence)):
        if max_token_id == model.eot:
            is_broke = True
            break
        results.append(max_token_id.item())
        tokens = torch.tensor([[results[-1]]])
        mask = torch.zeros([model.n_text_ctx]).to(tokens.device)
        mask[:model.n_text_ctx - offset[0] - 1] = -torch.inf

        if save_data:
            os.makedirs(f"calibrations_{model_type}/decoder_loop/tokens", exist_ok=True)
            os.makedirs(f"calibrations_{model_type}/decoder_loop/n_layer_self_k_cache", exist_ok=True)
            os.makedirs(f"calibrations_{model_type}/decoder_loop/n_layer_self_v_cache", exist_ok=True)
            os.makedirs(f"calibrations_{model_type}/decoder_loop/n_layer_cross_k", exist_ok=True)
            os.makedirs(f"calibrations_{model_type}/decoder_loop/n_layer_cross_v", exist_ok=True)
            os.makedirs(f"calibrations_{model_type}/decoder_loop/positional_embedding", exist_ok=True)
            os.makedirs(f"calibrations_{model_type}/decoder_loop/mask", exist_ok=True)

            np.save(f"./calibrations_{model_type}/decoder_loop/tokens/{name}_{i}.npy", tokens)
            np.save(f"./calibrations_{model_type}/decoder_loop/n_layer_self_k_cache/{name}_{i}.npy", n_layer_self_k_cache)
            np.save(f"./calibrations_{model_type}/decoder_loop/n_layer_self_v_cache/{name}_{i}.npy", n_layer_self_v_cache)
            np.save(f"./calibrations_{model_type}/decoder_loop/n_layer_cross_k/{name}_{i}.npy", n_layer_cross_k)
            np.save(f"./calibrations_{model_type}/decoder_loop/n_layer_cross_v/{name}_{i}.npy", n_layer_cross_v)
            np.save(f"./calibrations_{model_type}/decoder_loop/positional_embedding/{name}_{i}.npy", positional_embedding[offset[0] : offset[0] + tokens.shape[-1]])
            np.save(f"./calibrations_{model_type}/decoder_loop/mask/{name}_{i}.npy", mask)

        logits, n_layer_self_k_cache, n_layer_self_v_cache = model.run_decoder(
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            positional_embedding=positional_embedding[offset[0] : offset[0] + tokens.shape[-1]],
            mask=mask,
            dynamic=False,
        )
        offset += 1
        logits = logits[0, -1]
        model.suppress_tokens(logits, is_initial=False)
        max_token_id = logits.argmax(dim=-1)
        # print("token: ", results[-1])
    if not is_broke:
        results = []
    return results


def min_distance(word1: str, word2: str) -> int:
 
    row = len(word1) + 1
    column = len(word2) + 1
 
    cache = [ [0]*column for i in range(row) ]
 
    for i in range(row):
        for j in range(column):
 
            if i ==0 and j ==0:
                cache[i][j] = 0
            elif i == 0 and j!=0:
                cache[i][j] = j
            elif j == 0 and i!=0:
                cache[i][j] = i
            else:
                if word1[i-1] == word2[j-1]:
                    cache[i][j] = cache[i-1][j-1]
                else:
                    replace = cache[i-1][j-1] + 1
                    insert = cache[i][j-1] + 1
                    remove = cache[i-1][j] + 1
 
                    cache[i][j] = min(replace, insert, remove)
 
    return cache[row-1][column-1]


def main():
    args = get_args()
    model_type = args.model
    max_num = args.max_num
    encoder_filename = f"{model_type}/{model_type}-encoder.onnx"
    decoder_dynamic_filename = f"{model_type}/{model_type}-decoder-main.onnx"
    decoder_static_filename = f"{model_type}/{model_type}-decoder-loop.onnx"
    pe_file = f"{model_type}/{model_type}-positional-embedding.npy"
    token_file = f"{model_type}/{model_type}-tokens.txt"
    model = OnnxModel(encoder_filename, decoder_dynamic_filename, decoder_static_filename)

    if args.language is not None:
        if model.is_multilingual is False and args.language != "en":
            print(f"This model supports only English. Given: {args.language}")
            return

        if args.language not in model.lang2id:
            print(f"Invalid language: {args.language}")
            print(f"Valid values are: {list(model.lang2id.keys())}")
            return

        # [sot, lang, task, notimestamps]
        model.sot_sequence[1] = model.lang2id[args.language]
    elif model.is_multilingual is True:
        assert False
        print("detecting language")
        lang = model.detect_language(n_layer_cross_k, n_layer_cross_v)
        model.sot_sequence[1] = lang

    if args.task is not None:
        if model.is_multilingual is False and args.task != "transcribe":
            print("This model supports only English. Please use --task=transcribe")
            return
        assert args.task in ["transcribe", "translate"], args.task

        if args.task == "translate":
            model.sot_sequence[2] = model.translate

    positional_embedding = torch.from_numpy(np.load(pe_file))
    token_table = load_tokens(token_file)

    dataset = []
    all_character_num = 0
    all_character_error_num = 0
    with open(args.gt_file, "r") as f:
        for i, line in enumerate(f):
            if max_num >= 0 and i >= max_num:
                break
            line = line.strip()
            name, gt = line.split(' ')
            character_num = len(gt)
            dataset.append({
                "name": name,
                "gt": gt,
                "character_num": character_num,
            })
            all_character_num += character_num
    random.seed(0)
    random.shuffle(dataset)

    print(f"Generating data, total {len(dataset)}")
    for idx, data in tqdm(enumerate(dataset)):
        if max_num <= 0:
            save_data = True
        else:
            save_data = idx < max_num
        sound_file = args.dataset_path + "/" + data["name"] + ".wav"
        print(sound_file)
        results = forward(model_type, model, sound_file, positional_embedding, save_data)
    
        s = b""
        for i in results:
            if i in token_table:
                s += base64.b64decode(token_table[i])
        # print(s.decode().strip())
        pd = zhconv.convert(s.decode().strip(), 'zh-hans')
        pd = re.sub(r'[^\w\s]', '', pd)
        character_error_num = min_distance(data["gt"], pd)
        character_error_rate = character_error_num / data["character_num"] * 100
        data.update({
            "pd": pd,
            "character_error_num": character_error_num,
            "character_error_rate": character_error_rate,
        })
        all_character_error_num += character_error_num

        print(f"{idx}: ")
        print(data["gt"])
        print(data["pd"])
        print(f"CER: {character_error_rate}%")

    total_character_error_rate = all_character_error_num / all_character_num * 100
    print(f"\ntotal CER: {total_character_error_rate}%")

    tar_dirs = [f"calibrations_{model_type}/encoder/mel", f"calibrations_{model_type}/decoder_main/tokens", f"calibrations_{model_type}/decoder_main/n_layer_cross_k",
                f"calibrations_{model_type}/decoder_main/n_layer_cross_v", f"calibrations_{model_type}/decoder_loop/tokens", f"calibrations_{model_type}/decoder_loop/n_layer_self_k_cache",
                f"calibrations_{model_type}/decoder_loop/n_layer_self_v_cache", f"calibrations_{model_type}/decoder_loop/n_layer_cross_k", f"calibrations_{model_type}/decoder_loop/n_layer_cross_v",
                f"calibrations_{model_type}/decoder_loop/positional_embedding", f"calibrations_{model_type}/decoder_loop/mask"]
    for td in tar_dirs:
        tar_filename = os.path.join(td, "..", os.path.basename(td) + ".tar.gz")
        tar = tarfile.open(tar_filename, "w:gz")
        for f in glob.glob(td + "/*.npy"):
            tar.add(f)
        tar.close()
        print(f"Save {tar_filename}")
    
    if args.save_report:
        with open('out_report.tsv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter='\t')
            row = ["name", "gt", "pd", "character_num", "character_error_num", "character_error_rate(%)"]
            writer.writerow(row)
            for data in dataset:
                row = [
                    data["name"],
                    data["gt"],
                    data["pd"],
                    data["character_num"],
                    data["character_error_num"],
                    data["character_error_rate"],
                ]
                writer.writerow(row)
            writer.writerow([])
            writer.writerow(["total_character_error_rate(%): ", total_character_error_rate])


if __name__ == "__main__":
    main()
'''
python3 ./test_ax_loop_dataset.py --model tiny
python3 ./test_ax_loop_dataset.py --model small
'''