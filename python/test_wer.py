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
import logging
import re
import torch
import kaldi_native_fbank as knf
from languages import WHISPER_LANGUAGES


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
SOT_SEQUENCE = np.array([WHISPER_SOT,WHISPER_SOT + 1 + tuple(WHISPER_LANGUAGES).index("zh"),WHISPER_TRANSCRIBE,WHISPER_NO_TIMESTAMPS], dtype=np.int32)
WHISPER_N_TEXT_STATE_MAP = {
    "tiny": 384,
    "base": 512,
    "small": 768,
    "large": 1280,
    "large-v3": 1280,
    "turbo": 1280
}

def setup_logging():
    """配置日志系统，同时输出到控制台和文件"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "test_wer.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    
    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class AIShellDataset:
    def __init__(self, gt_path: str):
        """
        初始化数据集
        
        Args:
            json_path: voice.json文件的路径
        """
        self.gt_path = gt_path
        self.dataset_dir = os.path.dirname(gt_path)
        self.voice_dir = os.path.join(self.dataset_dir, "aishell_S0764")
        
        # 检查必要文件和文件夹是否存在
        assert os.path.exists(gt_path), f"gt文件不存在: {gt_path}"
        assert os.path.exists(self.voice_dir), f"aishell_S0764文件夹不存在: {self.voice_dir}"
        
        # 加载数据
        self.data = []
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                audio_path, gt = line.split(" ")
                audio_path = os.path.join(self.voice_dir, audio_path + ".wav")
                self.data.append({"audio_path": audio_path, "gt": gt})

        # 使用logging而不是print
        logger = logging.getLogger()
        logger.info(f"加载了 {len(self.data)} 条数据")
    
    def __iter__(self):
        """返回迭代器"""
        self.index = 0
        return self
    
    def __next__(self):
        """返回下一个数据项"""
        if self.index >= len(self.data):
            raise StopIteration
        
        item = self.data[self.index]
        audio_path = item["audio_path"]
        ground_truth = item["gt"]
        
        self.index += 1
        return audio_path, ground_truth
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    

class CommonVoiceDataset:
    """Common Voice数据集解析器"""
    
    def __init__(self, tsv_path: str):
        """
        初始化数据集
        
        Args:
            json_path: voice.json文件的路径
        """
        self.tsv_path = tsv_path
        self.dataset_dir = os.path.dirname(tsv_path)
        self.voice_dir = os.path.join(self.dataset_dir, "clips")
        
        # 检查必要文件和文件夹是否存在
        assert os.path.exists(tsv_path), f"{tsv_path}文件不存在: {tsv_path}"
        assert os.path.exists(self.voice_dir), f"voice文件夹不存在: {self.voice_dir}"
        
        # 加载JSON数据
        self.data = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                line = line.strip()
                splits = line.split("\t")
                audio_path = splits[1]
                gt = splits[2]
                audio_path = os.path.join(self.voice_dir, audio_path)
                self.data.append({"audio_path": audio_path, "gt": gt})
        
        # 使用logging而不是print
        logger = logging.getLogger()
        logger.info(f"加载了 {len(self.data)} 条数据")
    
    def __iter__(self):
        """返回迭代器"""
        self.index = 0
        return self
    
    def __next__(self):
        """返回下一个数据项"""
        if self.index >= len(self.data):
            raise StopIteration
        
        item = self.data[self.index]
        audio_path = item["audio_path"]
        ground_truth = item["gt"]
        
        self.index += 1
        return audio_path, ground_truth
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

def get_args():
    parser = argparse.ArgumentParser(
        prog="whisper",
        description="Test WER on dataset"
    )
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["aishell", "common_voice"], help="Test dataset")
    parser.add_argument("--gt_path", "-g", type=str, required=True, help="Test dataset ground truth file")
    parser.add_argument("--max_num", type=int, default=-1, required=False, help="Maximum test data num")
    parser.add_argument("--model_type", "-t", type=str, choices=["tiny", "base", "small", "large", "large-v3", "turbo"], required=True, help="model type, only support tiny, base and small currently")
    parser.add_argument("--model_path", "-p", type=str, required=False, default="../models", help="model path for *.axmodel, tokens.txt, positional_embedding.bin")
    parser.add_argument("--language", "-l", type=str, required=False, default="zh", help="Target language, support en, zh, ja, and others. See languages.py for more options.")
    return parser.parse_args()


def print_args(args):
    logger = logging.getLogger()
    logger.info(f"dataset: {args.dataset}")
    logger.info(f"gt_path: {args.gt_path}")
    logger.info(f"max_num: {args.max_num}")
    logger.info(f"model_type: {args.model_type}")
    logger.info(f"model_path: {args.model_path}")
    logger.info(f"language: {args.language}")


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


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = librosa.load(filename, sr=WHISPER_SAMPLE_RATE)
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
    encoder = axe.InferenceSession(required_files[0])
    # Load decoder main
    decoder_main = axe.InferenceSession(required_files[1])
    # Load decoder loop
    decoder_loop = axe.InferenceSession(required_files[2])
    # Load position embedding
    pe = np.fromfile(required_files[3], dtype=np.float32)
    # Load tokens
    tokens = []
    with open(required_files[4], "r") as f:
        for line in f:
            line = line.strip()
            tokens.append(line.split(" ")[0])

    return encoder, decoder_main, decoder_loop, pe, tokens


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


def choose_language(lang):
    if lang not in WHISPER_LANGUAGES.keys():
        raise Exception(f"Unknown language: {lang}. Check languages.py for correct options.")
    SOT_SEQUENCE[1] = WHISPER_SOT + 1 + tuple(WHISPER_LANGUAGES.keys()).index(lang)


def main():
    # 设置日志系统
    logger = setup_logging()

    args = get_args()
    print_args(args)

    dataset_type = args.dataset.lower()
    if dataset_type == "aishell":
        dataset = AIShellDataset(args.gt_path)
    elif dataset_type == "common_voice":
        dataset = CommonVoiceDataset(args.gt_path)
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    max_num = args.max_num

    # Choose language
    choose_language(args.language)

    # Load models and other stuff
    start = time.time()
    encoder, decoder_main, decoder_loop, pe, token_table = load_models(args.model_path, args.model_type)
    logger.info(f"Load models take {(time.time() - start) * 1000}ms")
    WHISPER_N_TEXT_STATE = WHISPER_N_TEXT_STATE_MAP[args.model_type]


    # Iterate over dataset
    references = []
    hyp = []
    all_character_error_num = 0
    all_character_num = 0
    wer_file = open("wer.txt", "w")
    max_data_num = max_num if max_num > 0 else len(dataset)
    for n, (audio_path, reference) in enumerate(dataset):
        # Preprocess
        if "large" in args.model_type or "turbo" in args.model_type:
            n_mels = 128
        else:
            n_mels = 80
        mel = compute_feature(audio_path, n_mels=n_mels)

        # Run encoder
        x = encoder.run(None, input_feed={"mel": mel[None, ...]})
        n_layer_cross_k, n_layer_cross_v = x

        # Run decoder_main
        x = decoder_main.run(None, input_feed={
            "tokens": SOT_SEQUENCE[None, ...],
            "n_layer_cross_k": n_layer_cross_k,
            "n_layer_cross_v": n_layer_cross_v
        })
        logits, n_layer_self_k_cache, n_layer_self_v_cache = x

        # Decode token
        logits = logits[0, -1, :]
        logits = supress_tokens(logits, is_initial=True)
        # logits.tofile("logits.bin")
        max_token_id = np.argmax(logits)
        output_tokens = []

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

            # Decode token
            offset += 1
            logits = supress_tokens(logits.flatten(), is_initial=False)
            max_token_id = np.argmax(logits)
        
        s = b""
        for i in output_tokens:
            s += base64.b64decode(token_table[i])

        hypothesis = s.decode('utf-8', errors='ignore').strip()

        if args.language == "zh":
            try:
                hypothesis = zhconv.convert(hypothesis, 'zh-hans')
            except:
                hypothesis = ""

        hypothesis = re.sub(r'[^\w\s]', '', hypothesis)
        character_error_num = min_distance(reference, hypothesis)
        character_num = len(reference)
        character_error_rate = character_error_num / character_num * 100

        all_character_error_num += character_error_num
        all_character_num += character_num

        hyp.append(hypothesis)
        references.append(reference)
        
        line_content = f"({n+1}/{max_data_num}) {os.path.basename(audio_path)}  reference: {reference}  hypothesis: {hypothesis}  WER: {character_error_rate}%"
        wer_file.write(line_content + "\n")
        logger.info(line_content)

        if n + 1 >= max_data_num:
            break

    total_character_error_rate = all_character_error_num / all_character_num * 100

    logger.info(f"Total WER: {total_character_error_rate}%")
    wer_file.write(f"Total WER: {total_character_error_rate}%")
    wer_file.close()

if __name__ == "__main__":
    main()
