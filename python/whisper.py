import axengine as axe 
import numpy as np
import librosa
import os
from typing import Union
from whisper_tokenizer import *
import json
from dataclasses import dataclass
import zhconv


NEG_INF = float("-inf")

@dataclass
class WhisperConfig:
    n_mels          : int = 0
    sample_rate     : int = 0
    n_fft           : int = 0
    hop_length      : int = 0

    sot             : int = 0
    eot             : int = 0
    blank_id        : int = 0
    no_timestamps   : int = 0
    no_speech       : int = 0
    translate       : int = 0
    transcribe      : int = 0
    n_vocab         : int = 0
    n_text_ctx      : int = 0
    n_text_state    : int = 0

    sot_sequence    : np.ndarray = field(default_factory=lambda: np.array([0,0,0,0], dtype=np.int32))


class Whisper:
    def __init__(self, model_type: str, model_path: str, language: str, task: str):
        assert task in ["translate", "transcribe"]

        self.language = language
        self.task = task
        self.encoder, self.decoder_main, self.decoder_loop, self.pe, self.tokenizer, model_config = \
            self.load_model(model_type, model_path, language, task)
        self.config = self.load_config(model_config)


    def load_model(self, model_type, model_path, language, task):
        encoder_path = f"{model_type}/{model_type}-encoder.axmodel"
        decoder_main_path = f"{model_type}/{model_type}-decoder-main.axmodel"
        decoder_loop_path = f"{model_type}/{model_type}-decoder-loop.axmodel"
        pe_path = f"{model_type}/{model_type}-positional_embedding.bin"
        model_config_file = f"{model_type}/{model_type}_config.json"

        required_files = [os.path.join(model_path, i) for i in (encoder_path, decoder_main_path, decoder_loop_path, pe_path, model_config_file)]
        # Check file existence
        for i, file_path in enumerate(required_files):
            assert os.path.exists(file_path), f"{file_path} NOT exist"

        # Load encoder
        encoder = axe.InferenceSession(required_files[0], providers=['AxEngineExecutionProvider'])
        # Load decoder main
        decoder_main = axe.InferenceSession(required_files[1], providers=['AxEngineExecutionProvider'])
        # Load decoder loop
        decoder_loop = axe.InferenceSession(required_files[2], providers=['AxEngineExecutionProvider'])
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

        return encoder, decoder_main, decoder_loop, pe, tokenizer, model_config
    

    def load_config(self, model_config):
        config = WhisperConfig
        config.n_mels = model_config["n_mels"]
        config.sample_rate = 16000
        config.n_fft = 480
        config.hop_length = 160

        config.sot = model_config["sot"]
        config.eot = model_config["eot"]
        config.blank_id = model_config["blank_id"]
        config.no_timestamps = model_config["no_timestamps"]
        config.no_speech = model_config["no_speech"]
        config.translate = model_config["translate"]
        config.transcribe = model_config["transcribe"]
        config.n_vocab = model_config["n_vocab"]
        config.n_text_ctx = model_config["n_text_ctx"]
        config.n_text_state = model_config["n_text_state"]

        lang_token = model_config["all_language_tokens"][model_config["all_language_codes"].index(self.language)]
        task_token = config.transcribe if self.task == "transcribe" else config.translate
        config.sot_sequence = np.array([config.sot, lang_token, task_token, config.no_timestamps], dtype=np.int32)

        return config
    

    def load_audio(self, audio: str):
        data, sample_rate = librosa.load(audio, sr=self.config.sample_rate)
        samples = np.ascontiguousarray(data)
        return samples, sample_rate


    def compute_feature(self, audio: np.ndarray, padding = 480000):
        if padding > 0:
            audio = np.concatenate((audio, np.zeros((padding,), dtype=np.float32)), axis=-1)

        mel = librosa.feature.melspectrogram(y=audio, 
                                             sr=self.config.sample_rate, 
                                             n_fft=self.config.n_fft, 
                                             hop_length=self.config.hop_length, 
                                             window="hann", 
                                             center=True, 
                                             pad_mode="reflect", 
                                             power=2.0, 
                                             n_mels=self.config.n_mels)
        
        log_spec = np.log10(np.maximum(mel, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        mel = (log_spec + 4.0) / 4.0

        target = 3000
        if mel.shape[1] > target:
            # -50 so that there are some zero tail paddings.
            mel = mel[:, : target]
            mel[:, -50:] = 0

        # We don't need to pad it to 30 seconds now!
        if mel.shape[1] < target:
            mel = np.concatenate((mel, np.zeros((self.config.n_mels, target - mel.shape[1]), dtype=np.float32)), axis=-1)

        return mel
    

    def supress_tokens(self, logits, is_initial):
        if is_initial:
            logits[self.config.eot] = NEG_INF
            logits[self.config.blank_id] = NEG_INF

        logits[self.config.no_timestamps] = NEG_INF
        logits[self.config.sot] = NEG_INF
        logits[self.config.no_speech] = NEG_INF

        if self.task == "transcribe":
            logits[self.config.translate] = NEG_INF
        else:
            logits[self.config.transcribe] = NEG_INF
        return logits


    def run(self, audio: Union[str, np.ndarray]) -> str:
        if isinstance(audio, str):
            audio, sample_rate = self.load_audio(audio)

        mel = self.compute_feature(audio)

        # Run encoder
        x = self.encoder.run(None, input_feed={"mel": mel[None, ...]})
        n_layer_cross_k, n_layer_cross_v = x

        # Run decoder_main
        x = self.decoder_main.run(None, input_feed={
            "tokens": self.config.sot_sequence[None, ...],
            "n_layer_cross_k": n_layer_cross_k,
            "n_layer_cross_v": n_layer_cross_v
        })
        logits, n_layer_self_k_cache, n_layer_self_v_cache = x

        # Decode token
        logits = logits[0, -1, :]
        logits = self.supress_tokens(logits, is_initial=True)
        # logits.tofile("logits.bin")
        max_token_id = np.argmax(logits)
        output_tokens = []

        # Position embedding offset
        offset = self.config.sot_sequence.shape[0]

        # Autoregressively run decoder until token meets EOT
        for i in range(self.config.n_text_ctx - self.config.sot_sequence.shape[0]):
            if max_token_id >= self.config.eot:
                break

            output_tokens.append(max_token_id)

            mask = np.zeros((self.config.n_text_ctx,), dtype=np.float32)
            mask[: self.config.n_text_ctx - offset - 1] = NEG_INF

            # Run decoder_loop
            x = self.decoder_loop.run(None, input_feed={
                "tokens": np.array([[output_tokens[-1]]], dtype=np.int32),
                "in_n_layer_self_k_cache": n_layer_self_k_cache,
                "in_n_layer_self_v_cache": n_layer_self_v_cache,
                "n_layer_cross_k": n_layer_cross_k,
                "n_layer_cross_v": n_layer_cross_v,
                "positional_embedding": self.pe[offset * self.config.n_text_state : (offset + 1) * self.config.n_text_state][None, ...],
                "mask": mask
            })
            logits, n_layer_self_k_cache, n_layer_self_v_cache = x

            # Decode token
            offset += 1
            logits = self.supress_tokens(logits.flatten(), is_initial=False)
            max_token_id = np.argmax(logits)
        
        text = self.tokenizer.decode(output_tokens)

        if self.language == "zh":
            try:
                sim_zh = zhconv.convert(text, 'zh-hans')
                return sim_zh
            except:
                return text
            
        return text

        
