import axengine as axe
import numpy as np
import librosa
import os
from typing import Union
from whisper_tokenizer import *
import json
from dataclasses import dataclass
import zhconv


@dataclass
class WhisperConfig:
    n_mels: int = 0
    sample_rate: int = 0
    n_fft: int = 0
    hop_length: int = 0

    sot: int = 0
    eot: int = 0
    blank_id: int = 0
    no_timestamps: int = 0
    no_speech: int = 0
    translate: int = 0
    transcribe: int = 0
    n_vocab: int = 0
    n_text_ctx: int = 0
    n_text_state: int = 0

    sot_sequence: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0, 0], dtype=np.int32)
    )


class Whisper:
    def __init__(self, model_type: str, model_path: str, language: str, task: str):
        assert task in ["translate", "transcribe"]

        self.language = language
        self.task = task
        self.encoder, self.decoder, self.tokenizer, model_config = self.load_model(
            model_type, model_path, language, task
        )
        self.config = self.load_config(model_config)

    def load_model(self, model_type, model_path, language, task):
        encoder_path = f"{model_type}/{model_type}-encoder.axmodel"
        decoder_path = f"{model_type}/{model_type}-decoder.axmodel"
        model_config_file = f"{model_type}/{model_type}_config.json"
        token_file = f"{model_type}/{model_type}-tokens.txt"

        required_files = [
            os.path.join(model_path, i)
            for i in (encoder_path, decoder_path, model_config_file, token_file)
        ]
        # Check file existence
        for i, file_path in enumerate(required_files):
            assert os.path.exists(file_path), f"{file_path} NOT exist"

        # Load encoder
        encoder = axe.InferenceSession(
            required_files[0], providers=["AxEngineExecutionProvider"]
        )
        # Load decoder main
        decoder = axe.InferenceSession(
            required_files[1], providers=["AxEngineExecutionProvider"]
        )
        # Load tokens
        model_config = json.load(open(required_files[2], "r"))
        model_config["all_language_tokens"] = [
            int(i) for i in model_config["all_language_tokens"].split(",")
        ]
        model_config["all_language_codes"] = [
            i for i in model_config["all_language_codes"].split(",")
        ]
        tokenizer = get_tokenizer(
            model_config["is_multilingual"],
            num_languages=len(model_config["all_language_codes"]),
            language=language,
            task=task,
        )

        self.id2token = self.load_tokens(required_files[3])

        return encoder, decoder, tokenizer, model_config

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
        config.n_text_layer = model_config["n_text_layer"]

        lang_token = model_config["all_language_tokens"][
            model_config["all_language_codes"].index(self.language)
        ]
        task_token = (
            config.transcribe if self.task == "transcribe" else config.translate
        )
        config.sot_sequence = np.array(
            [config.sot, lang_token, task_token, config.no_timestamps], dtype=np.int32
        )

        return config

    def load_tokens(self, filename):
        tokens = dict()
        with open(filename, "r") as f:
            for line in f:
                t, i = line.split()
                tokens[int(i)] = t
        return tokens

    def load_audio(self, audio: str):
        data, sample_rate = librosa.load(audio, sr=self.config.sample_rate)
        samples = np.ascontiguousarray(data)
        return samples, sample_rate

    def compute_feature(self, audio: np.ndarray):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_mels=self.config.n_mels,
        )

        log_spec = np.log10(np.maximum(mel, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        mel = (log_spec + 4.0) / 4.0

        target = 3000
        if mel.shape[1] > target:
            # -50 so that there are some zero tail paddings.
            mel = mel[:, :target]
            mel[:, -50:] = 0

        # We don't need to pad it to 30 seconds now!
        if mel.shape[1] < target:
            mel = np.concatenate(
                (
                    mel,
                    np.zeros(
                        (self.config.n_mels, target - mel.shape[1]), dtype=np.float32
                    ),
                ),
                axis=-1,
            )

        return mel[np.newaxis, ...]

    def run_encoder(
        self,
        mel: np.ndarray,
    ) -> List[np.ndarray]:
        cross_kv = self.encoder.run(
            None,
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
            None,
            feed,
        )
        return out

    def get_self_cache(self) -> List[np.ndarray]:
        self_cache = []
        batch_size = 1
        for i in range(self.config.n_text_layer):
            k = np.zeros(
                (batch_size, self.config.n_text_ctx, self.config.n_text_state),
                dtype=np.float32,
            )
            v = np.zeros(
                (batch_size, self.config.n_text_ctx, self.config.n_text_state),
                dtype=np.float32,
            )
            self_cache.extend([k, v])
        return self_cache

    def causal_mask_1d(self, n: int, L: int):
        """
        Returns a 1-D int mask of shape (L,) with:
        0 -> allowed
        1 -> masked (will be converted to -inf later)
        """
        mask = np.ones((L,), dtype=np.int32)
        if n > 0:
            mask[:n] = 0
        return mask

    def run(self, audio: Union[str, np.ndarray]) -> str:
        if isinstance(audio, str):
            audio, sample_rate = self.load_audio(audio)

        mel = self.compute_feature(audio)

        cross_kv = self.run_encoder(mel)

        self_kv = self.get_self_cache()

        offset = np.array([0], dtype=np.int32)
        for t in self.config.sot_sequence:
            token = np.array([[t]], dtype=np.int32)  # sot
            mask = self.causal_mask_1d(offset.item(), self.config.n_text_ctx)

            out = self.run_decoder([token] + self_kv + cross_kv + [offset, mask])

            for i in range(1, len(out)):
                self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]

            offset += 1

        idx = out[0][0, 0].argmax()

        eot = self.config.eot

        ans = []

        while idx != eot and offset.item() < self.config.n_text_ctx:
            ans.append(idx)
            token = np.array([[idx]], dtype=np.int32)

            mask = self.causal_mask_1d(offset.item(), self.config.n_text_ctx)

            out = self.run_decoder([token] + self_kv + cross_kv + [offset, mask])

            for i in range(1, len(out)):
                self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]

            offset += 1
            idx = out[0][0, 0].argmax()

        # print(ans)

        s = b""
        for i in ans:
            if i in self.id2token:
                s += base64.b64decode(self.id2token[i])

        text = s.decode().strip()

        if self.language == "zh":
            try:
                sim_zh = zhconv.convert(text, "zh-hans")
                return sim_zh
            except:
                return text

        return text
