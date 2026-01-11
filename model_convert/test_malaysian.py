import torch
from transformers.models.whisper import tokenization_whisper

tokenization_whisper.TASK_IDS = ["translate", "transcribe", 'transcribeprecise']

from transformers import (
    WhisperFeatureExtractor, 
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    WhisperTokenizerFast
)
import soundfile as sf
import numpy as np
from typing import Tuple
import whisper
import base64


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


sr = 16000
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    'mesolitica/Malaysian-whisper-large-v3-turbo-v3'
)
processor = WhisperProcessor.from_pretrained(
    'mesolitica/Malaysian-whisper-large-v3-turbo-v3'
)
tokenizer = WhisperTokenizerFast.from_pretrained(
    'mesolitica/Malaysian-whisper-large-v3-turbo-v3'
)
model = WhisperForConditionalGeneration.from_pretrained(
    'mesolitica/Malaysian-whisper-large-v3-turbo-v3', 
    dtype = torch.float32,
).cpu()

assembly = load_audio('./assembly.mp3')
assembly = assembly[: 16000 * 30]

print(f"n_mels: {model.config.num_mel_bins}")
print(f"new token <|transcribeprecise|> is {tokenizer.convert_tokens_to_ids('<|transcribeprecise|>')}")
print(f"base64 of <|transcribeprecise|> is {base64.b64encode(b'<|transcribeprecise|>')}")

with torch.no_grad():
    # p = processor([assembly], return_tensors='pt')
    # p['input_features'] = p['input_features'].to(torch.float32)

    feature = compute_feat('./assembly.mp3', model.config.num_mel_bins)
    r = model.generate(
        feature,
        output_scores=True,
        return_dict_in_generate=True,
        return_timestamps=True, 
        task = 'transcribeprecise',
    )
    
print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(r['sequences'][0])))
