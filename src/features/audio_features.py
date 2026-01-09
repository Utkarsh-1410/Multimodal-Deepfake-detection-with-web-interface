from __future__ import annotations

import numpy as np
import soundfile as sf
from python_speech_features import mfcc
from typing import Optional


def load_audio_mono(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # basic resample if needed (nearest neighbor for speed)
    if sr != target_sr:
        ratio = target_sr / sr
        idx = (np.arange(int(len(audio) * ratio)) / ratio).astype(np.int64)
        idx = np.clip(idx, 0, len(audio) - 1)
        audio = audio[idx]
        sr = target_sr
    return audio.astype(np.float32), sr


def compute_mfcc(audio: np.ndarray, sr: int, winlen: float = 0.025, winstep: float = 0.01, numcep: int = 13) -> np.ndarray:
    return mfcc(audio, samplerate=sr, winlen=winlen, winstep=winstep, numcep=numcep)


