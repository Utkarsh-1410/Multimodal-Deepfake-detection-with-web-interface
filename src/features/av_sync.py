from __future__ import annotations

import numpy as np


def simple_av_sync_score(mouth_open_series: np.ndarray, audio_energy_series: np.ndarray) -> float:
    n = min(len(mouth_open_series), len(audio_energy_series))
    if n == 0:
        return 0.0
    a = (mouth_open_series[:n] - mouth_open_series[:n].mean())
    b = (audio_energy_series[:n] - audio_energy_series[:n].mean())
    denom = (a.std() + 1e-6) * (b.std() + 1e-6)
    return float(np.clip((a * b).mean() / denom, -1.0, 1.0))


