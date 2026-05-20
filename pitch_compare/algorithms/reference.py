"""Wrappers em torno das implementações de pitch detection do librosa."""
from __future__ import annotations
import time
import numpy as np
import librosa


def yin_librosa(audio: np.ndarray, sr: int, *, fmin: float = 65.0, fmax: float = 1000.0,
                frame_length: int = 2048, hop_length: int = 512, **_) -> dict:
    t0 = time.perf_counter()
    f0 = librosa.yin(audio.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr,
                     frame_length=frame_length, hop_length=hop_length)
    elapsed = time.perf_counter() - t0
    voiced = (~np.isnan(f0)) & (f0 > 0)
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return {"f0": f0.astype(np.float64), "voiced": voiced, "times": times,
            "elapsed_s": elapsed}


def pyin_librosa(audio: np.ndarray, sr: int, *, fmin: float = 65.0, fmax: float = 1000.0,
                 frame_length: int = 2048, hop_length: int = 512, **_) -> dict:
    t0 = time.perf_counter()
    f0, voiced_flag, voicing_prob = librosa.pyin(
        audio.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
    )
    elapsed = time.perf_counter() - t0
    voiced = voiced_flag.astype(bool)
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    f0_clean = np.where(voiced, f0, np.nan).astype(np.float64)
    return {"f0": f0_clean, "voiced": voiced, "voicing_prob": voicing_prob,
            "times": times, "elapsed_s": elapsed}


def autocorr_librosa(audio: np.ndarray, sr: int, *, fmin: float = 65.0, fmax: float = 1000.0,
                     frame_length: int = 2048, hop_length: int = 512,
                     voicing_threshold: float = 0.3, **_) -> dict:
    """Autocorrelação via librosa.autocorrelate, frame-a-frame.

    librosa não tem detector de pitch baseado em autocorrelação puro;
    este wrapper usa librosa.autocorrelate em cada frame e faz a busca
    de pico (mantém a semântica próxima à versão from-scratch)."""
    t0 = time.perf_counter()
    audio = audio.astype(np.float64)
    frames = librosa.util.frame(np.pad(audio, frame_length // 2), frame_length=frame_length,
                                 hop_length=hop_length)
    n_frames = frames.shape[1]
    f0 = np.full(n_frames, np.nan)
    voiced = np.zeros(n_frames, dtype=bool)
    min_lag = max(1, int(np.floor(sr / fmax)))
    max_lag = min(frame_length - 1, int(np.ceil(sr / fmin)))
    window = np.hanning(frame_length)
    for i in range(n_frames):
        frame = frames[:, i] * window
        if np.max(np.abs(frame)) < 1e-8:
            continue
        acf = librosa.autocorrelate(frame, max_size=max_lag + 1)
        if acf[0] <= 0:
            continue
        acf /= acf[0]
        seg = acf[min_lag:max_lag + 1]
        if len(seg) == 0:
            continue
        peak = int(np.argmax(seg))
        if seg[peak] < voicing_threshold:
            continue
        lag = min_lag + peak
        f0_hz = sr / lag
        if fmin <= f0_hz <= fmax:
            f0[i] = f0_hz
            voiced[i] = True
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    elapsed = time.perf_counter() - t0
    return {"f0": f0, "voiced": voiced, "times": times, "elapsed_s": elapsed}
