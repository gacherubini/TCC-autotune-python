"""Geração de sinais sintéticos com ground truth para benchmark."""
from __future__ import annotations
import numpy as np

SignalTuple = tuple[np.ndarray, int, np.ndarray, np.ndarray]


def _frame_times(n_samples: int, sr: int, hop_length: int) -> int:
    return 1 + (n_samples - 1) // hop_length


def _per_frame_constant(value: float, n_samples: int, sr: int, hop_length: int) -> np.ndarray:
    n = _frame_times(n_samples, sr, hop_length)
    return np.full(n, value, dtype=np.float64)


def gen_sine(f0: float, dur: float, sr: int, hop_length: int = 512) -> SignalTuple:
    n = int(round(dur * sr))
    t = np.arange(n) / sr
    audio = np.sin(2 * np.pi * f0 * t).astype(np.float64)
    f0_arr = _per_frame_constant(f0, n, sr, hop_length)
    voicing = np.ones_like(f0_arr, dtype=bool)
    return audio, sr, f0_arr, voicing


def gen_harmonic(f0: float, dur: float, sr: int, hop_length: int = 512,
                 n_harmonics: int = 5) -> SignalTuple:
    n = int(round(dur * sr))
    t = np.arange(n) / sr
    audio = np.zeros(n, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        audio += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
    audio /= np.max(np.abs(audio))
    f0_arr = _per_frame_constant(f0, n, sr, hop_length)
    voicing = np.ones_like(f0_arr, dtype=bool)
    return audio, sr, f0_arr, voicing


def gen_noisy(signal: SignalTuple, snr_db: float, seed: int = 0) -> SignalTuple:
    audio, sr, f0, voicing = signal
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(audio))
    power_signal = np.mean(audio ** 2)
    power_noise_target = power_signal / (10 ** (snr_db / 10.0))
    noise *= np.sqrt(power_noise_target / np.mean(noise ** 2))
    return audio + noise, sr, f0, voicing


def gen_vibrato(f0: float, dur: float, sr: int, hop_length: int = 512,
                vib_rate: float = 5.0, vib_depth_cents: float = 50.0) -> SignalTuple:
    n = int(round(dur * sr))
    t = np.arange(n) / sr
    cents = vib_depth_cents * np.sin(2 * np.pi * vib_rate * t)
    f_inst = f0 * (2.0 ** (cents / 1200.0))
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    audio = np.sin(phase).astype(np.float64)
    nf = _frame_times(n, sr, hop_length)
    idx = np.minimum(np.arange(nf) * hop_length, n - 1)
    f0_arr = f_inst[idx]
    voicing = np.ones_like(f0_arr, dtype=bool)
    return audio, sr, f0_arr, voicing


def gen_glissando(f0_start: float, f0_end: float, dur: float, sr: int,
                  hop_length: int = 512) -> SignalTuple:
    n = int(round(dur * sr))
    t = np.arange(n) / sr
    f_inst = f0_start + (f0_end - f0_start) * (t / dur)
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    audio = np.sin(phase).astype(np.float64)
    nf = _frame_times(n, sr, hop_length)
    idx = np.minimum(np.arange(nf) * hop_length, n - 1)
    f0_arr = f_inst[idx]
    voicing = np.ones_like(f0_arr, dtype=bool)
    return audio, sr, f0_arr, voicing
