"""Métricas de avaliação de detecção de pitch."""
from __future__ import annotations
import time
import tracemalloc
from typing import Callable
import numpy as np


def _to_cents(f0_hz: np.ndarray, ref_hz: float = 10.0) -> np.ndarray:
    """Converte Hz para cents (1200 * log2(f/ref))."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200.0 * np.log2(np.where(f0_hz > 0, f0_hz, np.nan) / ref_hz)


def _abs_cents_diff(f0_pred: np.ndarray, f0_true: np.ndarray) -> np.ndarray:
    return np.abs(_to_cents(f0_pred) - _to_cents(f0_true))


def rpa(f0_pred: np.ndarray, f0_true: np.ndarray, voicing_true: np.ndarray,
        tol_cents: float = 50.0) -> float:
    """Raw Pitch Accuracy: % de frames voiced com erro <= tol_cents."""
    voiced = voicing_true.astype(bool)
    if not np.any(voiced):
        return 0.0
    diff = _abs_cents_diff(f0_pred[voiced], f0_true[voiced])
    return float(np.nanmean(diff <= tol_cents))


def rca(f0_pred: np.ndarray, f0_true: np.ndarray, voicing_true: np.ndarray,
        tol_cents: float = 50.0) -> float:
    """Raw Chroma Accuracy: igual a RPA mas ignora oitavas."""
    voiced = voicing_true.astype(bool)
    if not np.any(voiced):
        return 0.0
    diff = _abs_cents_diff(f0_pred[voiced], f0_true[voiced])
    diff_mod_octave = np.minimum(diff % 1200.0, 1200.0 - (diff % 1200.0))
    return float(np.nanmean(diff_mod_octave <= tol_cents))


def gpe(f0_pred: np.ndarray, f0_true: np.ndarray, voicing_true: np.ndarray,
        tol: float = 0.2) -> float:
    """Gross Pitch Error: % de frames voiced com |f_pred/f_true - 1| > tol."""
    voiced = voicing_true.astype(bool)
    if not np.any(voiced):
        return 0.0
    fp = f0_pred[voiced]
    ft = f0_true[voiced]
    valid = (fp > 0) & (ft > 0)
    if not np.any(valid):
        return 0.0
    rel = np.abs(fp[valid] / ft[valid] - 1.0)
    return float(np.mean(rel > tol))


def temporal_stability(f0_pred: np.ndarray, voiced_pred: np.ndarray) -> float:
    """Desvio padrão das diferenças frame-a-frame em cents (regiões voiced)."""
    voiced = voiced_pred.astype(bool)
    if voiced.sum() < 2:
        return 0.0
    f0 = np.where(voiced, f0_pred, np.nan)
    cents = _to_cents(f0)
    diffs = np.diff(cents)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return 0.0
    return float(np.std(diffs))


def compute_cost(fn: Callable, audio: np.ndarray, sr: int, n_runs: int = 5) -> dict:
    """Mede tempo médio (s) e pico de memória (MB) ao chamar fn(audio, sr)."""
    times = []
    peaks = []
    for _ in range(n_runs):
        tracemalloc.start()
        t0 = time.perf_counter()
        fn(audio, sr)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)
        peaks.append(peak)
    return {
        "tempo_s": float(np.mean(times)),
        "tempo_s_std": float(np.std(times)),
        "mem_mb": float(max(peaks)) / (1024 ** 2),
    }
