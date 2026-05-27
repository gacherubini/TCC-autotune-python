"""Suavizacao causal de trajetorias de f0 para autotune ao vivo.

Todas as funcoes operam em cents (escala log perceptual) para evitar
enviesamento por escala linear de Hz.

Latencia adicional (em frames de saida):
  - causal_median: 0 (so usa passado)
  - centered_median(window=k): (k-1)//2 frames de look-ahead
  - ema: 0 (recursivo causal)
"""
from __future__ import annotations
import numpy as np


_REF_HZ = 10.0


def _to_cents(f0_hz: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200.0 * np.log2(np.where(f0_hz > 0, f0_hz, np.nan) / _REF_HZ)


def _from_cents(cents: np.ndarray) -> np.ndarray:
    out = _REF_HZ * np.power(2.0, cents / 1200.0)
    return np.where(np.isnan(cents), 0.0, out)


def causal_median(f0_hz: np.ndarray, window: int = 3) -> np.ndarray:
    """Mediana sobre janela [i-window+1, i]. Zero frames de look-ahead."""
    cents = _to_cents(f0_hz)
    out = np.copy(cents)
    for i in range(len(cents)):
        start = max(0, i - window + 1)
        win = cents[start:i + 1]
        valid = win[~np.isnan(win)]
        if len(valid) > 0:
            out[i] = np.median(valid)
    return _from_cents(out)


def centered_median(f0_hz: np.ndarray, window: int = 3) -> np.ndarray:
    """Mediana sobre janela [i-w/2, i+w/2]. Look-ahead = (window-1)//2 frames."""
    cents = _to_cents(f0_hz)
    out = np.copy(cents)
    half = window // 2
    for i in range(len(cents)):
        start = max(0, i - half)
        end = min(len(cents), i + half + 1)
        win = cents[start:end]
        valid = win[~np.isnan(win)]
        if len(valid) > 0:
            out[i] = np.median(valid)
    return _from_cents(out)


def ema(f0_hz: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average causal em cents.

    output[i] = alpha * x[i] + (1 - alpha) * output[i-1].
    Amostras invalidas (f0 <= 0) repetem o ultimo valor valido.
    """
    cents = _to_cents(f0_hz)
    out = np.copy(cents)
    last_valid = np.nan
    for i in range(len(cents)):
        if np.isnan(cents[i]):
            out[i] = last_valid
        elif np.isnan(last_valid):
            out[i] = cents[i]
            last_valid = cents[i]
        else:
            out[i] = alpha * cents[i] + (1 - alpha) * last_valid
            last_valid = out[i]
    return _from_cents(out)
