"""SWIPE' (Camacho 2008), variante single-FFT em numpy/scipy puro."""
from __future__ import annotations
import time
import numpy as np
import librosa


def _hz_to_erb(hz: np.ndarray | float) -> np.ndarray | float:
    return 21.4 * np.log10(1.0 + 0.00437 * np.asarray(hz, dtype=np.float64))


def _erb_to_hz(erb: np.ndarray | float) -> np.ndarray | float:
    return (np.power(10.0, np.asarray(erb, dtype=np.float64) / 21.4) - 1.0) / 0.00437


def _candidate_grid(fmin: float, fmax: float,
                    step_erb: float = 1.0 / 48.0
                    ) -> tuple[np.ndarray, np.ndarray, float]:
    """Grade de candidatos de pitch espacada uniformemente em ERB.

    Retorna (hz, erb, step_erb) para permitir interpolacao parabolica em ERB.
    """
    erb_min = float(_hz_to_erb(fmin))
    erb_max = float(_hz_to_erb(fmax))
    n = int(np.floor((erb_max - erb_min) / step_erb)) + 1
    erbs = erb_min + np.arange(n) * step_erb
    return (np.asarray(_erb_to_hz(erbs), dtype=np.float64), erbs, step_erb)


def _build_kernel(candidate_hz: float, freqs_hz: np.ndarray,
                  nyquist: float) -> np.ndarray:
    """Kernel harmonic-sieve com remoção de média (zero-mean).

    Soma lobes cos² positivos em cada harmonico k*p (peso 1/sqrt(k)), depois
    subtrai a média sobre o suporte ativo do kernel (bins até o ultimo
    harmonico). Resultado: sum(kernel) ≈ 0 → ruído branco produz strength ≈ 0
    em qualquer p. Sem isso, kernels para p pequeno acumulam mais lobes e
    enviesam o argmax para fmax sob ruido.
    """
    p = float(candidate_hz)
    kernel = np.zeros_like(freqs_hz)
    last_harmonic_hz = 0.0
    k = 1
    while k * p < nyquist:
        weight = 1.0 / np.sqrt(k)
        d = np.abs(freqs_hz - k * p)
        mask = d < p / 2.0
        kernel[mask] += weight * np.cos(np.pi * d[mask] / p) ** 2
        last_harmonic_hz = k * p
        k += 1
    # zero-mean sobre o suporte ativo (de p/2 até último harmonico + p/2)
    support_mask = (freqs_hz >= p / 2.0) & (freqs_hz <= last_harmonic_hz + p / 2.0)
    n_support = int(support_mask.sum())
    if n_support > 0:
        kernel[support_mask] -= kernel[support_mask].mean()
    return kernel


def _build_kernel_matrix(candidates: np.ndarray, freqs_hz: np.ndarray,
                         nyquist: float) -> np.ndarray:
    return np.vstack([_build_kernel(p, freqs_hz, nyquist) for p in candidates])


def _parabolic_erb_interp(strengths: np.ndarray, idx: int,
                           candidate_erbs: np.ndarray, step_erb: float) -> float:
    """Interpolacao parabolica em ERB; retorna f0 em Hz."""
    if idx <= 0 or idx >= len(strengths) - 1:
        return float(_erb_to_hz(candidate_erbs[idx]))
    y_m, y0, y_p = strengths[idx - 1], strengths[idx], strengths[idx + 1]
    denom = y_m - 2.0 * y0 + y_p
    if abs(denom) < 1e-12:
        return float(_erb_to_hz(candidate_erbs[idx]))
    delta = 0.5 * (y_m - y_p) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    erb_refined = float(candidate_erbs[idx]) + delta * step_erb
    return float(_erb_to_hz(erb_refined))


def swipe_scipy(audio: np.ndarray, sr: int, *, fmin: float = 65.0,
                fmax: float = 1000.0, frame_length: int = 2048,
                hop_length: int = 512, voicing_threshold: float = 0.2,
                fft_zero_pad: int = 1, **_) -> dict:
    """SWIPE' (variante single-FFT, harmonic-sieve) — frame-based, sem Viterbi.

    fft_zero_pad multiplica o tamanho da FFT (zero-padding). Default 1 (sem
    padding) — empiricamente nao melhora a precisao em senoides puras (viés
    domina) e custa tempo em realtime.
    """
    t0 = time.perf_counter()
    audio = audio.astype(np.float64)
    nyquist = sr / 2.0

    padded = np.pad(audio, frame_length // 2)
    frames = librosa.util.frame(padded, frame_length=frame_length,
                                 hop_length=hop_length)
    n_frames = frames.shape[1]
    window = np.hanning(frame_length)

    n_fft = int(frame_length * fft_zero_pad)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    candidates_hz, candidate_erbs, step_erb = _candidate_grid(fmin, fmax)
    kernels = _build_kernel_matrix(candidates_hz, freqs, nyquist)

    f0 = np.full(n_frames, np.nan)
    voiced = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        frame = frames[:, i] * window
        if np.max(np.abs(frame)) < 1e-8:
            continue
        spec = np.abs(np.fft.rfft(frame, n=n_fft))
        spec_sqrt = np.sqrt(spec)
        norm = np.linalg.norm(spec_sqrt)
        if norm < 1e-12:
            continue
        spec_sqrt /= norm
        strengths = kernels @ spec_sqrt
        idx = int(np.argmax(strengths))
        peak = float(strengths[idx])
        if peak < voicing_threshold:
            continue
        f0[i] = _parabolic_erb_interp(strengths, idx, candidate_erbs, step_erb)
        voiced[i] = True

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    elapsed = time.perf_counter() - t0
    return {"f0": f0, "voiced": voiced, "times": times, "elapsed_s": elapsed}
