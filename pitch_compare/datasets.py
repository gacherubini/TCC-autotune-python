"""Carregamento de datasets sintéticos e do Vocadito."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterator
import numpy as np
import soundfile as sf

from . import synth


def iter_synthetic(sr: int = 22050, hop_length: int = 512) -> Iterator[dict]:
    """Gera os sinais sintéticos canônicos para benchmark."""
    cases: list[tuple[str, callable]] = [
        ("sine_220", lambda: synth.gen_sine(220.0, 2.0, sr, hop_length)),
        ("sine_440", lambda: synth.gen_sine(440.0, 2.0, sr, hop_length)),
        ("sine_880", lambda: synth.gen_sine(880.0, 2.0, sr, hop_length)),
        ("harmonic_220_5h", lambda: synth.gen_harmonic(220.0, 2.0, sr, hop_length, 5)),
        ("harmonic_440_5h", lambda: synth.gen_harmonic(440.0, 2.0, sr, hop_length, 5)),
        ("noisy_440_snr20", lambda: synth.gen_noisy(
            synth.gen_sine(440.0, 2.0, sr, hop_length), 20.0)),
        ("noisy_440_snr10", lambda: synth.gen_noisy(
            synth.gen_sine(440.0, 2.0, sr, hop_length), 10.0)),
        ("vibrato_440", lambda: synth.gen_vibrato(440.0, 2.0, sr, hop_length, 5.0, 50.0)),
        ("glissando_220_440", lambda: synth.gen_glissando(220.0, 440.0, 2.0, sr, hop_length)),
    ]
    for name, gen in cases:
        audio, sr_out, f0, voicing = gen()
        yield {"name": name, "audio": audio, "sr": sr_out,
               "f0_true": f0, "voicing_true": voicing}


def load_vocadito_sample(wav_path: Path | str, f0_path: Path | str) -> dict:
    """Carrega um par WAV + CSV de F0 do Vocadito.

    Áudio: 44.1kHz mono. CSV: duas colunas (tempo_s, f0_hz); f0=0 = unvoiced.
    """
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    times, f0_hz = [], []
    with open(f0_path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            times.append(float(row[0]))
            f0_hz.append(float(row[1]))
    src_times = np.array(times, dtype=np.float64)
    f0_true = np.array(f0_hz, dtype=np.float64)
    return {"name": Path(wav_path).stem, "audio": audio.astype(np.float64),
            "sr": sr, "f0_true": f0_true, "voicing_true": f0_true > 0,
            "src_times": src_times}


def iter_vocadito(data_dir: Path | str) -> Iterator[dict]:
    data_dir = Path(data_dir)
    audio_dir = data_dir / "Audio"
    f0_dir = data_dir / "Annotations" / "F0"
    if not audio_dir.exists() or not f0_dir.exists():
        raise FileNotFoundError(
            f"Vocadito não encontrado em {data_dir}. "
            f"Esperado: {audio_dir} e {f0_dir}.")
    for wav in sorted(audio_dir.glob("*.wav")):
        f0_csv = f0_dir / f"{wav.stem}_f0.csv"
        if f0_csv.exists():
            yield load_vocadito_sample(wav, f0_csv)


def align_f0_to_times(src_times: np.ndarray, src_f0: np.ndarray,
                      src_voicing: np.ndarray,
                      target_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reamostra anotação (tempos src) para o grid de tempo dos algoritmos."""
    f0_out = np.interp(target_times, src_times, src_f0, left=0.0, right=0.0)
    src_hop = src_times[1] - src_times[0] if len(src_times) > 1 else 1.0
    idx = np.clip(np.round(target_times / src_hop).astype(int), 0, len(src_voicing) - 1)
    voicing_out = src_voicing[idx]
    return f0_out, voicing_out
