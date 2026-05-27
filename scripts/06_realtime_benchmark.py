"""Benchmark de viabilidade real-time: custo por frame, latencia, estabilidade pos-smoothing.

Diferenca vs scripts 02/03:
  - Mede tempo POR FRAME (nao por audio inteiro).
  - Varre frame_length em {512, 1024, 2048} para ver tradeoff latencia x precisao.
  - Aplica suavizacao causal e centrada em cima da f0 estimada, e remede a
    estabilidade temporal (a metrica que mais afeta a qualidade percebida
    do autotune ao vivo).
  - Reporta latencia total = buffering do frame + look-ahead do smoother.

Saida: results/realtime.csv
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from pitch_compare.algorithms.reference import (
    autocorr_librosa, yin_librosa, pyin_librosa,
)
from pitch_compare.algorithms.swipe import swipe_scipy
from pitch_compare.datasets import iter_vocadito, align_f0_to_times
from pitch_compare.metrics import temporal_stability, rpa, gpe
from pitch_compare.smoothing import causal_median, centered_median, ema


# (nome, fn, causal?). pyin nao e causal de verdade (Viterbi olha sequencia inteira).
# swipe e frame-based puro (sem Viterbi), entao e causal.
ALGORITHMS = [
    ("autocorr", autocorr_librosa, True),
    ("yin",      yin_librosa,      True),
    ("pyin",     pyin_librosa,     False),
    ("swipe",    swipe_scipy,      True),
]

FRAME_LENGTHS = [512, 1024, 2048]

# (nome, funcao, look_ahead_frames)
SMOOTHERS = [
    ("none",              lambda f: f,                       0),
    ("median3_causal",    lambda f: causal_median(f, 3),     0),
    ("median5_causal",    lambda f: causal_median(f, 5),     0),
    ("median3_centered",  lambda f: centered_median(f, 3),   1),
    ("median5_centered",  lambda f: centered_median(f, 5),   2),
    ("ema_a0.3",          lambda f: ema(f, 0.3),             0),
]


def _safe_fmin(frame_length: int, sr: int, requested: float = 65.0) -> float:
    """Menor fmin compativel com (frame_length, sr) para librosa.yin/pyin.

    librosa.yin exige aproximadamente frame_length >= sr / fmin. Adicionamos
    pequena margem para evitar arredondamento na borda.
    """
    minimum = sr / float(frame_length) + 2.0  # margem de seguranca
    return float(max(requested, minimum))


def _measure_per_frame_ms(fn, audio, sr, frame_length, hop_length, fmin, fmax,
                          n_repeats=3):
    """Roda fn n_repeats vezes, retorna tempo mediano por frame em ms."""
    times = []
    result = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn(audio, sr, fmin=fmin, fmax=fmax,
                    frame_length=frame_length, hop_length=hop_length)
        times.append(time.perf_counter() - t0)
    n_frames = len(result["f0"])
    if n_frames == 0:
        return float("nan"), result
    return float(np.median(times) / n_frames * 1000.0), result


def _process_sample(sample, args):
    """Roda todas as combinacoes em um sample, retorna lista de rows."""
    rows = []
    audio = sample["audio"]
    sr = sample["sr"]
    hop_length = args.hop

    for algo_name, fn, is_causal in ALGORITHMS:
        for frame_length in FRAME_LENGTHS:
            # ajusta fmin para o frame_length atual (librosa.yin exige frame_length >= sr/fmin)
            fmin = _safe_fmin(frame_length, sr, requested=65.0)
            fmax = 1000.0
            # tempo por frame
            per_frame_ms, result = _measure_per_frame_ms(
                fn, audio, sr, frame_length, hop_length, fmin, fmax,
                n_repeats=args.repeats,
            )
            f0_pred = result["f0"]
            voiced_pred = result["voiced"]

            # alinha ground truth para o grid de saida do algoritmo
            f0_true_aligned, voicing_aligned = align_f0_to_times(
                sample["src_times"], sample["f0_true"],
                sample["voicing_true"], result["times"])
            n = min(len(f0_pred), len(f0_true_aligned))
            f0_pred_c = f0_pred[:n]
            voiced_c = voiced_pred[:n]
            f0_true_c = f0_true_aligned[:n]
            voicing_true_c = voicing_aligned[:n]

            frame_buffer_ms = frame_length / sr * 1000.0
            hop_ms = hop_length / sr * 1000.0

            for sm_name, sm_fn, sm_lookahead in SMOOTHERS:
                f0_sm = sm_fn(f0_pred_c)
                stab = temporal_stability(f0_sm, voiced_c)
                rpa_v = rpa(f0_sm, f0_true_c, voicing_true_c)
                gpe_v = gpe(f0_sm, f0_true_c, voicing_true_c)

                lookahead_ms = sm_lookahead * hop_ms
                # latencia minima de produzir a 1a estimativa:
                # frame_buffer (esperar 1 frame inteiro) + smoothing lookahead
                latency_ms = frame_buffer_ms + lookahead_ms

                rows.append({
                    "arquivo": sample["name"],
                    "algoritmo": algo_name,
                    "causal": int(is_causal),
                    "frame_length": frame_length,
                    "sr": sr,
                    "hop_length": hop_length,
                    "fmin_efetivo": round(fmin, 1),
                    "smoothing": sm_name,
                    "lookahead_frames": sm_lookahead,
                    "frame_buffer_ms": round(frame_buffer_ms, 2),
                    "lookahead_ms": round(lookahead_ms, 2),
                    "latencia_ms": round(latency_ms, 2),
                    "tempo_por_frame_ms": round(per_frame_ms, 4),
                    "rpa": round(rpa_v, 4),
                    "gpe": round(gpe_v, 4),
                    "stab_cents": round(stab, 3),
                })
            print(f"  {algo_name:>9} fl={frame_length:>4} fmin={fmin:>5.1f}: "
                  f"{per_frame_ms:.3f} ms/frame, "
                  f"frame_buf={frame_buffer_ms:.1f} ms")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/Vocadito", type=Path)
    parser.add_argument("--limit", type=int, default=3,
                        help="Numero de arquivos do Vocadito (default: 3)")
    parser.add_argument("--hop", type=int, default=256,
                        help="hop_length em amostras (default: 256)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="results/realtime.csv", type=Path)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for i, sample in enumerate(iter_vocadito(args.data_dir)):
        if args.limit and i >= args.limit:
            break
        print(f"--- {i}: {sample['name']}  "
              f"({len(sample['audio'])/sample['sr']:.1f}s @ {sample['sr']} Hz)")
        all_rows.extend(_process_sample(sample, args))

    if not all_rows:
        print("Nenhum sample processado.")
        return

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSalvo: {args.out} ({len(all_rows)} linhas)")

    _print_summary(all_rows)


def _print_summary(rows: list[dict]) -> None:
    """Imprime um resumo agregando por (algoritmo, frame_length, smoothing)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r["algoritmo"], r["frame_length"], r["smoothing"])
        groups[key].append(r)

    print("\n=== Resumo (mediana sobre arquivos) ===")
    print(f"{'algo':>9} {'fl':>5} {'smoothing':>18} "
          f"{'lat(ms)':>8} {'t/frame(ms)':>12} {'stab(c)':>9} {'RPA':>6}")
    print("-" * 80)
    for key in sorted(groups.keys()):
        rs = groups[key]
        latency = rs[0]["latencia_ms"]
        per_frame = float(np.median([r["tempo_por_frame_ms"] for r in rs]))
        stab = float(np.median([r["stab_cents"] for r in rs]))
        rpa_v = float(np.median([r["rpa"] for r in rs]))
        algo, fl, sm = key
        print(f"{algo:>9} {fl:>5} {sm:>18} "
              f"{latency:>8.2f} {per_frame:>12.4f} {stab:>9.2f} {rpa_v:>6.3f}")


if __name__ == "__main__":
    main()
