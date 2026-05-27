"""Roda todos os algoritmos no dataset Vocadito e salva CSV.

Pré-requisito: extrair o Vocadito em ./data/Vocadito/
(Audio/ e Annotations/F0/).
Fonte: https://zenodo.org/records/5578807 (CC-BY-4.0)
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

from pitch_compare.datasets import iter_vocadito, align_f0_to_times
from pitch_compare.algorithms.reference import autocorr_librosa, yin_librosa, pyin_librosa
from pitch_compare.algorithms.swipe import swipe_scipy
from pitch_compare.metrics import rpa, rca, gpe, temporal_stability, compute_cost


ALGORITHMS = [
    ("autocorr", autocorr_librosa),
    ("yin", yin_librosa),
    ("pyin", pyin_librosa),
    ("swipe", swipe_scipy),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/Vocadito", type=Path)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = todos; >0 = primeiros N arquivos")
    parser.add_argument("--out", default="results/vocadito.csv", type=Path)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, sample in enumerate(iter_vocadito(args.data_dir)):
        if args.limit and i >= args.limit:
            break
        print(f"--- {i}: {sample['name']}  ({len(sample['audio'])/sample['sr']:.1f}s)")

        for algo_name, fn in ALGORITHMS:
            result = fn(sample["audio"], sample["sr"], fmin=65, fmax=1000)
            f0_true_aligned, voicing_aligned = align_f0_to_times(
                sample["src_times"], sample["f0_true"],
                sample["voicing_true"], result["times"])
            n = min(len(result["f0"]), len(f0_true_aligned))
            f0_pred = result["f0"][:n]
            voiced_pred = result["voiced"][:n]
            f0_true = f0_true_aligned[:n]
            voicing_true = voicing_aligned[:n]

            cost = compute_cost(lambda a, s: fn(a, s, fmin=65, fmax=1000),
                                sample["audio"], sample["sr"], n_runs=1)

            row = {
                "arquivo": sample["name"],
                "algoritmo": algo_name,
                "rpa": rpa(f0_pred, f0_true, voicing_true),
                "rca": rca(f0_pred, f0_true, voicing_true),
                "gpe": gpe(f0_pred, f0_true, voicing_true),
                "stab_cents": temporal_stability(f0_pred, voiced_pred),
                "tempo_s": cost["tempo_s"],
                "mem_mb": cost["mem_mb"],
                "duracao_s": len(sample["audio"]) / sample["sr"],
            }
            rows.append(row)
            print(f"  {algo_name:>9}: RPA={row['rpa']:.3f} GPE={row['gpe']:.3f} "
                  f"t={row['tempo_s']:.3f}s")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSalvo: {args.out} ({len(rows)} linhas)")


if __name__ == "__main__":
    main()
