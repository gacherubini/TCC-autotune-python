"""Roda todos os algoritmos em todos os sinais sintéticos e salva CSV."""
from __future__ import annotations
import csv
from pathlib import Path

from pitch_compare.datasets import iter_synthetic
from pitch_compare.algorithms.reference import autocorr_librosa, yin_librosa, pyin_librosa
from pitch_compare.algorithms.swipe import swipe_scipy
from pitch_compare.metrics import rpa, rca, gpe, temporal_stability, compute_cost


ALGORITHMS = [
    ("autocorr", autocorr_librosa),
    ("yin", yin_librosa),
    ("pyin", pyin_librosa),
    ("swipe", swipe_scipy),
]


def _align(f0_pred, voiced_pred, f0_true, voicing_true):
    n = min(len(f0_pred), len(f0_true))
    return (f0_pred[:n], voiced_pred[:n], f0_true[:n], voicing_true[:n])


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    rows = []

    for sample in iter_synthetic():
        print(f"--- sinal: {sample['name']}")
        for algo_name, fn in ALGORITHMS:
            result = fn(sample["audio"], sample["sr"], fmin=65, fmax=1000)
            f0_pred, voiced_pred, f0_true, voicing_true = _align(
                result["f0"], result["voiced"],
                sample["f0_true"], sample["voicing_true"])

            cost = compute_cost(lambda a, s: fn(a, s, fmin=65, fmax=1000),
                                sample["audio"], sample["sr"], n_runs=3)

            row = {
                "sinal": sample["name"],
                "algoritmo": algo_name,
                "rpa": rpa(f0_pred, f0_true, voicing_true),
                "rca": rca(f0_pred, f0_true, voicing_true),
                "gpe": gpe(f0_pred, f0_true, voicing_true),
                "stab_cents": temporal_stability(f0_pred, voiced_pred),
                "tempo_s": cost["tempo_s"],
                "mem_mb": cost["mem_mb"],
            }
            rows.append(row)
            print(f"  {algo_name:>9}: RPA={row['rpa']:.3f} "
                  f"GPE={row['gpe']:.3f} t={row['tempo_s']:.3f}s")

    csv_path = out_dir / "synthetic.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSalvo: {csv_path}")


if __name__ == "__main__":
    main()
