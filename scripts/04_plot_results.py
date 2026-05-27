"""Gera figuras comparativas a partir dos CSVs de benchmark."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def _load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return [
            {k: (float(v) if k in {"rpa", "rca", "gpe", "stab_cents",
                                   "tempo_s", "mem_mb", "duracao_s"} else v)
             for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def _aggregate(rows: list[dict], metric: str) -> dict[str, tuple[float, float]]:
    """Retorna {algoritmo: (mean, std)} para `metric`."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["algoritmo"]].append(r[metric])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in groups.items()}


def _bar_plot(data: dict, title: str, ylabel: str, out_path: Path) -> None:
    labels = sorted(data.keys())
    means = [data[k][0] for k in labels]
    stds = [data[k][1] for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    for source, csv_path in [("synthetic", Path("results/synthetic.csv")),
                              ("vocadito", Path("results/vocadito.csv"))]:
        if not csv_path.exists():
            print(f"skip: {csv_path} não existe")
            continue
        rows = _load_csv(csv_path)
        for metric, ylabel in [("rpa", "RPA (proporção)"),
                                ("gpe", "GPE (proporção)"),
                                ("stab_cents", "Estabilidade temporal (cents)"),
                                ("tempo_s", "Tempo (s)")]:
            data = _aggregate(rows, metric)
            _bar_plot(data, f"{metric.upper()} — {source}", ylabel,
                      fig_dir / f"{source}_{metric}.png")
            print(f"salvo: {fig_dir / f'{source}_{metric}.png'}")


if __name__ == "__main__":
    main()
