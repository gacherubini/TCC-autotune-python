"""Gera tabela síntese em markdown com média ± desvio por algoritmo."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np


METRICS = [("rpa", "RPA"), ("rca", "RCA"), ("gpe", "GPE"),
           ("stab_cents", "Estab. (cents)"), ("tempo_s", "Tempo (s)"),
           ("mem_mb", "Mem (MB)")]

METRIC_DESCRIPTIONS = [
    ("RPA (Raw Pitch Accuracy)",
     "Proporção de frames sonoros (voiced) em que o pitch estimado está a menos de 50 cents (meio semitom) do valor de referência. **Maior é melhor.**"),
    ("RCA (Raw Chroma Accuracy)",
     "Igual à RPA, porém ignorando erros de oitava — mede se a classe de altura (dó, ré, mi…) foi acertada, independentemente da oitava. **Maior é melhor.**"),
    ("GPE (Gross Pitch Error)",
     "Proporção de frames sonoros em que o erro relativo |f_pred / f_true − 1| ultrapassa 20% — conta os erros grosseiros (pulos de oitava, trocas de nota). **Menor é melhor.**"),
    ("Estab. (cents)",
     "Estabilidade temporal: desvio padrão das diferenças frame-a-frame da f0 estimada (em cents) nas regiões sonoras. Mede o tremor/jitter da curva de pitch. **Menor é melhor.**"),
    ("Tempo (s)",
     "Tempo médio (em segundos) de execução do algoritmo sobre o áudio, medido em várias rodadas. **Menor é melhor.**"),
    ("Mem (MB)",
     "Pico de memória alocada (em megabytes) durante a execução do algoritmo, medido com `tracemalloc`. **Menor é melhor.**"),
]

REALTIME_METRIC_DESCRIPTIONS = [
    ("Latência (ms)",
     "Latência mínima para produzir a primeira estimativa de f0 em modo streaming. Inclui o buffer do frame (`frame_length / sr`) e o look-ahead do suavizador. **Menor é melhor** — autotune ao vivo tolera no máximo ~20–30 ms antes de incomodar o cantor."),
    ("Tempo/frame (ms)",
     "Tempo de CPU para processar 1 frame, isolado. Precisa ser menor que o intervalo entre frames (`hop_length / sr`) para ser viável em tempo real. **Menor é melhor.**"),
    ("frame_length",
     "Tamanho da janela de análise em amostras. Frame maior → mais precisão e robustez, mas mais latência. Também limita a menor f0 detectável (`fmin ≥ sr / frame_length`)."),
    ("smoothing",
     "Filtro causal/centrado aplicado em cima da f0 bruta para reduzir tremor (`median3_causal`, `median5_causal`, `median3_centered`, `median5_centered`, `ema_a0.3`). Suavizadores centrados ganham estabilidade ao custo de look-ahead (latência extra)."),
]


def _render_glossary() -> str:
    lines = ["## Descrição das métricas\n"]
    for name, desc in METRIC_DESCRIPTIONS:
        lines.append(f"- **{name}** — {desc}")
    return "\n".join(lines) + "\n"


def _render_realtime_glossary() -> str:
    lines = ["### Métricas específicas do benchmark realtime\n"]
    for name, desc in REALTIME_METRIC_DESCRIPTIONS:
        lines.append(f"- **{name}** — {desc}")
    return "\n".join(lines) + "\n"


def _load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [
            {k: (float(v) if k in {"rpa", "rca", "gpe", "stab_cents",
                                   "tempo_s", "mem_mb", "duracao_s"} else v)
             for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def _aggregate(rows: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m, _ in METRICS:
            groups[r["algoritmo"]][m].append(r[m])
    return {
        key: {m: (float(np.mean(v)), float(np.std(v))) for m, v in metrics.items()}
        for key, metrics in groups.items()
    }


def _render_table(agg: dict, title: str) -> str:
    lines = [f"## {title}\n",
             "| Algoritmo | " + " | ".join(label for _, label in METRICS) + " |",
             "|---|" + "|".join(["---"] * len(METRICS)) + "|"]
    for algo, stats in sorted(agg.items()):
        cells = []
        for m, _ in METRICS:
            mean, std = stats[m]
            cells.append(f"{mean:.3f} ± {std:.3f}")
        lines.append(f"| {algo} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


_REALTIME_NUMERIC = {
    "causal", "frame_length", "sr", "hop_length", "fmin_efetivo",
    "lookahead_frames", "frame_buffer_ms", "lookahead_ms", "latencia_ms",
    "tempo_por_frame_ms", "rpa", "gpe", "stab_cents",
}


def _load_realtime_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [
            {k: (float(v) if k in _REALTIME_NUMERIC else v) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def _aggregate_realtime(rows: list[dict]) -> list[dict]:
    """Agrega por (algoritmo, frame_length, smoothing) usando mediana sobre arquivos."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["algoritmo"], int(r["frame_length"]), r["smoothing"])
        groups[key].append(r)
    out = []
    for (algo, fl, sm), rs in groups.items():
        out.append({
            "algoritmo": algo,
            "frame_length": fl,
            "smoothing": sm,
            "causal": int(rs[0]["causal"]),
            "fmin_efetivo": rs[0]["fmin_efetivo"],
            "latencia_ms": float(np.median([r["latencia_ms"] for r in rs])),
            "tempo_por_frame_ms": float(np.median([r["tempo_por_frame_ms"] for r in rs])),
            "stab_cents": float(np.median([r["stab_cents"] for r in rs])),
            "rpa": float(np.median([r["rpa"] for r in rs])),
            "gpe": float(np.median([r["gpe"] for r in rs])),
        })
    out.sort(key=lambda r: (r["latencia_ms"], r["stab_cents"]))
    return out


def _render_realtime_table(agg: list[dict]) -> str:
    header = ("| Algoritmo | Causal? | frame_length | fmin (Hz) | smoothing "
              "| Latência (ms) | Tempo/frame (ms) | Estab. (cents) | RPA | GPE |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = ["## Benchmark realtime\n",
             "_Linhas ordenadas por **latência crescente** e depois **estabilidade crescente**. "
             "Combinações mais próximas do topo são as mais promissoras para autotune ao vivo._\n",
             header, sep]
    for r in agg:
        causal = "sim" if r["causal"] else "não"
        lines.append(
            f"| {r['algoritmo']} | {causal} | {r['frame_length']} | "
            f"{r['fmin_efetivo']:.1f} | {r['smoothing']} | "
            f"{r['latencia_ms']:.2f} | {r['tempo_por_frame_ms']:.4f} | "
            f"{r['stab_cents']:.2f} | {r['rpa']:.3f} | {r['gpe']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _render_realtime_top(agg: list[dict], max_latency_ms: float = 30.0,
                          top_n: int = 5) -> str:
    """Destaca as melhores combinações abaixo de um teto de latência."""
    candidates = [r for r in agg if r["latencia_ms"] <= max_latency_ms]
    candidates.sort(key=lambda r: r["stab_cents"])
    candidates = candidates[:top_n]
    if not candidates:
        return ""
    lines = [f"### Top {len(candidates)} candidatos com latência ≤ {max_latency_ms:.0f} ms\n",
             "Ordenado por **estabilidade** (menor jitter = autotune mais limpo).\n",
             "| Posição | Algoritmo | frame_length | smoothing | Latência (ms) | Estab. (cents) | RPA |",
             "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(candidates, 1):
        lines.append(
            f"| {i} | {r['algoritmo']} | {r['frame_length']} | {r['smoothing']} | "
            f"{r['latencia_ms']:.2f} | {r['stab_cents']:.2f} | {r['rpa']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    out_path = Path("results/summary.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = ["# Síntese dos benchmarks de detecção de pitch\n",
              _render_glossary()]
    for source, csv_path in [("Sinais sintéticos", Path("results/synthetic.csv")),
                              ("Vocadito", Path("results/vocadito.csv"))]:
        if not csv_path.exists():
            chunks.append(f"## {source}\n\n_CSV não encontrado: {csv_path}_\n")
            continue
        rows = _load_csv(csv_path)
        agg = _aggregate(rows)
        chunks.append(_render_table(agg, source))

    realtime_path = Path("results/realtime.csv")
    if realtime_path.exists():
        rt_rows = _load_realtime_csv(realtime_path)
        rt_agg = _aggregate_realtime(rt_rows)
        chunks.append(_render_realtime_glossary())
        chunks.append(_render_realtime_top(rt_agg, max_latency_ms=30.0, top_n=5))
        chunks.append(_render_realtime_table(rt_agg))
    else:
        chunks.append(f"## Benchmark realtime\n\n_CSV não encontrado: {realtime_path}_\n")

    out_path.write_text("\n".join(chunks), encoding="utf-8")
    print(f"Salvo: {out_path}")


if __name__ == "__main__":
    main()
