# CLAUDE.md — guia para agentes

Contexto operacional deste repositório. Leia antes de mexer no código.

## O que é

Bancada experimental que compara **quatro algoritmos de detecção de pitch** (autocorrelação,
YIN, pYIN, SWIPE′) para fundamentar a escolha do algoritmo do protótipo de autotune. É a
parte de *metodologia* de um TCC (PUCRS, 2026) — não é código de produção.

**Repositório irmão:** [`TCC-autotune-cpp`](https://github.com/gacherubini/TCC-autotune-cpp)
— o protótipo em C++/JUCE. **A documentação técnica completa do trabalho vive lá**, em
[`docs/`](https://github.com/gacherubini/TCC-autotune-cpp/tree/main/docs); comece por
`docs/README.md`.

## O resultado que este repo existe para produzir

Em precisão média (RPA) os três métodos temporais empatam. O que os separa é a
**estabilidade temporal** — o desvio-padrão das diferenças de F0 entre quadros consecutivos:
pYIN 29,4 cents vs. YIN 671,6 no Vocadito. Como o deslocamento de pitch converte oscilação
entre quadros em artefato audível, **a estabilidade é o critério decisivo**, não a precisão.

Se for alterar métricas ou algoritmos, preserve essa conclusão ou refute-a com dados — ela é
a primeira contribuição declarada do TCC.

## Estrutura

```
pitch_compare/
  algorithms/reference.py   wrappers finos sobre librosa (autocorr, yin, pyin)
  algorithms/swipe.py       SWIPE′ do zero — a única implementação original aqui
  synth.py                  geradores de sinal com ground truth exato
  datasets.py               Vocadito + align_f0_to_times()
  metrics.py                RPA, RCA, GPE, estabilidade, tempo, memória
  smoothing.py              suavizadores causais/centrados, com look-ahead declarado
scripts/02..06              pipeline de experimentos, numerado na ordem de execução
tests/                      pytest
```

Todo algoritmo devolve o mesmo dicionário: `{"f0", "voiced", "times", "elapsed_s"}`.
Mantenha esse contrato ao adicionar um quinto algoritmo.

## Rodar

```bash
pip install -e ".[dev]"
pytest
python scripts/02_run_synthetic_benchmark.py   # não precisa de dataset
python scripts/03_run_dataset_benchmark.py     # precisa do Vocadito em data/Vocadito/
```

O **Vocadito** não é versionado: baixe de <https://zenodo.org/records/5578807> e extraia em
`data/Vocadito/` (com `Audio/` e `Annotations/F0/`).

## Armadilhas conhecidas

- **`data/` e `results/` estão no `.gitignore`.** Nenhum CSV de resultado é versionado — os
  números do TCC não são reproduzíveis a partir do repo sem rodar tudo de novo. **Versionar
  os CSVs de `results/` é uma pendência conhecida** (são pequenos).
- **Sempre opere em cents, nunca em Hz**, ao suavizar ou medir distância de pitch. A mesma
  diferença em Hz vale intervalos musicais diferentes em regiões graves e agudas.
  `metrics.py` e `smoothing.py` já fazem isso; mantenha.
- **`librosa.yin` exige `frame_length ≥ sr/fmin`.** Ao varrer `frame_length` para baixo, o
  `fmin` pedido fica ilegal — é o que `_safe_fmin()` em `06_realtime_benchmark.py` resolve.
- **Alinhe o ground truth antes de comparar.** Cada algoritmo tem seu próprio `hop`, então a
  anotação do Vocadito precisa passar por `align_f0_to_times()`.
- **O SWIPE′ usa kernel zero-mean sobre o suporte ativo.** Sem essa remoção de média, o
  `argmax` fica enviesado para `fmax` sob ruído. Não "simplifique" isso.

## Convenções

- **Idioma:** código, docstrings e comentários em **português** (sem acentos nos docstrings,
  por consistência com o que já existe). Mensagens de commit em **inglês**.
- **Scripts numerados** (`02_`, `03_`…) refletem a ordem de execução do pipeline. Mantenha.
- Sem dependências novas sem necessidade.
