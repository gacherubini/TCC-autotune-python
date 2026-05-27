# Design — Adicionar SWIPE' como 4º algoritmo de detecção de pitch

**Data:** 2026-05-26
**Autor:** Guilherme (TCC PUCRS 2026)
**Status:** aprovado para implementação

## Contexto

O pacote `pitch_compare` hoje compara 3 detectores de pitch via librosa: `autocorr_librosa`, `yin_librosa`, `pyin_librosa`. Os benchmarks atuais (sintético, Vocadito, realtime) mostram que **pyin** é o melhor candidato pra autotune ao vivo (RPA 0.916 com 23 ms de latência e estabilidade 12.78 cents usando frame_length=1024 + EMA), mas é caro: ~1.3 ms/frame e ~80 MB de pico de memória.

Os 3 algoritmos atuais pertencem todos à família temporal (autocorrelação ou variantes). O comparativo do TCC ficaria mais defensável incluindo um detector de **paradigma diferente** que possa de fato competir com pyin em qualidade sem o custo do Viterbi.

## Objetivo

Adicionar **SWIPE'** (Camacho 2008) ao benchmark como 4º algoritmo, integrado de forma idêntica aos 3 existentes, sem novas dependências externas além do que o pacote já tem (numpy/scipy/librosa).

## Não-objetivos

- Não vamos implementar o SWIPE' original em sua forma completa (P²-WSF com múltiplos tamanhos de janela interpolados). Usamos a variante "SWIPE-simple" com FFT única do tamanho do frame, justificada na seção de Algoritmo.
- Não vamos adicionar pós-processamento dinâmico (Viterbi/HMM). Mantém comparação honesta com yin/autocorr (também sem Viterbi).
- Não vamos validar a implementação contra outra biblioteca de SWIPE (ex: pysptk) — a validação vem dos testes sintéticos + benchmark Vocadito.
- Não vamos commitar os CSVs/PNGs regenerados como parte deste design. Essa é uma etapa separada após a implementação rodar.

## Arquitetura e integração

### Novo arquivo: `pitch_compare/algorithms/swipe.py`

Função pública única, mesma assinatura dos wrappers em `reference.py`:

```python
def swipe_scipy(audio, sr, *, fmin=65.0, fmax=1000.0,
                frame_length=2048, hop_length=512,
                voicing_threshold=0.2, **_) -> dict:
    """SWIPE' (Camacho 2008), variante single-FFT em scipy/numpy puro.

    Retorna {"f0": np.ndarray, "voiced": np.ndarray[bool],
             "times": np.ndarray, "elapsed_s": float}.
    """
```

**Por que arquivo separado em vez de adicionar a `reference.py`:** `reference.py` é descrito como "wrappers em torno das implementações de pitch detection do librosa". SWIPE' é implementação from-scratch — semanticamente diferente. Mantém a fronteira clara.

**Dependências:** zero novas. `numpy.fft` + `scipy.signal` (scipy já está em `pyproject.toml` como dep direta).

### Integração nos scripts (mudança de 1 linha cada)

- `scripts/02_run_synthetic_benchmark.py` → adicionar `("swipe", swipe_scipy)` em `ALGORITHMS`
- `scripts/03_run_dataset_benchmark.py` → idem
- `scripts/06_realtime_benchmark.py` → adicionar `("swipe", swipe_scipy, True)` (causal: True, pois é frame-based puro sem Viterbi)

### Sem mudança

- `scripts/04_plot_results.py` — agrega por `algoritmo` dinamicamente
- `scripts/05_summary_report.py` — idem
- `scripts/run_all.ps1` — orquestrador

## Algoritmo

SWIPE' é um detector espectral. Para cada candidato de pitch `f`, constrói-se um kernel no domínio da frequência que tem peso positivo nos harmônicos de `f` (modulado por `cos²` decaindo com `1/√k`) e peso negativo perto dos sub-harmônicos. O pitch estimado é o `f` que maximiza o produto interno `<kernel(f), √|X(f)|>` por frame.

### Estrutura

```python
def _erb_to_hz(erb): ...
def _hz_to_erb(hz): ...
def _candidate_grid(fmin, fmax, step_erb=1/24) -> np.ndarray:
    # grade de candidatos em escala ERB (perceptual)

def _build_kernel_matrix(candidates, freqs) -> np.ndarray:
    # matriz [n_cand, n_freqs] com kernels SWIPE' (cos² nos harmônicos,
    # decaimento 1/sqrt(k)). Pré-computada uma vez por chamada.

def swipe_scipy(audio, sr, *, fmin, fmax, frame_length, hop_length,
                voicing_threshold, **_) -> dict:
    # 1. enquadra: mesma estratégia do autocorr_librosa
    #    (pad + librosa.util.frame com frame_length, hop_length)
    # 2. |FFT|^(1/2) por frame, janela de Hann
    # 3. produto matricial: strength = kernels @ spec_sqrt  →  [n_cand, n_frames]
    # 4. argmax por frame + interpolação parabólica (3 pontos) para precisão sub-bin
    # 5. voiced = strength_max > voicing_threshold
    # 6. retorna dict no contrato padrão
```

### Simplificações vs paper original

- **Single FFT size** em vez de múltiplas FFTs com tamanhos proporcionais ao período candidato (P²-WSF). Justificativa: o `frame_length` no benchmark é controlado externamente pelo script realtime (varre {512, 1024, 2048}) para medir tradeoff latência×precisão. Usar uma única FFT por candidato mantém essa variável independente e comparável aos outros algoritmos. Esta variante "single-FFT" é a usada em `pysptk.sptk.swipe` e outras implementações Python comuns.
- **Grade ERB** de candidatos com passo `1/24` ERB (padrão do paper).
- **Sem Viterbi**: voicing decidido por threshold sobre o pico de strength por frame.

### Vetorização e custo esperado

- Kernels pré-computados uma vez (`[n_candidates, n_freqs]`).
- Loop sobre frames calcula FFT, depois produto matricial → eficiente em numpy.
- Custo dominado pela FFT (~mesma ordem do pyin), com vantagem de **memória menor** (sem matriz de transição do Viterbi).

### Cuidados

- Considerar harmônicos do kernel até Nyquist (`sr/2`). Bins acima de Nyquist têm contribuição zero.
- `voicing_threshold` default 0.2 (calibrável via parâmetro nomeado).
- Tratamento de áudio silencioso (`max(abs(frame)) < 1e-8`): `voiced = False`, `f0 = NaN` (mesma convenção de `autocorr_librosa`).

## Testes

**Novo arquivo:** `tests/test_swipe.py`. Segue o padrão de `tests/test_reference.py` (smoke + sanity, sem mocks, asserts diretos).

Casos:
1. `test_swipe_detects_440hz_pure_sine` — senoide 440 Hz, mediana do erro absoluto em cents < 5.
2. `test_swipe_detects_220hz_harmonic` — 5 parciais em 220 Hz, garante que pega a fundamental e não h2/h3 (esse é o ponto forte de SWIPE' vs autocorr ingênuo).
3. `test_swipe_returns_expected_keys` — contrato `{f0, voiced, times, elapsed_s}`.
4. `test_swipe_voicing_zero_on_silence` — áudio de zeros não crasha, `voiced.sum() == 0`.
5. `test_swipe_tracks_glissando` — `gen_glissando(220, 440, 1.0, 22050)`, mediana do erro absoluto em cents < 50 (tolerância maior porque f0 muda dentro do frame).
6. `test_swipe_robust_to_noise` — `gen_noisy(sine 440, SNR=10dB)`, RPA > 0.8.

**Fora de escopo:** comparação numérica contra outra implementação de SWIPE (não temos baseline na suite); testes de performance (responsabilidade dos scripts de benchmark).

## Regeneração de benchmarks

Após implementação + testes verdes, rodar `.\scripts\run_all.ps1` regenera:

- `results/synthetic.csv` — +9 linhas (1 por sinal sintético)
- `results/vocadito.csv` — +40 linhas (1 por áudio do Vocadito)
- `results/realtime.csv` — +54 linhas (3 frame_lengths × 6 smoothers × 3 arquivos)
- `results/figures/*.png` — todas regeradas (agora com 4 barras)
- `results/summary.md` — regenerado com 4ª linha nas tabelas

**Esses artefatos não fazem parte deste design** — ficam num commit separado após a implementação rodar.

## Critérios de sucesso

Cenários possíveis pós-benchmark:

| Cenário | Resultado de SWIPE' vs pyin | Narrativa pro TCC |
|---|---|---|
| Ideal | stab ≤ pyin **E** memória < pyin | "Achamos alternativa espectral mais leve com qualidade comparável" |
| Neutro | qualidade ~pyin, custo ~pyin | "Paradigma espectral confirma resultado do temporal; pyin segue como escolha" |
| Ruim | abaixo de pyin em tudo | "Validamos pyin como teto realista entre métodos clássicos sem deep learning" |

Em qualquer dos 3 a contribuição ao TCC se sustenta: ou ampliamos o leque ou fortalecemos a justificativa do pyin.

## Resumo dos arquivos tocados

**Novos:**
- `pitch_compare/algorithms/swipe.py`
- `tests/test_swipe.py`

**Modificados (1 linha cada na lista `ALGORITHMS`):**
- `scripts/02_run_synthetic_benchmark.py`
- `scripts/03_run_dataset_benchmark.py`
- `scripts/06_realtime_benchmark.py`

**Não modificados (mas regerados depois, fora deste design):**
- `results/synthetic.csv`, `results/vocadito.csv`, `results/realtime.csv`
- `results/figures/*.png`
- `results/summary.md`
