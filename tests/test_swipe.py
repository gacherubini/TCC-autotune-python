import numpy as np

from pitch_compare.synth import gen_sine, gen_harmonic, gen_noisy, gen_glissando
from pitch_compare.algorithms.swipe import swipe_scipy


def test_swipe_detects_440hz_pure_sine():
    # Single-FFT spectral pitch detectors (SWIPE-simple) tem precisao limitada
    # em senoide pura: resolucao de bin (10.77 Hz @ frame_length=2048/sr=22050)
    # e a concavidade do sqrt() introduzem viés de ~7-10 cents. Limite realista
    # 15 cents (≈ 1/8 de semitom). Para precisao sub-5-cents em senoides puras
    # use o time-domain (yin/pyin).
    audio, sr, f0_true, _ = gen_sine(440.0, dur=1.0, sr=22050)
    result = swipe_scipy(audio, sr, fmin=65, fmax=1000)
    f0 = result["f0"]
    valid = (f0 > 0) & ~np.isnan(f0)
    n = min(len(f0), len(f0_true))
    diff = 1200 * np.abs(np.log2(f0[:n][valid[:n]] / f0_true[:n][valid[:n]]))
    assert np.median(diff) < 15.0


def test_swipe_detects_220hz_harmonic():
    # SWIPE deve pegar a fundamental, NAO h2 (440) ou h3 (660).
    # Este e o ponto-forte do SWIPE vs detectores baseados em pico-de-FFT puro.
    audio, sr, f0_true, _ = gen_harmonic(220.0, dur=1.0, sr=22050, n_harmonics=5)
    result = swipe_scipy(audio, sr, fmin=65, fmax=1000)
    f0 = result["f0"]
    valid = (f0 > 0) & ~np.isnan(f0)
    # mediana deve estar perto de 220, nao 440/660/etc
    median_f0 = float(np.median(f0[valid]))
    error_cents = 1200 * abs(np.log2(median_f0 / 220.0))
    assert error_cents < 15.0, f"got {median_f0:.1f} Hz (off by {error_cents:.1f} cents)"


def test_swipe_returns_expected_keys():
    audio, sr, _, _ = gen_sine(440.0, dur=1.0, sr=22050)
    result = swipe_scipy(audio, sr, fmin=65, fmax=1000)
    for key in ("f0", "voiced", "times", "elapsed_s"):
        assert key in result


def test_swipe_voicing_zero_on_silence():
    sr = 22050
    audio = np.zeros(sr, dtype=np.float64)
    result = swipe_scipy(audio, sr, fmin=65, fmax=1000)
    assert result["voiced"].sum() == 0


def test_swipe_tracks_glissando():
    audio, sr, f0_true, _ = gen_glissando(220.0, 440.0, dur=1.0, sr=22050)
    result = swipe_scipy(audio, sr, fmin=65, fmax=1000)
    f0_pred = result["f0"]
    voiced = result["voiced"]
    n = min(len(f0_pred), len(f0_true))
    valid = voiced[:n] & (f0_pred[:n] > 0) & ~np.isnan(f0_pred[:n])
    diff = 1200 * np.abs(np.log2(f0_pred[:n][valid] / f0_true[:n][valid]))
    assert np.median(diff) < 50.0  # tolerancia maior: f0 muda dentro do frame


def test_swipe_robust_to_noise():
    # SWIPE espectral em senoide pura ruidosa: sem harmonicos reais para a
    # soma cruzada (1/sqrt(k)) ajudar, cada frame depende de um único pico
    # espectral. Time-domain (yin/autocorr) atinge RPA=1.0 ate SNR=10dB; SWIPE
    # precisa SNR≥30dB para mesma faixa. Em voz real (harmonicos verdadeiros)
    # SWIPE deve performar melhor — sera visto no benchmark Vocadito.
    clean = gen_sine(440.0, dur=1.0, sr=22050)
    noisy_audio, sr, f0_true, voicing = gen_noisy(clean, snr_db=30.0)
    result = swipe_scipy(noisy_audio, sr, fmin=65, fmax=1000)
    from pitch_compare.metrics import rpa
    n = min(len(result["f0"]), len(f0_true))
    assert rpa(result["f0"][:n], f0_true[:n], voicing[:n], tol_cents=50.0) > 0.8
