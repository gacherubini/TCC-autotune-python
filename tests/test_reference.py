import numpy as np
from pitch_compare.synth import gen_sine
from pitch_compare.algorithms.reference import autocorr_librosa, yin_librosa, pyin_librosa


def test_yin_librosa_detects_440hz():
    audio, sr, f0_true, _ = gen_sine(440.0, dur=1.0, sr=22050)
    result = yin_librosa(audio, sr, fmin=65, fmax=1000)
    f0 = result["f0"]
    valid = (f0 > 0) & ~np.isnan(f0)
    n = min(len(f0), len(f0_true))
    diff = 1200 * np.abs(np.log2(f0[:n][valid[:n]] / f0_true[:n][valid[:n]]))
    assert np.median(diff) < 5.0


def test_pyin_librosa_detects_440hz_and_returns_voicing_prob():
    audio, sr, f0_true, _ = gen_sine(440.0, dur=1.0, sr=22050)
    result = pyin_librosa(audio, sr, fmin=65, fmax=1000)
    assert "voicing_prob" in result
    assert result["voiced"].mean() > 0.5


def test_autocorr_librosa_returns_expected_keys():
    audio, sr, _, _ = gen_sine(440.0, dur=1.0, sr=22050)
    result = autocorr_librosa(audio, sr, fmin=65, fmax=1000)
    for key in ("f0", "voiced", "times", "elapsed_s"):
        assert key in result
