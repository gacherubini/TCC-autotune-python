import numpy as np
import pytest
from pitch_compare.metrics import rpa, rca, gpe, temporal_stability, compute_cost


def test_rpa_perfect_prediction():
    f0_true = np.array([220.0, 220.0, 440.0, 440.0])
    voicing = np.array([True, True, True, True])
    assert rpa(f0_true, f0_true, voicing) == pytest.approx(1.0)


def test_rpa_ignores_unvoiced_frames():
    f0_pred = np.array([220.0, 0.0, 440.0])
    f0_true = np.array([220.0, 999.0, 440.0])
    voicing = np.array([True, False, True])
    assert rpa(f0_pred, f0_true, voicing) == pytest.approx(1.0)


def test_rpa_all_wrong_returns_zero():
    f0_true = np.array([220.0, 440.0])
    f0_pred = np.array([1000.0, 2000.0])
    voicing = np.array([True, True])
    assert rpa(f0_pred, f0_true, voicing) == 0.0


def test_rca_accepts_octave_error():
    f0_true = np.array([220.0, 220.0])
    f0_pred = np.array([440.0, 110.0])
    voicing = np.array([True, True])
    assert rpa(f0_pred, f0_true, voicing) == 0.0
    assert rca(f0_pred, f0_true, voicing) == pytest.approx(1.0)


def test_gpe_detects_gross_errors():
    f0_true = np.array([220.0, 220.0, 220.0])
    f0_pred = np.array([220.0, 440.0, 110.0])
    voicing = np.array([True, True, True])
    assert gpe(f0_pred, f0_true, voicing) == pytest.approx(2 / 3)


def test_temporal_stability_zero_for_constant_signal():
    f0 = np.full(10, 440.0)
    voicing = np.ones(10, dtype=bool)
    assert temporal_stability(f0, voicing) == pytest.approx(0.0)


def test_temporal_stability_positive_for_jumpy_signal():
    f0 = np.array([440.0, 440.0, 880.0, 440.0, 440.0])
    voicing = np.ones(5, dtype=bool)
    assert temporal_stability(f0, voicing) > 100.0  # cents


def test_compute_cost_returns_time_and_memory():
    def fn(audio, sr):
        return {"f0": np.zeros(100), "voiced": np.zeros(100, dtype=bool)}
    audio = np.zeros(22050)
    cost = compute_cost(fn, audio, sr=22050, n_runs=3)
    assert "tempo_s" in cost
    assert "mem_mb" in cost
    assert cost["tempo_s"] > 0
    assert cost["mem_mb"] >= 0
