import numpy as np
from pitch_compare.synth import gen_sine, gen_harmonic, gen_noisy, gen_vibrato, gen_glissando


def _dominant_freq(audio, sr):
    """FFT pico — só pra sanity check."""
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    return freqs[np.argmax(spec)]


def test_gen_sine_returns_expected_shape():
    audio, sr, f0, voicing = gen_sine(440.0, dur=1.0, sr=22050, hop_length=512)
    assert sr == 22050
    assert len(audio) == 22050
    n_frames = len(f0)
    assert n_frames == len(voicing)
    assert np.all(voicing)
    assert np.allclose(f0, 440.0)


def test_gen_sine_fft_peak_matches():
    audio, sr, _, _ = gen_sine(440.0, dur=1.0, sr=22050, hop_length=512)
    assert abs(_dominant_freq(audio, sr) - 440.0) < 2.0


def test_gen_harmonic_has_fundamental_and_harmonics():
    audio, sr, _, _ = gen_harmonic(220.0, dur=1.0, sr=22050, hop_length=512, n_harmonics=5)
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    for k in range(1, 6):
        target = 220.0 * k
        idx = np.argmin(np.abs(freqs - target))
        assert spec[idx] > spec.mean() * 5


def test_gen_noisy_reduces_snr():
    clean, sr, f0, voicing = gen_sine(440.0, dur=1.0, sr=22050, hop_length=512)
    noisy_audio, sr2, f0_n, voicing_n = gen_noisy((clean, sr, f0, voicing), snr_db=10.0)
    assert sr2 == sr
    noise = noisy_audio - clean
    power_signal = np.mean(clean ** 2)
    power_noise = np.mean(noise ** 2)
    snr_measured = 10 * np.log10(power_signal / power_noise)
    assert abs(snr_measured - 10.0) < 1.0


def test_gen_vibrato_f0_oscillates():
    audio, sr, f0, voicing = gen_vibrato(440.0, dur=1.0, sr=22050, hop_length=512,
                                          vib_rate=5.0, vib_depth_cents=50.0)
    assert f0.max() > 440.0 + 5.0
    assert f0.min() < 440.0 - 5.0


def test_gen_glissando_f0_increases_monotonically():
    audio, sr, f0, voicing = gen_glissando(220.0, 440.0, dur=1.0, sr=22050, hop_length=512)
    assert f0[0] == 220.0
    assert abs(f0[-1] - 440.0) < 5.0
    assert np.all(np.diff(f0) >= -1e-6)
