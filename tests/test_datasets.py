import numpy as np
from pitch_compare.datasets import iter_synthetic, load_vocadito_sample


def test_iter_synthetic_yields_named_samples():
    samples = list(iter_synthetic(sr=22050, hop_length=512))
    assert len(samples) >= 5
    names = {s["name"] for s in samples}
    assert "sine_440" in names
    for s in samples:
        assert "audio" in s
        assert "sr" in s
        assert "f0_true" in s
        assert "voicing_true" in s


def test_load_vocadito_sample_parses_csv(tmp_path):
    import soundfile as sf
    sr = 44100
    audio = np.zeros(sr)
    wav_path = tmp_path / "vocadito_99.wav"
    sf.write(wav_path, audio, sr)
    f0_path = tmp_path / "vocadito_99_f0.csv"
    # 100 frames a ~5.8ms: tempos crescentes, F0 constante 440Hz
    lines = [f"{i * 0.0058},{440.0 if i % 2 == 0 else 0.0}" for i in range(100)]
    f0_path.write_text("\n".join(lines))
    sample = load_vocadito_sample(wav_path, f0_path)
    assert sample["sr"] == sr
    assert sample["voicing_true"].sum() == 50
    assert np.allclose(sample["f0_true"][sample["voicing_true"]], 440.0)
