from experiments.tts.profile_qwen3_tts_gpu_sentence_lengths import (
    EXPECTED_LENGTHS,
    TEXTS,
    _validate_text_lengths,
)
from experiments.tts.profiling_utils import linear_fit, percentile


def test_gpu_profile_has_five_texts_in_each_required_length_band() -> None:
    _validate_text_lengths()
    assert set(TEXTS) == {"short", "medium", "long"}
    for tier, texts in TEXTS.items():
        lower, upper = EXPECTED_LENGTHS[tier]
        assert len(texts) == 5
        assert all(lower <= len(text) <= upper for text in texts)
        assert len(set(texts)) == 5


def test_profile_statistics_are_deterministic() -> None:
    assert percentile([1, 2, 3, 4, 5], 0.95) == 4.8
    fitted = linear_fit([10, 20, 30], [2, 4, 6])
    assert fitted["intercept"] == 0.0
    assert fitted["slope_per_char"] == 0.2
    assert fitted["r_squared"] == 1.0
