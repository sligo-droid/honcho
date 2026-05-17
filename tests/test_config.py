import pytest

from src.config import ConfiguredModelSettings, DeriverSettings


def _make_deriver_settings(
    *,
    MAX_INPUT_TOKENS: int = 25000,
    MAX_CUSTOM_INSTRUCTIONS_TOKENS: int = 2000,
    REPRESENTATION_BATCH_MAX_TOKENS: int = 1024,
    REPRESENTATION_BATCH_MAX_AGE_SECONDS: int = 300,
) -> DeriverSettings:
    return DeriverSettings(
        MODEL_CONFIG=ConfiguredModelSettings(
            model="gpt-5.4-mini",
            transport="openai",
        ),
        MAX_INPUT_TOKENS=MAX_INPUT_TOKENS,
        MAX_CUSTOM_INSTRUCTIONS_TOKENS=MAX_CUSTOM_INSTRUCTIONS_TOKENS,
        REPRESENTATION_BATCH_MAX_TOKENS=REPRESENTATION_BATCH_MAX_TOKENS,
        REPRESENTATION_BATCH_MAX_AGE_SECONDS=REPRESENTATION_BATCH_MAX_AGE_SECONDS,
    )


def test_deriver_defaults_enable_custom_instructions_at_supported_cap() -> None:
    settings = _make_deriver_settings()

    assert settings.MAX_INPUT_TOKENS == 25000
    assert settings.MAX_CUSTOM_INSTRUCTIONS_TOKENS == 2000


def test_custom_instructions_tokens_can_be_disabled_with_zero() -> None:
    settings = _make_deriver_settings(MAX_CUSTOM_INSTRUCTIONS_TOKENS=0)

    assert settings.MAX_CUSTOM_INSTRUCTIONS_TOKENS == 0


def test_custom_instructions_tokens_cannot_exceed_supported_cap() -> None:
    with pytest.raises(ValueError, match="less than or equal to 2000"):
        _make_deriver_settings(MAX_CUSTOM_INSTRUCTIONS_TOKENS=2001)


def test_representation_batch_max_age_seconds_is_configurable() -> None:
    settings = _make_deriver_settings(REPRESENTATION_BATCH_MAX_AGE_SECONDS=900)

    assert settings.REPRESENTATION_BATCH_MAX_AGE_SECONDS == 900


def test_representation_batch_max_age_seconds_can_disable_age_flush() -> None:
    settings = _make_deriver_settings(REPRESENTATION_BATCH_MAX_AGE_SECONDS=0)

    assert settings.REPRESENTATION_BATCH_MAX_AGE_SECONDS == 0
