"""Installed Forge Spectrum settings bridge; CPU only."""

from types import SimpleNamespace

import pytest

from pi_ideogram_forge.spectrum_settings import read_native_spectrum


def _script(start=2):
    return type("SpectrumForForge", (), {"args_from": start})()


def _request(values, scripts=None):
    return SimpleNamespace(script_args=("x", "y", *values), scripts=scripts)


VALID = (True, 0.25, 6, 0.5, 2, 0.0, 6, 0.9)


def test_reads_exact_installed_slice_from_runner():
    runner = SimpleNamespace(alwayson_scripts=[_script()])
    assert read_native_spectrum(runner, _request(VALID)) == {
        "w": 0.25, "m": 6, "lam": 0.5, "window_size": 2,
        "flex_window": 0.0, "warmup_steps": 6, "stop_caching_step": 0.9,
    }


def test_gui_and_api_use_p_script_args_via_p_scripts_fallback():
    script_runner = SimpleNamespace(alwayson_scripts=[_script()])
    assert read_native_spectrum(SimpleNamespace(alwayson_scripts=[]), _request(VALID, script_runner))["m"] == 6


def test_disabled_native_slice_skips_even_when_truncated():
    runner = SimpleNamespace(alwayson_scripts=[_script()])
    assert read_native_spectrum(runner, _request((False,))) is None


@pytest.mark.parametrize("values", [
    (True,),
    ("true", 0.25, 6, 0.5, 2, 0.0, 6, 0.9),
    (True, float("nan"), 6, 0.5, 2, 0.0, 6, 0.9),
    (True, 1.01, 6, 0.5, 2, 0.0, 6, 0.9),
    (True, 0.25, 6.0, 0.5, 2, 0.0, 6, 0.9),
    (True, 0.25, 9, 0.5, 2, 0.0, 6, 0.9),
    (True, 0.25, 6, float("inf"), 2, 0.0, 6, 0.9),
    (True, 0.25, 6, 0.5, 0, 0.0, 6, 0.9),
    (True, 0.25, 6, 0.5, 2, 2.1, 6, 0.9),
    (True, 0.25, 6, 0.5, 2, 0.0, 21, 0.9),
    (True, 0.25, 6, 0.5, 2, 0.0, 6, -0.1),
])
def test_enabled_malformed_native_slice_is_explicit_error(values):
    runner = SimpleNamespace(alwayson_scripts=[_script()])
    with pytest.raises(ValueError, match="Forge Spectrum"):
        read_native_spectrum(runner, _request(values))
