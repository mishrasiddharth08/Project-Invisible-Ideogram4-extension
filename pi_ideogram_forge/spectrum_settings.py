"""Read Forge Neo's installed Spectrum controls from the native script slice."""

from __future__ import annotations

import math
import numbers


_KEYS = ("w", "m", "lam", "window_size", "flex_window", "warmup_steps", "stop_caching_step")


def _spectrum_script(runner, p):
    sources = (runner, getattr(p, "scripts", None))
    for source in sources:
        scripts = getattr(source, "alwayson_scripts", None)
        if scripts is None and isinstance(source, (list, tuple)):
            scripts = source
        for script in scripts or ():
            if type(script).__name__ == "SpectrumForForge":
                return script
    return None


def _number(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"Forge Spectrum {name} must be a finite number from {low} to {high}.")
    value = float(value)
    if not low <= value <= high:
        raise ValueError(f"Forge Spectrum {name} must be from {low} to {high}; received {value}.")
    return value


def _integer(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"Forge Spectrum {name} must be an integer from {low} to {high}.")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"Forge Spectrum {name} must be from {low} to {high}; received {value}.")
    return value


def read_native_spectrum(runner, p):
    """Return validated installed-Spectrum settings, or None when it is disabled."""
    script = _spectrum_script(runner, p)
    if script is None:
        return None

    args = getattr(p, "script_args", ()) or ()
    start = getattr(script, "args_from", None)
    if not isinstance(start, int) or isinstance(start, bool) or start < 0 or start >= len(args):
        return None

    enabled = args[start]
    if enabled is False:
        return None
    if enabled is not True:
        raise ValueError("Forge Spectrum enable must be true or false.")

    values = tuple(args[start:start + 8])
    if len(values) != 8:
        raise ValueError("Forge Spectrum is enabled, but its native 8-value script_args slice is incomplete.")

    _, w, m, lam, window, flex, warmup, stop = values
    checked = (
        _number("w", w, 0.0, 1.0),
        _integer("m", m, 1, 8),
        _number("lam", lam, 0.0, 2.0),
        _integer("window_size", window, 1, 10),
        _number("flex_window", flex, 0.0, 2.0),
        _integer("warmup_steps", warmup, 0, 20),
        _number("stop_caching_step", stop, 0.0, 1.0),
    )
    return dict(zip(_KEYS, checked))
