"""Add Ideogram4 to Forge's native UI Preset without modifying enum members.

Forge's preset handler uses string option keys, so an additive public choices
wrapper and engine-owned options are sufficient. Core files, stock defaults,
enum membership and sibling extensions' recipe tables remain untouched.
"""
from __future__ import annotations

from copy import copy, deepcopy
from functools import wraps

PRESET_NAME = "ideogram4"
TAG = "[Invisible-I4]"


def _wrap_choices(arch):
    original = arch.choices
    if getattr(original, "_i4_injected", False):
        return

    @wraps(original)
    def choices(*args, **kwargs):
        result = list(original(*args, **kwargs))
        if PRESET_NAME not in result:
            result.append(PRESET_NAME)
        return result

    choices._i4_injected = True
    arch.choices = staticmethod(choices)


def _register_options(presets, opts):
    # Copy the plain-image SD template using Forge's real component classes.
    # Do NOT add an enum member: later sibling register() calls must stay safe.
    template = {}
    presets.register(template)
    ours = {}
    for key, info in template.items():
        if key.startswith("sd_"):
            target = PRESET_NAME + key[2:]
        elif key in ("forge_checkpoint_sd", "forge_additional_modules_sd",
                     "forge_unet_storage_dtype_sd"):
            target = key[:-2] + PRESET_NAME
        else:
            continue
        cloned = copy(info)
        cloned.default = deepcopy(info.default)
        if getattr(info, "section", (None,))[0] == "ui_sd":
            cloned.section = ("ui_ideogram4", "IDEOGRAM4")
        if target.endswith("_sampler"):
            cloned.default = "Euler"
        elif target.endswith("_scheduler"):
            cloned.default = "Simple"
        elif target.endswith("_step"):
            cloned.default = 20
        elif target.endswith("_cfg"):
            cloned.default = 7.0
        elif target.endswith(("_width", "_height")):
            cloned.default = 1024
        if cloned.component is not None and not callable(cloned.component):
            raise TypeError(f"Unsupported Forge setting component: {target}")
        ours[target] = cloned
    required = {"forge_checkpoint_ideogram4", "forge_additional_modules_ideogram4",
                "forge_unet_storage_dtype_ideogram4", "ideogram4_t2i_step",
                "ideogram4_t2i_cfg", "ideogram4_t2i_width", "ideogram4_t2i_height"}
    if not required.issubset(ours):
        raise RuntimeError("Forge's preset option layout changed; registration skipped")
    for key, info in ours.items():
        if key not in opts.data_labels:
            opts.add_option(key, info)


def _report(*args, **kwargs):
    try:
        from modules_forge import main_entry
        raw = main_entry.ui_forge_preset.choices
        names = [c[1] if isinstance(c, (tuple, list)) else c for c in raw]
        status = "OK" if PRESET_NAME in names else "MISSING"
        print(f"{TAG} UI Preset 'ideogram4' served to browser: {status}")
    except Exception as error:
        print(f"{TAG} UI Preset verification unavailable: {error}")


def install() -> bool:
    """Register at script import, before Forge builds its UI. Safe to repeat."""
    try:
        from modules import shared
        from modules_forge import presets
        arch = presets.PresetArch
        arch.choices()  # Probe compatibility before touching anything.
        _register_options(presets, shared.opts)
        _wrap_choices(arch)
        key = "forge_checkpoint_ideogram4"
        current = getattr(shared.opts, "sd_model_checkpoint", None)
        if not shared.opts.data.get(key) and current and "ideogram" in str(current).lower():
            shared.opts.data[key] = current
            # Preserve this checkpoint's active modules on the first switch.
            modules_key = "forge_additional_modules_ideogram4"
            if not shared.opts.data.get(modules_key):
                shared.opts.data[modules_key] = list(
                    getattr(shared.opts, "forge_additional_modules", None) or [])
        # Store the registration marker on the host class, not this module:
        # script/module reloads must not accumulate callbacks or choice entries.
        if not getattr(arch, "_i4_report_installed", False):
            try:
                from modules import script_callbacks
                script_callbacks.on_app_started(_report)
                arch._i4_report_installed = True
            except (ImportError, AttributeError):
                pass
        print(f"{TAG} UI Preset 'ideogram4' registered (no Forge core-file edits)")
        return True
    except Exception as error:
        print(f"{TAG} WARNING: Ideogram4 UI Preset registration failed: {error}")
        return False
