"""Forge entrypoint for the version-bound Ideogram 4 compatibility route.

Uses Forge's native checkpoint, prompt, Generate, gallery, API and Settings.
Creates no extension accordion, model panel, private tab or private Generate.
Generate never downloads model files; Settings provides manual source links.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXT_DIR = Path(__file__).resolve().parent.parent
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))
_FORGE_ROOT = _EXT_DIR.parent.parent
if str(_FORGE_ROOT) not in sys.path:
    sys.path.append(str(_FORGE_ROOT))
_VENDORED = _FORGE_ROOT / "modules_forge/packages"
if _VENDORED.is_dir() and str(_VENDORED) not in sys.path:
    sys.path.append(str(_VENDORED))

import pi_ideogram_lib.settings as pi_settings  # noqa: E402,F401
from pi_ideogram_forge.paths import models_dirs as _models_dirs  # noqa: E402
from pi_ideogram_forge.selection import (  # noqa: E402,F401
    _ci_is_ideogram4, _current_ideogram_file, _resolve_ci_for_name,
    _selected_checkpoint_name, _selected_ci_is_i4, is_ideogram4_selected,
)
from pi_ideogram_forge.dropdown import install_dropdown_hook  # noqa: E402
from pi_ideogram_forge.routing import (  # noqa: E402,F401
    _enforce_loadable_policy, _policy_for_current, _routing_markdown,
)
from pi_ideogram_forge.run import (  # noqa: E402,F401
    _ACTIVE_LORAS, _maybe_download_adapters, _next_bypass_anchor,
    _ORIGINAL_RUN, _PIPELINE_CACHE, _resolve_run_opts, _wrapped_run,
    generate, install_wrapper,
)
from pi_ideogram_forge.generate import (  # noqa: E402,F401
    generate as _generate_fn, _looks_blocked, _resolve_sampling,
    _resolve_seed, _safe_comment,
)
from pi_ideogram_forge.native_settings import install_native_settings  # noqa: E402
from pi_ideogram_forge.ui_defaults import install as install_ui_defaults  # noqa: E402
import pi_ideogram_forge.lora as lora  # noqa: E402

TAG = "[Invisible-I4]"

try:
    from pi_ideogram_forge import legacy_preset as forge_preset  # noqa: E402
    forge_preset.install()
except Exception as _e:
    print(f"{TAG} WARNING: Forge preset compatibility unavailable: {_e}")

try:
    from modules import scripts
    _FORGE_OK = True
except Exception as _e:
    print(f"{TAG} WARNING: Forge imports failed: {_e}")
    scripts = None
    _FORGE_OK = False


def _probe_hardware_once() -> None:
    from pi_ideogram_lib.hardware import configure_once
    result = configure_once()
    if not result or not result.get("changed"):
        return
    probe, profile, defaults = result["probe"], result["profile"], result["defaults"]
    for warning in result.get("warnings", []):
        print(f"{TAG} WARNING: {warning}")
    print(f"{TAG} hardware: profile={profile} vram={probe.get('vram_total_gb')}GB "
          f"recommended quant={defaults['quant_auto']} max_size={defaults['max_size']} "
          f"uncond-LoRA={defaults['use_uncond_lora_default']}")


try:
    _probe_hardware_once()
except Exception:
    pass
try:
    lora.set_models_dirs(_models_dirs())
except Exception:
    pass


if _FORGE_OK and scripts is not None:

    class Ideogram4EngineScript(scripts.Script):
        _i4_alwayson = True

        def title(self):
            return "Ideogram 4 (Invisible)"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            return []


    install_native_settings()
    install_ui_defaults()
    install_wrapper()
    install_dropdown_hook()
