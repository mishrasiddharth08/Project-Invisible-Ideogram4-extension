"""PROJECT INVISIBLE - SINGLE OWNER of resolved run settings.

Every Ideogram 4 generation runs from ONE dict produced by resolve() here.
Nothing else decides what a run does: generate.py reads only this dict,
engine.py builds it from the accordion UI (or config defaults) and persists
UI changes through persist(), install.py reads config for boot-time defaults.

Ownership map (see ARCHITECTURE.md):
  config.py            -> raw persistent store (key/value, hand-editable)
  settings.py          -> UI intent -> resolved run settings (THIS module)
  generate.py          -> consumes a resolved settings dict, nothing else
  engine.py            -> builds settings from Gradio values + owns Forge state

Data flow (one direction, no cycles):
  Gradio accordion values (p.script_args slice)
      -> settings.from_ui(values)
      -> settings.resolve(cfg, ui_values)   # full dict, typed + defaulted
      -> generate(p, opts=that_dict)
  The accordion is the ONLY source of per-run intent; config.json supplies
  defaults for first runs and for direct (non-UI) callers.

This module has no Forge imports. config is imported lazily so tests and
standalone probes never need the extension root on sys.path at import time.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# UI component order. MUST match Script.ui() return order AND Script.process()
# (both live in scripts/engine.py). Kept here so engine + tests share ONE list.
# --------------------------------------------------------------------------- #
# The filter-bypass panel is deliberately TWO controls: pick a LoRA, set its
# strength. It used to be eight (on/off, a master mix, and six "advanced"
# levers), which was a lot of surface for something most people touch once.
#
# The simplification is also the more correct design. A bypass LoRA is a
# weight delta applied for one step - it does not touch the sampler, the
# scheduler or the step count, so the official recipe stays exactly official
# with the panel in use. The old master slider drove sigma smoothing, which
# INSERTED steps ("Default 20" really sampled 22) - a silent deviation nobody
# asked for. The levers still exist in bypass.py for anyone who wants them
# from config.json; they are simply no longer in the way.
#
# The density guard (three bounding boxes in the JSON caption) is NOT a lever
# here: it is prompt-side, always on, and it is what actually beats the gray
# card in most cases. See pi_ideogram_lib/decompose.py.
UI_KEYS: tuple[str, ...] = (
    "preset",
    "quant_override",
    "uncond_mode",
    "json_pass_through",
    "prompt_upsampling",
    "bypass_lora_name",
    "bypass_lora_strength",
    # The adapter picker: any Ideogram 4 adapter, applied exactly as a
    # <lora:name:strength> tag would be. Distinct from bypass_lora_name, which
    # lists filter-bypass adapters ONLY and drives the first-step bypass path.
    "adapter_choice",
    "adapter_strength",
    # Spectrum: forecast some steps instead of computing them. A UI component
    # rather than config-only because it is a speed/exactness trade a person
    # makes per run, and because config-only levers are invisible - this one
    # shipped unreachable and nobody could tell it was never firing.
    "spectrum_enabled",
)

# Shown when no bypass LoRA is selected.
BYPASS_NONE_LABEL = "None (official recipe, no bypass)"

PRESET_AUTO_LABEL = "Auto (follow steps slider)"
PRESET_CHOICES = ["Auto (follow steps slider)", "V4_QUALITY_48", "V4_DEFAULT_20",
                  "V4_TURBO_12", "TurboTime-4", "Instant-8"]

# extra keys resolve() adds that are NOT UI components (engine/generate state)
_DERIVED_KEYS = ("sigma_smooth_steps", "use_uncond_lora", "turbo", "debanner_enabled",
                 "debanner_strength", "magic_prompt_enabled", "magic_prompt_provider",
                 "ideogram_api_key", "openrouter_api_key", "lora_runtime")

_SIGMA_SMOOTH_STEPS = 2  # subdivide the first sigma interval into N jumps (P2)


def _load_cfg() -> dict:
    """Current config.json content (never raises; defaults on any failure)."""
    try:
        import pi_ideogram_lib.config as _cfg_mod
        return _cfg_mod.load()
    except Exception:
        return {}


def _ui_default(cfg: dict, ui_key: str) -> Any:
    """Map a config key to the matching UI default. Config uses the panel
    wording filter_bypass_enabled for the UI key filter_bypass."""
    key = ui_key
    if ui_key == 'spectrum_enabled' and cfg.get('first_block_cache_enabled', False):
        return 'first_block'
    if ui_key == "filter_bypass":
        key = "filter_bypass_enabled"
    val = cfg.get(key)
    if val is None:
        # fall back to config.DEFAULTS so keys missing from an old config.json
        # still produce the same defaults the UI showed before
        try:
            import pi_ideogram_lib.config as _cfg_mod
            val = _cfg_mod.DEFAULTS.get(key)
        except Exception:
            val = None
    if ui_key == "preset" and val == "Auto":
        return PRESET_AUTO_LABEL
    return val


def ui_defaults(cfg: dict | None = None) -> dict:
    """UI component default values, read from config.json once. Keys == UI_KEYS."""
    cfg = cfg if cfg is not None else _load_cfg()
    out: dict = {}
    for k in UI_KEYS:
        out[k] = _ui_default(cfg, k)
    return out


def _bypass_on(cfg: dict) -> bool:
    """Is filter bypass actually engaged, per config.json?

    Named LoRA + master switch + a non-'None' selection. Empty name = off =
    the official recipe untouched, which is what the dropdown's default label
    already says; this makes config.json say the same thing.
    """
    name = str(cfg.get("bypass_lora_name") or "").strip()
    if not name or name == BYPASS_NONE_LABEL:
        return False
    return bool(cfg.get("filter_bypass_enabled", True)) and bool(cfg.get("bypass_lora", True))


def from_ui(values, cfg: dict | None = None) -> dict:
    """Raw Gradio component values -> typed dict with keys == UI_KEYS (+ the
    derived sigma_smooth_steps). `values` must be exactly len(UI_KEYS).

    Strict on purpose. A wrong-length tuple means the caller and UI_KEYS
    disagree about ORDER, so every value would be read off the wrong
    component - a preset landing in the quant field, a strength landing in a
    checkbox. resolve() catches this and falls back to config defaults, which
    is a worse run than the user asked for but a CORRECT one. Padding a short
    tuple was tried and reverted: it accepted `("only-one",)` and used the
    garbage as the preset instead of falling back.
    """
    values = tuple(values)
    if len(values) != len(UI_KEYS):
        raise ValueError(f"expected {len(UI_KEYS)} UI values, got {len(values)}")
    cfg = cfg if cfg is not None else _load_cfg()

    preset = str(values[0])
    if preset == PRESET_AUTO_LABEL:
        preset = "Auto"

    adapter_choice = str(values[7] or "")
    adapter_strength = float(values[8]) if adapter_choice else 1.0
    lora_name = str(values[5] or "")
    lora_on = bool(lora_name) and lora_name != BYPASS_NONE_LABEL
    lora_strength = float(values[6]) if lora_on else 0.0

    # The two visible controls drive the internal flags. Everything that would
    # change the SAMPLING SCHEDULE stays off: a bypass LoRA is a one-step
    # weight delta, so the official steps/CFG/shift are untouched while it is
    # in use. Advanced levers remain reachable from config.json for anyone who
    # wants them, and default off.
    cfg_get = cfg.get
    # Slot 9 remains backward compatible with the former Spectrum checkbox.
    # New dropdown values choose ONE approximation, never silently stack them.
    acceleration = values[9]
    if acceleration not in (True, False, None, 'off', 'spectrum', 'first_block'):
        raise ValueError('Unknown acceleration mode')
    return {
        "preset": preset,
        "quant_override": str(values[1]),
        "uncond_mode": str(values[2]),
        "json_pass_through": bool(values[3]),
        "prompt_upsampling": bool(values[4]),
        "bypass_lora_name": lora_name if lora_on else "",
        "lora_runtime": bool(cfg_get("lora_runtime", True)),
        "bypass_lora_strength": lora_strength,
        "adapter_choice": adapter_choice,
        "adapter_strength": adapter_strength,
        "spectrum_enabled": acceleration is True or acceleration == 'spectrum',
        "first_block_cache_enabled": acceleration == 'first_block',
        # derived
        "filter_bypass": lora_on,
        "bypass_lora": lora_on,
        "bypass_strength": 0.0,
        "sigma_smooth": bool(cfg_get("sigma_smooth", False)),
        "sigma_smooth_steps": _SIGMA_SMOOTH_STEPS if cfg_get("sigma_smooth", False) else 0,
        "layer_dampen": bool(cfg_get("layer_dampen", False)),
        "dampen_cond_mult": _num(cfg, "dampen_cond_mult", 0.4),
        "dampen_uncond_mult": _num(cfg, "dampen_uncond_mult", 0.1),
    }


def chosen_variant(cfg: dict | None = None) -> str:
    """The variant the spec-F picker targets: config.chosen_variant when set,
    else the registry's recommendation for the probed GPU. Never None."""
    cfg = cfg if cfg is not None else _load_cfg()
    chosen = str(cfg.get("chosen_variant") or "")
    if chosen:
        return chosen
    try:
        from pi_ideogram_lib.quant_registry import recommended_for_vram
        return recommended_for_vram(cfg.get("hw_vram_gb"))
    except Exception:
        return "fp8_scaled"


def persist_chosen_variant(quant_id: str, cfg: dict | None = None) -> None:
    """Picker 'variant' dropdown change -> config.json (same persist flow as
    the accordion values; resolve() then carries it at generate time)."""
    cfg = cfg if cfg is not None else _load_cfg()
    try:
        import pi_ideogram_lib.config as _cfg_mod
        _cfg_mod.save({"chosen_variant": str(quant_id)})
    except Exception:
        pass


# Below this, "Auto" prefers ONE transformer + the uncond LoRA. A 32 GB card
# fits the dual path and still runs it badly (94% occupancy, 2.8x slower
# forwards), so the bar is comfort, not capacity.
DUAL_TRANSFORMER_COMFORT_GB = 40.0


def _uncond_decision(cfg: dict) -> bool:
    """True when the run uses the Ostris uncond LoRA instead of the 9.3B
    transformer. Order: explicit mode wins, else the VRAM-profile default."""
    mode = str(cfg.get("uncond_mode") or "Auto")
    if mode == "Ostris uncond LoRA":
        return True
    if mode == "Full transformer":
        return False
    forced = cfg.get("use_uncond_lora_auto", None)
    if forced is not None:
        return bool(forced)

    # AUTO, decided from the card. The two-transformer path holds 2 x 9.3 GB
    # of weights; measured on a 31.8 GB RTX 5090 at 1024x1024 it peaked at
    # 30.02 GB - 94% occupancy - and the forwards ran ~2.8x slower than the
    # same layers benchmarked with the card free (2.42 s vs 0.85 s). That is
    # allocator pressure, not arithmetic: "fits" and "runs well" are not the
    # same threshold.
    #
    # The uncond-replacement LoRA gives the reference implementation's shape -
    # ONE transformer, the unconditional pass through the same weights - for
    # ~9 GB less. Below DUAL_TRANSFORMER_COMFORT_GB there is no headroom to
    # spend on a second copy, so Auto takes the single-model path.
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return False
    return total_gb < DUAL_TRANSFORMER_COMFORT_GB


def _resolve_base(cfg: dict) -> dict:
    preset = str(cfg.get("preset") or "V4_DEFAULT_20")
    if preset == PRESET_AUTO_LABEL or preset == "Auto":
        preset = "Auto"
    sigma_smooth = bool(cfg.get("sigma_smooth", True))
    return {
        "preset": preset,
        "quant_override": str(cfg.get("quant_override") or "auto"),
        "uncond_mode": str(cfg.get("uncond_mode") or "Auto"),
        "json_pass_through": bool(cfg.get("json_pass_through", False)),
        "prompt_upsampling": bool(cfg.get("prompt_upsampling", False)),
        # The NAME is the authority, exactly as in from_ui(): "no LoRA picked"
        # is the same statement however the settings arrived. Two independent
        # booleans defaulting True while the name defaulted "" meant the config
        # path (API calls, and any run without a UI slice) reported
        # "bypass on, nothing named" - enough for wanted_adapters() to request
        # the gray adapter and start a Civitai download for a run the user had
        # not asked to bypass anything.
        "filter_bypass": _bypass_on(cfg),
        # Default 0.0 = the OFFICIAL sampling schedule. Anything above 0 inserts
        # extra interpolated sigmas, so "Default 20" would sample 22 - a silent
        # deviation from the published recipe (ideogram-oss/ComfyUI-Ideogram4:
        # 20 steps = 18 main + 2 polish, CFG 7->3, mu 0.0, std 1.75).
        # NB `or` would swallow a deliberate 0.0, so test for None explicitly.
        "bypass_strength": _num(cfg, "bypass_strength", 0.0),
        "bypass_lora": _bypass_on(cfg),
        "bypass_lora_strength": _num(cfg, "bypass_lora_strength", -0.25),
        "bypass_lora_name": str(cfg.get("bypass_lora_name") or ""),
        "adapter_choice": str(cfg.get("adapter_choice") or ""),
        "adapter_strength": _num(cfg, "adapter_strength", 1.0),
        "lora_runtime": bool(cfg.get("lora_runtime", True)),
        "sigma_smooth": sigma_smooth,
        "sigma_smooth_steps": _SIGMA_SMOOTH_STEPS if sigma_smooth else 0,
        "layer_dampen": bool(cfg.get("layer_dampen", False)),
        "dampen_cond_mult": _num(cfg, "dampen_cond_mult", 0.4),
        "dampen_uncond_mult": _num(cfg, "dampen_uncond_mult", 0.1),
        # ---- derived (non-UI) ------------------------------------------------
        "use_uncond_lora": _uncond_decision(cfg),
        "turbo": str(cfg.get("preset") or "") in ("TurboTime-4", "Instant-8"),
        # OFF unless explicitly enabled: see config.py DEFAULTS for why.
        "debanner_enabled": bool(cfg.get("debanner_enabled", False)),
        "debanner_strength": _num(cfg, "debanner_strength", 0.6),
        # OFF unless explicitly enabled: see config.py DEFAULTS for why.
        "spectrum_enabled": bool(cfg.get("spectrum_enabled", False)),
        "first_block_cache_enabled": bool(cfg.get("first_block_cache_enabled", False)) and not bool(cfg.get("spectrum_enabled", False)),
        "allow_quality_reducing_oom_recovery": bool(cfg.get("allow_quality_reducing_oom_recovery", False)),
        "spectrum_degree": int(_num(cfg, "spectrum_degree", 1)),
        "spectrum_ridge": _num(cfg, "spectrum_ridge", 0.1),
        "spectrum_history": int(_num(cfg, "spectrum_history", 6)),
        "spectrum_warmup": int(_num(cfg, "spectrum_warmup", 4)),
        "spectrum_tail": int(_num(cfg, "spectrum_tail", 2)),
        "spectrum_stride": int(_num(cfg, "spectrum_stride", 3)),
        "spectrum_blend": _num(cfg, "spectrum_blend", 1.0),
        "spectrum_max_growth": _num(cfg, "spectrum_max_growth", 2.0),
        "magic_prompt_enabled": bool(cfg.get("magic_prompt_enabled", False)),
        "magic_prompt_provider": str(cfg.get("magic_prompt_provider", "ideogram-4-v1")),
        "ideogram_api_key": str(cfg.get("ideogram_api_key", "")),
        "openrouter_api_key": str(cfg.get("openrouter_api_key", "")),
        "chosen_variant": chosen_variant(cfg),
    }


def _num(cfg: dict, key: str, default: float) -> float:
    """Read a numeric setting WITHOUT swallowing a deliberate 0.

    `float(cfg.get(k, d) or d)` looks harmless and is not: `0.0 or 0.35` is
    0.35, so a user who set the value to zero silently got the default back.
    Zero is meaningful for every setting this is used on - bypass_strength 0
    means the official schedule, dampen mult 0 means skip the block,
    debanner_strength 0 means off - so those settings were simply unreachable.
    """
    val = cfg.get(key)
    if val is None or val == "":
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def resolve(cfg: dict | None = None, ui_values=None) -> dict:
    """FULL resolved run settings. ui_values (optional, raw Gradio values)
    override the config-derived defaults. Never raises: any failure falls back
    to config defaults (and config.load() never raises)."""
    cfg = cfg if cfg is not None else _load_cfg()
    opts = _resolve_base(cfg)
    if ui_values is not None:
        try:
            ui = from_ui(ui_values, cfg)
            ui["use_uncond_lora"] = _uncond_decision(dict(cfg, uncond_mode=ui["uncond_mode"]))
            ui["turbo"] = ui["preset"] in ("TurboTime-4", "Instant-8")
            # CONFIG-ONLY LEVERS RIDE ALONG WITH THE UI RUN.
            #
            # `opts = ui` REPLACES the config-derived dict, so anything
            # _resolve_base computed that from_ui does not produce is dropped.
            # This used to be a hand-written list of seven keys, and a
            # hand-written list drifts: every config-only setting added after
            # it was silently discarded on every UI run - which is every real
            # generation. Spectrum shipped that way and could not be switched
            # on at all, enabled in config.json or not.
            #
            # setdefault, not assignment: a UI component always outranks the
            # config value behind it.
            for _k, _v in opts.items():
                ui.setdefault(_k, _v)
            opts = ui
        except Exception:
            pass  # malformed values -> config defaults (never crash a generate)
    return opts


def persist(ui_values, cfg: dict | None = None) -> None:
    """Remember last-used UI values in config.json (spec M: persist last
    quant/preset/bypass). ui_values is the raw len(UI_KEYS) tuple."""
    ui_values = tuple(ui_values)
    if len(ui_values) != len(UI_KEYS):
        return
    ui = from_ui(ui_values)
    updates = {
        "preset": ui["preset"],
        "quant_override": ui["quant_override"],
        "uncond_mode": ui["uncond_mode"],
        "json_pass_through": ui["json_pass_through"],
        "prompt_upsampling": ui["prompt_upsampling"],
        "filter_bypass_enabled": ui["filter_bypass"],
        "bypass_strength": ui["bypass_strength"],
        "bypass_lora": ui["bypass_lora"],
        "bypass_lora_strength": ui["bypass_lora_strength"],
        "adapter_choice": ui.get("adapter_choice", ""),
        "adapter_strength": ui.get("adapter_strength", 1.0),
        "spectrum_enabled": ui.get("spectrum_enabled", False),
        "first_block_cache_enabled": ui.get("first_block_cache_enabled", False),
        "bypass_lora_name": ui.get("bypass_lora_name", ""),
        "sigma_smooth": ui["sigma_smooth"],
        "layer_dampen": ui["layer_dampen"],
        "dampen_cond_mult": ui["dampen_cond_mult"],
        "dampen_uncond_mult": ui["dampen_uncond_mult"],
    }
    try:
        import pi_ideogram_lib.config as _cfg_mod
        _cfg_mod.save(updates)
    except Exception:
        pass
