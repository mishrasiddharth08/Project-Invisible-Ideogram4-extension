"""Native Forge Settings bridge for the version-bound Ideogram 4 route."""

from __future__ import annotations

import html
import json
from pathlib import Path

from pi_ideogram_lib import config as pi_config
from pi_ideogram_lib import settings as pi_settings


PREFIX = "pi_i4_"
NATIVE_TO_CONFIG = {
    PREFIX + "preset": "preset",
    PREFIX + "quant_override": "quant_override",
    PREFIX + "uncond_mode": "uncond_mode",
    PREFIX + "json_pass_through": "json_pass_through",
    PREFIX + "prompt_upsampling": "prompt_upsampling",
    PREFIX + "bypass_lora_name": "bypass_lora_name",
    PREFIX + "bypass_lora_strength": "bypass_lora_strength",
    PREFIX + "adapter_choice": "adapter_choice",
    PREFIX + "adapter_strength": "adapter_strength",
    PREFIX + "chosen_variant": "chosen_variant",
    PREFIX + "text_encoder_override": "text_encoder_override",
    PREFIX + "lora_runtime": "lora_runtime",
    PREFIX + "auto_escalate_blocked": "auto_escalate_blocked",
    PREFIX + "oom_retry_uncond_lora": "oom_retry_uncond_lora",
    PREFIX + "allow_quality_reducing_oom_recovery": "allow_quality_reducing_oom_recovery",
}


def _legacy_defaults() -> dict:
    """Migrate existing extension config values; use native Auto on a fresh install."""
    cfg = pi_config.load()
    try:
        path = Path(pi_config.CONFIG_PATH)
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        has_legacy_preset = isinstance(raw, dict) and "preset" in raw
    except Exception:
        has_legacy_preset = False
    if not has_legacy_preset:
        cfg["preset"] = "Auto"
    return cfg


def _models_dirs():
    try:
        from pi_ideogram_forge.paths import models_dirs
        return models_dirs()
    except Exception:
        return []


def _choices(kind: str, current: str = "") -> list[str]:
    try:
        if kind == "quant":
            from pi_ideogram_lib.quant_registry import all_quant_ids
            values = ["auto", *all_quant_ids()]
        elif kind == "variant":
            from pi_ideogram_lib.quant_registry import all_quant_ids
            values = ["", *all_quant_ids()]
        elif kind == "adapter":
            from pi_ideogram_lib.detect import ADAPTER_NONE_LABEL, adapter_choices
            values = [ADAPTER_NONE_LABEL, *adapter_choices(_models_dirs())]
        else:
            from pi_ideogram_lib.detect import bypass_lora_choices
            values = [pi_settings.BYPASS_NONE_LABEL, *bypass_lora_choices(_models_dirs())]
    except Exception:
        values = ["auto"] if kind == "quant" else [""]
    if current and current not in values:
        values.append(current)
    return values


def _manual_setup_html() -> str:
    from pi_ideogram_assets.links import ADAPTERS, CHECKPOINTS, MODULES, UNCOND

    rows = []
    for label, filename, url, destination, note in (*CHECKPOINTS, *MODULES, *ADAPTERS, *UNCOND):
        rows.append(
            f"<li><a href='{html.escape(url, quote=True)}' target='_blank'>{html.escape(label)}</a>"
            f" — <code>{html.escape(filename)}</code> → <code>{html.escape(destination)}</code>"
            f"<br><small>{html.escape(note)}</small></li>"
        )
    return (
        "<b>Manual model setup only.</b> Generate never downloads files. "
        "This is a community, version-bound compatibility route because this Forge Neo host has no "
        "native Ideogram backend/preset. Save files to the named folders, then refresh Forge's model lists."
        "<ul>" + "".join(rows) + "</ul>"
    )


def named_options(p=None) -> dict:
    """Return one request snapshot. API override_settings wins over saved Forge options."""
    cfg = _legacy_defaults()
    try:
        from modules import shared
        saved = getattr(shared.opts, "data", {}) or {}
    except Exception:
        saved = {}
    overrides = getattr(p, "override_settings", None) or {}
    for native_name, config_name in NATIVE_TO_CONFIG.items():
        if native_name in overrides:
            cfg[config_name] = overrides[native_name]
        elif config_name in overrides:  # legacy/API compatibility
            cfg[config_name] = overrides[config_name]
        elif native_name in saved:
            cfg[config_name] = saved[native_name]

    # Extension-owned approximations are forbidden. Forge's built-in Spectrum
    # remains a host concern and is not claimed compatible with this takeover.
    cfg["spectrum_enabled"] = False
    cfg["first_block_cache_enabled"] = False
    return cfg


def resolve_run_options(p=None) -> dict:
    return pi_settings.resolve(cfg=named_options(p))


def register_native_settings() -> None:
    """Register persistent controls in Forge Settings (also Quicksettings-eligible)."""
    try:
        import gradio as gr
        from modules import shared
    except Exception:
        return

    cfg = _legacy_defaults()
    generation = ("pi_i4_generation", "Ideogram 4 compatibility")
    advanced = ("pi_i4_setup", "Ideogram 4 setup & advanced")

    def add(name, info):
        info.category_id = "sd"
        if name not in shared.opts.data_labels:
            shared.opts.add_option(name, info)

    add(PREFIX + "preset", shared.OptionInfo(
        cfg.get("preset", "Auto"), "Recipe", gr.Dropdown,
        {"choices": ["Auto", "V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12", "TurboTime-4", "Instant-8"]},
        section=generation).info("Auto follows Forge's native Steps and CFG; named recipes apply Ideogram schedules."))
    add(PREFIX + "quant_override", shared.OptionInfo(
        cfg.get("quant_override", "auto"), "Quantization override", gr.Dropdown,
        lambda: {"choices": _choices("quant", str(cfg.get("quant_override") or ""))}, section=generation))
    add(PREFIX + "uncond_mode", shared.OptionInfo(
        cfg.get("uncond_mode", "Auto"), "Unconditional path", gr.Dropdown,
        {"choices": ["Auto", "Full transformer", "Ostris uncond LoRA"]}, section=generation))
    add(PREFIX + "json_pass_through", shared.OptionInfo(
        bool(cfg.get("json_pass_through", False)), "Use raw prompt (skip structured JSON)", section=generation))
    add(PREFIX + "prompt_upsampling", shared.OptionInfo(
        bool(cfg.get("prompt_upsampling", False)), "Enable prompt enhancement", section=generation))
    add(PREFIX + "adapter_choice", shared.OptionInfo(
        cfg.get("adapter_choice") or "", "Adapter", gr.Dropdown,
        lambda: {"choices": _choices("adapter", str(cfg.get("adapter_choice") or ""))}, section=generation))
    add(PREFIX + "adapter_strength", shared.OptionInfo(
        float(cfg.get("adapter_strength", 1.0)), "Adapter strength", gr.Slider,
        {"minimum": 0.0, "maximum": 1.5, "step": 0.05}, section=generation))
    add(PREFIX + "bypass_lora_name", shared.OptionInfo(
        cfg.get("bypass_lora_name") or pi_settings.BYPASS_NONE_LABEL, "Filter-bypass adapter", gr.Dropdown,
        lambda: {"choices": _choices("bypass", str(cfg.get("bypass_lora_name") or ""))}, section=generation))
    add(PREFIX + "bypass_lora_strength", shared.OptionInfo(
        float(cfg.get("bypass_lora_strength", -0.25)), "Filter-bypass strength", gr.Slider,
        {"minimum": -1.0, "maximum": 0.25, "step": 0.05}, section=generation))

    add(PREFIX + "setup_info", shared.OptionHTML(_manual_setup_html()))
    shared.opts.data_labels[PREFIX + "setup_info"].section = advanced
    shared.opts.data_labels[PREFIX + "setup_info"].category_id = "sd"
    add(PREFIX + "spectrum_info", shared.OptionHTML(
        "<b>Acceleration:</b> Forge Neo's installed Spectrum controls are bridged to its real "
        "<code>SpectrumNode.patch</code> implementation. This is a version-bound compatibility bridge, "
        "not a claim that Ideogram has a full native Forge engine. No private Spectrum controls are added."))
    shared.opts.data_labels[PREFIX + "spectrum_info"].section = advanced
    shared.opts.data_labels[PREFIX + "spectrum_info"].category_id = "sd"
    add(PREFIX + "chosen_variant", shared.OptionInfo(
        cfg.get("chosen_variant") or "", "Preferred model variant", gr.Dropdown,
        lambda: {"choices": _choices("variant", str(cfg.get("chosen_variant") or ""))}, section=advanced))
    add(PREFIX + "text_encoder_override", shared.OptionInfo(
        cfg.get("text_encoder_override", ""), "Text encoder file/folder override", gr.Textbox, section=advanced))
    add(PREFIX + "lora_runtime", shared.OptionInfo(
        bool(cfg.get("lora_runtime", True)), "Use exact runtime LoRA path", section=advanced))
    add(PREFIX + "auto_escalate_blocked", shared.OptionInfo(
        bool(cfg.get("auto_escalate_blocked", False)), "Retry once after a blocked-image result", section=advanced))
    add(PREFIX + "oom_retry_uncond_lora", shared.OptionInfo(
        bool(cfg.get("oom_retry_uncond_lora", True)), "Retry OOM with unconditional LoRA path", section=advanced))
    add(PREFIX + "allow_quality_reducing_oom_recovery", shared.OptionInfo(
        bool(cfg.get("allow_quality_reducing_oom_recovery", False)), "Allow quality-reducing OOM recovery", section=advanced))


def install_native_settings() -> None:
    try:
        from modules import script_callbacks
        script_callbacks.on_ui_settings(register_native_settings)
    except Exception:
        pass
