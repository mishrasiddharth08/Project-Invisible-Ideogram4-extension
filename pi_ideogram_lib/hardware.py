"""PROJECT INVISIBLE - hardware autodetection for Ideogram 4.

Runs once at first launch (and re-runs when the GPU/driver changes, detected
via a fingerprint in config.json). Everything degrades gracefully: no torch,
no CUDA, unreadable files -> safe defaults, never an exception.

Profiles come from the official + community benches baked into the registry
(see quant_registry.py); the probe here only supplies measured numbers.
"""

from __future__ import annotations

import shutil
from pathlib import Path

TAG = "[Invisible-I4]"


def probe() -> dict:
    """Measure the machine. Never raises. All values are floats (GB) or None."""
    out: dict = {
        "vram_total_gb": None,
        "vram_free_gb": None,
        "ram_gb": None,
        "disk_free_gb": None,
        "cuda_cap": None,        # (major, minor) or None
        "cuda_driver": None,
        "device_name": None,
        "cuda_available": False,
    }
    # --- VRAM / CUDA -------------------------------------------------------
    try:
        import torch
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            out['device_index'] = index
            out['backend'] = 'rocm' if getattr(torch.version, 'hip', None) else 'cuda'
            out["cuda_available"] = True
            out["device_name"] = getattr(props, "name", "CUDA device")
            out["vram_total_gb"] = round(props.total_memory / 1024**3, 1)
            try:
                free, _total = torch.cuda.mem_get_info(index)
                out["vram_free_gb"] = round(free / 1024**3, 1)
            except Exception:
                pass
            try:
                major = int(getattr(props, "major", 0))
                minor = int(getattr(props, "minor", 0))
                out["cuda_cap"] = (major, minor)
            except Exception:
                pass
            try:
                out["cuda_driver"] = torch.cuda.get_device_name(index)
            except Exception:
                pass
    except Exception:
        pass
    # --- system RAM --------------------------------------------------------
    try:
        import os
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                out["ram_gb"] = round(pages * page_size / 1024**3, 1)
    except Exception:
        pass
    if out["ram_gb"] is None:
        try:
            import psutil  # Forge ships psutil
            out["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
        except Exception:
            pass
    # --- free disk on the models volume ------------------------------------
    for d in _model_dirs_candidates():
        try:
            usage = shutil.disk_usage(d)
            out["disk_free_gb"] = round(usage.free / 1024**3, 1)
            break
        except Exception:
            continue
    return out


def _model_dirs_candidates() -> list[Path]:
    """Best-effort list of directories whose volume hosts the models."""
    cands: list[Path] = []
    try:
        from modules.paths_internal import models_path
        cands.append(Path(models_path))
    except Exception:
        pass
    # extension-relative fallbacks
    try:
        ext = Path(__file__).resolve().parent.parent
        cands.append(ext.parent.parent / "models")
    except Exception:
        pass
    try:
        cands.append(Path.cwd())
    except Exception:
        pass
    return cands


def vram_profile(vram_gb: float | None) -> str:
    """Bucket measured VRAM into the spec's profile names.

    Profiles recommend portable FP8 and full guidance at every VRAM tier.
    Memory management streams blocks below approximately 24 GiB. These are
    recommendations, not physical validation of every GPU or resolution.
    """
    if vram_gb is None or vram_gb <= 0:
        return "unknown"
    if vram_gb <= 8.0:
        return "<=8"
    if vram_gb < 12.0:
        return "8-12"
    if vram_gb < 16.0:
        return "12-16"
    if vram_gb < 24.0:
        return "16-24"
    return ">=24"


def profile_defaults(profile: str) -> dict:
    """Sensible defaults per profile. User override wins (engine reads config)."""
    table = {
        "<=8": {
            "quant_auto": "fp8_scaled",       # in-process block streaming
            "max_size": 512,
            "turbo_default": False,
            "use_uncond_lora_default": False,
            "quality_preset": "V4_DEFAULT_20",
            "te_quantized": True,
        },
        "8-12": {
            "quant_auto": "fp8_scaled",       # portable on RTX 30/40/50
            "max_size": 1024,
            "turbo_default": False,
            "use_uncond_lora_default": False,
            "quality_preset": "V4_DEFAULT_20",
            "te_quantized": True,
        },
        "12-16": {
            "quant_auto": "fp8_scaled",
            "max_size": 1024,
            "turbo_default": False,
            "use_uncond_lora_default": False,  # streamed original uncond
            "quality_preset": "V4_DEFAULT_20",
            "te_quantized": False,
        },
        "16-24": {
            "quant_auto": "fp8_scaled",
            "max_size": 1536,
            "turbo_default": False,
            "use_uncond_lora_default": False,  # full dual transformer
            "quality_preset": "V4_DEFAULT_20",
            "te_quantized": False,
        },
        ">=24": {
            "quant_auto": "fp8_scaled",
            "max_size": 2048,
            "turbo_default": False,
            "use_uncond_lora_default": False,  # dual transformer pinned
            "quality_preset": "V4_QUALITY_48",
            "te_quantized": False,
        },
        "unknown": {
            "quant_auto": "fp8_scaled",
            "max_size": 1024,
            "turbo_default": False,
            "use_uncond_lora_default": False,
            "quality_preset": "V4_DEFAULT_20",
            "te_quantized": False,
        },
    }
    return table.get(profile, table["unknown"])


def fingerprint(p: dict) -> str:
    """Stable machine fingerprint; config saves it and re-probes when it changes."""
    return "|".join(str(p.get(k, "")) for k in
                    ("vram_total_gb", "cuda_cap", "device_name", "ram_gb"))


def check_warnings(p: dict) -> list[str]:
    """User-facing warnings (system RAM, disk, Blackwell-only quants). Never raises."""
    warns: list[str] = []
    if p.get("cuda_available") and p.get("ram_gb") is not None and p['ram_gb'] < 32:
        warns.append(f"System RAM {p.get('ram_gb')}GB is limited for the full Ideogram model set; "
                     "CPU offloading needs tens of GB and may be slow or fail. Close other applications.")
    if (p.get("disk_free_gb") or 999) < 40:
        warns.append(f"Only ~{p.get('disk_free_gb')}GB free on the models volume — a full "
                     "cond+uncond+TE+VAE set needs ~40 GB. Blocking large downloads until space is freed.")
    cap = p.get("cuda_cap")
    if cap is not None and cap[0] < 10:
        warns.append(f"CUDA capability {cap[0]}.{cap[1]} < 10.0: nvfp4_mixed is Blackwell-only and "
                     "will be hidden from the recommended list.")
    return warns


def configure_once() -> dict | None:
    """SINGLE probe -> config.json owner, shared by engine (first launch / GPU
    change) and install.py (every boot). Fingerprint-gated and idempotent.

    Returns {"changed", "probe", "profile", "defaults", "warnings"} or None
    when the probe could not run (caller logs its own prefix). Never raises.
    """
    try:
        import pi_ideogram_lib.config as pi_config
        p = probe()
        fp = fingerprint(p)
        cfg = pi_config.load()
        if cfg.get("hw_fingerprint") == fp and cfg.get("hw_vram_gb"):
            prof = cfg.get("hw_profile") or vram_profile(p.get("vram_total_gb"))
            return {"changed": False, "probe": p, "profile": prof,
                    "defaults": profile_defaults(prof), "warnings": []}
        prof = vram_profile(p.get("vram_total_gb"))
        pd = profile_defaults(prof)
        pi_config.save({
            "hw_fingerprint": fp,
            "hw_vram_gb": p.get("vram_total_gb"),
            "hw_profile": prof,
            "use_uncond_lora_auto": bool(pd["use_uncond_lora_default"]),
            "quant_auto": pd["quant_auto"],
        })
        return {"changed": True, "probe": p, "profile": prof,
                "defaults": pd, "warnings": check_warnings(p)}
    except Exception as e:
        print(f"{TAG} hardware probe skipped: {e}")
        return None
