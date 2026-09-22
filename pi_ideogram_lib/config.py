"""PROJECT INVISIBLE - persistent settings for the Ideogram 4 engine.

config.json lives next to this file. STRICTLY NO UI: everything here is edited
by hand (optional) and everything falls back to safe defaults. The engine
derives all generation parameters from the STOCK Forge controls; only optional
secrets/flags live here. Never raises at import.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_THIS_FILE = Path(__file__).resolve().parent.parent
CONFIG_PATH = _THIS_FILE / "config.json"

_LOCK = threading.Lock()

DEFAULTS: dict[str, Any] = {
  # optional Magic Prompt (official implementation); OFF by default; never required
  "magic_prompt_enabled": False,
  "magic_prompt_provider": "ideogram-4-v1",   # ideogram-4-v1 | claude-sonnet-v1 | claude-opus-v1
  "ideogram_api_key": "",
  "openrouter_api_key": "",
  # experimental: fp8 tensor-core GEMM (torch._scaled_mm) instead of the
  # dequant matmul. Measured SLOWER on torch 2.13/cu130 at DiT layer sizes
  # (per-call f32 activation quantize), so default False. Flip to True only
  # after re-benching on a newer torch (rowwise fp8 support).
  "fp8_fast_gemm": False,
  # Debanner: MrJackSpade correction tensor, applied as a hook on DiT blocks
  # 25-28 (auto-downloads on first use).
  #
  # DEFAULT OFF, deliberately. Until this build the feature was wired to the
  # model's 128-channel OUTPUT while the tensor is 4608-wide residual-stream
  # data, so the merge always threw and was swallowed - it has never once run,
  # despite printing "4 blocks applied". Now that it genuinely applies, its
  # effect on real images is unvalidated, and defaulting it ON would silently
  # change every user's output. Turn it on once you have compared a pair of
  # images at a fixed seed.
  "debanner_enabled": False,
  "debanner_strength": 0.6,
  # SPECTRUM: forecast some steps instead of computing them (spectrum.py).
  # OFF by default. It is an APPROXIMATION - forecast steps skip the
  # transformer, so the same seed no longer reproduces the same image - and
  # turning that on for someone without their asking is not a decision this
  # file gets to make.
  #
  # Measured on a 5090, int8_convrot, 1024x1024, 20 steps, fixed seed:
  # these defaults forecast 5 of 20 steps for 14.7s -> 11.1s (1.32x). The
  # picture stays the same picture, fine texture goes slightly softer.
  # stride 2 reaches 1.66x and is NOT the default because it is
  # prompt-dependent: it made the best portrait of the test set and smeared a
  # hand study into merged fingers at identical settings.
  #   degree      1 = straight line, 2 = curve. Higher fits harder and
  #               overshoots sooner. 1 is what survived testing.
  #   stride      3 = forecast every third eligible step. 2 is the most it
  #               will ever skip, and the setting that broke hands.
  #   warmup/tail steps at the start/end always computed for real.
  #               Composition is set early and detail late; both are the
  #               worst possible places to guess.
  #   max_growth  refuse a forecast that would multiply the velocity
  #               magnitude by more than this in a step, and compute it for
  #               real instead.
  "spectrum_enabled": False,
  "first_block_cache_enabled": False,
  "allow_quality_reducing_oom_recovery": False,
  "spectrum_degree": 1,
  "spectrum_ridge": 0.1,
  "spectrum_history": 6,
  "spectrum_warmup": 4,
  "spectrum_tail": 2,
  "spectrum_stride": 3,
  "spectrum_blend": 1.0,
  "spectrum_max_growth": 2.0,
  # when a run comes back as a gray 'blocked' image, re-run it ONCE at the next
  # bypass-strength anchor (same seed) so the panel finishes the job instead of
  # handing the user a gray image + advice. False = hint-only (old behaviour).
  # OFF by default. When an image comes back as the gray card this used to
  # silently run a SECOND full generation - "I selected 1 image and it was
  # ready for the 2nd". Worse, for as long as the bypass dropdown was inert
  # the retry could not change anything: it escalated a strength no code read
  # and burned a minute to produce the same card. The blocked image is now
  # reported with what to do about it instead. Set True to opt back in.
  "auto_escalate_blocked": False,
  # when a DUAL-transformer run dies with torch.cuda.OutOfMemoryError, retry it
  # ONCE at the same prompt/seed/settings on the cond-only + Ostris uncond-LoRA
  # path (~9 GB lighter, adapter auto-downloaded). Never fires on a run that is
  # already single-transformer (turbo / uncond-LoRA). False = old behaviour
  # (batch/resolution degradation only).
  "oom_retry_uncond_lora": True,
  # ---- filter bypass panel (spec section P) ----
  # Civitai API key for adapter downloads. Being signed in to the website
  # does not help: the download is a server-side API call with no browser
  # session. config.load() drops any key not listed here, so this entry is
  # Version of the quant_registry mirror below; a bump refreshes it from code.
  "quant_registry_version": 0,
  # Apply LoRAs to QUANTISED checkpoints as an exact runtime side path
  # (cosine 1.000000) instead of merging into fp8/int8 weights (cosine 0.57 -
  # ~43% of the adapter arrives as quantisation noise, which is the "body
  # horror" people report with character LoKrs on fp8).
  #
  # DEFAULT OFF, deliberately. The side path is the better answer for image
  # quality, but it holds the adapter's factors on the card and allocates per
  # hooked module - 204 modules x 2 trunks on top of ~16 GB of resident
  # weights OOMed a 32 GB card at 1024px, through every rung of the recovery
  # ladder. Its per-module cost was measured (+16-86 MB); its cost across a
  # whole loaded model was NOT, and shipping it on by default on that basis
  # was wrong. Set true once you have headroom (single-transformer /
  # uncond-LoRA profile, or a smaller canvas) and want the exact adapter.
  "lora_runtime": True,
  # Point this at a Qwen3-VL-8B text encoder to use instead of whatever Forge
  # has selected. A single .safetensors OR a Hugging Face folder (config.json +
  # model.safetensors.index.json + shards) - the folder case is the only way to
  # use an abliterated/uncensored encoder, since Forge's module dropdown lists
  # single files only. Must be the 8B: hidden 4096, 36 layers. Empty = off.
  "text_encoder_override": "",
  # The adapter picker's remembered selection (a label from
  # detect.adapter_choices) and its strength.
  "adapter_choice": "",
  "adapter_strength": 1.0,
  # The Ideogram 4 checkpoint that last generated, by absolute path. Forge
  # stores ours by display label ('Ideogram4 - <name>'), which does not
  # survive a rename; this is the fallback detect.restore_checkpoint() uses
  # when that label no longer resolves. Empty until the first generation.
  "last_checkpoint": "",
  "filter_bypass_enabled": True,      # checkbox: master on/off
  # 0.0 = the OFFICIAL sampling schedule, untouched.
  #
  # Anything above 0 turns on sigma smoothing, which INSERTS extra interpolated
  # sigmas - so "Default 20" would really sample 22. The official reference
  # (ideogram-oss/ComfyUI-Ideogram4) specifies 20 steps = 18 main + 2 polish,
  # CFG 7.0 -> 3.0, mu 0.0, std 1.75, and describes no such smoothing. Shipping
  # it on by default meant every stock run silently deviated from the published
  # recipe in exactly the parameters people compare against.
  #
  # Gray-screen protection does NOT depend on this: the JSON density guard in
  # json_prompt.py is prompt-side and always on. Raise this slider only if you
  # actually meet the gray card.
  # Filter bypass is now TWO controls: pick a LoRA, set its strength.
  # Empty name = off = the official recipe, untouched.
  "bypass_lora_name": "",
  "bypass_strength": 0.0,            # master slider 0.0-1.0 (0 = everything off)
  "bypass_lora": True,                # P1: gray-screen bypass LoRA
  "bypass_lora_strength": -0.25,      # LoRA weight (recipe overrides; -1.00..+0.25)
  # OFF: sigma smoothing INSERTS interpolated steps, so "Default 20" would
  # sample 22 - a silent deviation from the official recipe. It is no longer a
  # panel control; set it here only if you deliberately want it.
  "sigma_smooth": False,               # P2: smooth first sigmas
  "layer_dampen": False,              # P3: layer dampening (advanced, default OFF)
  "dampen_cond_mult": 0.4,
  "dampen_uncond_mult": 0.1,
  # ---- generation defaults ----
  "preset": "V4_DEFAULT_20",          # V4_QUALITY_48 | V4_DEFAULT_20 | V4_TURBO_12 | TurboTime-4 | Instant-8 | Auto
  "quant_override": "auto",           # auto | registry id
  "uncond_mode": "Auto",              # Auto | Full transformer | Ostris uncond LoRA
  "use_uncond_lora": None,             # None=auto (VRAM profile) | True | False
  "turbo_time": None,                  # None=auto (VRAM profile) | True | False
  "json_pass_through": False,          # send the prompt verbatim, no JSON wrap
  "prompt_upsampling": False,          # local prompt_upsampling (off by default on <=12 GB)
  # ---- hardware fingerprint (re-probe when it changes) ----
  "hw_fingerprint": "",
  "hw_profile": "unknown",
  "hw_vram_gb": None,
  # null = let "Auto" decide from the card (see settings.DUAL_TRANSFORMER_
  # COMFORT_GB). true/false force it. This was False, and because
  # config.load() merges every default in, the key was ALWAYS present -
  # so a hardware-aware Auto could never run.
  "use_uncond_lora_auto": None,
  "quant_auto": "fp8_scaled",
  "chosen_variant": None,             # spec F picker: registry id picked in the UI (None = recommended for GPU)
  # informational
  "warned_uncond_missing": False,
  "warned_gray_lora_missing": False,
  "warned_turbotime_missing": False,
  "warned_uncond_lora_missing": False,
  # startup may auto-download enabled-but-missing adapters (set False to keep
  # first boot fully offline; feature toggles still download at generation time)
  "startup_auto_download": True,
  # quantization registry (spec section R) - refreshed by pi_ideogram_lib.quant_registry
  "quant_registry": None,
}


def load() -> dict[str, Any]:
  try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
      data = json.load(f)
    if not isinstance(data, dict):
      return dict(DEFAULTS)
    out = dict(DEFAULTS)
    out.update({k: v for k, v in data.items() if k in DEFAULTS})
    return out
  except Exception:
    return dict(DEFAULTS)


def save(updates: dict[str, Any]) -> dict[str, Any]:
  with _LOCK:
    current = load()
    current.update({k: v for k, v in updates.items() if k in DEFAULTS})
    try:
      tmp = CONFIG_PATH.with_suffix(".json.tmp")
      with open(tmp, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
      os.replace(tmp, CONFIG_PATH)
    except Exception as e:
      print(f"[Invisible-I4] Could not save settings: {e}")
    return current


def get(key: str, default: Any = None) -> Any:
  return load().get(key, DEFAULTS.get(key, default))
