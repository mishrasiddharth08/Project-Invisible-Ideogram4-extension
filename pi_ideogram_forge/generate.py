# Project Invisible: namespaced Forge adapter; no core-file edits.
"""Ideogram 4 generation orchestration (Phase 3 extraction from engine.py).

This module owns generate() and all helpers it needs:
- generate(p, opts, ...) -> Processed   (opts = RESOLVED run settings)
- _resolve_sampling(p) -> (steps, schedule, mu, std, preset_name)
- _looks_blocked(pil_img) -> bool
- _safe_comment(p, text) -> None

DESIGN: This module has ONE Forge dependency direction — it imports
Forge modules (shared, Processed, forge_images) directly. It does NOT
import from engine.py (which would create a circular dependency).
engine.py passes engine-specific state as arguments to generate().
Run settings arrive pre-resolved from pi_ideogram_lib.settings; this
module never reads config.json or the Gradio layer.

DOS/CONSOLE (spec Q): a live tqdm step bar goes to stderr exactly like
stock Forge models ("Ideogram4  7/20  [██████----]  00:12<00:24"), while
modules.shared.state keeps the WebUI progress bar + live preview in sync.
A one-line header is printed before sampling; elapsed ms + peak VRAM after
VAE decode.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

TAG = "[Invisible-I4]"

# Run settings come in via `opts`, produced by pi_ideogram_lib.settings
# (single owner of UI->settings mapping). generate() never reads config.json
# or the Gradio layer directly.

# ---------------------------------------------------------------------------
# Sampling schedule resolver (pure function — no Forge deps)
# ---------------------------------------------------------------------------

def _resolve_seed(p) -> int:
    """Turn Forge's seed field into a concrete seed. -1 (or None) means random."""
    raw = getattr(p, "seed", None)
    try:
        seed = int(raw) if raw is not None else -1
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        try:
            from modules.processing import get_fixed_seed
            seed = int(get_fixed_seed(-1))
        except Exception:
            import random
            seed = random.randint(0, 2**32 - 1)
    try:
        p.seed = seed
    except Exception:
        pass
    return seed


def _resolve_sampling(p) -> tuple[int, tuple[float, ...], float, float, str]:
    """Map stock steps/CFG to the official Ideogram 4 recipe.

    - exact official steps (48/20/12) -> that preset wholesale
    - anything else -> custom: main CFG from slider, 3.0 polish tail
    - CFG slider at 1.0 -> distilled path: cond-only, no uncond pass
    """
    from pi_ideogram_lib.sampler_configs import PRESETS, build_custom_schedule

    steps = max(1, int(getattr(p, "steps", 20)))
    try:
        cfg = float(getattr(p, "cfg_scale", 7.0) or 7.0)
    except (TypeError, ValueError):
        cfg = 7.0
    for key, preset in PRESETS.items():
        if key in ("V4_FAST_20", "V4_INSTANT_8"):
            continue
        if steps == preset.num_steps and cfg >= 1.5:
            return preset.num_steps, preset.guidance_schedule, preset.mu, preset.std, key
    if cfg <= 1.0 + 1e-6:
        return steps, (1.0,) * steps, 0.5, 1.75, "distilled-cfg1"
    schedule = build_custom_schedule(steps, main_cfg=cfg, tail_cfg=3.0, tail_fraction=0.15)
    return steps, schedule, 0.0, 1.75, "custom"


from pi_ideogram_lib.settings import PRESET_AUTO_LABEL


def _resolve_preset(opts: dict, p) -> tuple[int, tuple[float, ...], float, float, str]:
    """Accordion preset wins; stock slider mapping is the fallback.

    Presets: V4_QUALITY_48 | V4_DEFAULT_20 | V4_TURBO_12 | TurboTime-4 | Instant-8 | Auto
    """
    from pi_ideogram_lib.sampler_configs import (
        DISTILL_PRESETS, PRESETS, build_custom_schedule)
    preset = str(opts.get("preset", "Auto") or "Auto")
    if preset == "TurboTime-4":
        # Ostris TurboTime: 2-8 steps, DEFAULT 4. The stock steps slider only
        # counts when it is already inside the distill range (user override).
        steps = int(getattr(p, "steps", 4) or 4)
        steps = steps if 2 <= steps <= 8 else 4
        return steps, (1.0,) * steps, 0.5, 1.75, "TurboTime-4"
    if preset == "Instant-8":
        return 8, (1.0,) * 8, 0.5, 1.75, "Instant-8"
    if preset in PRESETS:
        pr = PRESETS[preset]
        # THE STEPS SLIDER IS HONOURED, not discarded. This used to return
        # pr.num_steps unconditionally, so setting the slider to 25 under
        # "Default 20" silently ran 20 and the console only said
        # "steps 20 (V4_DEFAULT_20)". A visible control that does nothing is
        # worse than no control.
        #
        # What the preset actually owns is the SHAPE of the recipe - a short
        # low-CFG polish tail after a main run at gw=7 (stored in loop-index
        # order, index 0 = final step) - plus mu/std. That shape is
        # scale-free, so it is rebuilt at the requested length with the same
        # tail PROPORTION rather than thrown away. At the preset's own step
        # count this reproduces the stored schedule exactly, so the default
        # path is unchanged.
        #
        # Distilled recipes are excluded on purpose: their step count is a
        # property of the weights, not a preference (see distill_turbo_policy
        # and the TurboTime/Instant branches above).
        want = int(getattr(p, "steps", 0) or 0)
        if want and want != pr.num_steps and preset not in DISTILL_PRESETS:
            tail = sum(1 for g in pr.guidance_schedule if g != pr.guidance_schedule[-1])
            if 0 < tail < pr.num_steps:
                sched = build_custom_schedule(
                    want, main_cfg=pr.guidance_schedule[-1],
                    tail_cfg=pr.guidance_schedule[0],
                    tail_fraction=tail / pr.num_steps)
                return want, sched, pr.mu, pr.std, preset
        return pr.num_steps, pr.guidance_schedule, pr.mu, pr.std, preset
    return _resolve_sampling(p)


def distill_turbo_policy(preset_name: str, steps: int, distilled: bool
                         ) -> tuple[int, tuple[float, ...], float, float, str, bool, str | None]:
    """Spec H stack guard: an already-distilled / Instant / Fast base must
    NEVER receive the TurboTime LoRA on top (double-distillation is silent
    misbehavior). Pure decision, returns
    (steps, schedule, mu, std, preset_name, turbo_merge_ok, note).

    - TurboTime-4 on a distilled base -> snaps to the Instant-8 cond-only
      recipe (8 steps, CFG 1.0, no uncond, NO TurboTime LoRA) + note.
    - TurboTime-4 on a regular base -> the recipe, merge allowed.
    - Instant-8 (any base) -> cond-only distill; the TurboTime LoRA is never
      the Instant recipe's business (merge NOT allowed).
    """
    if preset_name == "TurboTime-4":
        if distilled:
            return (8, (1.0,) * 8, 0.5, 1.75, "Instant-8", False,
                    "base checkpoint looks like an already-distilled / Instant merge "
                    "(filename or header metadata) - TurboTime LoRA NOT merged (stacking "
                    "would double-distill). Running the Instant-8 cond-only recipe at CFG 1.0.")
        return (steps, (1.0,) * steps, 0.5, 1.75, "TurboTime-4", True, None)
    if preset_name == "Instant-8":
        return 8, (1.0,) * 8, 0.5, 1.75, "Instant-8", False, None
    return steps, (1.0,) * steps, 0.5, 1.75, preset_name, False, None


# ---------------------------------------------------------------------------
# Image quality heuristics
# ---------------------------------------------------------------------------

def _find_turbo_adapter(loras, lora_mod, models_dirs):
    """Return (name, strength, path) of a step-distill LoRA in the tag list.

    The step count a run should use is a property of the ADAPTER FILE, so it is
    knowable before anything is loaded. Asking the pipeline instead was a
    circular dependency: the pipeline cache key contains `turbo`, so the answer
    was needed to build the very object being asked for it.

    Returns None when no distill adapter is tagged. Never raises - a LoRA that
    cannot be read simply is not a turbo adapter for this purpose, and the real
    diagnostics come later from apply_lora_set().
    """
    from pi_ideogram_lora.ideogram4_lora import classify_adapter
    for name, strength, _policy in (loras or []):
        try:
            path = lora_mod.find_lora_file(name, models_dirs)
            if path is None:
                continue
            if classify_adapter(path).get("kind") == "turbo_time":
                return (name, float(strength), str(path))
        except Exception:
            continue
    return None


def _looks_blocked(pil_img) -> bool:
    """Detect gray 'safety filter blocked' images via row-band analysis."""
    try:
        import numpy as np
        a = np.asarray(pil_img.convert("L"), dtype=np.float32)
        if a.std() >= 25:
            return False
        row_std = a.std(axis=1)
        band = row_std > (row_std.mean() * 3 + 5)
        runs, in_run, start = [], False, 0
        for i, b in enumerate(band):
            if b and not in_run:
                start = i
                in_run = True
            elif not b and in_run:
                runs.append((start, i))
                in_run = False
        if in_run:
            runs.append((start, a.shape[0]))
        big = [r for r in runs if r[1] - r[0] >= 8]
        return bool(band.mean() > 0.005 and len(big) >= 1)
    except Exception:
        return False


def _require_guidance_assets(pipe, guidance_schedule, *, want_uncond):
    """Stop a guided Forge run when its requested guidance weights are absent."""
    if not any(float(scale) != 1.0 for scale in guidance_schedule):
        return  # CFG 1 recipes do not run an unconditional pass.
    if want_uncond:
        if getattr(pipe, "unconditional_transformer", None) is None:
            raise ValueError(
                "The matching unconditional transformer is missing. Add its weights "
                "using Model Setup, or select Ostris uncond LoRA with its adapter installed."
            )
    elif not getattr(pipe, "_i4_uncond_lora", None):
        raise ValueError(
            "The unconditional guidance LoRA is missing. Install "
            "ideogram_4_unconditional_lora_r16.safetensors using Model Setup, "
            "or select Full transformer with matching weights."
        )


def _save_images(p, images, seeds, prompts, infotexts, *, shared, forge_images,
                 announce_stop: bool = False) -> bool:
    """Write finished images to disk. Returns True if anything was written.

    ONE owner of the save decision, called from two places: as each image is
    decoded (so a crash or a kill cannot lose finished work), and once more at
    the end for anything the first pass could not write.

    Forge's own gate is
        save_samples() = opts.samples_save and not do_not_save_samples
                         and (opts.save_incomplete_images
                              or not state.interrupted and not state.skipped)
    so pressing Stop discards everything, including images that had already
    finished. That default is right for a stock sampler, where interrupting
    mid-denoise hands back a half-denoised picture. It is wrong here: an image
    reaches this function only after its final VAE decode.

    The two settings that really mean "do not write files" - samples_save and
    do_not_save_samples - are still obeyed. Only the interrupted-batch clause
    is set aside, and out loud.
    """
    if not images:
        return False
    try:
        save_ok = bool(p.save_samples())
    except Exception:
        save_ok = False
    if not save_ok:
        try:
            stopped = bool(shared.state.interrupted or shared.state.skipped)
            allowed = bool(shared.opts.samples_save) and not bool(
                getattr(p, "do_not_save_samples", False))
        except Exception:
            stopped = allowed = False
        if stopped and allowed:
            save_ok = True
            if announce_stop:
                print(f"{TAG} you pressed Stop - saving the {len(images)} image(s) that had "
                      "already finished (they are complete, not partially denoised)")
    if not save_ok:
        return False
    for i, img in enumerate(images):
        try:
            forge_images.save_image(
                img, p.outpath_samples, "",
                seeds[i] if i < len(seeds) else 0,
                prompts[i] if i < len(prompts) else str(p.prompt),
                shared.opts.samples_format,
                info=infotexts[i] if i < len(infotexts) else "", p=p)
        except Exception as e:
            print(f"{TAG} WARNING: could not save an image ({e})")
            return False
    return True


def _safe_comment(p, text: str) -> None:
    """Append a comment through Forge's generation API. Never raises."""
    try:
        if hasattr(p, "comment") and callable(p.comment):
            p.comment(text)
        else:
            c = getattr(p, "comments", None)
            if isinstance(c, list):
                c.append(text)
            elif isinstance(c, dict):
                c[text] = 1
    except Exception:
        pass


BLOCKED_HINT = (
    "Note: the official Ideogram 4 weights contain a baked-in safety filter "
    "that can misfire on sparse captions, showing a gray 'Image blocked by "
    "safety filter' image. In order of effectiveness: (1) pick a Bypass LoRA "
    "in the Ideogram 4 Controls accordion - Gray_000002000.safetensors at "
    "-0.25 is the author's recipe, and it is applied on the first step only; "
    "(2) write a richer prompt, a few full sentences rather than a phrase; "
    "(3) supply your own elements with bounding boxes via raw JSON. "
    "No second generation was started for you - that is off by default now "
    "(config auto_escalate_blocked)."
)


def _peak_vram_gb() -> float:
    try:
        import torch
        return torch.cuda.max_memory_allocated(0) / 1024**3
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate(
    p,
    *,
    opts: dict | None = None,
    active_loras=None,
    models_dirs=None,
    get_pipeline_fn=None,
    lora_mod=None,
    json_prompt_mod=None,
    init_images=None,
    denoising_strength: float = 0.75,
    resize_mode: int = 0,
) -> object:  # returns Processed
    """Run the Ideogram 4 generation pipeline.

    Args:
        p: Forge's StableDiffusionProcessing object.
        opts: RESOLVED run settings - produced by pi_ideogram_lib.settings
              (single owner). None falls back to defaults resolved from
              config.json, so direct/non-UI callers behave like the UI with
              its default values.
        active_loras: Mutable list the engine owns; updated with the resolved
              LoRA set for the pipeline cache key.
        models_dirs, get_pipeline_fn, lora_mod, json_prompt_mod: as before.

    Returns:
        Forge Processed object with generated images.
    """
    try:
        from modules import shared, images as forge_images
        from modules.processing import Processed
        _FORGE_OK = True
    except Exception:
        _FORGE_OK = False

    if opts is None:
        from pi_ideogram_lib.settings import resolve as _resolve_settings
        opts = _resolve_settings()
    opts = dict(opts or {})

    cond_path = _get_ideogram_file()
    if cond_path is None:
        raise RuntimeError("Ideogram 4 is not the selected checkpoint.")
    # UNCOND HALF REFUSAL (playtest defect 1): the uncond file is a diffusion
    # model the pipeline loads from its own folder and pairs automatically - it
    # is never itself a checkpoint. If one is selected through ANY path, refuse
    # with a pointer to its matching conditional file instead of loading
    # nonsense. (The dropdown no longer offers it; this is the deep guard.)
    from pi_ideogram_forge.selection import uncond_refusal_message as _uncond_refusal
    _uncond_refusal_text = _uncond_refusal(cond_path)
    if _uncond_refusal_text:
        raise ValueError("Ideogram 4: " + _uncond_refusal_text)

    # --- distilled-base detection (spec H stack guard) ------------------------
    distilled = False
    try:
        from pi_ideogram_lib.quant_registry import sniff_distilled
        _dist = sniff_distilled(cond_path)
        distilled = bool(_dist.get("distilled"))
    except Exception:
        pass

    # --- quant context for the DOS header ----------------------------------
    try:
        from pi_ideogram_lib.quant_registry import sniff_quant
        quant = sniff_quant(cond_path)
        quant_label = quant.get("id", "unknown")
    except Exception:
        quant_label = "unknown"

    # --- prompt: strip stock lora tags, keep the rest as the user's scene text ---
    user_text, loras = lora_mod.split_lora_tags(str(p.prompt or ""), models_dirs)

    # THE ADAPTER PICKER is exactly a <lora:name:strength> tag, chosen from a
    # list instead of typed. Resolving it here - through the same
    # split_lora_tags path - means every downstream behaviour (turbo detection,
    # uncond replacement, the runtime side path, the pipeline cache key) treats
    # it identically to a typed tag, with no second code path to keep in step.
    _pick = str(opts.get("adapter_choice") or "").strip()
    if _pick:
        from pi_ideogram_lib.detect import ADAPTER_NONE_LABEL, adapter_name_of
        _name = adapter_name_of(_pick)
        if _name and _pick != ADAPTER_NONE_LABEL:
            _stem = Path(_name).stem
            if not any(n == _stem or n == _name for n, _s, _pol in loras):
                _strength = opts.get("adapter_strength", 1.0)
                _st = float(1.0 if _strength is None else _strength)
                _extra_text, _extra = lora_mod.split_lora_tags(
                    f"<lora:{_stem}:{_st}>", models_dirs)
                if _extra:
                    loras.extend(_extra)
                    print(f"{TAG} adapter picker: {_stem} at strength {_st}")
                else:
                    print(f"{TAG} adapter picker: '{_stem}' could not be resolved "
                          "under the LoRA folders - ignoring it (your image still generates).")

    if not user_text.strip():
        raise ValueError("The prompt box is empty. Type your scene description first "
                         "(<lora:...> tags alone are not a prompt).")
    if active_loras is not None:
        active_loras.clear()
        active_loras.extend(loras)

    # --- turbo mode decision --------------------------------------------------
    # SELECTING a distill LoRA is itself the request. A turbo adapter is
    # trained for a specific, small step count; leaving the run on the 20-step
    # default it was never meant for wastes most of the speed-up and looks
    # like the adapter "did nothing". So a staged turbo LoRA turns turbo mode
    # on even when the preset dropdown was left alone.
    #
    # This has to be decided from the TAG LIST, not from the pipeline. The
    # pipeline does not exist yet (it is built below, and its cache key
    # depends on `turbo`), and pipe._i4_turbo_lora is only staged later still,
    # by apply_lora_set. Reading it here raised UnboundLocalError on every
    # generation; classify_adapter() answers the same question from the file.
    _turbo_lora_staged = _find_turbo_adapter(loras, lora_mod, models_dirs)
    turbo = (bool(opts.get("turbo"))
             or str(opts.get("preset", "")) in ("TurboTime-4", "Instant-8")
             or bool(_turbo_lora_staged))
    # resolve() derives turbo=False for ordinary recipes; it is not a user
    # veto of a deliberately selected TurboTime adapter.

    num_steps, guidance_schedule, mu, std, preset_name = _resolve_preset(opts, p)

    # Step count comes from the ADAPTER, not from a constant: ostris/TurboTime
    # carries ss_output_name "ideogram_turbo_8_v1", so it asks for 8. Read it
    # rather than assume, and fall back only when the file says nothing.
    _turbo_steps = None
    if turbo and _turbo_lora_staged:
        try:
            from pi_ideogram_lora.ideogram4_lora import (
                DISTILL_DEFAULT_STEPS, adapter_step_hint,
            )
            _turbo_steps = adapter_step_hint(_turbo_lora_staged[2]) or DISTILL_DEFAULT_STEPS
        except Exception:
            _turbo_steps = None
    if turbo and preset_name not in ("TurboTime-4", "Instant-8"):
        # TurboTime requested (toggle or a staged LoRA): snap to the distill
        # recipe, using the adapter's own step count when it declares one.
        _want = _turbo_steps if _turbo_steps else num_steps
        num_steps = max(2, min(12, int(_want)))
        guidance_schedule = (1.0,) * num_steps
        mu, std = 0.5, 1.75
        preset_name = "TurboTime-4"
        if _turbo_steps:
            print(f"{TAG} distill LoRA selected -> steps auto-set to {num_steps} "
                  f"and CFG to 1.0 (the adapter's own recipe)")

    # DISTILL STACK GUARD (spec H): the detected distilled base routes the
    # turbo path - TurboTime-4 snaps to Instant-8 with a clear note, and the
    # TurboTime LoRA merge is refused. Never silent.
    turbo_merge_ok = False
    if turbo:
        _s, _sc, _m, _st, _pn, turbo_merge_ok, _note = distill_turbo_policy(
            preset_name, num_steps, distilled)
        num_steps, guidance_schedule, mu, std, preset_name = _s, _sc, _m, _st, _pn
        if _note:
            print(f"{TAG} {_note}")
            try:
                _safe_comment(p, _note)
            except Exception:
                pass

    width = max(256, min(2048, int(p.width)))
    height = max(256, min(2048, int(p.height)))
    width -= width % 16
    height -= height % 16
    if max(width, height) / min(width, height) > 6.0:
        raise ValueError(f"Aspect ratio above 6:1 is not supported by Ideogram 4 ({width}x{height}).")

    if p.negative_prompt and str(p.negative_prompt).strip():
        print(f"{TAG} NOTE: Ideogram 4's official negative branch is image-only (asymmetric CFG); "
              f"the negative prompt is not used by the official pipeline: '{str(p.negative_prompt)[:80]}'")

    # --- prompt mode: raw JSON / pass-through / freeform wrap ------------------
    pass_through = bool(opts.get("json_pass_through", False))
    if pass_through:
        caption_str = user_text
        mode = "pass-through"
    elif json_prompt_mod._looks_like_json(user_text):
        caption_str, _cap, warnings = json_prompt_mod.build_prompt(
            mode="raw-json", user_text=user_text, target_size=(width, height))
        if warnings:
            raise ValueError("Your pasted JSON failed official schema validation: " + "; ".join(warnings))
        mode = "raw-json"
    else:
        caption_str, _cap, warnings = json_prompt_mod.build_prompt(
            mode="freeform-wrap", user_text=user_text, target_size=(width, height),
            density=int(opts.get("caption_density", 3) or 3))
        mode = "freeform-wrap"
    # SAY SO WHEN THE PRESET DISCARDS THE SLIDER. A named preset pins its own
    # step count (_resolve_preset returns pr.num_steps), so moving the stock
    # Steps slider to 25 did nothing and the console only said "steps 20
    # (V4_DEFAULT_20)" - which names the preset but never admits it overrode
    # the number the user just typed. Silent override is the same defect
    # class as a feature that logs nothing when it is off.
    # The slider now DRIVES the step count for the non-distilled presets
    # (_resolve_preset rescales the preset's CFG tail to the requested
    # length), so the old "your slider is IGNORED" line would itself be a
    # lie. It fires only where the override genuinely does not apply: a
    # distilled recipe, whose step count is a property of the weights.
    _slider = int(getattr(p, "steps", 0) or 0)
    _override = (f"  <- Steps slider ({_slider}) not applied: '{preset_name}' "
                 "is a DISTILLED recipe whose step count is baked into the "
                 "weights"
                 if _slider and _slider != num_steps else "")
    # THE SAMPLER AND SCHEDULER DROPDOWNS DO NOT APPLY EITHER. Ideogram 4 is a
    # flow-matching model integrated by this pipeline's own single-step Euler
    # update (z = z + v * delta) on its own sigma schedule; nothing reads
    # p.sampler_name or p.scheduler. A preset that stores "iPNDM / SGM
    # Uniform" therefore changes nothing, and saying so beats letting someone
    # tune a control that is not connected.
    #
    # It is also why the RES4LYF compatibility note about Spectrum does not
    # bite here: that warning is about MULTISTEP samplers reusing previous-step
    # model output on top of Spectrum's own caching. This integrator keeps no
    # such history, so there is no double-approximation to compound.
    _samp = str(getattr(p, "sampler_name", "") or "")
    _sched = str(getattr(p, "scheduler", "") or "")
    if _samp:
        _override += (f"\n{TAG}         note: sampler '{_samp}'"
                      + (f" / scheduler '{_sched}'" if _sched else "")
                      + " is not used - Ideogram 4 has its own flow-matching "
                        "integrator and sigma schedule")
    print(f"{TAG} prompt mode: {mode} | steps {num_steps} ({preset_name}) "
          f"| {width}x{height}{_override}")

    # --- optional local prompt_upsampling (P4, off by default on low VRAM) ----
    if opts.get("prompt_upsampling", False) and mode == "freeform-wrap":
        try:
            from pi_ideogram_lib.upsample import local_upsample
            up = local_upsample(user_text)
            if up and up != user_text:
                caption_str, _cap, _w = json_prompt_mod.build_prompt(
                    mode="freeform-wrap", user_text=up, target_size=(width, height))
                print(f"{TAG} local prompt_upsampling expanded {len(user_text)} -> {len(up)} chars")
        except Exception as e:
            print(f"{TAG} prompt_upsampling unavailable ({e}) - continuing with the local wrap")

    # --- optional magic prompt via config.json (never required, default OFF) ---
    if opts.get("magic_prompt_enabled", False):
        try:
            from pi_ideogram_lib.magic_prompt import MAGIC_PROMPTS, aspect_ratio_from_size
            prov = opts.get("magic_prompt_provider", "ideogram-4-v1")
            key = opts.get("ideogram_api_key" if prov == "ideogram-4-v1" else "openrouter_api_key", "")
            if key:
                expanded = MAGIC_PROMPTS[prov](api_key=key).expand(
                    user_text, aspect_ratio=aspect_ratio_from_size(width, height))
                cap = json.loads(expanded) if expanded.strip().startswith("{") else None
                if isinstance(cap, dict):
                    hld = cap.get("high_level_description", "")
                    if user_text.strip().lower() not in str(hld).lower():
                        cap["high_level_description"] = (user_text.strip() + " " + str(hld)).strip()
                    caption_str = json.dumps(cap, ensure_ascii=False, separators=(",", ":"))
                    print(f"{TAG} Magic Prompt ({prov}) applied; your text kept first (dominance)")
        except Exception as e:
            print(f"{TAG} Magic Prompt skipped ({e}) - continuing with the local wrap")

    # --- uncond decision: full transformer vs Ostris uncond LoRA ----------------
    # (resolved by settings.py from uncond_mode / VRAM profile; never None)
    use_uncond_lora = bool(opts.get("use_uncond_lora", False))

    # SELECTING THE UNCOND-REPLACEMENT ADAPTER *IS* CHOOSING THE LORA PATH.
    #
    # That adapter exists to stand in for the 9.3 GB unconditional
    # transformer. Loading both is a contradiction, and an expensive one: on a
    # 31.8 GB card an fp8 pair (9.3 + 9.3) plus the encoder filled the card to
    # 31.8/31.8 and the run died with "OOM: 0.0GB free" before step 1.
    #
    # The adapter can arrive from the accordion's Adapter picker or from a
    # <lora:...> tag; either way it settles the question, so the uncond half
    # is not loaded at all.
    if not use_uncond_lora:
        for _n, _s, _policy in (loras or []):
            if str(_policy) == "uncond_replace":
                use_uncond_lora = True
                print(f"{TAG} '{_n}' is the uncond-replacement adapter - it REPLACES the "
                      "unconditional transformer, so that 9.3 GB half is not loaded")
                break

    want_uncond = (not turbo) and (not use_uncond_lora)

    # --- pipeline (cache key includes LoRA set + turbo + uncond mode) ----------
    pipe = get_pipeline_fn(cond_path, want_uncond=want_uncond, turbo=turbo)
    if use_uncond_lora and not turbo and models_dirs:
        # UNTAGGED uncond-LoRA runs (OOM retry swaps to this path with no
        # <lora:...> tag in the prompt): auto-stage the Ostris uncond-
        # replacement adapter when it is on disk so the uncond pass is real
        # instead of the zeroed-text approximation. A prompt tag still wins -
        # apply_lora_set below overwrites the staged tuple with its strength.
        try:
            if getattr(pipe, "_i4_uncond_lora", None) is None:
                _p = lora_mod.find_uncond_replacement_lora(models_dirs)
                if _p is not None:
                    pipe._i4_uncond_lora = (Path(_p).stem, 1.0, str(_p))
                    print(f"{TAG} Ostris uncond-replacement LoRA auto-staged "
                          f"({Path(_p).name}) for the uncond pass")
        except Exception:
            pass
    # The OOM ladder can ask for the cheaper merge path. Changing the mode has
    # to INVALIDATE the "already applied" marker, or the retry would reuse the
    # very adapters whose memory triggered the OOM and fail identically.
    _want_runtime = bool(opts.get("lora_runtime", True))
    # ONE-WAY. Merging writes the delta INTO the weights; going back to the
    # runtime side path on the same pipeline would then apply the adapter
    # twice. Once a pipeline has merged, it stays merged for its lifetime -
    # a different LoRA set produces a different cache key and a fresh load.
    if getattr(pipe, "_i4_lora_merged", False):
        _want_runtime = False
    if not _want_runtime:
        pipe._i4_lora_merged = True
    if bool(getattr(pipe, "_i4_lora_runtime", True)) != _want_runtime:
        for _ad in (getattr(pipe, "_i4_runtime_adapters", None) or []):
            try:
                _ad.remove()
            except Exception:
                pass
        pipe._i4_runtime_adapters = []
        pipe._i4_applied_lora_key = None
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass
    pipe._i4_lora_runtime = _want_runtime
    lora_logs = lora_mod.apply_lora_set(pipe, loras)
    _require_guidance_assets(pipe, guidance_schedule, want_uncond=want_uncond)
    debanner_on = opts.get("debanner_enabled", True)
    debanner_str = float(opts.get("debanner_strength", 0.6))

    # --- SPECTRUM (opt-in step forecasting) ------------------------------------
    # OFF unless config.json says otherwise. It trades exactness for speed:
    # forecast steps skip the transformer, so the same seed gives a slightly
    # different image. That is not a setting to turn on for someone.
    from pi_ideogram_lib.spectrum import SpectrumConfig
    spectrum_cfg = SpectrumConfig(
        enabled=bool(opts.get("spectrum_enabled", False)),
        degree=int(opts.get("spectrum_degree", 1)),
        ridge=float(opts.get("spectrum_ridge", 0.1)),
        history=int(opts.get("spectrum_history", 6)),
        warmup=int(opts.get("spectrum_warmup", 4)),
        tail=int(opts.get("spectrum_tail", 2)),
        stride=int(opts.get("spectrum_stride", 3)),
        blend=float(opts.get("spectrum_blend", 1.0)),
        max_growth=float(opts.get("spectrum_max_growth", 2.0)),
    ).sanitised()

    # --- FILTER BYPASS recipe (spec P1-P7) -------------------------------------
    from pi_ideogram_lib.bypass import log_line, recipe as bypass_recipe
    bypass_enabled = bool(opts.get("filter_bypass", True))
    master_strength = float(opts.get("bypass_strength", 0.0) or 0.0)
    if not bypass_enabled:
        master_strength = 0.0
    from pi_ideogram_lib.bypass import dampen_cfg as build_dampen_cfg  # P3 owner
    dampen_cfg = build_dampen_cfg(opts.get("dampen_cond_mult"), opts.get("dampen_uncond_mult"))
    r = bypass_recipe(master_strength,
                      layer_dampen_enabled=bool(opts.get("layer_dampen", False)),
                      dampen_cfg=dampen_cfg,
                      sigma_smooth_steps=int(opts.get("sigma_smooth_steps", 2) or 2))

    # Gray bypass LoRA: staged tag, the accordion's dropdown, or none.
    #
    # THE DROPDOWN USED TO DO NOTHING. Everything here hung off `r`, which is
    # built from opts["bypass_strength"] - and settings.from_ui sets that to
    # 0.0 unconditionally (a leftover from the redesign that removed the master
    # slider). So selecting "Gray_000002000.safetensors" produced:
    #
    #     bypass_lora_name  'Gray_000002000.safetensors'
    #     filter_bypass     True
    #     bypass_strength   0.0        -> recipe enabled False, lora_strength 0.0
    #
    # a named LoRA that nothing loaded, a panel that reported "bypass=off", and
    # an auto-escalation to -0.6 that moved a number no code read. The one lever
    # this project has against the gray card was never connected.
    #
    # The dropdown now applies its own file at its own strength, independently
    # of the master-strength abstraction the UI no longer exposes.
    bypass_lora_path = None
    bypass_lora_strength = r["lora_strength"]
    staged = getattr(pipe, "_i4_bypass_lora", None)
    if staged:
        bypass_lora_path = Path(staged[2])
        bypass_lora_strength = float(staged[1])  # <lora:...> tag strength wins
    elif str(opts.get("bypass_lora_name") or "").strip():
        from pi_ideogram_lib.detect import bypass_lora_path as _resolve_bypass_name
        _picked = str(opts["bypass_lora_name"]).strip()
        try:
            cand = _resolve_bypass_name(_picked, models_dirs)
        except Exception:
            cand = None
        if cand is not None:
            bypass_lora_path = Path(cand)
            amount = opts.get("bypass_lora_strength")
            bypass_lora_strength = -0.25 if amount is None else float(amount)
        else:
            print(f"{TAG} bypass LoRA '{_picked}' selected but not resolvable on disk - "
                  "running the official recipe with no bypass")
    elif r["enabled"] and bool(opts.get("bypass_lora", True)):
        from pi_ideogram_lib.detect import bypass_lora_file as _find_bypass_lora_file
        bypass_lora_path = _find_bypass_lora_file(models_dirs)
    if bypass_lora_path is not None and not bypass_lora_path.exists():
        bypass_lora_path = None
    if bypass_lora_path is None:
        bypass_lora_strength = r["lora_strength"]

    # Report what will ACTUALLY happen. log_line() reads `r`, which does not
    # know about a dropdown selection, so on its own it kept printing
    # "bypass=off strength=0.00" while a bypass LoRA was in play.
    _line = log_line(r, lora_file=bypass_lora_path.name if bypass_lora_path else None)
    if bypass_lora_path is not None and not r["enabled"]:
        _line = (f"bypass={'on' if bypass_lora_strength else 'off'} (dropdown) lora={bypass_lora_strength:+.2f} "
                 f"({bypass_lora_path.name}) first_step_only=yes "
                 f"sigma_smooth={'yes' if r['sigma_smooth'] else 'no'} "
                 f"layer_dampen={'yes' if r['layer_dampen'] else 'no'}")
    print(f"{TAG} {_line}")
    # STATE IS REPORTED WHETHER OR NOT IT IS ON. Spectrum printed a line only
    # when it was active, so an ordinary log could not distinguish "off",
    # "on but every step declined" and "wired up wrong and silently dead" -
    # and it shipped in the third state. Absence of a line is not evidence.
    _spx = ("ON (" + f"stride {spectrum_cfg.stride}, degree {spectrum_cfg.degree}"
            + ", approximate)") if spectrum_cfg.enabled else "OFF"
    _bypass_configured = bool(r['enabled'] or (bypass_lora_path is not None and bypass_lora_strength))
    print(f"{TAG} filter bypass: {'ON' if _bypass_configured else 'OFF'} | "
          f"debanner: {'ON (' + str(debanner_str) + ')' if debanner_on else 'OFF'} | "
          f"Spectrum: {_spx}")

    # --- layer dampening (P3, soft-fail) ----------------------------------------
    dampen_handles: list = []
    if r["layer_dampen"]:
        from pi_ideogram_lib.bypass import install_layer_dampen
        cfg_ = r["dampen_cfg"] or dampen_cfg
        dampen_handles += install_layer_dampen(pipe.conditional_transformer, cfg_["cond"], cfg_["cond_mult"])
        if pipe.unconditional_transformer is not None:
            dampen_handles += install_layer_dampen(pipe.unconditional_transformer, cfg_["uncond"], cfg_["uncond_mult"])
        if dampen_handles:
            print(f"{TAG} layer dampen: {len(dampen_handles)} hooks installed "
                  f"(cond x{cfg_['cond_mult']} layers {cfg_['cond']}, uncond x{cfg_['uncond_mult']} layers {cfg_['uncond']})")
        else:
            print(f"{TAG} layer dampen: no addressable blocks - soft-failed, continuing without it")
    try:
        return _run_sampling(
            p, pipe, caption_str, width, height, num_steps, guidance_schedule, mu, std,
            preset_name, quant_label, loras, r, bypass_lora_path, bypass_lora_strength,
            want_uncond, turbo, debanner_on, debanner_str, mode, cond_path,
            turbo_merge_ok=turbo_merge_ok, distilled=distilled,
            spectrum_cfg=spectrum_cfg,
            first_block_cache=0.08 if opts.get('first_block_cache_enabled', False) else 0.0,
            _FORGE_OK=_FORGE_OK, shared=shared, Processed=Processed, forge_images=forge_images,
            lora_logs=lora_logs, active_loras=active_loras,
            init_images=init_images, denoising_strength=denoising_strength,
            resize_mode=resize_mode,
        )
    finally:
        from pi_ideogram_lib.bypass import remove_layer_dampen
        remove_layer_dampen(dampen_handles)


def _run_sampling(
    p, pipe, caption_str, width, height, num_steps, guidance_schedule, mu, std,
    preset_name, quant_label, loras, r, bypass_lora_path, bypass_lora_strength,
    want_uncond, turbo, debanner_on, debanner_str, mode, cond_path,
    *, turbo_merge_ok=False, distilled=False, spectrum_cfg=None, first_block_cache=0.0,
    _FORGE_OK, shared, Processed, forge_images, lora_logs, active_loras,
    init_images=None, denoising_strength=0.75, resize_mode=0,
):
    """The actual sampling loop with the DOS tqdm bar. Split out so the
    dampen-hook cleanup in generate() always runs."""

    # --- TurboTime distill LoRA: permanent merge (cache key carries turbo). ----
    # Merged ONLY on a regular base that actually asked for the TurboTime
    # recipe (turbo_merge_ok). A distilled/Instant base or the Instant-8
    # preset never merges it (spec H stack guard - see distill_turbo_policy).
    if turbo and turbo_merge_ok:
        turbo_lora_path = getattr(pipe, "_i4_turbo_lora", None)
        # The preset alone must be enough. _i4_turbo_lora is only staged by
        # apply_lora_set when the user TYPES <lora:ideogram_4_turbotime_v1:..>
        # in the prompt, so picking "TurboTime-4" from the dropdown used to
        # print "the TurboTime LoRA is missing" and silently fall back to a
        # slow cond-only distill - even with the file sitting in
        # models/Ideogram4/loras. Look it up on disk before giving up.
        if not turbo_lora_path:
            try:
                from pi_ideogram_assets.downloader import HF_TURBOTIME, _on_disk
                from pi_ideogram_forge.paths import models_dirs
                found = _on_disk(HF_TURBOTIME[1], models_dirs())
                if found:
                    turbo_lora_path = (found.stem, 1.0, str(found))
                    pipe._i4_turbo_lora = turbo_lora_path
                    print(f"{TAG} TurboTime preset: found {found.name} on disk "
                          "(no prompt tag needed)")
            except Exception as e:
                print(f"{TAG} note: TurboTime lookup skipped ({e})")
        if turbo_lora_path:
            # Through pi_ideogram_forge.lora, NOT apply_lora_to_model directly: that
            # shortcut merged into quantised weights, merged again on every
            # repeat, and left the delta on a pipeline shared with full-CFG
            # runs. See lora.apply_turbo_adapter.
            from pi_ideogram_forge.lora import apply_turbo_adapter
            try:
                for line in apply_turbo_adapter(pipe, turbo_lora_path[0],
                                                turbo_lora_path[2],
                                                float(turbo_lora_path[1])):
                    print(f"{TAG} {line}")
                print(f"{TAG} TurboTime active - steps {num_steps}, CFG 1.0, no uncond")
            except Exception as e:
                print(f"{TAG} WARNING: TurboTime LoRA could not be applied ({e}) - running cond-only distill")
        else:
            print(f"{TAG} WARNING: TurboTime preset selected but the TurboTime LoRA is missing - "
                  "running cond-only distill at CFG 1.0 (2-8 steps). Enable the toggle to auto-download it.")
    elif turbo and getattr(pipe, "_i4_turbo_lora", None) is not None:
        reason = ("already-distilled base" if distilled
                  else f"{preset_name} is the cond-only Instant recipe")
        print(f"{TAG} TurboTime LoRA present but NOT merged ({reason}) - no stacking; "
              f"running cond-only distill at CFG 1.0")

    if not (turbo and turbo_merge_ok):
        # LEAVING THE DISTILL PRESET. A pipeline is cached and reused across
        # preset changes, so a TurboTime adapter installed for a 4-step run is
        # still installed when the next V4_DEFAULT_20 run reuses that object -
        # a 20-step full-CFG sample through distilled weights, which is what
        # "always body horror" looked like. Take it off.
        try:
            from pi_ideogram_forge.lora import remove_turbo_adapter
            for line in remove_turbo_adapter(pipe):
                print(f"{TAG} {line}")
        except Exception:
            pass

    # --- uncond-replacement LoRA (replaces the 9.3B uncond transformer) --------
    # PER-PASS, NOT MERGED. ostris's card is explicit that this adapter is
    # "used on the conditional model DURING THE UNCONDITIONAL PASS". Merging it
    # leaves it live for the conditional pass too, so the prompt-following pass
    # runs through weights trained to produce unconditional output and CFG then
    # subtracts two near-identical results. A runtime side path lets the
    # pipeline switch it on for the uncond forward alone.
    uncond_undo = None
    uncond_lora_path = getattr(pipe, "_i4_uncond_lora", None)
    if (uncond_lora_path and not turbo and not want_uncond
            and any(float(scale) != 1.0 for scale in guidance_schedule)):
        from pi_ideogram_lora.ideogram4_lora import load_lora_state_dict
        from pi_ideogram_lora.runtime_lora import RuntimeLoraAdapter
        adapter = None
        try:
            sd = load_lora_state_dict(uncond_lora_path[2])
            adapter = RuntimeLoraAdapter(pipe.conditional_transformer, sd,
                                         strength=float(uncond_lora_path[1]),
                                         name="uncond-replacement")
            n = adapter.install()
            if n == 0:
                raise RuntimeError("no module of this trunk matched the adapter")
            pipe.uncond_adapter = adapter
            uncond_undo = adapter.remove
            print(f"{TAG} {adapter.report()} - active on the UNCOND pass only; "
                  "the 9.3B uncond transformer is not loaded (low-VRAM profile)")
            lora_logs.append(adapter.report())
        except Exception as e:
            pipe.uncond_adapter = None
            if adapter is not None:
                adapter.remove()
            raise ValueError(
                "The unconditional guidance LoRA could not be applied. "
                "Check or replace ideogram_4_unconditional_lora_r16.safetensors "
                "in Model Setup, or select Full transformer with matching weights. "
                f"Generation stopped: {e}"
            ) from e

    # Load once per request, but install/undo Gray separately for EVERY image.
    # Runtime hooks work for dense and quantized trunks without weight drift.
    from pi_ideogram_lora.first_step import FirstStepAdapters
    from pi_ideogram_lora.ideogram4_lora import load_lora_state_dict
    gray_steps = None
    first_step_undo = None
    if bypass_lora_path is not None and bypass_lora_strength:
        gray_steps = FirstStepAdapters(
            [pipe.conditional_transformer, pipe.unconditional_transformer],
            load_lora_state_dict(bypass_lora_path), bypass_lora_strength)
        print(f"{TAG} Gray bypass configured at {bypass_lora_strength:+.2f}; "
              "first step of each image (runtime adapter)")

    # --- steps (bypass adds the sigma-smooth steps) -----------------------------
    extra_steps = int(r["sigma_smooth_steps"]) if r["sigma_smooth"] else 0
    actual_steps = num_steps + extra_steps
    bypass_note = f" (smooth +{extra_steps})" if extra_steps else ""

    # --- DOS header (spec Q) -----------------------------------------------------
    gw_main = float(guidance_schedule[-1]) if guidance_schedule else 7.0
    gw_tail = float(guidance_schedule[0]) if guidance_schedule else 3.0
    # IMAGE COUNT, stated up front. A Config-Preset can set the batch count
    # without that being obvious in the UI - KREA-2 PRESET-1 sets
    # txt2img_batch_count: 6 - and the only symptom was "I asked for one image
    # and it would not stop". The number and its two factors now appear before
    # sampling starts, so the answer is on screen rather than in a preset file.
    _ni = max(1, int(getattr(p, "n_iter", 1) or 1))
    _bs = max(1, int(getattr(p, "batch_size", 1) or 1))
    _tot = _ni * _bs
    _n_msg = (f"{_tot} image" if _tot == 1 else
              f"{_tot} images ({_ni} batch(es) x {_bs})")
    print(f"{TAG} [PROJECT INVISIBLE - Ideogram4] quant={quant_label}  size={width}x{height}  "
          f"steps={actual_steps}  cfg={gw_main:g}\u2192{gw_tail:g}  preset={preset_name}  "
          f"making {_n_msg}")
    print(f"{TAG} seed={_resolve_seed(p)}  loras={', '.join(n for n, _s, _p in loras) or 'none'}  "
          f"uncond={'full' if want_uncond else ('LoRA' if not turbo else 'none (turbo)')}")

    # PROVENANCE. The presets, scheduler, dual-model CFG, TE tap, VAE and
    # caption schema are the official ideogram-oss recipe. The filter-bypass
    # levers and the debanner are COMMUNITY additions, and one of them is on by
    # default (bypass_strength 0.35 -> sigma smoothing, which is why "20 steps"
    # runs 22). That is a legitimate default, but the user should never have to
    # infer it from a step count - say plainly which recipe is running.
    _mods = []
    if actual_steps != num_steps:
        _mods.append(f"sigma smoothing (+{actual_steps - num_steps} steps)")
    if bypass_lora_path and bypass_lora_strength:
        _mods.append(f"bypass LoRA {bypass_lora_strength:+.2f} first-step-only")
    if isinstance(r, dict) and r.get("layer_dampen"):
        _mods.append("layer dampen")
    if debanner_on:
        _mods.append(f"debanner {debanner_str:g}")
    if _mods:
        print(f"{TAG} recipe: OFFICIAL {preset_name} + COMMUNITY " + ", ".join(_mods))
        print(f"{TAG}         set Bypass strength to 0 (and debanner off) for the "
              "unmodified official recipe")
    else:
        print(f"{TAG} recipe: OFFICIAL {preset_name}, unmodified")

    # --- generate -----------------------------------------------------------------
    n_iter = max(1, int(p.n_iter))
    batch_size = max(1, int(p.batch_size))
    total = n_iter * batch_size
    all_images, all_seeds, all_prompts, all_infotexts = [], [], [], []
    init_images = list(init_images or [])
    is_img2img = bool(init_images)
    denoising_strength = float(denoising_strength)
    if is_img2img:
        try:
            p.extra_generation_params["Denoising strength"] = denoising_strength
            p.extra_generation_params["Resize mode"] = resize_mode
        except Exception:
            pass
    # Images already written by the per-image save, so the end-of-run pass
    # does not write them a second time.
    already_saved: list = []
    interrupted = False
    base_seed = _resolve_seed(p)
    blocked = False

    if _FORGE_OK:
        # call_queue.py already called shared.state.begin(); never call it again.
        shared.state.job_count = total
        shared.state.sampling_steps = actual_steps
        shared.state.textinfo = f"Ideogram 4 | {preset_name} | {width}x{height}"

    # The peak-VRAM counter is CUMULATIVE SINCE PROCESS START unless it is
    # reset. Printing it next to "sampling + decode done" therefore reported
    # the PARALLEL LOAD's peak (4 models at once, plus fp8 dequant temporaries)
    # as if it were the sampling peak - on a 31.8 GB card it read 34.33 GB,
    # which looks like a spill and is not one. Reset here so the number means
    # what the line says it means.
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    try:
        pipe._phase_reset = True   # new request: phase totals start from zero
    except Exception:
        pass
    t0 = time.perf_counter()
    # CONSOLE BAR - identical to what every other checkpoint prints.
    #
    # Stock Forge builds its bars with tqdm DEFAULTS (see
    # modules/shared_total_tqdm.py: tqdm.tqdm(desc=..., total=...,
    # file=shared.progress_print_out) - no bar_format), so the familiar
    # "NN%|####| n/total [elapsed<remaining, it/s]" comes from tqdm itself.
    # A custom bar_format is what made this one look foreign: no percentage,
    # no rate. Match the stock construction instead of imitating it:
    #   * tqdm defaults, no bar_format
    #   * write to shared.progress_print_out, not sys.stderr directly
    #   * honour --disable-console-progressbars and the multiple_tqdm option
    #   * drive shared.total_tqdm too, so the "Total progress" bar across a
    #     batch behaves the same as on any other model
    bar = None
    _use_console_bar = True
    try:
        if _FORGE_OK:
            _use_console_bar = not getattr(shared.cmd_opts, "disable_console_progressbars", False)
    except Exception:
        _use_console_bar = True
    if _use_console_bar:
        try:
            from tqdm import tqdm
            _out = sys.stderr
            try:
                if _FORGE_OK and getattr(shared, "progress_print_out", None) is not None:
                    _out = shared.progress_print_out
            except Exception:
                pass
            bar = tqdm(total=actual_steps, file=_out, dynamic_ncols=True, leave=False, position=0)
        except Exception:
            bar = None
    # Forge's own across-the-batch bar. reset() reads state.job_count *
    # state.sampling_steps, both set just above.
    if _FORGE_OK:
        try:
            shared.total_tqdm.clear()
            shared.total_tqdm.reset()
        except Exception:
            pass

    try:
        for n in range(n_iter):
            if interrupted:
                break
            for b in range(batch_size):
                seed = base_seed + n * batch_size + b
                output_index = n * batch_size + b
                image_actual_steps = [actual_steps]
                if _FORGE_OK:
                    shared.state.job_no = n * batch_size + b
                    if getattr(shared.opts, 'live_previews_enable', True):
                        from .preview import publish_start
                        publish_start(shared.state, width, height)
                    shared.state.sampling_step = 0
                    shared.state.textinfo = (
                        f"Ideogram 4 | Image {output_index + 1}/{total} | Step 0/{actual_steps}")

                # RESET THE PER-IMAGE BAR. It is created once, outside this
                # loop, so without this its elapsed clock keeps running across
                # the whole batch while `bar.n` restarts at every image. tqdm
                # then divides total elapsed by the current image's step count
                # and reports nonsense:
                #
                #   2/20 [06:09<55:23, 184.65s/it]      <- this bar
                #   82/120 [06:09<02:51,  4.50s/it]     <- the real rate
                #
                # 184 s/step reads as a catastrophic slowdown; the run was
                # doing 4.5 s/step. Same class of defect as the phase timers:
                # arithmetic across mismatched scopes.
                if bar is not None:
                    try:
                        bar.reset(total=actual_steps)
                        if total > 1:
                            bar.set_description(f"Ideogram4 image {n * batch_size + b + 1}/{total}")
                    except Exception:
                        pass

                def cb(step, _total):
                    if bar is not None:
                        bar.n = step + 1
                        bar.refresh()
                    if _FORGE_OK:
                        try:
                            shared.total_tqdm.update()
                        except Exception:
                            pass
                    if _FORGE_OK:
                        shared.state.sampling_step = step + 1
                        shared.state.textinfo = (
                            f"Ideogram 4 | Image {output_index + 1}/{total} | "
                            f"Step {step + 1}/{_total}")
                        if shared.state.interrupted:
                            raise InterruptedError()

                def _on_actual_steps(actual):
                    actual = int(actual)
                    image_actual_steps[0] = actual
                    if bar is not None:
                        try:
                            bar.reset(total=actual)
                        except Exception:
                            pass
                    if _FORGE_OK:
                        shared.state.sampling_steps = actual
                        try:
                            shared.total_tqdm.updateTotal(total * actual)
                        except Exception:
                            pass

                # PREVIEW CADENCE - Forge's own settings, which this ignored.
                # Someone who turned live previews OFF still paid for them, and
                # `actual_steps // 8` means EVERY step at 12 steps or fewer.
                # Measured at 1280x1728: 35-39 s per generation, ~27% of the run.
                _PREVIEW_EVERY = 1
                _PREVIEW_ON = True
                if _FORGE_OK:
                    try:
                        _PREVIEW_ON = bool(getattr(shared.opts, "live_previews_enable", True))
                        _every = int(getattr(shared.opts, "show_progress_every_n_steps", 0) or 0)
                        if _every < 0:
                            _PREVIEW_ON = False   # -1 = only after the batch completes
                        elif _every > 0:
                            _PREVIEW_EVERY = _every
                    except Exception:
                        pass
                if not _PREVIEW_ON:
                    print(f"{TAG} live previews off (Forge setting) - skipping preview decodes")
                # Our own counter, deliberately NOT shared.state.preview_step:
                # that field is Forge's gate for its set_current_image() path
                # and is reset at job boundaries, which would silently shift
                # our preview cadence mid-generation. We publish current_image
                # directly, so we owe Forge nothing here but an id bump.
                _preview_tick = [0]

                def _preview(z_latent):
                    if not _FORGE_OK or not _PREVIEW_ON:
                        return
                    _preview_tick[0] += 1
                    if _preview_tick[0] % _PREVIEW_EVERY != 0:
                        return
                    try:
                        # LINEAR PROGRESS: preview on EVERY scheduled step (the
                        # old 1.0 s wall-clock throttle skipped steps, so the
                        # image and bar jumped instead of advancing one notch
                        # per step). Forge's show_progress_every_n_steps still
                        # controls the cadence via _PREVIEW_EVERY above.
                        #
                        # COMPLETE PICTURE AT THE LAST STEP: built-in models
                        # show the finished image the moment sampling ends, not
                        # a thumbnail that lingers until the decode phase. On
                        # the final step we decode the preview at full canvas
                        # size, so the last preview IS the final image.
                        _last = shared.state.sampling_step >= max(1, image_actual_steps[0])
                        _px = max(width, height) if _last else 256
                        img = pipe.decode_preview(z_latent, preview_pixels=_px)
                        if img is not None:
                            shared.state.current_image = img
                            shared.state.id_live_preview += 1
                    except Exception:
                        pass

                try:
                    if gray_steps is not None:
                        gray_steps.begin()
                        first_step_undo = gray_steps.finish
                    pipeline_kwargs = {}
                    if is_img2img:
                        source = init_images[output_index % len(init_images)]
                        pipeline_kwargs.update(
                            init_image=forge_images.resize_image(
                                resize_mode, source, width, height),
                            strength=denoising_strength,
                        )
                    imgs = pipe(
                        caption_str, height=height, width=width, num_steps=num_steps,
                        guidance_schedule=guidance_schedule, mu=mu, std=std, seed=seed,
                        raise_on_caption_issues=False, step_callback=cb,
                        preview_callback=_preview,
                        on_actual_steps=_on_actual_steps,
                        filter_bypass=r["enabled"],
                        bypass_extra_steps=0,
                        sigma_smooth_steps=extra_steps,
                        first_step_undo=first_step_undo,
                        debanner_enabled=debanner_on, debanner_strength=debanner_str,
                        spectrum=spectrum_cfg,
                        first_block_cache=first_block_cache,
                        **pipeline_kwargs,
                    )
                except InterruptedError:
                    interrupted = True
                    break
                finally:
                    if first_step_undo is not None:
                        # pipe consumes it on step 1; on interruption it may not
                        # have run - undo defensively (idempotent-ish via clone)
                        try:
                            first_step_undo()
                        except Exception:
                            pass
                        first_step_undo = None
                if _FORGE_OK and imgs:
                    from .preview import publish_final
                    publish_final(shared.state, imgs[-1], image_actual_steps[0])
                all_images.extend(imgs)
                all_seeds.append(seed)
                all_prompts.append(str(p.prompt))
                used_steps = image_actual_steps[0]
                info = (
                    f"{str(p.prompt)}\n"
                    f"Model: Ideogram 4 ({cond_path.name})\n"
                    f"Steps: {used_steps} ({preset_name}{bypass_note}) | "
                    f"CFG schedule: main {gw_main:g}, tail {gw_tail:g}\n"
                    f"Size: {width}x{height} | Seed: {seed}\n"
                    f"Sampler: euler + Ideogram4Scheduler (logit-normal, mu={mu}, std={std})\n"
                    f"VAE: flux2-vae | TE: qwen3vl_8b 13-layer tap | Prompt mode: {mode}"
                    + (f"\nMode: img2img | Denoising strength: {denoising_strength:g} | "
                       f"Resize mode: {resize_mode}" if is_img2img else "")
                    + (f"\nLoRAs: {', '.join(n for n, _s, _p in loras)}" if loras else "")
                )
                all_infotexts.append(info)
                # SAVE NOW, not at the end of the batch. An image that has been
                # decoded is finished work; holding it in a list until every
                # other image in the batch is done means a crash, an OOM or a
                # process kill loses all of them. Writing here also means the
                # file is on disk before the NEXT image starts sampling.
                if _FORGE_OK:
                    _saved_now = _save_images(p, imgs, [seed] * len(imgs),
                                              [str(p.prompt)] * len(imgs),
                                              [info] * len(imgs),
                                              shared=shared, forge_images=forge_images)
                    already_saved.extend(imgs if _saved_now else [])
                if imgs and _looks_blocked(imgs[0]):
                    blocked = True
    finally:
        if bar is not None:
            bar.close()
        if _FORGE_OK:
            try:
                shared.total_tqdm.clear()
            except Exception:
                pass
        # uncond-replacement LoRA: remove after the whole run
        pipe.uncond_adapter = None
        if uncond_undo is not None:
            try:
                uncond_undo()
            except Exception:
                pass

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # NB: t0 is taken before the generation loop, so this covers SAMPLING +
    # decode for every image in the batch - not the VAE decode alone. It used
    # to be printed as "VAE decode done", which reads as a 168 MB decode taking
    # 200 s and sent at least one investigation chasing a VRAM problem that did
    # not exist. Say what is actually being measured.
    print(f"{TAG} sampling + decode done: {elapsed_ms:.0f} ms "
          f"({len(all_images)} image(s)) | peak VRAM {_peak_vram_gb():.2f} GB")
    # WHERE the time went, not just how much. A single total invites guessing;
    # this line ends the guessing.
    _ph = getattr(pipe, "_phase_ms", None)
    if _ph:
        # The transformer forwards are timed with an explicit synchronize, so
        # they are REAL GPU time rather than queue-submission time. Everything
        # else inside "sampling" is scheduler arithmetic and the CFG blend.
        _fc, _fu = _ph.pop("fwd_cond", 0.0), _ph.pop("fwd_uncond", 0.0)
        if interrupted:
            # The pipeline's end-of-loop timer is not reached on interruption.
            _ph['steps'] = max(_ph['steps'], elapsed_ms - _ph['encode'] - _ph['decode'])
        if _fc or _fu:
            print(f"{TAG} forwards: cond {_fc / 1000:.1f}s | uncond {_fu / 1000:.1f}s | "
                  f"rest of sampling {max(0.0, _ph['steps'] - _fc - _fu) / 1000:.1f}s")
        # Preview time is already inside sampling, not a separate addition.
        _known = _ph['encode'] + _ph['steps'] + _ph['decode']
        print(f"{TAG} breakdown: text-encode {_ph['encode'] / 1000:.1f}s | "
              f"sampling {_ph['steps'] / 1000:.1f}s | "
              f"previews {_ph['preview'] / 1000:.1f}s | "
              f"final decode {_ph['decode'] / 1000:.1f}s | "
              f"elsewhere {max(0.0, elapsed_ms - _known) / 1000:.1f}s")

    if blocked:
        if _FORGE_OK:
            _safe_comment(p, BLOCKED_HINT)
        print(f"{TAG} {BLOCKED_HINT}")

    # SAVING AN INTERRUPTED BATCH.
    #
    # Forge's own gate is
    #     save_samples() = opts.samples_save and not do_not_save_samples
    #                      and (opts.save_incomplete_images
    #                           or not state.interrupted and not state.skipped)
    # so pressing Stop discards everything, including images that had already
    # finished. That default is right for a stock sampler, where interrupting
    # mid-denoise hands back a half-denoised picture.
    #
    # It is wrong here. In this loop an image reaches all_images only after its
    # final VAE decode, and the interrupt does `break` BEFORE that append - so
    # every image we are holding is complete. A user who stopped a six-image
    # batch they never asked for should not also lose the three that finished.
    #
    # The two settings that actually express "do not write files" are still
    # obeyed; only the interrupted-batch clause is set aside, and out loud.
    # Anything the per-image save already wrote is skipped here, so a batch is
    # never written twice. This pass exists for the images that could not be
    # saved as they finished - the interrupted case, where the state flags are
    # only set after the loop has broken.
    _pending = [(i, img) for i, img in enumerate(all_images) if img not in already_saved]
    if _pending and _FORGE_OK:
        _seeds = [all_seeds[i] if i < len(all_seeds) else base_seed for i, _ in _pending]
        _prompts = [all_prompts[i] if i < len(all_prompts) else str(p.prompt) for i, _ in _pending]
        _infos = [all_infotexts[i] if i < len(all_infotexts) else "" for i, _ in _pending]
        _save_images(p, [img for _, img in _pending], _seeds, _prompts, _infos,
                     shared=shared, forge_images=forge_images, announce_stop=True)

    print(f"{TAG} {'Interrupted' if interrupted else 'Done'}: {len(all_images)} completed image(s) at {width}x{height}, {preset_name}")
    for line in lora_logs:
        print(f"{TAG} {line}")

    if _FORGE_OK:
        return Processed(p, all_images, seed=base_seed,
                         info=all_infotexts[0] if all_infotexts else "",
                         all_prompts=all_prompts, all_seeds=all_seeds,
                         infotexts=all_infotexts)
    return None


def _chain_undo(*undos):
    """Compose several undo closures into one (all called, all guarded)."""
    def _run():
        for u in undos:
            if u is None:
                continue
            try:
                u()
            except Exception:
                pass
    return _run


# ---------------------------------------------------------------------------
# Checkpoint detection - the ACTIVE Ideogram 4 file (pi_ideogram_forge/selection.py owns
# current-selection identity). Resolved lazily so this module imports in any
# order; when Forge is absent the caller passes get_pipeline_fn/selected_path
# explicitly and this stays None.
# ---------------------------------------------------------------------------
try:
    from pi_ideogram_forge.selection import _current_ideogram_file as _get_ideogram_file  # type: ignore[assignment]
except Exception:
    _get_ideogram_file = None  # type: ignore[assignment]
