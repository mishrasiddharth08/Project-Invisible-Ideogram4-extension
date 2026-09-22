# Project Invisible: namespaced Forge adapter; no core-file edits.
"""PROJECT INVISIBLE - the invisible takeover (one owner: what happens when
Generate is pressed while an Ideogram 4 checkpoint is selected).

Owns the session state that makes one generation behave like the last one did
without leaking into the next:
    _ORIGINAL_RUN   - the stock ScriptRunner.run we wrap (passthrough)
    _ACTIVE_LORAS   - LoRA set resolved for the CURRENT generate (cache key input)
    _PIPELINE_CACHE - loaded pipelines keyed by all model files, LoRAs and mode
    _DL_ATTEMPTED   - adapter kinds already attempted this session (download once)

Data flows ONE way and is documented in ARCHITECTURE.md:
    engine.py (_wrapped_run entry) -> this module resolves the accordion slice
    via settings (pi_ideogram_lib/settings.py) -> pi_ideogram_forge/generate.py over a
    RESOLVED dict -> pipeline + LoRA owners.  This module never reads Gradio
    values or config.json itself; pi_ideogram_forge/generate.py never reads either.

scripts/engine.py imports this module for the Script UI and installs the
wrapper at boot (install_wrapper); pi_ideogram_forge/selection.py answers "is an
Ideogram 4 checkpoint selected right now".  Forge imports (modules.shared /
modules.processing) are guarded so this module imports standalone; every
Forge-interface function still runs verbatim inside Forge.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

_EXT_DIR = Path(__file__).resolve().parent.parent
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))

try:
    from modules import shared
    from modules.processing import Processed, StableDiffusionProcessing
    _FORGE_OK = True
except Exception as _e:
    print(f"[Invisible-I4] WARNING: run.py Forge imports failed: {_e}")
    shared = None
    Processed = None
    StableDiffusionProcessing = object
    _FORGE_OK = False

import pi_ideogram_lib.settings as pi_settings  # noqa: E402  (single owner of run settings)
from pi_ideogram_lib import json_prompt  # noqa: E402
import pi_ideogram_forge.lora as lora  # noqa: E402
from pi_ideogram_forge.paths import models_dirs  # noqa: E402
from pi_ideogram_forge.selection import (  # noqa: E402
    _current_ideogram_file,
    is_ideogram4_selected,
)
from pi_ideogram_forge.generate import (  # noqa: E402
    generate as _generate_fn,
    _looks_blocked,
    _resolve_seed,
    _safe_comment,
)
from pi_ideogram_forge.routing import _enforce_loadable_policy  # noqa: E402

TAG = "[Invisible-I4]"

_ORIGINAL_RUN = None
_ORIGINAL_PROCESS_IMAGES = None

# Engine-owned session state (single owner per piece, see ARCHITECTURE.md):
_ACTIVE_LORAS: list[tuple[str, float, str]] = []  # (name, strength, policy)
_DL_ATTEMPTED: set[str] = set()

_PIPELINE_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# pipeline cache (one owner of loaded Ideogram4Pipeline instances)
# --------------------------------------------------------------------------- #

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _compute_dtype():
    from pi_ideogram_lib.device_policy import compute_dtype
    return compute_dtype('cuda' if _cuda_available() else 'cpu')


def _file_identity(path):
    """Cheap cache identity; no multi-gigabyte hashing on the Generate path."""
    if path is None:
        return None
    p = Path(path).resolve()
    try:
        if p.is_dir():
            # HF encoder folders may replace a shard without changing the
            # folder timestamp. Include the config, tokenizer and weight files.
            return (str(p), tuple((str(f.relative_to(p)), f.stat().st_size,
                                  f.stat().st_mtime_ns)
                                 for f in sorted(p.rglob("*")) if f.is_file()
                                 and f.suffix in {".json", ".safetensors", ".bin", ".model"}))
        stat = p.stat()
        return (str(p), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return (str(p), None, None)


def _get_pipeline(selected_path: Path, want_uncond: bool = True, turbo: bool = False):
    """Cache key includes the active LoRA set + turbo flag + uncond mode, so
    deltas never leak between generations with different settings.

    REVERSE PAIRING: if the user selected the *unconditional* half, we swap in
    its conditional twin. want_uncond=False skips loading the 9.3B uncond
    transformer (turbo / uncond-LoRA profiles); turbo also never needs it."""
    from pi_ideogram_lib.detect import (find_matching_cond, find_matching_uncond,
                                        is_unconditional_file, model_search_dirs,
                                        resolve_te_vae)
    from pi_ideogram_lib.pipeline import Ideogram4Pipeline

    search = model_search_dirs(models_dirs())
    if is_unconditional_file(selected_path):
        twin = find_matching_cond(selected_path, search)
        if twin is not None:
            print(f"{TAG} you picked the unconditional file; running the official pair "
                  f"(cond: {twin.name} + uncond: {selected_path.name})")
            cond_path, uncond_path = twin, selected_path
            want_uncond = True  # the user explicitly wants the pair
        else:
            print(f"{TAG} WARNING: no conditional twin for {selected_path.name}; using it as cond "
                  "in degraded single-model mode")
            cond_path, uncond_path = selected_path, None
    else:
        cond_path = selected_path
        uncond_path = None
        if want_uncond and not turbo:
            uncond_path = find_matching_uncond(selected_path, search)
            if uncond_path is None:
                print(f"{TAG} WARNING: no matching unconditional checkpoint for {cond_path.name}; "
                      "the Ostris uncond LoRA profile can replace it (enable 'Ostris uncond LoRA')")

    # Remember it BY PATH for the next launch (see detect.remember_checkpoint):
    # Forge saves our checkpoints by display label, which is one rename away
    # from not resolving.
    try:
        from pi_ideogram_lib.detect import remember_checkpoint
        for line in remember_checkpoint(cond_path):
            print(f"{TAG} {line}")
    except Exception:
        pass

    # Resolve BEFORE looking in the cache. A different TE or VAE is a different
    # pipeline, even when the checkpoint and prompt have not changed.
    te, vae, _te_vae_notes = resolve_te_vae(models_dirs())
    lora_key = tuple((str(n), float(s), str(policy),
                      _file_identity(lora.find_lora_file(str(n), models_dirs())))
                     for n, s, policy in _ACTIVE_LORAS)
    key = (_file_identity(cond_path), _file_identity(uncond_path),
           _file_identity(te), _file_identity(vae), lora_key, bool(turbo))
    if (key in _PIPELINE_CACHE and _PIPELINE_CACHE[key].loaded
            and not getattr(_PIPELINE_CACHE[key], "_i4_turbo_merged", False)):
        return _PIPELINE_CACHE[key]

    # REUSE ACROSS THE TURBO BOUNDARY. `turbo` is part of the key because a
    # turbo pipeline deliberately skips loading the 9.3 GB uncond transformer.
    # But the reverse direction costs nothing: a turbo run samples at CFG 1.0,
    # and the loop only touches the uncond model when `gw_i != 1.0`, so a
    # FULL pipeline already on the card can serve a turbo request untouched.
    #
    # Without this, flipping the preset from V4_DEFAULT_20 to Instant-8 -
    # exactly what someone does when they want the fast path - evicted ~27 GB
    # and paid a full 17 s reload to run a recipe the loaded weights could
    # already produce. The wrong direction (turbo pipeline asked for a real
    # CFG run) still reloads: those weights genuinely are not present.
    if turbo or not want_uncond:
        for k, cached in list(_PIPELINE_CACHE.items()):
            try:
                if (not cached.loaded
                        or k[0] != key[0] or k[2:5] != key[2:5] or k[-1]):
                    continue
                # A pipeline whose TurboTime adapter was MERGED (bf16 trunk)
                # can never be given back: nothing un-merges it. Reusing it
                # here is how a distill delta reached full-CFG generations.
                # Quantised trunks take the removable runtime path instead and
                # are safe to share - see lora.apply_turbo_adapter.
                if getattr(cached, "_i4_turbo_merged", False):
                    continue
            except Exception:
                continue
            # THE OOM LADDER'S WHOLE POINT IS TO FREE THE UNCOND HALF.
            #
            # Reusing the cached pipeline as-is defeated it: the dual-
            # transformer run died with "OOM: 0.0GB free", switched to the
            # cond-only path, reused the SAME object - still holding the
            # 9.3 GB unconditional transformer - and OOM'd again with exactly
            # as much memory as before. Drop that half here so the retry
            # genuinely has room, and evict the dual-path cache entry so a
            # later full-CFG run reloads it rather than finding it missing.
            if not want_uncond and getattr(cached, "unconditional_transformer", None) is not None:
                try:
                    # Streaming hooks hold block references too; detach them
                    # before dropping the model so its RAM can really be freed.
                    memory = getattr(cached, "memory", None)
                    if memory is not None:
                        memory.forget_model(cached.unconditional_transformer)
                    retained = []
                    for adapter in getattr(cached, "_i4_runtime_adapters", None) or []:
                        if getattr(adapter, "_model", None) is cached.unconditional_transformer:
                            adapter.remove()
                        else:
                            retained.append(adapter)
                    cached._i4_runtime_adapters = retained
                    adapter = None
                    cached.unconditional_transformer = None
                    _PIPELINE_CACHE.pop(k, None)
                    import gc as _gc
                    _gc.collect()
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                    print(f"{TAG} released the unconditional transformer (~9.3 GB) - this run "
                          "uses the uncond-LoRA path and does not need it")
                except Exception:
                    continue
            print(f"{TAG} reusing the loaded pipeline for the distill run "
                  "(its uncond half is simply unused at CFG 1.0) - no reload")
            # Never leave a full-CFG alias pointing at an object which a later
            # TurboTime run can permanently modify on a BF16 checkpoint.
            for alias, value in list(_PIPELINE_CACHE.items()):
                if value is cached:
                    _PIPELINE_CACHE.pop(alias, None)
            _PIPELINE_CACHE[key] = cached
            return cached

    for k in list(_PIPELINE_CACHE):
        try:
            _PIPELINE_CACHE[k].unload()
        except Exception:
            pass
        _PIPELINE_CACHE.pop(k, None)

    # Forge Neo remembers the VAE/TE selected for this checkpoint and restores
    # them on the next launch; honour that selection instead of re-picking from
    # disk. Only when nothing is selected do we fall back to the scan.
    for _n in _te_vae_notes:
        print(f"{TAG} {_n}")
    missing = [lbl for lbl, p in (("conditional checkpoint", cond_path),
                                  ("text encoder (qwen3vl_8b_*.safetensors under models/text_encoder)", te),
                                  ("VAE (flux2-vae.safetensors under models/VAE)", vae)) if p is None]
    if missing:
        raise FileNotFoundError("[Invisible-I4] Missing required files: " + ", ".join(missing) +
                                ".\nSee https://huggingface.co/Comfy-Org/Ideogram-4 for the ungated "
                                "single files (qwen3vl_8b_fp8_scaled.safetensors + flux2-vae.safetensors).")

    pipe = Ideogram4Pipeline(cond_path=cond_path, uncond_path=uncond_path, te_path=te, vae_path=vae,
                             device="cuda" if _cuda_available() else "cpu", dtype=_compute_dtype())
    pipe._i4_intentional_single = (uncond_path is None)
    pipe.load()
    _PIPELINE_CACHE[key] = pipe
    return pipe


# --------------------------------------------------------------------------- #
# accordion values -> resolved run settings (entry point of the data flow)
# --------------------------------------------------------------------------- #

def _our_alwayson_script(runner):
    """Find THIS extension's always-visible Script instance inside a ScriptRunner.
    Forge sets args_from/args_to on it during setup_ui; they index our accordion
    component values inside p.script_args."""
    for s in getattr(runner, "alwayson_scripts", ()) or ():
        if getattr(s, "_i4_alwayson", False):
            return s
    return None


def _resolve_run_opts(runner, p) -> dict:
    """Resolved run settings for THIS generation - the ONLY place the accordion
    values enter the engine.

    Always-visible process() only runs inside processing.process_images, which
    the takeover skips, so we slice our component values straight out of
    p.script_args using args_from/args_to - the exact tuple process() would
    have received. Any failure (missing script, short arg list, bad values)
    falls back to config defaults; this never crashes a generate."""
    ui_values = None
    try:
        script = _our_alwayson_script(runner)
        if script is not None:
            lo = int(getattr(script, "args_from", -1))
            hi = int(getattr(script, "args_to", -1))
            arr = getattr(p, "script_args", None)
            if lo >= 0 and hi > lo and isinstance(arr, (tuple, list)) and len(arr) >= hi:
                cand = tuple(arr[lo:hi])
                if len(cand) == len(pi_settings.UI_KEYS):
                    ui_values = cand
    except Exception:
        ui_values = None
    try:
        opts = pi_settings.resolve(ui_values=ui_values)
        if ui_values is not None:
            pi_settings.persist(ui_values)  # remember last-used values (spec M)
        return opts
    except Exception:
        return pi_settings.resolve()


# --------------------------------------------------------------------------- #
# adapter auto-download (download-once per session, spec F + P1)
# --------------------------------------------------------------------------- #

def _maybe_download_adapters(opts: dict) -> None:
    """Name the adapter a feature needs, and where to get it. Never downloads.

    This used to fetch files over the network the first time a feature was
    switched on. It no longer does anything of the sort: the Model Setup panel
    carries clickable links and the destination folder, downloads happen in the
    browser where the user is already signed in, and there is no API key
    anywhere in this extension.

    Kept as a function because the takeover calls it once per generation and a
    missing adapter is worth one clear line rather than a silent fallback.
    """
    try:
        from pi_ideogram_lib.detect import lora_search_dirs
        from pi_ideogram_assets.downloader import wanted_adapters, _on_disk, ASSET_LABELS
        from pi_ideogram_assets.links import ADAPTERS
    except Exception:
        return
    try:
        wanted = list(wanted_adapters(opts) or [])
    except Exception:
        return
    if not wanted:
        return
    by_kind = {"turbotime": 0, "uncond": 1, "gray": 2}
    dirs = models_dirs()
    for kind in wanted:
        if kind in _DL_ATTEMPTED:
            continue
        idx = by_kind.get(kind)
        if idx is None:
            continue
        label, filename, url, dest, _note = ADAPTERS[idx]
        try:
            if _on_disk(filename, dirs) is not None:
                continue
        except Exception:
            pass
        _DL_ATTEMPTED.add(kind)  # say it once per session, not once per image
        print(f"{TAG} {label} is not installed. Download {filename} from "
              f"{url} and drop it into {dest} (subfolders are searched).")


def generate(p, **kw):
    """Direct entry for harnesses/tests: single-arg generate(p) runs the resolved
    config defaults with engine-owned dependencies bound. Forge UI generations
    go through ScriptRunner.run -> _wrapped_run, never through here."""
    kw.setdefault("opts", pi_settings.resolve())
    kw.setdefault("models_dirs", models_dirs())
    kw.setdefault("get_pipeline_fn", _get_pipeline)
    kw.setdefault("lora_mod", lora)
    kw.setdefault("json_prompt_mod", json_prompt)
    kw.setdefault("active_loras", _ACTIVE_LORAS)
    return _generate_fn(p, **kw)


# --------------------------------------------------------------------------- #
# OOM recovery ladder (playtest defect 2) + blocked-image auto-escalation + run
# --------------------------------------------------------------------------- #

def _generate_with_oom_recovery(p, opts: dict, *, orig_w: int, orig_h: int,
                                orig_bs: int, generation_kwargs=None):
    """One generate() with the full OOM recovery ladder (single owner of the
    OOM decision). Order:

      1. full settings exactly as requested;
      2. OOM #1 on the DUAL-transformer path -> ONE automatic retry at the SAME
         prompt/seed/settings on the cond-only + Ostris uncond-LoRA path (~9 GB
         lighter; the LoRA auto-downloads through the existing machinery).
         Config `oom_retry_uncond_lora=false` disables the swap; a NON-OOM
         failure of the swapped retry surfaces the ORIGINAL OOM; a run already
         on the single-transformer path (turbo / uncond-LoRA) never swaps;
      3. any further OOM (swapped path, already-single runs, flag off) runs the
         Forge degradation ladder: batch_size halve -> batch 1 + 0.75x
         resolution (16-aligned) -> a graceful error-card Processed (never
         raised to the UI).

    Non-OOM errors always raise (unchanged). Returns Processed."""

    def _attempt(run_opts):
        """One generate() attempt. Returns Processed, or the OOM exception for
        the ladder to decide (non-OOM errors propagate)."""
        try:
            return _generate_fn(p, opts=run_opts, active_loras=_ACTIVE_LORAS,
                                models_dirs=models_dirs(), get_pipeline_fn=_get_pipeline,
                                lora_mod=lora, json_prompt_mod=json_prompt,
                                **(generation_kwargs or {}))
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" not in msg:
                raise
            try:
                import torch
                free, total = torch.cuda.mem_get_info(0)
                print(f"{TAG} OOM: {free / 1024**3:.1f}GB free / {total / 1024**3:.1f}GB total")
            except Exception:
                pass
            return e

    def _swap_allowed(run_opts) -> bool:
        """Dual-transformer run + config flag on => eligible for the one
        cond-only + uncond-LoRA path swap. Never on an already-single run."""
        try:
            import pi_ideogram_lib.config as _cfg
            if not _cfg.get("oom_retry_uncond_lora", True):
                return False
        except Exception:
            pass
        if bool(run_opts.get("turbo")):
            return False
        if bool(run_opts.get("use_uncond_lora")):
            return False
        return True

    p.batch_size = orig_bs
    p.width, p.height = orig_w, orig_h
    out = _attempt(opts)
    if not isinstance(out, Exception):
        return out
    first_oom = out
    if not opts.get('allow_quality_reducing_oom_recovery', False):
        raise RuntimeError('Ideogram 4 ran out of memory. Your resolution, guidance model and adapter precision '
                           'were NOT reduced. Close other GPU jobs, select portable FP8, or explicitly choose '
                           'a smaller canvas. Quality-reducing automatic recovery is off.') from first_oom
    current_opts = opts
    if _swap_allowed(opts):
        current_opts = dict(opts)
        current_opts["use_uncond_lora"] = True
        print(f"{TAG} OOM on dual-transformer path -> retried on the uncond-LoRA path "
              "(cond-only), seed preserved")
        try:
            _maybe_download_adapters(current_opts)  # ostris uncond LoRA (spec E/F)
        except Exception:
            pass
        try:
            out = _attempt(current_opts)
            if not isinstance(out, Exception):
                return out
        except Exception:
            # the swapped retry died for a non-OOM reason - the truth the user
            # needs is the ORIGINAL OOM, not the retry's new failure
            raise first_oom
    # RUNTIME LORA -> MERGE. The exact side path keeps adapters out of the
    # quantised weights (cosine 1.000000 vs 0.57 for an fp8 merge), but it
    # holds the factors on the card and allocates per hooked module. On a
    # ~16 GB-resident dual-transformer run at 1024px that was enough to OOM.
    #
    # Merging instead is measurably worse - ~43% of the adapter arrives as
    # quantisation noise - but it is a QUALITY loss, and the alternative here
    # is no image at all. Degrade the adapter before degrading the picture:
    # this rung comes BEFORE batch and resolution cuts.
    if current_opts.get("lora_runtime") is not False:
        current_opts = dict(current_opts)
        current_opts["lora_runtime"] = False
        print(f"{TAG} OOM recovery: applying LoRAs by MERGE instead of the runtime "
              "side path (frees the adapter's GPU factors; the merge loses some "
              "adapter fidelity on quantised weights)")
        try:
            out = _attempt(current_opts)
            if not isinstance(out, Exception):
                return out
        except Exception:
            raise first_oom

    # batch halve (Forge degradation ladder)
    new_bs = max(1, int(orig_bs) // 2)
    p.batch_size = new_bs
    print(f"{TAG} OOM recovery: batch_size {orig_bs} -> {new_bs}")
    out = _attempt(current_opts)
    if not isinstance(out, Exception):
        return out
    # batch 1 + resolution *0.75 (16-aligned)
    p.batch_size = 1
    new_w = max(256, int(orig_w * 0.75)) - (max(256, int(orig_w * 0.75)) % 16)
    new_h = max(256, int(orig_h * 0.75)) - (max(256, int(orig_h * 0.75)) % 16)
    p.width, p.height = new_w, new_h
    print(f"{TAG} OOM recovery: resolution {orig_w}x{orig_h} -> {new_w}x{new_h}, batch_size=1")
    out = _attempt(current_opts)
    if not isinstance(out, Exception):
        return out
    oom_msg = (f"Ideogram 4: out of VRAM at {p.width}x{p.height} batch={p.batch_size}. "
               "Try a lower resolution, the uncond-LoRA mode, or a smaller quant.")
    _safe_comment(p, oom_msg)
    try:
        seed = int(p.seed) if p.seed is not None else -1
    except (TypeError, ValueError):
        seed = -1
    return Processed(p, [], seed, oom_msg, infotexts=[oom_msg])


# The bypass panel is a LoRA + a strength, so escalation walks the LoRA
# strength, not a master mix. Negative is the working direction (the Civitai
# author's recipe is -0.25); -0.60 is the deeper retry for a stubborn card.
BYPASS_ANCHORS: tuple[float, ...] = (-0.25, -0.60)


def _next_bypass_anchor(strength: float) -> float | None:
    """Next bypass-LoRA strength strictly STRONGER (more negative) than
    `strength`. None means we are already at the strongest anchor."""
    try:
        cur = float(strength)
    except (TypeError, ValueError):
        cur = 0.0
    for anchor in BYPASS_ANCHORS:
        if cur > anchor + 1e-6:
            return anchor
    return None


def _merged_lora_active(proc, opts) -> bool:
    """Is a style/character adapter merged into the weights for this run?"""
    try:
        if bool(_ACTIVE_LORAS):
            return True
    except Exception:
        pass
    return False


def _auto_escalate_blocked(p, opts: dict, proc, run_once) -> object:
    """BLOCKED-IMAGE AUTO-RETRY: when a run comes back as a gray 'Image blocked'
    image, re-run it ONCE at the next bypass-strength anchor (same seed) so the
    checkbox/slider mechanism finishes the job instead of handing the user a gray
    image plus advice.

    Respects the master checkbox (bypass OFF = never escalate - the user asked
    for the raw output) and the strength ceiling (1.0 = already at max, hint
    only). Can be disabled entirely with config auto_escalate_blocked=false.
    Never raises: any failure keeps the original Processed."""
    try:
        try:
            import pi_ideogram_lib.config as pi_config
            if not pi_config.get("auto_escalate_blocked", True):
                return proc
        except Exception:
            pass
        imgs = list(getattr(proc, "images", None) or [])
        if not imgs or not _looks_blocked(imgs[0]):
            return proc
        # THE USER PRESSED STOP. A retry here reads as "I asked for one image
        # and it kept going", because the second run starts before the first
        # result is even on screen and Forge's own interrupt has already been
        # consumed. Whatever we think of the image, they asked us to stop.
        try:
            from modules import shared as _shared
            if (getattr(_shared.state, "interrupted", False)
                    or getattr(_shared.state, "skipped", False)
                    or getattr(_shared.state, "stopping_generation", False)):
                print(f"{TAG} image came back blocked, but you pressed Stop - not retrying.")
                return proc
        except Exception:
            pass
        # TWO ladders, and the second one is the important one.
        #
        # With a bypass LoRA selected, retry at the next strength anchor. But
        # this used to RETURN IMMEDIATELY when no LoRA was selected - which is
        # the normal case, since the gray-bypass LoRA is gated behind a Civitai
        # login and most installs do not have it. So the one remedy that is
        # always available, and is the one the project measured as effective,
        # was never tried: caption DENSITY. The gray card is a sparse-caption
        # false positive (a brass compass on a nautical chart triggered it), so
        # the answer is more structure in the caption, not a heavier weight
        # delta. It is prompt-side, needs no download, and leaves the sampler,
        # the schedule and the weights exactly as the official recipe has them.
        retry_opts = dict(opts)
        anchor = _next_bypass_anchor(opts.get("bypass_lora_strength", 0.0))
        if opts.get("bypass_lora_name") and anchor is not None:
            retry_opts["bypass_lora_strength"] = anchor
            print(f"{TAG} blocked image detected - auto-escalating bypass LoRA strength to {anchor} "
                  f"(same seed, one retry)")
        else:
            density = int(opts.get("caption_density", 3) or 3)
            if density >= 6:
                print(f"{TAG} blocked image detected - caption density already at {density}; "
                      "try a richer prompt, or raw JSON with your own elements")
                return proc
            retry_opts["caption_density"] = 6
            print(f"{TAG} blocked image detected - retrying with a DENSER caption "
                  f"({density} -> 6 bounding boxes, same seed). No bypass LoRA needed: "
                  "the gray card is a sparse-caption misfire.")
        proc2 = run_once(retry_opts)
        try:
            imgs2 = list(getattr(proc2, "images", None) or [])
            if imgs2 and _looks_blocked(imgs2[0]):
                # "Blocked" is a GUESS from pixel statistics: a flat, low-variance
                # image. A LoRA merged into quantised weights produces exactly
                # that too - measured at cosine 0.57, i.e. ~43% of the adapter
                # arriving as noise across 204 modules. Naming the safety filter
                # as the cause when a merged adapter is loaded would send the
                # user after the wrong thing, so say what is actually known.
                extra = ""
                if opts.get("bypass_lora_name") or _merged_lora_active(proc, opts):
                    extra = (" NOTE: a LoRA is merged into these quantised weights. "
                             "Merging is lossy on fp8/int8 (cosine 0.57), and a damaged "
                             "model looks the same to this detector as a filter card. "
                             "Generate once WITHOUT the LoRA to tell the two apart.")
                print(f"{TAG} still flat/grey after the denser caption - "
                      "this may be the safety filter OR a lossy LoRA merge." + extra)
        except Exception:
            pass
        return proc2 if proc2 is not None else proc
    except Exception:
        return proc


def _img2img_generation_kwargs(p):
    """Validate Forge img2img state and return optional pipeline arguments."""
    init_images = list(getattr(p, "init_images", None) or [])
    if not init_images:
        return {}
    class_names = {c.__name__ for c in type(p).__mro__}
    is_img2img = (bool(getattr(p, "is_img2img", False)) or
                  any("Img2Img" in name for name in class_names))
    if not is_img2img:
        raise ValueError("Ideogram 4 init images are accepted only from Forge's img2img tab.")
    for name in ("image_mask", "mask", "mask_for_overlay", "latent_mask"):
        if getattr(p, name, None) is not None:
            raise ValueError("Ideogram 4 experimental img2img does not support masks or inpainting.")
    resize_mode = int(getattr(p, "resize_mode", 0) or 0)
    if resize_mode == 3:
        raise ValueError("Ideogram 4 experimental img2img does not support resize mode 3. "
                         "Choose resize, crop and resize, or resize and fill.")
    raw_strength = getattr(p, "denoising_strength", None)
    strength = 0.75 if raw_strength is None else float(raw_strength)
    import math
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("Denoising strength must be between 0 and 1.")
    return dict(init_images=init_images, denoising_strength=strength,
                resize_mode=resize_mode)


def _take_over(runner, p):
    from . import lifecycle
    from modules import shared
    with lifecycle.lock:
        lifecycle.release_changed(_PIPELINE_CACHE, shared.opts.sd_model_checkpoint)
        lifecycle.selection = shared.opts.sd_model_checkpoint
        try:
            return _take_over_locked(runner, p)
        finally:
            lifecycle.release_changed(_PIPELINE_CACHE, shared.opts.sd_model_checkpoint)


def _take_over_locked(runner, p):
    """Run the official Ideogram 4 pipeline for `p` and return a Processed.

    Single owner of the takeover body. Two entry points reach it:
      * _wrapped_run           - the stock Generate button (ScriptRunner.run)
      * _wrapped_process_images - the API and any direct process_images caller

    `runner` may be None: _resolve_run_opts only needs it to slice accordion
    values out of p.script_args, and falls back to the persisted config
    defaults when there is no ScriptRunner (which is exactly the API case).
    """
    if True:
        try:
            generation_kwargs = _img2img_generation_kwargs(p)
            print(f"{TAG} stock Generate -> official Ideogram 4 pipeline (invisible takeover)")
            # ROUTING (audit R): the DETECTED format is the run-time authority -
            # refuse a confident non-in-process format with the documented fallback,
            # log the merge/loader policy line for everything else.
            try:
                _enforce_loadable_policy(_current_ideogram_file())
            except ValueError:
                raise
            except Exception as e:
                print(f"{TAG} [routing] policy check skipped ({e}) - proceeding")
            # Resolve the accordion values -> run settings (single owner: settings.py)
            opts = _resolve_run_opts(runner, p)
            _maybe_download_adapters(opts)
            # Fix the seed ONCE so an auto-escalation retry re-runs the SAME image
            try:
                _resolve_seed(p)
            except Exception:
                pass
            orig_w, orig_h, orig_bs = int(p.width), int(p.height), int(p.batch_size)

            def _run_with_recovery(run_opts):
                return _generate_with_oom_recovery(p, run_opts,
                                                   orig_w=orig_w, orig_h=orig_h, orig_bs=orig_bs,
                                                   generation_kwargs=generation_kwargs)

            proc = _run_with_recovery(opts)
            # gray 'blocked' output -> one same-seed retry at the next strength anchor
            return _auto_escalate_blocked(p, opts, proc, _run_with_recovery)
        except InterruptedError:
            raise
        except Exception as e:
            print(f"{TAG} generation failed: {e}")
            traceback.print_exc()
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (1024, 256), "white")
            ImageDraw.Draw(img).text((16, 16), f"Ideogram 4 generation error:\n{str(e)[:400]}", fill="black")
            _safe_comment(p, f"Ideogram 4: {e}")
            return Processed(p, [img], -1 if p.seed is None else int(p.seed), f"Ideogram 4 error: {e}")


def _wrapped_run(self, p, *args, **kwargs):
    """Stock Generate button. txt2img.py skips process_images when this returns
    a Processed, so the two entry points never both fire for one generation.

    *args/**kwargs are forwarded VERBATIM. This wrapper must stay transparent
    to a Forge Neo update: pinning today's signature would turn a new upstream
    parameter into a TypeError on every non-Ideogram generation - i.e. this
    extension breaking models it has nothing to do with.
    """
    if is_ideogram4_selected():
        return _take_over(self, p)
    # stock behavior for every other checkpoint
    return _ORIGINAL_RUN(self, p, *args, **kwargs)


def _wrapped_process_images(p, *args, **kwargs):
    """Guard for callers that reach processing.process_images DIRECTLY.

    modules/api/api.py does exactly that (`processed = process_images(p)`),
    never going through ScriptRunner.run - so before this hook existed, an API
    txt2img against an Ideogram 4 checkpoint bypassed the takeover entirely and
    died inside Forge's own loader with 'Failed to recognize model...'. Any
    extension that calls process_images itself had the same hole.

    Every other checkpoint passes straight through untouched.
    """
    if is_ideogram4_selected():
        print(f"{TAG} process_images() called directly (API or extension) -> "
              "official Ideogram 4 pipeline")
        # No ScriptRunner here, so no accordion values: _take_over falls back to
        # the persisted config defaults, which is the documented API behaviour.
        return _take_over(None, p)
    return _ORIGINAL_PROCESS_IMAGES(p, *args, **kwargs)


def _install_process_images_guard():
    """Rebind processing.process_images in known Forge entry points only.

    `modules/api/api.py` imports the FUNCTION by name
    (`from modules.processing import ... process_images`), so patching only
    modules.processing would leave api.py's own binding pointing at the
    original. We patch the source and known API/txt2img/img2img aliases that
    still hold that same function. Unrelated extensions are never scanned or
    rewritten. New upstream entry points require an explicit compatibility review.

    Fail-soft: any problem here leaves the stock function in place; only the
    API path loses coverage, exactly as before.
    """
    global _ORIGINAL_PROCESS_IMAGES
    try:
        import sys as _sys
        from modules import processing as _processing

        if getattr(_processing, "_i4_pi_wrapped", False):
            return
        original = _processing.process_images
        _ORIGINAL_PROCESS_IMAGES = original
        _processing.process_images = _wrapped_process_images
        _processing._i4_pi_wrapped = True

        rebound = ["modules.processing"]
        # Only known Forge entry points; never rewrite another extension's
        # module bindings, even when it imported the same function object.
        for name in ('modules.api.api', 'modules.txt2img', 'modules.img2img'):
            mod = _sys.modules.get(name)
            if mod is None or mod is _processing:
                continue
            try:
                if getattr(mod, "process_images", None) is original:
                    mod.process_images = _wrapped_process_images
                    rebound.append(name)
            except Exception:
                continue
        print(f"{TAG} process_images guard installed for {len(rebound)} module(s) "
              f"({', '.join(rebound[:4])}{'...' if len(rebound) > 4 else ''}) - "
              "API txt2img now routes through the official pipeline too")
    except Exception as e:
        print(f"{TAG} WARNING: process_images guard not installed ({e}); the API "
              "path stays uncovered, stock behaviour is unchanged")


def install_wrapper():
    global _ORIGINAL_RUN
    try:
        import inspect
        from modules.scripts import ScriptRunner
        inspect.signature(ScriptRunner.run).bind(object(), object())
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        print(f'{TAG} Ideogram integration disabled: Forge ScriptRunner API changed ({error}). Stock Forge is unchanged.')
        return False
    if getattr(ScriptRunner, "_i4_wrapped", False):
        return
    _ORIGINAL_RUN = ScriptRunner.run
    ScriptRunner.run = _wrapped_run
    ScriptRunner._i4_wrapped = True
    _install_process_images_guard()
    from .lifecycle import install as install_lifecycle
    install_lifecycle(_PIPELINE_CACHE)
    print(f"{TAG} Invisible takeover installed: select an Ideogram 4 checkpoint in the normal "
          "dropdown ('Ideogram4 - <name>') and press Generate. LoRAs/LoKrs via the usual "
          "<lora:name:strength> prompt tags. Filter-bypass panel inside the accordion.")
