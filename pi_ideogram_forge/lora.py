# Project Invisible: namespaced Forge adapter; no core-file edits.
"""LoRA tag parsing, file discovery, and application for Ideogram 4.

Extracted from engine.py (Phase 2 of architecture decomposition).
This module has ZERO Forge dependencies — it receives everything it needs
as function arguments, making it independently testable.

Adapter contract (spec section H):
  - native style/character LoRAs are applied to BOTH trunks by default
    (dual-trunk lock): with a separate uncond transformer loaded they merge
    into both; in single-transformer (uncond-LoRA) mode both passes share
    the same weights, so symmetry is automatic.
  - TurboTime -> cond trunk only, distill recipe (2-8 steps, CFG 1.0).
  - Ostris Unconditional LoRA -> replaces the 9.3B uncond transformer;
    applied to the cond transformer right before sampling, undone after.
  - Gray-screen bypass LoRA -> first-step-only, negative strength (handled
    by generate.py via merge_lora_undoable; never merged permanently).
  - Foreign LoRAs are REFUSED before any weight is touched.
"""

from __future__ import annotations

import re
from pathlib import Path

TAG = "[Invisible-I4]"

LORA_TAG_RE = re.compile(r"<(?:lora|lyco):([^:>]+)(?::([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?))?[^>]*>", re.IGNORECASE)


def lora_search_dirs(models_dirs: list[Path]) -> list[Path]:
    """models/Lora + every Ideogram 4 spelling's /loras and /lora subfolder
    (layout owner: pi_ideogram_lib.detect)."""
    from pi_ideogram_lib.detect import lora_search_dirs as _detect_dirs
    return _detect_dirs(models_dirs)


def find_lora_file(name: str, models_dirs: list[Path]) -> Path | None:
    """Search all Ideogram 4 LoRA dirs (plus any Lora subfolder) for a matching
    .safetensors file whose stem equals `name`."""
    target = name[:-12] if name.lower().endswith(".safetensors") else name
    for d in lora_search_dirs(models_dirs):
        for cand in d.rglob("*.safetensors"):
            if cand.stem.casefold() == target.casefold():
                return cand
    return None


def find_uncond_replacement_lora(models_dirs: list[Path]) -> Path | None:
    """The Ostris uncond-replacement adapter on disk, whatever its filename
    (the classifier is the authority). Used to auto-stage the adapter for
    UNTAGGED uncond-LoRA runs (the OOM retry path swaps to cond-only + this
    LoRA with no <lora:...> tag in the prompt). None when absent."""
    from pi_ideogram_lora.ideogram4_lora import classify_adapter
    for d in lora_search_dirs(models_dirs):
        for cand in d.rglob("*.safetensors"):
            try:
                if classify_adapter(cand).get("kind") == "uncond_replacement":
                    return cand
            except Exception:
                continue
    return None


def split_lora_tags(prompt: str, models_dirs: list[Path]) -> tuple[str, list[tuple[str, float, str]]]:
    """Extract stock <lora:name:strength> / <lyco:...> tags from the prompt.

    Returns (clean_prompt_text, [(name, strength, policy)]).
    policy is the adapter trunk policy from classification:
      'dual' | 'cond' | 'uncond_replace' | 'gray_first_step'

    Missing or foreign adapters stop the request before expensive model loading.
    A successful generation must not silently omit a requested character/style.
    """
    from pi_ideogram_lora.ideogram4_lora import classify_adapter

    loras: list[tuple[str, float, str]] = []
    clean = prompt
    for m in list(LORA_TAG_RE.finditer(prompt)):
        name = m.group(1).strip()
        try:
            strength = float(m.group(2)) if m.group(2) else 1.0
        except ValueError:
            strength = 1.0
        path = find_lora_file(name, models_dirs)
        if path is None:
            raise ValueError(f"LoRA '{name}' was not found. Install it in models/Lora or "
                             "remove its prompt tag. No image was generated without your adapter.")
        info = classify_adapter(path)
        if info.get("kind") == "incompatible":
            raise ValueError(f"LoRA '{name}' cannot run on Ideogram 4: {info.get('reason', 'incompatible model')}. "
                             "Select an Ideogram 4 adapter in Style, or remove this tag. "
                             "Krea2, Flux and SDXL adapters require their own base models.")
        policy = info["trunk"]
        loras.append((name, strength, policy))
        clean = clean.replace(m.group(0), " ")
    return re.sub(r"\s+", " ", clean).strip(), loras


def lora_set_key(loras: list[tuple[str, float, str]]) -> str:
    """Stable identity for a LoRA set. Matches the engine's pipeline-cache key
    so "same pipe, same LoRAs" is decidable without touching any weights."""
    return "|".join(f"{n}:{s}:{policy}" for n, s, policy in loras) or "none"


def _runtime_enabled(pipe) -> bool:
    """Is the exact runtime side path allowed for this generation?

    Set False by the OOM ladder (see run.py) so a run that cannot afford the
    adapter's GPU factors still produces an image, by merging instead.
    """
    return bool(getattr(pipe, "_i4_lora_runtime", True))


def _is_quantised(model) -> bool:
    """Does this transformer hold quantised weights we must not merge into?

    Asked of the LIVE modules rather than the checkpoint filename, so it stays
    right for every format the loader supports (fp8_scaled, nf4, int8_convrot,
    w8a8, gguf...) and for anything added later. A bf16 model merges as before -
    merging is exact there, and free at runtime.
    """
    try:
        import torch
        for mod in model.modules():
            w = getattr(mod, "weight", None)
            if w is None:
                continue
            if getattr(w, "dtype", None) in (getattr(torch, "float8_e4m3fn", None),
                                             getattr(torch, "float8_e5m2", None),
                                             torch.int8, torch.uint8):
                return True
            if hasattr(mod, "weight_scale") or hasattr(mod, "quant_state"):
                return True
            # FORGE'S QuantizedTensor MASQUERADES AS BF16. This is the check
            # that was missing, and it cost every image after a LoRA merge.
            # On an int8_convrot checkpoint Forge hands back:
            #     type(w).__name__     QuantizedTensor
            #     w.dtype              torch.bfloat16   <- deliberately
            #     hasattr weight_scale False            <- scale is INSIDE w
            # so both tests above said "not quantised", the merge treated a
            # Hadamard-ROTATED int8 weight as plain bf16, and added an
            # unrotated delta straight into it. Not a lossy merge - a wrong
            # one. The model was destroyed and its own safety filter then
            # returned the gray "Image blocked by safety filter" card, which
            # sent us hunting a filter that was only the symptom.
            if any(c.__name__ == "QuantizedTensor" for c in type(w).__mro__):
                return True
            if getattr(mod, "quant_format", None) is not None:
                return True
            if getattr(mod, "layout_type", None) is not None:
                return True
    except Exception:
        pass
    return False


def _apply_runtime(pipe, name: str, path, strength: float, has_uncond: bool):
    """Install the adapter as a live side path on the trunk(s), never merged.

    Adapters are kept on the pipeline so they outlive this call; the pipeline
    cache key includes the LoRA set, so a different set means a fresh pipeline
    and these go away with it.
    """
    from pi_ideogram_lora.ideogram4_lora import load_lora_state_dict
    from pi_ideogram_lora.runtime_lora import RuntimeLoraAdapter

    logs: list[str] = []
    # Append this adapter. apply_lora_set removes the PREVIOUS set once,
    # before its loop; removing here made every LoRA erase the one before it.
    held = getattr(pipe, "_i4_runtime_adapters", None)
    if held is None:
        held = []
        pipe._i4_runtime_adapters = held

    total = 0
    targets = [("cond", pipe.conditional_transformer)]
    if has_uncond and getattr(pipe, "unconditional_transformer", None) is not None:
        targets.append(("uncond", pipe.unconditional_transformer))
    for label, model in targets:
        ad = RuntimeLoraAdapter(model, load_lora_state_dict(path), strength, name)
        n = ad.install()
        ad.enabled = True          # style/character LoRAs are on for every pass
        held.append(ad)
        total += n
        logs.append(f"LoRA '{name}' -> {n} modules on the {label} trunk "
                    "(runtime side path, exact - quantised weights left intact)")
    if not total:
        logs.append(f"LoRA '{name}' matched no modules at runtime")
    return total, logs


def apply_turbo_adapter(pipe, name: str, path, strength: float = 1.0) -> list[str]:
    """Apply the TurboTime distill adapter for THIS run - removably.

    THE BODY HORROR. generate.py used to call apply_lora_to_model() straight
    on pipe.conditional_transformer, which walked past everything apply_lora_set
    exists to decide. Three defects compounded:

    1. On a quantised trunk it MERGED - writing an unrotated delta into
       Hadamard-rotated int8 weights. Same corruption that made character
       LoRAs produce gray cards, arriving through a second door.
    2. Nothing was idempotent, so a second TurboTime run merged the same delta
       into the same weights AGAIN. The user's log shows exactly that:
       "TurboTime LoRA merged (204 modules)" twice on one cached pipeline.
    3. The merge was called permanent, which was true only while the cache key
       kept turbo pipelines separate. _get_pipeline later learned to reuse a
       non-turbo pipeline for a distill run - so the merge landed on the object
       BOTH keys point at, and every subsequent V4_DEFAULT_20 generation in
       that session ran on a TurboTime-merged model.

    So: quantised trunks get the exact runtime side path in their own slot
    (removable, so leaving the preset undoes it); a bf16 trunk still merges,
    but the pipeline is stamped `_i4_turbo_merged` so _get_pipeline refuses to
    hand it to a non-turbo run. Applying twice is a no-op either way.
    """
    from pi_ideogram_lora.ideogram4_lora import apply_lora_to_model, load_lora_state_dict
    from pi_ideogram_lora.runtime_lora import RuntimeLoraAdapter

    want = (str(path), float(strength))
    if getattr(pipe, "_i4_turbo_key", None) == want:
        return [f"TurboTime adapter already active ({name}); not re-applying"]

    remove_turbo_adapter(pipe)
    logs: list[str] = []
    model = pipe.conditional_transformer
    if _is_quantised(model):
        ad = RuntimeLoraAdapter(model, load_lora_state_dict(str(path)), float(strength), name)
        n = ad.install()
        ad.enabled = True
        pipe._i4_turbo_adapter = ad
        pipe._i4_turbo_key = want
        logs.append(f"TurboTime adapter -> {n} modules (runtime side path, exact - "
                    "quantised weights left intact, and removable when you leave the preset)")
    else:
        applied, sub = apply_lora_to_model(model, load_lora_state_dict(str(path)), float(strength))
        pipe._i4_turbo_merged = True
        pipe._i4_turbo_key = want
        logs.extend(sub)
        logs.append(f"TurboTime adapter merged into {applied} modules (bf16 trunk: exact). "
                    "This pipeline is now turbo-only and will not be reused for a full-CFG run.")
    return logs


def remove_turbo_adapter(pipe) -> list[str]:
    """Take the TurboTime adapter back off. No-op when none is installed.

    Only the runtime side path can be undone. A bf16 merge cannot, which is
    why the pipeline carrying one is marked and never reused for a non-turbo
    generation.
    """
    ad = getattr(pipe, "_i4_turbo_adapter", None)
    if ad is None:
        return []
    try:
        ad.remove()
    except Exception:
        pass
    pipe._i4_turbo_adapter = None
    pipe._i4_turbo_key = None
    return ["TurboTime adapter removed (left the distill preset)"]


def apply_lora_set(pipe, loras: list[tuple[str, float, str]]) -> list[str]:
    """Apply validated LoRA tags to the pipeline's transformers.

    IDEMPOTENT PER PIPELINE. The merge is ADDITIVE and destructive - it writes
    the delta straight into the live weights - and the engine caches pipelines
    keyed by (checkpoint, LoRA set). So a second generation with the same LoRAs
    reuses the same already-merged weights: re-applying here would stack the
    delta again. We record what a pipeline already carries and skip in that
    case; changing the LoRA set produces a different cache key, hence a freshly
    loaded pipeline.

    Dual-trunk lock: native style/character LoRAs merge into the uncond
    transformer too whenever a separate one is loaded. If a style LoRA would
    hit ONLY the cond trunk (uncond model present but skipped) we refuse it -
    single-trunk style application is the documented artifact source.

    Args:
        pipe: Ideogram4Pipeline with conditional_transformer (and optionally
              unconditional_transformer).
        loras: List of (name, strength, policy) tuples from split_lora_tags().

    Returns:
        List of human-readable log lines describing what was applied/skipped.
    """
    from pi_ideogram_lora.ideogram4_lora import (
        apply_lora_to_model,
        classify_adapter,
        inspect_lora,
        load_lora_state_dict,
        LoraRefused,
    )

    wanted = lora_set_key(loras)
    if getattr(pipe, "_i4_applied_lora_key", None) == wanted:
        # "merged into these weights" became wrong the moment quantised trunks
        # moved to the runtime side path - nothing is written into the weights
        # there. Say what is actually true.
        return [f"LoRA set already active on this pipeline ({wanted}); not re-applying"]

    logs: list[str] = []
    has_uncond = getattr(pipe, "unconditional_transformer", None) is not None
    # Replace the old set once, not once per incoming adapter. An identical
    # set already returned above and must retain its live hooks untouched.
    for adapter in getattr(pipe, "_i4_runtime_adapters", None) or []:
        adapter.remove()
    pipe._i4_runtime_adapters = []
    for name, strength, policy in loras:
        path = find_lora_file(name, _MODELS_DIRS_CACHE)
        if path is None:
            continue
        info = classify_adapter(path)
        kind = info["kind"]
        if kind == "incompatible" or not inspect_lora(path)["is_ideogram4"]:
            print(f"{TAG} WARNING: '<lora:{name}>' is not an Ideogram 4 LoRA/LoKr "
                  f"({info['reason']}) - tag skipped. SD / SDXL / Flux / Wan / Klein / "
                  "Krea2 LoRAs can never be applied to Ideogram 4; nothing was changed.")
            logs.append(f"skipped foreign LoRA '{name}'")
            continue
        if kind == "gray_bypass":
            # Handled by generate.py as a temporary first-step merge; never
            # persisted here. Recorded so the tag is not treated as a style LoRA.
            pipe._i4_bypass_lora = (name, float(strength), str(path))
            logs.append(f"gray-bypass LoRA '{name}' staged for first-step-only apply")
            continue
        if kind == "uncond_replacement":
            # Applied right before sampling by generate.py (replaces the uncond
            # transformer for the uncond pass). Staged, not merged here.
            pipe._i4_uncond_lora = (name, float(strength), str(path))
            logs.append(f"uncond-replacement LoRA '{name}' staged for the uncond pass")
            continue
        if kind == "turbo_time":
            pipe._i4_turbo_lora = (name, float(strength), str(path))
            logs.append(f"TurboTime distill LoRA '{name}' staged (steps snap to 2-8, CFG 1.0)")
            continue
        # native style/character LoRA: dual-trunk by default.
        #
        # ON A QUANTISED CHECKPOINT WE DO NOT MERGE. Writing the delta into
        # fp8 weights and requantising was measured on the real files as:
        #
        #     intended delta rms 0.000259
        #     requant error rms  0.000371     -> 1.44x the signal
        #     cosine(recovered, intended) 0.57
        #
        # So ~43% of what reached the weights was quantisation noise, spread
        # over 204 modules and both trunks. That is the "body horror", and it
        # shows up in ANY implementation that merges adapters into fp8 - which
        # is why the reference build has it too.
        #
        # The runtime side path is exact instead (cosine 1.000002 against the
        # dense delta) and, using the Kronecker mixed-product rather than a
        # materialised kron(w1, w2), costs +30% on the affected matmuls instead
        # of +101%. Correctness first: a fast wrong image is not a result.
        try:
            # opts["lora_runtime"] = False is the OOM ladder telling us to
            # take the memory-cheaper merge even though it is less exact.
            _quantised = _is_quantised(pipe.conditional_transformer)
            if _quantised and not _runtime_enabled(pipe):
                # The OOM ladder may ask for the merge to save memory. On a
                # quantised trunk that request cannot be honoured: merging is
                # not merely less exact there, it corrupts the weights (see
                # _is_quantised). Refuse and keep the exact path - an image
                # that needs a smaller batch beats a gray card.
                print(f"{TAG} note: merge was requested to save memory, but these weights are "
                      "quantised and cannot be merged into safely - keeping the exact runtime "
                      "side path. Lower the resolution or batch size if VRAM is tight.")
            if _quantised:
                applied, sub_logs = _apply_runtime(pipe, name, path, strength, has_uncond)
                logs.extend(sub_logs)
                continue
            sd = load_lora_state_dict(path)
            applied, sub_logs = apply_lora_to_model(pipe.conditional_transformer, sd, strength)
            logs.extend(sub_logs)
            if has_uncond:
                applied_u, sub_logs_u = apply_lora_to_model(
                    pipe.unconditional_transformer,
                    load_lora_state_dict(path),
                    strength,
                )
                logs.extend(sub_logs_u)
                logs.append(f"LoRA '{name}' applied to BOTH trunks (cond {applied} / uncond {applied_u} modules)")
            else:
                # Single-transformer mode (uncond-LoRA path): both passes share
                # these weights, so the dual-trunk lock is satisfied implicitly.
                logs.append(f"LoRA '{name}' applied to the shared transformer "
                            "(single-transformer uncond-LoRA mode: both passes use it)")
        except LoraRefused as e:
            print(f"{TAG} WARNING: {e} - tag skipped; your image still generates.")
            logs.append(f"skipped unusable LoRA '{name}'")
    try:
        pipe._i4_applied_lora_key = wanted
    except Exception:
        pass
    return logs


# --------------------------------------------------------------------------- #
# Module-level cache for models_dirs (set by engine.py at startup)
# --------------------------------------------------------------------------- #
_MODELS_DIRS_CACHE: list[Path] = []


def set_models_dirs(dirs: list[Path]) -> None:
    """Called by engine.py to provide the models directory list.
    apply_lora_set() needs this to find LoRA files on disk."""
    global _MODELS_DIRS_CACHE
    _MODELS_DIRS_CACHE = list(dirs)
