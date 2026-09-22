"""PROJECT INVISIBLE - quantization registry for Ideogram 4.

config.json owns the registry (key "quant_registry"); this module reads it
there and falls back to the embedded defaults below. Loaders are keyed by id;
the pipeline picks the loader by sniffing the actual file header, so the
registry and the file on disk can never disagree silently.

Ship list (official/proven, from ideogram-ai + Comfy-Org/Ideogram-4):
  fp8_scaled     - Comfy-Org single-file fp8 (also official HF fp8 layout)
  nf4            - bitsandbytes NF4 (ideogram-ai/ideogram-4-nf4)
  int8_convrot   - Comfy-Org int8 conv-rotated
  nvfp4_mixed    - Comfy-Org nvfp4 mixed (Blackwell; cap >= 10.0)
  bf16           - community dense bf16
  w8a8           - transformerlab w8a8 (if present)
  gguf_q4k/q8    - GGUF (only listed; in-process loader is external sd.cpp)
"""

from __future__ import annotations

import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# embedded defaults (mirrored into config.json on first launch)
# --------------------------------------------------------------------------- #

DEFAULT_REGISTRY: list[dict] = [
    {
        "id": "fp8_scaled",
        "files": ["ideogram4_fp8_scaled.safetensors", "ideogram4_Fp8Scaled.safetensors"],
        "vram_floor_gb": 12,
        "loader": "fp8",
        "notes": "Comfy-Org single-file / official HF fp8. Runs on ANY CUDA GPU (portable dequant path).",
        "recommended_badge": "Recommended",
        "downloadable": True,
        "in_process": True,
    },
    {
        "id": "nf4",
        "files": ["ideogram-4-nf4"],
        "vram_floor_gb": 8,
        "loader": "bnb4bit",
        "notes": "bitsandbytes NF4 single-file (packed quant_state.bitsandbytes__* layout): loader + kernels verified in-process on CUDA (bnb is present in the Forge venv). The official ideogram-ai/ideogram-4-nf4 repo is gated AND diffusers-layout - key-name compatibility with this loader is UNVERIFIED (file content gated); a Comfy-style single-file bnb NF4 export is the verified source. LoRA merge on bnb layers is NOT in-process.",
        "recommended_badge": "Best VRAM fit (8-12 GB)",
        "downloadable": True,
        "in_process": True,
    },
    {
        "id": "int8_convrot",
        "files": ["ideogram4_int8_convrot.safetensors"],
        "vram_floor_gb": 10,
        "loader": "int8_convrot",
        "notes": "int8 weights with a convrot (Hadamard) rotation, groupsize 256. Loads in-process through Forge's quantised ops (backend/quant_ops.py + Comfy-Kitchen kernels). LoRAs are never merged into it - they apply as an exact runtime side path.",
        "recommended_badge": "Good VRAM fit (10-16 GB)",
        "downloadable": True,
        "in_process": True,
    },
    {
        "id": "nvfp4_mixed",
        "files": ["ideogram4_nvfp4_mixed.safetensors"],
        "vram_floor_gb": 12,
        "loader": "nvfp4_mixed",
        "notes": "NVFP4 4-bit with two-level scaling (group 16). Loads in-process through Forge's quantised ops; needs Blackwell tensor cores, and Forge disables the format automatically when the GPU lacks them. LoRAs apply as an exact runtime side path.",
        "recommended_badge": "",
        "downloadable": True,
        "in_process": True,
    },
    {
        "id": "bf16",
        "files": [],
        "vram_floor_gb": 16,
        "loader": "bf16",
        "notes": "Community dense bf16 single-file. Largest footprint; best numerics.",
        "recommended_badge": "",
        "downloadable": False,
        "in_process": True,
    },
    {
        "id": "w8a8",
        "files": [],
        "vram_floor_gb": 10,
        "loader": "w8a8",
        "notes": "w8a8 is transformerlab's quantized layout - their tooling loads it, not this process. No in-process loader here; run with transformerlab's runtime or select fp8_scaled.",
        "recommended_badge": "",
        "downloadable": False,
        "in_process": False,
    },
    {
        "id": "gguf_q4k",
        "files": ["*Q4_K*.gguf", "*q4_k*.gguf"],
        "vram_floor_gb": 6,
        "loader": "gguf",
        "notes": "GGUF Q4_K runs in sd.cpp / llama.cpp, NOT this Forge process: drop the .gguf into sd.cpp's own models dir and run sd.cpp there. This extension loads single-file safetensors only - for in-Forge runs select fp8_scaled.",
        "recommended_badge": "Recommended (<=8 GB, external)",
        "downloadable": True,
        "in_process": False,
    },
    {
        "id": "gguf_q8",
        "files": ["*Q8_0*.gguf", "*q8_0*.gguf"],
        "vram_floor_gb": 10,
        "loader": "gguf",
        "notes": "GGUF Q8_0 runs in sd.cpp / llama.cpp, NOT this Forge process: drop the .gguf into sd.cpp's own models dir and run sd.cpp there. This extension loads single-file safetensors only - for in-Forge runs select fp8_scaled.",
        "recommended_badge": "",
        "downloadable": True,
        "in_process": False,
    },
]

# Bump when DEFAULT_REGISTRY changes meaning, so the config.json mirror is
# refreshed instead of pinning old capabilities forever.
# 2: int8_convrot / nvfp4_mixed / w8a8 load in-process via Forge's quant ops.
# Modern formats retain distinct identities; sharing Forge's loader does not
# make MXFP8 ordinary FP8 or W4A4 an INT8/convrot checkpoint.
for _id, _label in (
    ('fp16', 'Dense FP16 weights; computed in BF16 for the supported RTX 30/40/50 families.'),
    ('fp8_e5m2', 'FP8 E5M2 through Forge mixed-precision operations.'),
    ('mxfp8', 'Microscaled FP8 through Forge; hardware/kernel availability checked by the host.'),
    ('convrot_w4a4', 'Convrot W4A4 through Forge; original rotation and packed weight layout preserved.'),
    ('asym_w4a8_int8', 'Asymmetric W4A8 INT8 through Forge mixed-precision operations.'),
    ('forge_quant', 'Descriptor-defined mixed quantization through the installed Forge backend; unknown formats fail before loading.'),
):
    DEFAULT_REGISTRY.append(dict(id=_id, files=[], vram_floor_gb=4,
        loader='dense' if _id == 'fp16' else 'forge_quant_ops',
        notes=_label, recommended_badge='', downloadable=False, in_process=True))

REGISTRY_VERSION = 3

_REGISTRY_CACHE: list[dict] | None = None


def registry() -> list[dict]:
    """Registry from config.json if present and valid, else embedded defaults."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    data = None
    try:
        import pi_ideogram_lib.config as pi_config
        data = pi_config.get("quant_registry", None)
    except Exception:
        pass
    # STALENESS. config.json holds a MIRROR of DEFAULT_REGISTRY, written on
    # first launch and never revisited - so once a format's capabilities
    # changed in code, the copy on disk kept overriding it. int8_convrot and
    # nvfp4 stayed marked "no in-process loader" long after the loader existed,
    # and no amount of editing this file had any effect.
    #
    # A mirror needs a version. When REGISTRY_VERSION moves, the on-disk copy
    # is refreshed from code instead of silently winning.
    stale = False
    try:
        import pi_ideogram_lib.config as pi_config
        stale = int(pi_config.get("quant_registry_version", 0) or 0) < REGISTRY_VERSION
    except Exception:
        stale = False
    if (stale or not isinstance(data, list) or not data
            or not all(isinstance(e, dict) and e.get("id") for e in data)):
        data = list(DEFAULT_REGISTRY)
        try:
            import pi_ideogram_lib.config as pi_config
            pi_config.save({"quant_registry": data,
                            "quant_registry_version": REGISTRY_VERSION})
        except Exception:
            pass
    _REGISTRY_CACHE = data
    return data


def quant_by_id(quant_id: str) -> dict | None:
    for e in registry():
        if e["id"] == quant_id:
            return e
    return None


def all_quant_ids() -> list[str]:
    return [e["id"] for e in registry()]


# --------------------------------------------------------------------------- #
# sniffing: filename tokens + safetensors header markers
# --------------------------------------------------------------------------- #

def _stem_tokens(stem: str) -> set[str]:
    s = re.sub(r"[^a-z0-9]", " ", stem.lower())
    return {t for t in s.split() if t}


def _weight_dtypes(hdr: dict) -> set:
    """The dtypes actually stored for `*.weight` tensors, from the header only.

    This is the arbiter when a file's own quantisation descriptors disagree
    with its contents - see the note in sniff_quant.
    """
    out = set()
    for k, v in hdr.items():
        if k == "__metadata__" or not k.endswith(".weight"):
            continue
        if isinstance(v, dict) and v.get("dtype"):
            out.add(str(v["dtype"]))
    return out


def _comfy_formats_from_file(path, hdr: dict) -> set:
    """The quant formats a checkpoint declares about itself.

    Two places carry this, and a file may have either:

    * `__metadata__["_quantization_metadata"]` - a JSON blob mapping layer
      name to {"format": ..., "convrot": ...}. Forge Neo's converter and the
      Star converter both write it, and it is free to read (already in the
      header).
    * per-layer `<layer>.comfy_quant` U8 tensors holding the same JSON. Those
      are real tensor DATA, so reading them needs safe_open - the header only
      says a U8[72] exists there. Just one is read; the format is uniform.

    Returns an empty set for an unquantised or undeclared file. Never raises.
    """
    import json as _json
    out = set()
    meta = hdr.get("__metadata__") or {}
    blob = meta.get("_quantization_metadata")
    if blob:
        try:
            layers = (_json.loads(blob) or {}).get("layers", {})
            for conf in layers.values():
                if not isinstance(conf, dict):
                    continue
                fmt = str(conf.get("format", ""))
                if fmt:
                    out.add(fmt + ("+convrot" if conf.get("convrot") else ""))
        except Exception:
            pass
    if out:
        return out
    marker_keys = [k for k in hdr if k.endswith(".comfy_quant")]
    if not marker_keys:
        return out
    # Some producers (and our own fixtures) put the descriptor straight into
    # the header as a JSON string rather than as a U8 tensor. Read that first:
    # it needs no file access and is unambiguous when present.
    for k in marker_keys:
        v = hdr.get(k)
        if isinstance(v, (str, bytes, bytearray)):
            try:
                raw = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
                conf = _json.loads(raw)
                fmt = str(conf.get("format", ""))
                if fmt:
                    out.add(fmt + ("+convrot" if conf.get("convrot") else ""))
            except Exception:
                pass
    if out:
        return out
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            for k in marker_keys:
                raw = bytes(f.get_tensor(k).tolist()).rstrip(b"\x00")
                conf = _json.loads(raw.decode("utf-8"))
                fmt = str(conf.get("format", ""))
                if fmt:
                    out.add(fmt + ("+convrot" if conf.get("convrot") else ""))
    except Exception:
        pass
    return out


def sniff_quant(path: str | Path) -> dict:
    """Return {id, loader, in_process, notes, source: 'filename'|'header'}.

    Order: (1) safetensors header markers, (2) filename tokens, (3) unknown.
    Never raises.
    """
    path = Path(path)
    stem = path.stem
    toks = _stem_tokens(stem)

    # --- header markers (authoritative when present) -------------------------
    try:
        from pi_ideogram_lib.detect import read_safetensors_header
        hdr = read_safetensors_header(path)
        keys = list(hdr.keys())
        if any(".quant_state.bitsandbytes__" in k for k in keys):
            return _entry("nf4", "header")

        # ASK THE FILE WHAT IT IS, BEFORE GUESSING FROM weight_scale.
        #
        # This used to answer "any .weight_scale key -> fp8_scaled" FIRST, and
        # int8_tensorwise, nvfp4, mxfp8 and the convrot family all carry a
        # weight_scale. So an int8_convrot checkpoint announced itself as
        #     [routing] detected fp8_scaled (source=header) loader=fp8
        #               merge=fp8_dequant_requant
        # - wrong format, wrong loader, wrong merge policy, in the one line a
        # user reads to check what is running.
        #
        # The comfy_quant branch below could never have corrected it either:
        # read_safetensors_header returns each tensor's METADATA dict
        # ({"dtype": "U8", "shape": [72], ...}), not its bytes, so json.loads
        # on it always failed and comfy_formats was always empty.
        comfy_formats = _comfy_formats_from_file(path, hdr)
        wdtypes = _weight_dtypes(hdr)
        # A DESCRIPTOR CAN LIE; THE WEIGHTS CANNOT.
        # ideogram4_bf16.safetensors carries 211 comfy_quant markers that still
        # say float8_e4m3fn while every weight in it is BF16 - a leftover from
        # whatever produced it. Trusting the marker alone reported a plain bf16
        # checkpoint as fp8_scaled.
        #
        # The veto needs a POSITIVE contradiction, not merely an absence of
        # corroboration: a file must actually contain `*.weight` tensors and
        # every one of them must be a plain float type. Requiring corroboration
        # instead would refuse to identify any file whose weights we cannot see
        # (a header-only stub, a sharded checkpoint), which is a different
        # failure with the same silence.
        has_f8 = any(d.startswith("F8") for d in wdtypes)
        has_i8 = "I8" in wdtypes
        plain_weights = bool(wdtypes) and not (has_f8 or has_i8 or "U8" in wdtypes)
        if comfy_formats and not plain_weights:
            base_formats = {f.split('+')[0] for f in comfy_formats}
            if 'convrot_w4a4' in base_formats:
                return _entry('convrot_w4a4', 'header')
            if 'asym_w4a8_int8' in base_formats:
                return _entry('asym_w4a8_int8', 'header')
            if 'mxfp8' in base_formats:
                return _entry('mxfp8', 'header')
            if 'float8_e5m2' in base_formats:
                return _entry('fp8_e5m2', 'header')
            # nvfp4 first: a mixed-precision file declares int8 for some layers
            # and nvfp4 for others, and nvfp4 is the distinguishing one.
            if any("nvfp4" in f or "fp4" in f or "nf4" in f for f in comfy_formats):
                return _entry("nvfp4_mixed", "header")
            if any("int8" in f for f in comfy_formats):
                # ANY comfy_quant int8 -> int8_convrot, which is the registry
                # entry whose loader is Forge's quantised ops. Sending the
                # non-convrot variant to `w8a8` instead would have been wrong
                # twice over: w8a8 is transformerlab's layout and is marked
                # in_process=False, so a file Forge loads perfectly well would
                # have been reported as "external / manual only".
                return _entry("int8_convrot", "header")
            if any("fp8" in f or "float8" in f for f in comfy_formats):
                if any(f.endswith('+convrot') for f in comfy_formats):
                    return _entry('forge_quant', 'header')
                return _entry("fp8_scaled", "header")
            return _entry('forge_quant', 'header')
        # No veto here: ideogram4_bf16.safetensors, the file that motivated it,
        # carries NO weight_scale at all (BF16 weights + stale U8 descriptors),
        # so it never reaches this branch. A bare weight_scale really is the
        # original HF fp8 layout and nothing else.
        if any(k.endswith(".weight_scale") for k in keys):
            # No usable descriptors: the original HF fp8 layout, which is the
            # only quantisation that ships a bare weight_scale.
            return _entry("fp8_scaled", "header")
    except Exception:
        pass

    # --- filename tokens ------------------------------------------------------
    if path.suffix.lower() == ".gguf" or "gguf" in toks:
        q = "gguf_q4k" if any("q4" in t or "q4" in stem.lower() for t in toks) else "gguf_q8"
        return _entry(q, "filename")
    name_l = stem.lower()
    for quant_id in ('convrot_w4a4', 'asym_w4a8_int8', 'mxfp8', 'fp8_e5m2'):
        if quant_id in name_l.replace('-', '_'):
            return _entry(quant_id, 'filename')
    if "nvfp4" in name_l or "nvfp4" in toks:
        return _entry("nvfp4_mixed", "filename")
    if "int8" in name_l and "convrot" in name_l:
        return _entry("int8_convrot", "filename")
    if "nf4" in name_l or "nf4" in toks:
        return _entry("nf4", "filename")
    if "fp8" in name_l:
        return _entry("fp8_scaled", "filename")
    if "bf16" in name_l or "bfloat16" in name_l:
        return _entry("bf16", "filename")
    if 'fp16' in name_l:
        return _entry('fp16', 'filename')
    if "w8a8" in name_l:
        return _entry("w8a8", "filename")
    return {"id": "unknown", "loader": None, "in_process": False,
            "notes": "quant could not be sniffed from filename or header", "source": None}


def _entry(quant_id: str, source: str) -> dict:
    e = quant_by_id(quant_id) or {}
    out = dict(e)
    out["source"] = source
    return out


def recommended_for_vram(vram_gb: float | None) -> str:
    """Best registry id for the measured VRAM. Cap-aware (Blackwell check)."""
    from pi_ideogram_lib.hardware import vram_profile
    profile = vram_profile(vram_gb)
    mapping = {
        "<=8": "fp8_scaled",
        "8-12": "fp8_scaled",
        "12-16": "fp8_scaled",
        "16-24": "fp8_scaled",
        ">=24": "fp8_scaled",
        "unknown": "fp8_scaled",
    }
    return mapping.get(profile, "fp8_scaled")


def summarize(path: str | Path) -> str:
    """One-line human description of a checkpoint's quant, for the DOS header."""
    q = sniff_quant(path)
    name = Path(path).name
    return f"{name} [{q.get('id', 'unknown')}{' (in-process)' if q.get('in_process') else ''}]"


# --------------------------------------------------------------------------- #
# DISTILLED-CKPT SNIFFING (spec H): a checkpoint that is ALREADY a distilled /
# Instant / Fast merge must never receive the TurboTime LoRA on top (stacking).
# Signals: filename tokens first (instant/fast/distill/turbotime...), then any
# header metadata whose stringified values mention a distill marker. A negative
# is a claim only about the FILE NAME + header, so unknown stays undistilled.
# --------------------------------------------------------------------------- #

_DISTILLED_STEM_TOKENS = ("instant", "distill", "distilled",
                          "turbotime", "turbo-time", "turbo_time")
# substring-safe roots (checked dash-boundary to avoid e.g. 'fast' in 'superfast')
_DISTILLED_BOUNDARY_TOKENS = _DISTILLED_STEM_TOKENS + ("fast",)


def sniff_distilled(path: str | Path) -> dict:
    """Return {distilled: bool, signal: str | None, source: str | None}.

    True when the checkpoint file looks like an already-distilled / Instant /
    Fast merge (TurboTime-LoRA stacking must be refused). Header metadata is
    scanned when present; filename tokens otherwise. Never raises.
    """
    path = Path(path)
    stem = path.stem
    name_l = stem.lower().replace("_", "-")

    # --- header metadata markers (when a real header is readable) -------------
    try:
        from pi_ideogram_lib.detect import read_safetensors_header
        hdr = read_safetensors_header(path)
        meta = hdr.get("__metadata__") or {}
        if isinstance(meta, dict):
            for _k, v in meta.items():
                sv = str(v).lower()
                if any(t in sv for t in ("instant", "distill", "fast")):
                    return {"distilled": True, "signal": "header metadata", "source": "header"}
    except Exception:
        pass

    # --- filename tokens (dash-boundary, so 'fast' != 'superfast') -------------
    hyphenated = f"-{name_l}-"
    for t in _DISTILLED_BOUNDARY_TOKENS:
        norm = t.replace("_", "-")
        if f"-{norm}-" in hyphenated:
            return {"distilled": True, "signal": f"filename:{t}", "source": "filename"}
    return {"distilled": False, "signal": None, "source": None}


# --------------------------------------------------------------------------- #
# ROUTING (audit-gap R): the DETECTED format drives the extension's policy.
# sniff_quant answers "what format is this file?"; route_checkpoint answers
# "given that, how does THIS extension behave?" - which merge path LoRAs take,
# whether the file can run in-process at all, what the user should be told.
# Single owner: registry data + sniffing live here, engine only renders it.
# Where Forge owns the loader (stock sd_models checkpoint machinery) the
# extension does NOT pretend to replace it - it only routes what it controls
# (its own pipeline + LoRA merges + honest refusal of non-in-process formats).
# --------------------------------------------------------------------------- #

# merge strategy per registry id (id -> (merge_mode, human label))
_ROUTE_MERGE: dict[str, tuple[str, str]] = {
    'fp16': ('bf16_direct', 'Dense FP16 loaded in BF16; LoRAs use the dense merge path'),
    'fp8_e5m2': ('forge_quant_ops', 'FP8 E5M2 via Forge; runtime adapters'),
    'mxfp8': ('forge_quant_ops', 'MXFP8 via Forge; runtime adapters'),
    'convrot_w4a4': ('forge_quant_ops', 'Convrot W4A4 via Forge; runtime adapters'),
    'asym_w4a8_int8': ('forge_quant_ops', 'Asymmetric W4A8 INT8 via Forge; runtime adapters'),
    'forge_quant': ('forge_quant_ops', 'Descriptor-defined mixed precision via Forge; runtime adapters'),
    "bf16": ("bf16_direct", "bf16 weights: LoRAs merge in-place in bf16"),
    "fp8_scaled": ("fp8_dequant_requant", "FP8 (per-row weight_scale) weights: LoRAs dequantize \u2192 add \u2192 requantize"),
    "nf4": ("bnb_not_in_process", "NF4 loads via bitsandbytes; LoRA/LoKr merge on bnb layers is NOT in-process (skip logged) - apply LoRAs on the fp8/bf16 file of the same trunk"),
    # These load through FORGE's quantised ops (backend/quant_ops.py +
    # Comfy-Kitchen kernels), which this build now uses for any checkpoint
    # carrying per-layer `comfy_quant` descriptors. LoRAs are never merged
    # into them - the runtime side path applies adapters exactly instead.
    "int8_convrot": ("forge_quant_ops", "int8 + convrot rotation via Forge's quantised ops; LoRAs apply as an exact runtime side path (never merged)"),
    "nvfp4_mixed": ("forge_quant_ops", "NVFP4 (4-bit, Blackwell tensor cores) via Forge's quantised ops; LoRAs apply as an exact runtime side path"),
    "w8a8": ("external", "Unrecognized W8A8 layout; use a checkpoint with Forge-compatible quantization descriptors"),
    "gguf_q4k": ("external", "GGUF runs via sd.cpp / llama.cpp, NOT this Forge process"),
    "gguf_q8": ("external", "GGUF runs via sd.cpp / llama.cpp, NOT this Forge process"),
}


def route_checkpoint(path: str | Path | None, vram_gb: float | None = None) -> dict:
    """Detected-format -> extension routing policy. Reads the REAL file
    (header magic + filename, never filename alone for safetensors). Never
    raises: an unreadable/missing file routes as 'unknown' with a warning.

    Returns {path, id, source, loader, in_process, loadable_in_process,
    merge_mode, merge_label, note}. loadable_in_process is False ONLY for a
    CONFIDENT sniff of a format this build cannot load (int8 / nvfp4 / w8a8 /
    gguf); 'unknown' stays loadable because the weight-inspection loaders are
    the authority there."""
    if path is None:
        return {"path": None, "id": "unknown", "source": None, "loader": None,
                "in_process": False, "loadable_in_process": True,
                "merge_mode": "auto_weight_inspection", "note": "No checkpoint selected."}
    path = Path(path)
    base = {"path": str(path)}
    try:
        q = sniff_quant(path)
    except Exception:
        q = {"id": "unknown", "source": None, "loader": None, "in_process": False}
    qid = str(q.get("id") or "unknown")
    entry = quant_by_id(qid) or {}
    merge_mode, merge_label = _ROUTE_MERGE.get(
        qid, ("auto_weight_inspection",
              "format not recognized from name/header - loaders & LoRA merges fall back to "
              "weight inspection; if loading fails the file is likely an unsupported format"))
    in_process = bool(entry.get("in_process"))
    # unknown: keep the pipeline's own header checks authoritative, only warn
    loadable = True if qid == "unknown" else bool(in_process)
    # GGUF genuinely runs outside this process. The comfy_quant family does
    # not: Forge ships the loaders and the kernels, so refusing them was a
    # statement about this file, not about the format.
    blocked_ids = ("gguf_q4k", "gguf_q8", "w8a8")
    note = str(entry.get("notes") or "")
    if qid == "unknown":
        note = (merge_label + ". The file is registered as an Ideogram 4 checkpoint "
                "(name or model_type), but no known quant marker was found.")
    out = dict(base)
    out.update({
        "id": qid,
        "source": q.get("source"),
        "loader": entry.get("loader") or q.get("loader"),
        "in_process": in_process,
        "loadable_in_process": loadable,
        "merge_mode": merge_mode,
        "merge_label": merge_label,
        "blocked": qid in blocked_ids,
        "note": note,
        "fallback": recommended_for_vram(vram_gb) if qid in blocked_ids else None,
    })
    return out
