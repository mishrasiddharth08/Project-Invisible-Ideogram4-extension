"""PROJECT INVISIBLE - explicit asset utilities for Ideogram 4 (spec section F).

Scan-before-download: every ensure_*() function first checks the existing
Ideogram 4 dirs and only fetches what is missing. Downloads are resumable,
hash-verified, and print the same tqdm progress style as stock Forge in the
DOS window. HF gated repos print the exact accept-and-login steps and abort
cleanly.

Adapters (auto-download when the matching feature is enabled and the file is
missing):
  ostris/ideogram_4_turbotime_lora         -> ideogram_4_turbotime_v1.safetensors
  ostris/ideogram_4_unconditional_lora     -> ideogram_4_unconditional_lora_r16.safetensors
  Civitai 2750357 Gray_000002000.safetensors  (gray-screen bypass LoRA)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
import struct
from pathlib import Path

TAG = "[Invisible-I4]"

# --------------------------------------------------------------------------- #
# source table
# --------------------------------------------------------------------------- #

HF_TURBOTIME = ("ostris/ideogram_4_turbotime_lora", "ideogram_4_turbotime_v1.safetensors", None)
HF_UNCOND = ("ostris/ideogram_4_unconditional_lora", "ideogram_4_unconditional_lora_r16.safetensors", None)
CIVITAI_GRAY = (2750357, "Gray_000002000.safetensors", None)

# model_id -> preferred file per quant (used by the variant picker / 2-click setup)
HF_CHECKPOINTS = {
    "fp8_scaled": ("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_fp8_scaled.safetensors"),
    "uncond_fp8_scaled": ("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"),
    "int8_convrot": ("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_int8_convrot.safetensors"),
    "uncond_int8_convrot": ("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_unconditional_int8_convrot.safetensors"),
    "nvfp4_mixed": ("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_nvfp4_mixed.safetensors"),
    "uncond_nvfp4_mixed": ("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors"),
    "te": ("Comfy-Org/Ideogram-4", "text_encoders/qwen3vl_8b_fp8_scaled.safetensors"),
    "vae": ("Comfy-Org/Ideogram-4", "vae/flux2-vae.safetensors"),
}

# quants with a real single-file source in HF_CHECKPOINTS (the picker's
# actionable choices; everything else is honestly marked manual/gated)
SOURCED_QUANTS = ("fp8_scaled", "int8_convrot", "nvfp4_mixed")


class AdapterMissing(Exception):
    """Raised when a download cannot proceed (gated, no token, disk full...)."""


def _status(progress_cb, msg: str) -> None:
    print(f"{TAG} {msg}")
    if progress_cb:
        try:
            progress_cb(msg)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# location helpers
# --------------------------------------------------------------------------- #

def _lora_dirs(models_dirs) -> list[Path]:
    """LoRA dirs come from detect.py (single layout owner)."""
    try:
        from pi_ideogram_lib.detect import lora_search_dirs
        return lora_search_dirs(models_dirs)
    except Exception:
        return []


def wanted_adapters(opts: dict) -> list[str]:
    """SINGLE owner of the 'which adapter files does this run need?' decision
    (spec F + P1). Engine passes its resolved run settings at generation time;
    install.py passes the same keys read from config.json at boot. Returns a
    list of kinds ('turbotime' | 'uncond' | 'gray') in stable order."""
    wanted: list[str] = []
    preset = str(opts.get("preset", "") or "")
    turbo = bool(opts.get("turbo")) or preset in ("TurboTime-4", "Instant-8")
    use_uncond_lora = bool(opts.get("use_uncond_lora"))
    if turbo:
        wanted.append("turbotime")
    elif use_uncond_lora:
        wanted.append("uncond")
    if bool(opts.get("filter_bypass")) and bool(opts.get("bypass_lora")):
        wanted.append("gray")
    return wanted


# ONE home for Ideogram adapters, under Forge's own Lora tree - so they show
# up wherever LoRAs are expected instead of in a private models/Ideogram4/loras
# that only this extension knew about. That split is how the install ended up
# with three Ideogram folders (models/Ideogram 4, models/Ideogram4,
# models/Lora/IDEOGRAM 4) holding pieces of the same thing.
LORA_SUBDIR = "IDEOGRAM LORA"


def adapter_dest(models_dirs) -> Path:
    """models/Lora/IDEOGRAM LORA (created on demand)."""
    base = None
    for d in models_dirs or []:
        cand = Path(d) / "Lora" / LORA_SUBDIR
        try:
            cand.mkdir(parents=True, exist_ok=True)
            base = cand
            break
        except Exception:
            continue
    if base is None:
        base = (Path(models_dirs[0]) / "Lora" / LORA_SUBDIR) if models_dirs else Path.cwd()
        base.mkdir(parents=True, exist_ok=True)
    return base


def _is_complete_safetensors(path: Path) -> bool:
    """Cheap integrity check: is this a WHOLE safetensors file?

    Structural, not size-based: a valid file is valid at any size, so there is
    no arbitrary byte floor to produce false negatives on small adapters.

    A safetensors file starts with a uint64 header length, and the header plus
    the tensor payload must fit inside the file. An interrupted download leaves
    a name that matches but a body that does not, and without this check
    `_on_disk` would report it present forever - the adapter would then fail to
    load with an opaque safetensors error instead of simply being re-fetched.
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return False
            header_len = struct.unpack("<Q", raw)[0]
            if header_len <= 0 or 8 + header_len > size:
                return False
            header = json.loads(f.read(header_len))
        # The header declares where every tensor ends. A download cut short has
        # a valid header and a short body, so the size check is what catches it.
        end = 0
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            offsets = meta.get("data_offsets")
            if isinstance(offsets, (list, tuple)) and len(offsets) == 2:
                end = max(end, int(offsets[1]))
        return 8 + header_len + end <= size
    except Exception:
        return False


def _on_disk(filename: str, models_dirs) -> Path | None:
    stem = Path(filename).stem.lower()
    for d in _lora_dirs(models_dirs):
        for cand in d.rglob("*.safetensors"):
            if cand.stem.lower() == stem or cand.name == filename:
                if not _is_complete_safetensors(cand):
                    print(f"[Invisible-I4] {cand.name} is present but truncated/incomplete "
                          "- ignoring it so it can be fetched again")
                    continue
                return cand
    return None


# --------------------------------------------------------------------------- #
# HF downloads (resume + etag verify + console tqdm)
# --------------------------------------------------------------------------- #

def _hf() -> dict:
    try:
        import huggingface_hub  # noqa: F401
        return {"ok": True}
    except ImportError as e:
        return {"ok": False, "error": str(e)}


def hf_download(repo_id: str, filename: str, dest_dir: Path, *, token: str | None = None,
                progress_cb=None) -> Path:
    """huggingface_hub hf_hub_download: resumable, etag-verified, tqdm to stderr.

    Gated/private repos (401 / GatedRepoError / 404 for gated repos) raise
    AdapterMissing with the exact accept-and-login steps. Never loops.
    """
    hub = _hf()
    if not hub["ok"]:
        raise AdapterMissing(
            "huggingface_hub is not importable in the Forge venv - install it with "
            "'pip install huggingface_hub' inside the Forge environment, or drop the "
            "file into models/Ideogram4/loras manually."
        )
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except Exception as e:
        raise AdapterMissing(f"huggingface_hub import failed: {e}") from e

    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        _status(progress_cb, f"Downloading {repo_id}/{filename} (resume supported)...")
        path = hf_hub_download(
            repo_id=repo_id, filename=filename,
            local_dir=str(dest_dir), token=token or None,
        )
        _status(progress_cb, f"Downloaded {Path(path).name} -> {path}")
        return Path(path)
    except GatedRepoError:
        raise AdapterMissing(
            f"'{repo_id}' is a gated repository. To use it:\n"
            "  1) open https://huggingface.co/" + repo_id + " in a browser\n"
            "  2) click 'Agree and access repository' and accept the license\n"
            "  3) run `huggingface-cli login` in the Forge venv (or set the HF_TOKEN "
            "environment variable)\n"
            "Then retry. Nothing was downloaded; aborting cleanly."
        ) from None
    except RepositoryNotFoundError:
        raise AdapterMissing(
            f"'{repo_id}' returned 404. It is either private, renamed, or requires "
            "accepting the Ideogram 4 license gate first (steps above)."
        ) from None
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "authentication" in str(e).lower():
            raise AdapterMissing(
                f"'{repo_id}' needs authentication (401/403): run `huggingface-cli login` "
                "in the Forge venv or set the HF_TOKEN environment variable, then retry."
            ) from None
        raise AdapterMissing(f"Download of {repo_id}/{filename} failed: {e}") from e


# --------------------------------------------------------------------------- #
# Civitai (public model API, gray-screen bypass LoRA)
# --------------------------------------------------------------------------- #

def civitai_download(model_id: int, filename: str, dest_dir, *, token=None,
                     progress_cb=None):
    """Removed. Civitai assets this project uses are creator-gated.

    Downloading them needed an API key pasted into the UI and stored in
    config.json - a credential this extension should never hold, for a job the
    browser already does better (already signed in, resumable, and it shows
    the licence). The Model Setup panel prints the link and the destination
    folder instead.
    """
    raise AdapterMissing(
        f"{filename} is not downloaded automatically. Open "
        f"https://civitai.com/models/{model_id} in your browser (sign in - the "
        "creator gates this file), download it, and drop it into "
        "models/Lora/IDEOGRAM LORA. Subfolders are searched.")

def verify_sha256(path: Path, expected_hex: str | None) -> bool:
    """Full-file sha256. Returns True on match, or when no expected hash is
    known (nothing to verify against). Never raises."""
    if not expected_hex:
        return True
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except Exception:
        return False
    return h.hexdigest().lower() == str(expected_hex).lower()


# --------------------------------------------------------------------------- #
# adapter auto-download (one call per feature)
# --------------------------------------------------------------------------- #

def download_adapter(kind: str, models_dirs, *, token: str | None = None,
                     progress_cb=None) -> Path | None:
    """kind: 'turbotime' | 'uncond' | 'gray'. Returns the local file, or None
    when it is already present. Raises AdapterMissing on failure (never loops)."""
    dest = adapter_dest(models_dirs)
    if kind == "turbotime":
        repo, fname, _sha = HF_TURBOTIME
        found = _on_disk(fname, models_dirs)
        if found:
            return None
        return hf_download(repo, fname, dest, token=token, progress_cb=progress_cb)
    if kind == "uncond":
        repo, fname, _sha = HF_UNCOND
        found = _on_disk(fname, models_dirs)
        if found:
            return None
        return hf_download(repo, fname, dest, token=token, progress_cb=progress_cb)
    if kind == "gray":
        found = _on_disk(CIVITAI_GRAY[1], models_dirs)
        if found:
            return None
        return civitai_download(CIVITAI_GRAY[0], CIVITAI_GRAY[1], dest, token=token, progress_cb=progress_cb)
    raise AdapterMissing(f"unknown adapter kind: {kind}")


def ensure_adapters(wanted: list[str], models_dirs, *, progress_cb=None,
                    quiet_missing: bool = True) -> list[str]:
    """Auto-download the missing adapters in `wanted` (turbotime/uncond/gray).

    Returns the list of freshly downloaded files. A missing/invalid download
    prints its exact fallback but never crashes the caller.
    """
    downloaded: list[str] = []
    for kind in wanted:
        try:
            result = download_adapter(kind, models_dirs, progress_cb=progress_cb)
            if result is not None:
                downloaded.append(str(result))
        except AdapterMissing as e:
            if quiet_missing:
                print(f"{TAG} {e}")
            else:
                raise
        except Exception as e:
            print(f"{TAG} adapter '{kind}' download failed ({e}) - feature will warn at generation")
    return downloaded


# --------------------------------------------------------------------------- #
# variant picker (spec F: UI list of quant variants with Recommended badge)
# --------------------------------------------------------------------------- #

def list_variants(vram_gb: float | None) -> list[dict]:
    """Downloadable checkpoint variants fit for the measured VRAM, each with a
    Recommended badge driven by the registry. Sorted by VRAM floor."""
    from pi_ideogram_lib.quant_registry import registry
    from pi_ideogram_lib.hardware import vram_profile

    profile = vram_profile(vram_gb)
    rec = {
        "<=8": "gguf_q4k", "8-12": "nf4", "12-16": "fp8_scaled",
        "16-24": "fp8_scaled", ">=24": "fp8_scaled", "unknown": "fp8_scaled",
    }.get(profile)
    out = []
    for e in registry():
        if not e.get("downloadable"):
            continue
        eid = e["id"]
        label = eid.upper()
        if eid == "fp8_scaled":
            label = "Ideogram4 FP8 Scaled (9.3 GB) - runs on any CUDA GPU"
        elif eid == "nf4":
            label = "Ideogram4 NF4 (official gated) - best VRAM fit 8-12 GB"
        elif eid == "int8_convrot":
            label = "Ideogram4 INT8 ConvRot (9.3 GB)"
        elif eid == "nvfp4_mixed":
            label = "Ideogram4 NVFP4 Mixed (Blackwell only)"
        elif eid.startswith("gguf"):
            label = f"{eid.upper()} (external sd.cpp, not in-process)"
        out.append({
            "id": eid,
            "label": label,
            "files": e.get("files", []),
            "vram_floor_gb": e.get("vram_floor_gb"),
            "loader": e.get("loader"),
            "notes": e.get("notes", ""),
            "recommended": eid == rec,
            "badge": e.get("recommended_badge", "") if eid == rec else "",
            "in_process": bool(e.get("in_process")),
        })
    return sorted(out, key=lambda v: (v["vram_floor_gb"] or 0))


def gate_hint() -> str:
    """Printed once at startup when the gated official weights are missing."""
    return (
        "Ideogram 4 weights not found on disk. The official repositories are GATED:\n"
        "  1) open https://huggingface.co/ideogram-ai/ideogram-4-fp8 (and/or -nf4)\n"
        "  2) click 'Agree and access repository' and accept the Ideogram 4 license\n"
        "  3) run `huggingface-cli login` in the Forge venv (or set HF_TOKEN)\n"
        "Then restart Forge - the checkpoints will appear in the dropdown as "
        "'Ideogram4 - <name>'.\n"
        "Ungated alternative: Comfy-Org/Ideogram-4 single files (fp8_scaled / "
        "int8_convrot / nvfp4_mixed) - no login needed."
    )


# --------------------------------------------------------------------------- #
# asset status + pointed per-file fetch (spec F picker). OWNER of "what is on
# disk for a chosen variant" and "how one asset is fetched"; the UI (engine)
# only renders rows returned here and fires fetch_asset on explicit clicks.
# --------------------------------------------------------------------------- #

# asset ids the picker knows: checkpoint halves are 'checkpoint:cond' /
# 'checkpoint:uncond' for the CHOSEN quant; te/vae are quant-independent;
# the three adapters reuse download_adapter(kind).
ASSET_LABELS = {
    "checkpoint:cond": "Conditional model (cond)",
    "checkpoint:uncond": "Unconditional model (uncond)",
    "te": "Text encoder (Qwen3-VL-8B)",
    "vae": "VAE (flux2)",
    "adapter:turbotime": "TurboTime LoRA (Ostris)",
    "adapter:uncond": "Uncond-replacement LoRA (Ostris)",
    "adapter:gray": "Gray-bypass LoRA (Civitai 2750357)",
}

# ids currently downloading (set around fetch_asset; UI shows 'in progress').
ACTIVE: set[str] = set()


def _mkdir_first(models_dirs, *parts) -> Path:
    """First writable models dir + parts, created on demand.

    The writability probe used to be `(cand / "Ideogram4").mkdir(...)`, which
    CREATED models/Ideogram4 as a side effect - for a text encoder, for a VAE,
    for anything. Deleting that folder achieved nothing because the next
    download recreated it. Probe the target we actually intend to use, and
    create nothing else.
    """
    base = None
    for d in models_dirs or []:
        cand = Path(d)
        try:
            cand.mkdir(parents=True, exist_ok=True)
            if not os.access(cand, os.W_OK):
                continue
            base = cand
            break
        except Exception:
            continue
    if base is None:
        base = Path(models_dirs[0]) if models_dirs else Path.cwd()
    target = base
    for part in parts:
        target = target / part
    target.mkdir(parents=True, exist_ok=True)
    return target


def _asset_source(asset: str, quant: str) -> tuple[str, str] | None:
    """(repo_id, repo-relative path) for a fetchable asset; None when this
    build only supports manual placement (gated nf4 / external gguf)."""
    if asset in ("checkpoint:cond", "checkpoint:uncond"):
        key = quant if asset == "checkpoint:cond" else "uncond_" + quant
        return HF_CHECKPOINTS.get(key)
    return HF_CHECKPOINTS.get(asset)


def _manual_source_note(quant_id: str) -> str:
    """Exact manual-placement note for quants with no one-click single-file
    source (registry owns the format verdicts; this mirrors them for the row)."""
    if quant_id == "nf4":
        return ("NF4 loads in-process (bnb loader + kernels verified on this machine) "
                "from a single-file bnb checkpoint - drop it into models/Ideogram4/ "
                "(Comfy single-file bnb NF4 export; the official ideogram-ai gated "
                "repo is diffusers-layout, key names UNVERIFIED against this loader).")
    if quant_id.startswith("gguf"):
        return ("GGUF is external sd.cpp/llama.cpp only - drop the .gguf into sd.cpp's "
                "models dir and run sd.cpp there; for in-Forge runs select fp8_scaled.")
    return (f"no single-file source for '{quant_id}' in this build - run it in the "
            "external tool that owns its format (see the variant card), then select "
            "fp8_scaled here for in-Forge generation.")


def _dest_dir_for(models_dirs, asset: str) -> Path:
    """Canonical destination folder for a downloaded asset."""
    if asset == "checkpoint:cond" or asset == "checkpoint:uncond":
        # Checkpoints belong in Forge's checkpoint tree, next to every other
        # model, not in a private models/Ideogram4 the dropdown has to be
        # taught about separately.
        return _mkdir_first(models_dirs, "Stable-diffusion", "IDEOGRAM 4")
    if asset == "te":
        return _mkdir_first(models_dirs, "text_encoder")
    if asset == "vae":
        return _mkdir_first(models_dirs, "VAE")
    return adapter_dest(models_dirs)


def _find_named(basename: str, dirs) -> Path | None:
    for d in dirs or []:
        for cand in Path(d).rglob("*.safetensors"):
            if cand.name == basename or cand.stem.lower() == Path(basename).stem.lower():
                return cand
    return None


def _asset_present(asset: str, quant: str, models_dirs) -> Path | None:
    """On-disk file backing an asset, or None. Same scans the engine uses, so
    the status line and the checkpoint dropdown can never disagree."""
    try:
        from pi_ideogram_lib.detect import model_search_dirs, te_vae_search_dirs
    except Exception:
        return None
    if asset == "checkpoint:cond":
        source = _asset_source(asset, quant)
        if source:
            found = _find_named(Path(source[1]).name, model_search_dirs(models_dirs))
            if found:
                return found
        # accept any Ideogram4 file whose name carries this quant token
        q = quant
        for d in model_search_dirs(models_dirs):
            for cand in Path(d).rglob("*.safetensors"):
                low = cand.stem.lower()
                if "uncond" in low:
                    continue
                tok = low.replace("ideogram", "").replace("ideogram4", "").replace("_", "").replace("-", "")
                if q.replace("_", "") in tok and q.split("_")[0] in low:
                    return cand
        return None
    if asset == "checkpoint:uncond":
        source = _asset_source(asset, quant)
        if source:
            found = _find_named(Path(source[1]).name, model_search_dirs(models_dirs))
            if found:
                return found
        # Ostris uncond LoRA replaces the second transformer - accept it too
        try:
            found = _on_disk(HF_UNCOND[1], models_dirs)
        except Exception:
            found = None
        return found
    if asset == "te":
        from pi_ideogram_lib.detect import find_text_encoder
        return find_text_encoder(te_vae_search_dirs(models_dirs))
    if asset == "vae":
        from pi_ideogram_lib.detect import find_vae
        return find_vae(te_vae_search_dirs(models_dirs))
    if asset == "adapter:turbotime":
        return _on_disk(HF_TURBOTIME[1], models_dirs)
    if asset == "adapter:uncond":
        return _on_disk(HF_UNCOND[1], models_dirs)
    if asset == "adapter:gray":
        return _on_disk(CIVITAI_GRAY[1], models_dirs)
    return None


def _adapter_kind(asset: str) -> str | None:
    if asset.startswith("adapter:"):
        return asset.split(":", 1)[1]
    return None


def assets_status(models_dirs, quant_id: str) -> list[dict]:
    """Per-file rows for the picker: cond/uncond of `quant_id`, TE, VAE and the
    three adapters. Each row: {id, label, present, path, in_progress, note,
    actionable} where actionable=False means 'manual placement only'."""
    rows: list[dict] = []
    for asset in ("checkpoint:cond", "checkpoint:uncond"):
        source = _asset_source(asset, quant_id)
        path = _asset_present(asset, quant_id, models_dirs)
        if source is None:
            note = _manual_source_note(quant_id)
            rows.append({
                "id": asset, "label": ASSET_LABELS[asset], "present": False,
                "path": None, "in_progress": asset in ACTIVE, "actionable": False,
                "note": note,
            })
            continue
        repo, repo_path = source
        rows.append({
            "id": asset, "label": ASSET_LABELS[asset],
            "present": path is not None, "path": str(path) if path else None,
            "in_progress": asset in ACTIVE, "actionable": True,
            "note": f"{Path(repo_path).name} from {repo}" if path is None else "",
        })
    for asset in ("te", "vae"):
        path = _asset_present(asset, quant_id, models_dirs)
        rows.append({
            "id": asset, "label": ASSET_LABELS[asset], "present": path is not None,
            "path": str(path) if path else None, "in_progress": asset in ACTIVE,
            "actionable": True,
            "note": f"from Comfy-Org/Ideogram-4" if path is None else "",
        })
    for asset in ("adapter:turbotime", "adapter:uncond", "adapter:gray"):
        path = _asset_present(asset, quant_id, models_dirs)
        rows.append({
            "id": asset, "label": ASSET_LABELS[asset], "present": path is not None,
            "path": str(path) if path else None, "in_progress": asset in ACTIVE,
            "actionable": asset != 'adapter:gray', "note": "" if path is not None else "Select public files in Model Setup, or download manually",
        })
    return rows


def fetch_asset(asset: str, quant_id: str, models_dirs, *, progress_cb=None) -> Path | None:
    """Fetch ONE asset for the chosen quant. Returns the local file, or None
    when it is already on disk. Fires ONLY on an explicit picker click - this
    function is never called at boot or by generation. Resumes partial
    downloads (hf_hub_download etag/range; Civitai .part). Raises
    AdapterMissing with the exact fallback when a source is gated/manual."""
    if asset in ACTIVE:
        raise AdapterMissing(f"{ASSET_LABELS.get(asset, asset)} is already downloading - wait for it to finish")
    if asset.startswith("adapter:"):
        kind = asset.split(":", 1)[1]
        result = download_adapter(kind, models_dirs, progress_cb=progress_cb)
        return result
    source = _asset_source(asset, quant_id)
    if source is None:
        raise AdapterMissing(
            f"{ASSET_LABELS.get(asset, asset)} for '{quant_id}': " + _manual_source_note(quant_id)
            + " Nothing was changed."
        )
    present = _asset_present(asset, quant_id, models_dirs)
    if present is not None:
        _status(progress_cb, f"{ASSET_LABELS.get(asset, asset)} already present ({present}) - nothing to do")
        return None
    repo, repo_path = source
    dest_dir = _dest_dir_for(models_dirs, asset)
    basename = Path(repo_path).name
    ACTIVE.add(asset)
    try:
        path = hf_download(repo, repo_path, dest_dir, progress_cb=progress_cb)
        target = dest_dir / basename
        if path is not None and Path(path).resolve() != target.resolve():
            try:
                if target.exists():
                    target.unlink()
                os.replace(str(path), str(target))
            except Exception:
                pass  # cross-volume or lock: keep the hf layout file - it is still discoverable
        return target if target.exists() else Path(path)
    finally:
        ACTIVE.discard(asset)
