# Project Invisible: namespaced Forge adapter; no core-file edits.
"""PROJECT INVISIBLE - one owner of "which checkpoint is selected".

Answers three questions and nothing else:
  1. is this CheckpointInfo / filename an Ideogram 4 file?   (_ci_is_ideogram4)
  2. what does the UI currently point at?                    (_current_* , is_ideogram4_selected)
  3. does a given dropdown NAME resolve to an Ideogram 4 file?(_selected_ci_is_i4)

Everything that needs "is Ideogram 4 active right now" (the takeover in
pi_ideogram_forge/run.py, the accordion visibility in scripts/engine.py) calls THIS
module.  Forge imports (modules.sd_models / modules.shared) are lazy and
guarded so the module imports standalone; when Forge is absent every
"current selection" answer is None/False.

File identity (header truth) lives in pi_ideogram_lib/detect.py - the
layout/format owner.  This module only decides *which file the user picked*.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXT_DIR = Path(__file__).resolve().parent.parent
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))

from pi_ideogram_lib.detect import is_ideogram4_checkpoint_file

TAG = "[Invisible-I4]"


def _ci_is_ideogram4(ci) -> bool:
    """True when a CheckpointInfo points at an Ideogram 4 file (header-based,
    cheap; GGUF fall back to name tokens since they have no safetensors header)."""
    fn = getattr(ci, "filename", None)
    if not fn:
        return False
    try:
        if is_ideogram4_checkpoint_file(fn):
            return True
    except Exception:
        pass
    p = Path(fn)
    return p.suffix.lower() in (".gguf", ".ggml") and "ideogram" in p.name.lower()


def _current_checkpoint_info():
    try:
        from modules.sd_models import model_data
        return model_data.forge_loading_parameters.get("checkpoint_info", None)
    except Exception:
        return None


def _current_ideogram_file() -> Path | None:
    ci = _current_checkpoint_info()
    if ci is None:
        return None
    fn = getattr(ci, "filename", None)
    if not fn:
        return None
    meta = getattr(ci, "metadata", None) or {}
    if str(meta.get("model_type", "")).lower() in ("ideogram4_cond", "ideogram4_uncond", "ideogram4"):
        return Path(fn)
    try:
        if is_ideogram4_checkpoint_file(fn):
            return Path(fn)
    except Exception:
        pass
    # GGUF / unreadable-header files whose NAME marks them Ideogram 4
    p = Path(fn)
    if p.suffix.lower() in (".gguf", ".ggml") and "ideogram" in p.name.lower():
        return p
    return None


def is_ideogram4_selected() -> bool:
    return _current_ideogram_file() is not None


def _selected_checkpoint_name() -> str:
    """The checkpoint the UI currently points at (persisted across restarts)."""
    try:
        from modules import shared as _s
        return str(getattr(_s.opts, "sd_model_checkpoint", "") or "")
    except Exception:
        return ""


def _resolve_ci_for_name(ckpt_name):
    """CheckpointInfo for a dropdown value: aliases first, then a name/id match
    across the stock list (covers aliases lagging behind a typed selection)."""
    try:
        from modules import sd_models
        if not ckpt_name:
            return None
        ci = sd_models.checkpoint_aliases.get(ckpt_name)
        if ci is not None:
            return ci
        low = str(ckpt_name).lower()
        for cand in getattr(sd_models, "checkpoints_list", {}).values():
            if low == str(getattr(cand, "name", "")).lower() or \
               any(str(i).lower() == low for i in (getattr(cand, "ids", None) or [])):
                return cand
    except Exception:
        pass
    return None


def _selected_ci_is_i4(ckpt_name=None) -> bool:
    """Is the named checkpoint an Ideogram 4 file? Resolution order: the actual
    CheckpointInfo (header truth), then the name string itself (so the accordion
    appears even when the model-change event has not fired yet)."""
    name = ckpt_name if ckpt_name is not None else _selected_checkpoint_name()
    if not name:
        return False
    ci = _resolve_ci_for_name(name)
    if ci is not None:
        return _ci_is_ideogram4(ci)
    low = str(name).lower()
    return "ideogram4" in low


def uncond_refusal_message(file_path=None, search_dirs=None) -> str | None:
    """Plain-language generate-time refusal when the checkpoint that would run
    is the Ideogram 4 UNCONDITIONAL half: that file is a diffusion model the
    pipeline loads from its own folder and pairs automatically - it is never
    itself a checkpoint. Returns None for every other file (including the cond
    half and non-Ideogram4 checkpoints).

    When a matching conditional twin is discoverable the message NAMES it so
    the user can select it directly; the twin search scans only when the file
    exists (never for unit fakes). search_dirs is optional (detect's layout is
    the default)."""
    try:
        from pi_ideogram_lib.detect import (find_matching_cond, is_ideogram4_uncond_file,
                                            model_search_dirs)
        from pi_ideogram_forge.paths import models_dirs as _models_dirs
    except Exception:
        return None
    try:
        f = Path(file_path) if file_path is not None else _current_ideogram_file()
        if f is None or not is_ideogram4_uncond_file(f):
            return None
        msg = ("the checkpoint you selected is the UNCONDITIONAL half of the Ideogram 4 "
               "pair. It is a diffusion model the pipeline loads from its own folder and "
               "pairs automatically - it is not itself a checkpoint to generate from. ")
        twin = None
        if f.exists():
            if search_dirs is None:
                search_dirs = model_search_dirs(_models_dirs())
            twin = find_matching_cond(f, search_dirs)
        if twin is not None:
            msg += (f"Select '{twin.stem}' (its matching conditional file) in the "
                    "checkpoint dropdown and Generate again.")
        else:
            msg += ("Select the conditional Ideogram 4 checkpoint - the one WITHOUT "
                    "'uncond'/'unconditional' in its name - and Generate will pair it "
                    "automatically.")
        return msg
    except Exception:
        return None
