# Project Invisible: namespaced Forge adapter; no core-file edits.
"""PROJECT INVISIBLE - spec F variant picker: the Model Setup VIEW.

Pure helpers only: every function returns markdown strings (or a generator of
them) built from downloader state / registry recommendations - no gradio, no
Forge imports at module top, fully testable.  The gradio wiring (components,
.click/.change bindings) lives in Ideogram4EngineScript.ui() in
scripts/engine.py; state + fetch live in download/downloader.py
(assets_status / fetch_asset / ACTIVE / SOURCED_QUANTS); the chosen variant
persists through settings.persist_chosen_variant -> config.json ->
settings.resolve()['chosen_variant'] at generate time.  Downloads fire ONLY
from explicit button clicks - never at boot, never from a generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXT_DIR = Path(__file__).resolve().parent.parent
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))

try:
    from modules import shared
except Exception:
    shared = None

TAG = "[Invisible-I4]"


def _picker_cfg() -> dict:
    import pi_ideogram_lib.config as pi_config
    return pi_config.load()


def _picker_quant(cfg=None) -> str:
    """Quant the picker targets: persisted choice, else the GPU recommendation."""
    import pi_ideogram_lib.settings as pi_settings
    cfg = cfg if cfg is not None else _picker_cfg()
    return pi_settings.chosen_variant(cfg)


def _variant_card_markdown(cfg=None) -> str:
    """Recommended card + every registry variant, honestly flagged (registry
    notes embedded for anything NOT loadable in-process)."""
    import pi_ideogram_lib.settings as pi_settings
    from pi_ideogram_lib.quant_registry import recommended_for_vram, registry
    cfg = cfg if cfg is not None else _picker_cfg()
    vram = cfg.get("hw_vram_gb")
    rec = recommended_for_vram(vram)
    chosen = pi_settings.chosen_variant(cfg)
    head = f"**Recommended for this GPU{' (' + str(vram) + ' GB)' if vram else ''}: {rec}**"
    lines = [head, ""]
    for e in registry():
        if not e.get("downloadable"):
            continue
        eid = e["id"]
        tag = ""
        if eid == rec:
            tag = f" **[{e.get('recommended_badge') or 'Recommended'}]**"
        elif eid == chosen:
            tag = " *(your choice)*"
        loadable = "in-process ✓" if e.get("in_process") else "NOT in-process (documented fallback)"
        line = f"- `{eid}` — {loadable}{tag}"
        note = str(e.get("notes") or "").replace("|", "/")
        if not e.get("in_process") and note:
            line += " — " + note
        elif e.get("in_process") and note and ("UNVERIFIED" in note or "gated" in note.lower()):
            # e.g. nf4: loader verified, but the official source is gated + layout-UNVERIFIED
            line += " — " + (note[:200] + "…" if len(note) > 200 else note)
        lines.append(line)
    return "\n".join(lines)


def _picker_markdown(models_dirs, quant: str | None = None, cfg=None, note: str | None = None) -> str:
    """Live one-line status + per-file rows (the SAME state the download buttons
    act on). quant defaults to the chosen/recommended variant."""
    cfg = cfg if cfg is not None else _picker_cfg()
    quant = quant or _picker_quant(cfg)
    from pi_ideogram_assets.downloader import ASSET_LABELS, assets_status
    rows = assets_status(models_dirs, quant)
    marks = []
    lines: list[str] = []
    for row in rows:
        if row["present"]:
            mark = "✓"
        elif row.get("in_progress"):
            mark = "…"
        elif row.get("actionable"):
            mark = "○"
        else:
            mark = "⚠"
        marks.append(f"{row['label'].split(' (')[0].split(' — ')[-1].lower()}={mark}")
        detail = row["note"] or ""
        if row["present"] and row.get("path"):
            try:
                detail = Path(row["path"]).name
            except Exception:
                pass
        state = "present" if row["present"] else ("in progress (resume supported)" if row.get("in_progress") else ("missing" if row.get("actionable") else "manual only"))
        lines.append(f"- {row['label']}: {mark} {state} {('— ' + detail) if detail else ''}".rstrip())
    summary = " · ".join(marks)
    out = [f"**Variant {quant}** — {summary}", ""] + lines
    if note:
        out += ["", "**" + str(note)[:300] + "**"]
    return "\n".join(out)


def _on_variant_changed(quant: str, models_dirs=None) -> str:
    """Variant dropdown change: persist the choice (settings flow) + refresh."""
    import pi_ideogram_lib.settings as pi_settings
    from pi_ideogram_forge.paths import models_dirs as _default_dirs
    pi_settings.persist_chosen_variant(str(quant))
    return _picker_markdown(models_dirs if models_dirs is not None else _default_dirs(),
                            quant=str(quant))
