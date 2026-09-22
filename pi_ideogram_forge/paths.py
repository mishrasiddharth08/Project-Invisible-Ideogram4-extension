# Project Invisible: namespaced Forge adapter; no core-file edits.
"""PROJECT INVISIBLE - one owner of "where the model dirs are".

Every Forge-interface module (engine, run, picker, dropdown, selection)
needs the list of directories that hold checkpoints / text encoders / VAEs /
LoRAs in THIS Forge install.  That answer is Forge's models_path at runtime
(modules.paths_internal) with a sane fallback for standalone use - and it is
computed HERE, never re-derived.  The folder SPELLINGS inside those dirs are
owned by pi_ideogram_lib/detect.py (this module does not re-list them).

No Forge imports at module top: modules.paths_internal is probed lazily so
this module (and anything importing it) stays importable standalone.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXT_DIR = Path(__file__).resolve().parent.parent
if str(EXT_DIR) not in sys.path:
    sys.path.insert(0, str(EXT_DIR))

TAG = "[Invisible-I4]"


def models_dirs() -> list[Path]:
    """The Forge models root(s) for this install (runtime truth), or the
    conventional sibling-of-Forge layout when Forge is not importable."""
    try:
        from modules.paths_internal import models_path
        return [Path(models_path)]
    except Exception:
        return [EXT_DIR.parent.parent / "models"]
