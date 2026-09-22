# Project Invisible: namespaced Forge adapter; no core-file edits.
"""PROJECT INVISIBLE - one owner of the stock checkpoint dropdown.

Makes every Ideogram 4 file Forge lists read "Ideogram4 - <name>" (spec A/D):
  * files under models/Ideogram4/ (+ legacy spellings) that Forge never scans
    are registered from scratch as fresh CheckpointInfo entries;
  * files the STOCK scanner already found (models/Stable-diffusion/** incl.
    subfolders) get their existing entries renamed in place so no duplicate
    appears.

The hook targets modules.sd_models.list_models only (the loader API the spec
allows) and fails closed: no hook, no crash - stock-listed Ideogram 4 files
still work through header detection.  Which file the user ends up selecting is
pi_ideogram_forge/selection.py's question; where model dirs are is pi_ideogram_forge/paths.py's.
Nothing here loads weights: dropdown registration is selection metadata only,
and Forge's own sd_models loader is never replaced (see ARCHITECTURE.md).
"""

from __future__ import annotations

import os
from pathlib import Path

TAG = "[Invisible-I4]"


def _rename_ci_to_i4_label(ci, sd_models) -> bool:
    """Mutate a CheckpointInfo so the dropdown shows 'Ideogram4 - <file>' instead
    of the raw folder path, and re-register it under the new identity (same
    pattern as CheckpointInfo.calculate_shorthash). Returns True when renamed."""
    from pi_ideogram_forge.selection import _ci_is_ideogram4
    try:
        if not _ci_is_ideogram4(ci):
            return False
        name = str(getattr(ci, "name", "") or "")
        if "Ideogram4" in name or "Ideogram 4" in name:
            return False  # already labelled
        base = getattr(ci, "name_for_extra", None) or Path(getattr(ci, "filename", "")).name
        old_title = getattr(ci, "title", None) or name
        ci.name = f"Ideogram4 — {base}"
        ci.model_name = os.path.splitext(ci.name.replace("/", "_").replace("\\", "_"))[0]
        ci.title = f"{ci.name} [{ci.shorthash}]" if ci.shorthash else ci.name
        ci.short_title = f"{base} [{ci.shorthash}]" if ci.shorthash else base
        ci.ids = [ci.hash, ci.model_name, ci.title, ci.name, base,
                  f"{ci.name} [{ci.hash}]", f"{base} [{ci.hash}]"]
        if ci.shorthash:
            ci.ids += [ci.shorthash, ci.sha256, f"{ci.name} [{ci.shorthash}]",
                       f"{base} [{ci.shorthash}]"]
        replace = getattr(sd_models, "replace_key", None)
        if callable(replace):
            replace(sd_models.checkpoints_list, old_title, ci.title, ci)
        ci.register()
        return True
    except Exception:
        return False


_uncond_cache: dict = {}


def _maybe_ideogram_name(path) -> bool:
    """Cheap name gate - NO disk access.

    detect.is_ideogram4_uncond_file reads the safetensors header whenever the
    filename lacks 'uncond', which is the case for every ordinary checkpoint.
    That is fine for a one-off scan and ruinous on a hot path: calling it from
    the CheckpointInfo.register wrapper meant a header read per registration
    across the whole model folder, and a live Forge came up listing 3 of 58
    checkpoints before the API timed out.

    The Ideogram 4 uncond half always carries 'ideogram' or 'uncond' in its
    name in every published layout, so this gate only ever lets MORE through
    than needed - the header check behind it still decides. Anything else is
    rejected without touching the disk.
    """
    low = Path(str(path)).stem.lower()
    return "ideogram" in low or "uncond" in low


def _is_uncond_ideogram4_file(path) -> bool:
    """Is this the Ideogram 4 UNCONDITIONAL half? Identity lives in detect.py
    (single owner) - header truth when readable, name tokens otherwise. The
    uncond file is a diffusion model the pipeline pairs automatically - never
    a checkpoint the user should pick.

    Memoised on (path, size, mtime): the answer cannot change unless the file
    does, and this is called from a hot path.
    """
    from pi_ideogram_lib.detect import is_ideogram4_uncond_file

    try:
        st = Path(str(path)).stat()
        key = (str(path), st.st_size, int(st.st_mtime))
    except Exception:
        key = (str(path), -1, -1)
    hit = _uncond_cache.get(key)
    if hit is not None:
        return hit
    val = bool(is_ideogram4_uncond_file(path))
    _uncond_cache[key] = val
    return val


def _prune_uncond_entries(sd_models) -> int:
    """Remove every Ideogram 4 unconditional-half entry Forge has listed (own
    dirs + stock scan) from checkpoints_list and checkpoint_aliases, so the
    dropdown never offers it as a checkpoint. Returns how many were removed.
    Idempotent; never touches non-Ideogram4 files (an 'uncond'-named SD file
    stays listed)."""
    doomed = []
    for ci in list(getattr(sd_models, "checkpoints_list", {}).values()):
        fn = getattr(ci, "filename", None)
        # CHEAP NAME GATE FIRST. _is_uncond_ideogram4_file falls back to reading
        # the safetensors header whenever the filename lacks "uncond", and on
        # large checkpoints that header is megabytes. Running it over a whole
        # model folder cost 240 SECONDS on the first pass - measured, and it
        # was the entire reason Forge's "create ui" phase took four minutes
        # (2.9s with this extension disabled). The gate can only over-admit;
        # the header check behind it still decides.
        if fn and _maybe_ideogram_name(fn) and _is_uncond_ideogram4_file(fn):
            doomed.append(ci)
    cpl = getattr(sd_models, "checkpoints_list", None)
    aliases = getattr(sd_models, "checkpoint_aliases", None)

    # Remove by IDENTITY, not by ci.title. The register pass above renames
    # entries to "Ideogram4 - <name>", and the dict KEY is the renamed title
    # while ci.title can still hold the pre-rename one. Looking the key up from
    # the attribute then missed, `cpl.get(title) is ci` was False, the pop
    # silently did nothing - and the function still returned len(doomed), so it
    # reported "1 unconditional-half file hidden" while the entry stayed in the
    # dropdown and in /sdapi/v1/sd-models. Scanning for the object itself
    # cannot drift out of sync with whatever key it is filed under.
    removed = 0
    doomed_ids = {id(ci) for ci in doomed}
    if cpl is not None:
        for key in [k for k, v in list(cpl.items()) if id(v) in doomed_ids]:
            cpl.pop(key, None)
            removed += 1
    if aliases is not None:
        for key in [k for k, v in list(aliases.items()) if id(v) in doomed_ids]:
            aliases.pop(key, None)
    # Removing the entry is NOT enough on its own: the CheckpointInfo object
    # survives (aliases, the UI, whatever else holds it) and re-inserts itself
    # the moment anything hashes it.
    #
    #   CheckpointInfo.calculate_shorthash()   sd_models.py:106-111
    #       -> replace_key(checkpoints_list, old, new, self)   # line 27:
    #          d[new_key] = value  -- UNCONDITIONAL, re-adds even when the old
    #          key is gone
    #       -> self.register()                                 # line 88:
    #          checkpoints_list[self.title] = self
    #
    # That is why the uncond half kept reappearing in /sdapi/v1/sd-models after
    # a prune that had genuinely popped it. Neutralise the object itself so no
    # later caller can put it back. Instance attributes shadow the class
    # methods, and only ever on the uncond half.
    for ci in doomed:
        try:
            ci.register = lambda *a, **k: None
            ci.calculate_shorthash = lambda *a, **k: getattr(ci, "shorthash", None)
        except Exception:
            pass

    # Report what was ACTUALLY removed from the list the UI and the API read,
    # never the count we hoped to remove.
    return removed


def _register_ideogram4_dir_checkpoints() -> None:
    """Run the rename/register pass over BOTH sources (own dirs + stock list).
    Folder spellings come from pi_ideogram_lib.detect (single owner). The
    UNCONDITIONAL half is deliberately NOT offered as a checkpoint here - it
    is a diffusion model the pipeline loads from its own folder."""
    from pi_ideogram_lib.detect import ideogram4_ckpt_dirs
    from pi_ideogram_forge.paths import models_dirs
    try:
        import modules.sd_models as sd_models
    except Exception:
        return
    seen: set[str] = set()
    registered = 0
    # (1) own dirs: fresh CheckpointInfo + register (skip uncond halves)
    for d in ideogram4_ckpt_dirs(models_dirs()):
        for f in sorted(list(d.rglob("*.safetensors")) + list(d.rglob("*.gguf"))):
            r = str(f.resolve())
            if r in seen:
                continue
            seen.add(r)
            low = f.name.lower()
            if low.startswith(".") or any(t in low for t in (".vae.", "-vae", "vae-")) or \
               "loras" in f.parts or "lora" in str(f.relative_to(d)).lower():
                continue
            if _is_uncond_ideogram4_file(f):
                continue  # unconditional half: paired + loaded by the pipeline
            try:
                ci = sd_models.CheckpointInfo(str(f))
                if _rename_ci_to_i4_label(ci, sd_models):
                    registered += 1
            except Exception:
                continue
    # (2) stock-scanned entries (any checkpoint dir the stock scanner walks)
    for ci in list(getattr(sd_models, "checkpoints_list", {}).values()):
        r = str(Path(getattr(ci, "filename", "")).resolve())
        if r in seen:
            continue
        seen.add(r)
        if _rename_ci_to_i4_label(ci, sd_models):
            registered += 1
    # (3) unconditional halves Forge stock-listed are never selectable
    pruned = _prune_uncond_entries(sd_models)
    if registered:
        print(f"{TAG} Ideogram4 checkpoint(s) labelled 'Ideogram4 - <name>' in the dropdown "
              f"({registered} file(s))")
    if pruned:
        print(f"{TAG} {pruned} Ideogram4 unconditional-half file(s) hidden from the dropdown "
              "(diffusion models the pipeline pairs automatically - not checkpoints)")
    # RESTORE THE LAST SELECTION, now that every Ideogram 4 entry carries its
    # renamed ids. Forge stores our files by DISPLAY LABEL
    # ('Ideogram4 - <name>', em-dash separated), not by path as it does for
    # every other preset, so that string has to survive a rename, a JSON
    # round trip and this registration pass to resolve. When it does resolve,
    # restore_checkpoint() does nothing; it only repairs a selection that was
    # lost, never one the user changed on purpose.
    try:
        from pi_ideogram_lib.detect import restore_checkpoint
        for line in restore_checkpoint():
            print(f"{TAG} {line}")
    except Exception:
        pass


def _install_register_block() -> None:
    """Refuse registration of the Ideogram 4 unconditional half at the source.

    Pruning it from checkpoints_list can never hold on its own, and three
    successive attempts proved it:

      1. pop by ci.title            - the register pass renames the dict KEY,
                                      so the lookup missed and the pop did
                                      nothing (while still reporting success)
      2. pop by object identity     - genuinely removed it, and Forge put it
                                      straight back: calculate_shorthash()
                                      calls replace_key() (sd_models.py:27,
                                      `d[new_key] = value`, unconditional) and
                                      then self.register() (line 88)
      3. neutralise that object     - only protects THAT instance, and the
                                      stock scan builds a brand-new
                                      CheckpointInfo on every list_models()

    Every one of those paths ends in `CheckpointInfo.register()`, so that is
    where the decision belongs. One wrapper on the class method covers
    list_models, calculate_shorthash, forge_model_reload and anything added
    later, without our having to enumerate callers.

    Only the Ideogram 4 uncond half is refused - it is a diffusion model the
    pipeline pairs itself from disk by path, never something the user picks.
    Every other checkpoint registers exactly as before. Fail-closed: if the
    symbol moves, the wrapper is skipped and behaviour is stock.
    """
    try:
        import modules.sd_models as sd_models
        CheckpointInfo = getattr(sd_models, "CheckpointInfo", None)
        if CheckpointInfo is None or getattr(CheckpointInfo, "_i4_reg_blocked", False):
            return
        _orig_register = CheckpointInfo.register

        def _register(self, *a, **k):
            try:
                fn = getattr(self, "filename", "")
                # Cheap name gate before any disk access: register() runs for
                # every checkpoint in the folder, so this must not do I/O on
                # the 57 files that obviously are not ours.
                if _maybe_ideogram_name(fn) and _is_uncond_ideogram4_file(fn):
                    return  # never listable
            except Exception:
                pass
            return _orig_register(self, *a, **k)

        CheckpointInfo.register = _register
        CheckpointInfo._i4_reg_blocked = True
        print(f"{TAG} unconditional-half registration blocked at the source "
              "(CheckpointInfo.register); it can no longer come back via any path")
    except Exception as e:
        print(f"{TAG} WARNING: could not block uncond registration ({e}); the prune "
              "still runs, but Forge may re-list the unconditional half")


def install_dropdown_hook() -> None:
    """Hook modules.sd_models.list_models (loader API - allowed by the spec).
    Fail-closed: any exception disables the hook; stock presets untouched."""
    try:
        import modules.sd_models as sd_models
        if getattr(sd_models, "_i4_dropdown_hooked", False):
            return
        _orig = sd_models.list_models

        def _patched(*a, **k):
            # Transparent by construction: forward whatever Forge passes and
            # return whatever it returns, so a signature change upstream is a
            # no-op here rather than a crash in the model list.
            out = _orig(*a, **k)
            try:
                _register_ideogram4_dir_checkpoints()
            except Exception as error:
                print(f'{TAG} Ideogram dropdown registration skipped ({error}); stock model list preserved')
            return out

        sd_models.list_models = _patched
        sd_models._i4_dropdown_hooked = True
        _install_register_block()
        print(f"{TAG} checkpoint dropdown hook active: Ideogram4 files are labelled "
              "'Ideogram4 - <name>'")
    except Exception as e:
        print(f"{TAG} WARNING: checkpoint dropdown hook unavailable ({e}); Ideogram4 files "
              "in models/Stable-diffusion still work through header detection")
