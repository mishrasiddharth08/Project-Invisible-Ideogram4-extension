# Project Invisible: namespaced Forge adapter; no core-file edits.
"""PROJECT INVISIBLE - detected-format policy surface (audit-gap R).

The registry (pi_ideogram_lib/quant_registry.py) OWNS the detected format:
sniff_quant() (header magic + filename, header authoritative) and
route_checkpoint() -> {id, source, in_process, loadable_in_process,
merge_mode, merge_label, blocked, fallback}.  This module only RENDERS that
policy for the user (selection-time markdown in the Model Setup section) and
ENFORCES it at run time (a confident sniff of a format this build cannot load
refuses generation with the documented fallback - nothing is loaded, nothing
is changed).  Which file is selected is pi_ideogram_forge/selection.py's question.

Where Forge owns the loader (the stock sd_models machinery behind the
dropdown) this module never pretends to replace it: dropdown registration is
selection metadata only, actual weight loading is the extension pipeline, and
that split is stated on every Ideogram 4 run.
"""

from __future__ import annotations

from pathlib import Path

TAG = "[Invisible-I4]"


def _policy_for_current(path=None, cfg=None) -> dict:
    """Routing policy for the selected checkpoint (or a given path)."""
    from pi_ideogram_lib.quant_registry import route_checkpoint
    if path is None:
        from pi_ideogram_forge.selection import _current_ideogram_file
        path = _current_ideogram_file()
    if cfg is None:
        import pi_ideogram_lib.config as pi_config
        cfg = pi_config.load()
    try:
        return route_checkpoint(path, cfg.get("hw_vram_gb"))
    except Exception:
        return {"id": "unknown", "path": str(path) if path else None,
                "in_process": False, "loadable_in_process": True,
                "note": "routing check unavailable - proceeding with weight inspection"}


def _routing_markdown(path=None, cfg=None) -> str:
    """Selection-time surface: 'detected: fp8_scaled - in-process ✓' in the
    Model Setup accordion; prominent documented fallback for formats this
    build cannot load (never implied loadable)."""
    pol = _policy_for_current(path, cfg)
    if not pol.get("path") or pol["id"] == "unknown":
        if pol["id"] == "unknown" and pol.get("path"):
            return ("**detected: format not recognized** — "
                    + str(pol.get("note", "")))
        return "No Ideogram 4 checkpoint selected."
    if pol.get("in_process"):
        out = [f"**detected: {pol['id']} — in-process ✓**", str(pol.get("merge_label", ""))]
        # spec H: surface a distilled/Instant base so the TurboTime stacking
        # refusal at Generate time is never a surprise (registry owns sniffing)
        try:
            from pi_ideogram_lib.quant_registry import sniff_distilled
            if sniff_distilled(str(pol["path"])).get("distilled"):
                out.append("**base: already-distilled / Instant merge — TurboTime LoRA "
                           "stacking is disabled for this file**")
        except Exception:
            pass
        return "  \n".join(out)
    fallback = pol.get("fallback")
    if fallback:
        return (f"**⚠ detected: {pol['id']} — NO in-process loader in this build**  \n"
                f"{pol.get('merge_label', '')} {pol.get('note', '')}  \n"
                f"**Select the `{fallback}` variant instead (in-process ✓)** — Generate "
                "is refused for this file with these instructions.")
    return f"**⚠ detected: {pol['id']} — external / manual only**  \n{pol.get('note', '')}"


def _enforce_loadable_policy(path=None, cfg=None) -> dict:
    """RUN-TIME authority: a confident sniff of a format this build cannot
    load (int8 / nvfp4 / w8a8 / gguf) REFUSES generation with the documented
    fallback instead of silently proceeding. Everything else logs the routing
    line (merge path, loader split vs Forge) and returns the policy.
    Raises ValueError on refusal (the wrapper's error card shows it)."""
    pol = _policy_for_current(path, cfg)
    if pol.get("blocked"):
        raise ValueError(
            f"The selected Ideogram 4 file is `{pol['id']}` - this build has no in-process "
            f"loader for it. {pol.get('merge_label', '')} {pol.get('note', '')} "
            f"Select the '{pol.get('fallback') or 'fp8_scaled'}' variant (in-process ✓) "
            "or use the documented external path. Nothing was loaded or changed.")
    if pol["id"] == "unknown":
        print(f"{TAG} [routing] detected format unknown - the pipeline's weight inspection "
              "is the authority; unsupported formats fail at load with a clear error")
    else:
        print(f"{TAG} [routing] detected {pol['id']} (source={pol.get('source')}) "
              f"loader={pol.get('loader')} in_process=yes merge={pol.get('merge_mode')} "
              "vae=flux2 (forced) - weights load through the EXTENSION pipeline; Forge's "
              "SD loader never touches them (dropdown registration is selection metadata only)")
    return pol
