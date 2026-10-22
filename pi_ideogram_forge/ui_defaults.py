"""Native Forge sampler defaults for Ideogram checkpoint selection."""
from __future__ import annotations

TAG = "[Invisible-I4]"
EULER = "Euler"
NOTICE = ("Ideogram 4 selected: txt2img and img2img samplers were set to Euler. "
          "API requests are unchanged and must explicitly send sampler_name='Euler'.")


def _is_ideogram(checkpoint_name) -> bool:
    from pi_ideogram_forge.selection import _selected_ci_is_i4
    return _selected_ci_is_i4(checkpoint_name)


def _selection_values(checkpoint_name, detector=_is_ideogram):
    return (EULER, EULER) if detector(checkpoint_name) else (None, None)


def _on_selection(checkpoint_name):
    import gradio as gr
    values = _selection_values(checkpoint_name)
    if values[0] is None:
        skip = getattr(gr, "skip", lambda: {"__type__": "update"})
        return skip(), skip()
    info = getattr(gr, "Info", None)
    if callable(info):
        info(NOTICE)
    else:
        print(f"{TAG} NOTICE: {NOTICE}")
    return values


def _find_native_components(demo):
    wanted = {
        "setting_sd_model_checkpoint": "checkpoint",
        "txt2img_sampling": "txt2img",
        "img2img_sampling": "img2img",
    }
    found = {}
    for component in getattr(demo, "blocks", {}).values():
        name = wanted.get(getattr(component, "elem_id", None))
        if name and getattr(component, "is_rendered", False):
            if name in found and found[name] is not component:
                return None
            found[name] = component
    return found if len(found) == len(wanted) else None


def _bind_on_app_started(demo, _app=None) -> bool:
    if demo is None:
        return False
    if getattr(demo, "_i4_sampler_defaults_bound", False):
        return True
    components = _find_native_components(demo)
    if components is None:
        print(f"{TAG} WARNING: native sampler controls were not uniquely available; "
              "select Euler manually for Ideogram 4.")
        return False
    checkpoint = components["checkpoint"]
    change = getattr(checkpoint, "change", None)
    if not callable(change):
        label = str(getattr(checkpoint, "label", "Checkpoint") or "Checkpoint")
        if "Euler manually" not in label:
            checkpoint.label = f"{label} (Ideogram 4: select Euler manually)"
        print(f"{TAG} WARNING: native sampler selection API unavailable; select Euler "
              "manually in txt2img and img2img. API requests must send sampler_name='Euler'.")
        return False
    with demo:
        change(fn=_on_selection, inputs=[checkpoint],
               outputs=[components["txt2img"], components["img2img"]],
               queue=False, show_progress=False)
    demo._i4_sampler_defaults_bound = True
    return True


def install() -> bool:
    """Bind once after Forge has merged its final native Blocks tree."""
    try:
        from modules import script_callbacks
        marker = script_callbacks
        if getattr(marker, "_i4_sampler_defaults_app_started_installed", False):
            return True
        script_callbacks.on_app_started(_bind_on_app_started)
        marker._i4_sampler_defaults_app_started_installed = True
        return True
    except (ImportError, AttributeError) as error:
        print(f"{TAG} WARNING: native sampler defaults unavailable ({error}); select Euler "
              "manually. API requests must send sampler_name='Euler'.")
        return False
