"""Compatibility shim: native sampler and scheduler choices stay user-owned."""
from __future__ import annotations


def _selection_values(_checkpoint_name, detector=None):
    """Legacy seam retained without rewriting either native sampler control."""
    return None, None


def _sampler_value(_checkpoint_name, _sampler_name, detector=None):
    """Legacy seam retained without coercing the selected sampler."""
    return None


def _on_selection(_checkpoint_name):
    import gradio as gr
    return gr.skip(), gr.skip()


def _on_sampler_change(_checkpoint_name, _sampler_name):
    import gradio as gr
    return gr.skip()


def _bind_on_app_started(_demo, _app=None) -> bool:
    """No-op for compatibility with already-imported callers."""
    return True


def install() -> bool:
    """No callback registration: Forge's native selections pass through unchanged."""
    return True
