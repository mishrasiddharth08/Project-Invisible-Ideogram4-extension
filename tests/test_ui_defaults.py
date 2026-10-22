import os
from types import SimpleNamespace

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr

from pi_ideogram_forge import legacy_preset, ui_defaults


def test_selection_values_only_force_ideogram():
    yes = lambda name: name == "Ideogram4 - model"
    assert ui_defaults._selection_values("Ideogram4 - model", yes) == ("Euler", "Euler")
    assert ui_defaults._selection_values("SDXL", yes) == (None, None)


def _build_real_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            checkpoint = gr.Dropdown(elem_id="setting_sd_model_checkpoint")
        with gr.Column():
            txt2img = gr.Dropdown(elem_id="txt2img_sampling")
            img2img = gr.Dropdown(elem_id="img2img_sampling")
        gr.Dropdown(elem_id="setting_sd_model_checkpoint", render=False)
    return demo, checkpoint, txt2img, img2img


def test_real_gradio_final_registry_excludes_update_clone_and_binds_outside_context():
    demo, checkpoint, txt2img, img2img = _build_real_ui()
    assert ui_defaults._find_native_components(demo) == {
        "checkpoint": checkpoint, "txt2img": txt2img, "img2img": img2img}
    assert len(demo.fns) == 0
    assert ui_defaults._bind_on_app_started(demo) is True
    assert len(demo.fns) == 1
    assert ui_defaults._bind_on_app_started(demo) is True
    assert len(demo.fns) == 1
def test_stale_preset_sampler_values_are_replaced():
    opts = SimpleNamespace(data={"ideogram4_t2i_sampler": "DPM++ 2M",
                                 "ideogram4_i2i_sampler": "DDIM"})
    legacy_preset._enforce_sampler_defaults(opts)
    assert opts.data["ideogram4_t2i_sampler"] == "Euler"
    assert opts.data["ideogram4_i2i_sampler"] == "Euler"


def test_unsupported_native_event_api_gets_visible_manual_notice():
    demo, checkpoint, *_ = _build_real_ui()
    checkpoint.change = None
    assert ui_defaults._bind_on_app_started(demo) is False
    assert "select Euler manually" in checkpoint.label
