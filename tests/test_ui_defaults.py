import sys

from pi_ideogram_forge import ui_defaults


def test_install_registers_no_sampler_forcing_callbacks():
    assert ui_defaults.install() is True


def test_compatibility_handlers_do_not_rewrite_user_choices(monkeypatch):
    skip = object()
    fake_gradio = type('Gradio', (), {'skip': staticmethod(lambda: skip)})
    monkeypatch.setitem(sys.modules, 'gradio', fake_gradio)
    assert ui_defaults._selection_values('Ideogram4 - model') == (None, None)
    assert ui_defaults._sampler_value('Ideogram4 - model', 'DPM++ 2M') is None
    assert ui_defaults._on_selection('Ideogram4 - model') == (skip, skip)
    assert ui_defaults._on_sampler_change('Ideogram4 - model', 'DPM++ 2M') is skip
