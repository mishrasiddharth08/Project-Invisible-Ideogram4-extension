"""Build the actual Gradio controls without booting Forge or loading weights."""
# requires: cpu
import ast
import sys
from pathlib import Path
from unittest.mock import patch

EXT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXT))
import gradio as gr
from pi_ideogram_lib import config, settings


def main():
    tree = ast.parse((EXT / 'scripts/engine.py').read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {'_quickstart', '_recipe_line', '_bypass_help', '_bypass_status'}]
    ui = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'ui')
    namespace = dict(Path=Path, pi_settings=settings, _I4_PANELS=[],
                     is_ideogram4_selected=lambda: True, _models_dirs=lambda: [],
                     _picker_cfg=lambda: {}, _picker_quant=lambda cfg: 'fp8_scaled',
                     _selected_checkpoint_name=lambda: '', _resolve_ci_for_name=lambda name: None,
                     _picker_markdown=lambda *args: 'Model files',
                     _on_variant_changed=lambda *args: 'Model files')
    exec(compile(ast.Module(body=functions + [ui], type_ignores=[]), 'ui-under-test', 'exec'), namespace)
    cfg = dict(config.DEFAULTS)
    cfg.update(bypass_lora_name='Gray_000002000.safetensors', bypass_lora_strength=0.0)
    defaults = settings.ui_defaults
    with patch.object(settings, 'ui_defaults', lambda: defaults(cfg)), \
         patch('pi_ideogram_lib.detect.bypass_lora_choices', return_value=['Gray_000002000.safetensors']), \
         patch('pi_ideogram_lib.detect.adapter_choices', return_value=[]):
        with gr.Blocks() as demo:
            controls = namespace['ui'](None, False)
        callbacks = {block.fn.__name__: block.fn for block in demo.fns.values() if block.fn}
        selected, amount = callbacks['_recommended_bypass']()
        assert selected['value'] == 'Gray_000002000.safetensors' and amount['value'] == -0.25
        with patch('pi_ideogram_forge.lora.find_uncond_replacement_lora', return_value=Path('uncond.safetensors')):
            combo = callbacks['_combined_bypass']()
            assert combo[0]['value'] == 'Gray_000002000.safetensors' and combo[2]['value'] == 'Ostris uncond LoRA'
        with patch('pi_ideogram_forge.lora.find_uncond_replacement_lora', return_value=None):
            assert all('value' not in update for update in callbacks['_combined_bypass']())
        refreshed, help_text = callbacks['_refresh_bypasses']('deleted.safetensors')
        assert refreshed['value'] == settings.BYPASS_NONE_LABEL
        assert callbacks['_toggle_strength'](settings.BYPASS_NONE_LABEL)['visible'] is False
    assert len(controls) == len(settings.UI_KEYS)
    assert [c.value for c in controls][5:7] == ['Gray_000002000.safetensors', 0.0]
    assert settings.from_ui([c.value for c in controls], cfg)['bypass_lora_strength'] == 0.0
    tabs = [b.label for b in demo.blocks.values() if isinstance(b, gr.Tab)]
    assert tabs == ['Style', 'Filter bypass', 'Performance', 'Model Setup'], tabs
    # The panel must hand itself to the module-level registry, because that is
    # the only thing install_visibility_hook() has to wire to the checkpoint
    # dropdown. It used to import modules_forge.main_entry.ui_checkpoint right
    # here, which cannot work: Forge builds script UI before it creates that
    # dropdown, so the import raised, a bare except ate it, and the panel never
    # showed or hid on a checkpoint change. Assert the handoff, not the import.
    assert len(namespace['_I4_PANELS']) == 1, namespace['_I4_PANELS']
    panel, detect = namespace['_I4_PANELS'][0]
    assert isinstance(panel, gr.Accordion) and isinstance(detect, gr.Markdown)
    assert 'Off' in namespace['_bypass_status']('Gray_000002000.safetensors', 0.0)
    assert 'off (raw prompt)' in namespace['_bypass_status']('Gray_000002000.safetensors', -0.25, True)
    # Execute the real generation assignment: zero must survive the second read.
    gen = ast.parse((EXT / 'pi_ideogram_forge/generate.py').read_text(encoding='utf-8'))
    assignment = next(n for n in ast.walk(gen) if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.IfExp)
                      and any(isinstance(t, ast.Name) and t.id == 'bypass_lora_strength' for t in n.targets))
    for value, expected in [(0.0, 0.0), (None, -0.25), (-0.5, -0.5)]:
        scope = {'amount': value}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'strength-under-test', 'exec'), scope)
        assert scope['bypass_lora_strength'] == expected
    print('RESULT: 17 passed, 0 failed')


if __name__ == '__main__':
    main()
