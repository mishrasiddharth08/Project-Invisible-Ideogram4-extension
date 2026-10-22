"""CPU-only checks of the real Ideogram/Forge refiner entry-point contracts."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pi_ideogram_lib.refiner import fixed_step_counts, neutral_options


def functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class RefinerContractTests(unittest.TestCase):
    def test_fixed_steps_do_not_scale_twice(self):
        for strength in (.025, .125, .25, .5, 1.0):
            with self.subTest(strength=strength):
                schedule, active = fixed_step_counts(5, strength)
                self.assertEqual(active, 5)
                self.assertGreaterEqual(schedule, active)
                self.assertLessEqual(active / schedule, strength)
        self.assertEqual(fixed_step_counts(5, 0), (5, 0))

    def test_invalid_strength_is_rejected(self):
        for strength in (-1, 1.1, float('nan'), float('inf'), 5e-324):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                fixed_step_counts(8, strength)

    def test_neutral_profile_is_transient_and_keeps_memory_precision(self):
        original = dict(preset='V4_DEFAULT_20', turbo=True, adapter_choice='hidden',
                        quant_override='nvfp4', memory_mode='sequential', spectrum_enabled=True)
        resolved = neutral_options(original)
        self.assertEqual(original['preset'], 'V4_DEFAULT_20')
        self.assertEqual(resolved['preset'], 'Auto')
        self.assertFalse(resolved['spectrum_enabled'])
        self.assertFalse(resolved['turbo'])
        self.assertEqual(resolved['adapter_choice'], '')
        self.assertEqual((resolved['quant_override'], resolved['memory_mode']), ('nvfp4', 'sequential'))

    def test_refiner_cfg_and_steps_override_ordinary_recipe(self):
        fake = SimpleNamespace(PRESETS={}, DISTILL_PRESETS=(), build_custom_schedule=Mock())
        namespace = {'PRESET_AUTO_LABEL': 'Auto'}
        functions('pi_ideogram_forge/generate.py', ['_resolve_sampling', '_resolve_preset'], namespace)
        p = SimpleNamespace(_pi_refining=True, steps=5, cfg_scale=1)
        with patch.dict(sys.modules, {'pi_ideogram_lib.sampler_configs': fake}):
            steps, guidance, *_ = namespace['_resolve_preset']({'preset': 'V4_DEFAULT_20'}, p)
        self.assertEqual(steps, 5)
        self.assertEqual(guidance, (1.0,) * 5)

    def test_transient_ui_settings_are_not_persisted(self):
        settings = SimpleNamespace(UI_KEYS=('a', 'b'), resolve=Mock(return_value={'preset': 'Auto'}), persist=Mock())
        namespace = {'pi_settings': settings, '_our_alwayson_script': lambda runner: SimpleNamespace(args_from=0, args_to=2)}
        functions('pi_ideogram_forge/run.py', ['_resolve_run_opts'], namespace)
        namespace['_resolve_run_opts'](object(), SimpleNamespace(_pi_refining=True, script_args=(1, 2)))
        settings.persist.assert_not_called()
        namespace['_resolve_run_opts'](object(), SimpleNamespace(_pi_refining=False, script_args=(1, 2)))
        settings.persist.assert_not_called()  # native named request snapshots never rewrite saved options

    def test_refiner_failure_propagates_instead_of_error_image(self):
        namespace = {'TAG': '[test]', '_img2img_generation_kwargs': lambda p: {},
                     '_enforce_loadable_policy': lambda path: None, '_current_ideogram_file': lambda: None,
                     '_resolve_run_opts': lambda runner, p: {}, '_maybe_download_adapters': lambda opts: None,
                     '_resolve_seed': lambda p: None,
                     '_generate_with_oom_recovery': Mock(side_effect=RuntimeError('bad weights'))}
        functions('pi_ideogram_forge/run.py', ['_take_over_locked'], namespace)
        p = SimpleNamespace(_pi_refining=True, width=512, height=512, batch_size=1)
        with self.assertRaisesRegex(RuntimeError, 'bad weights'):
            namespace['_take_over_locked'](None, p)

    def test_refiner_oom_is_one_attempt_without_resize(self):
        generate = Mock(side_effect=RuntimeError('out of memory'))
        namespace = {'_generate_fn': generate, '_ACTIVE_LORAS': [], 'models_dirs': lambda: [],
                     '_get_pipeline': object(), 'lora': object(), 'json_prompt': object()}
        functions('pi_ideogram_forge/run.py', ['_generate_with_oom_recovery'], namespace)
        p = SimpleNamespace(_pi_refining=True, width=1024, height=768, batch_size=1)
        with self.assertRaisesRegex(RuntimeError, 'out of memory'):
            namespace['_generate_with_oom_recovery'](p, {}, orig_w=1024, orig_h=768, orig_bs=1)
        generate.assert_called_once()
        self.assertEqual((p.width, p.height), (1024, 768))


if __name__ == '__main__':
    unittest.main()
