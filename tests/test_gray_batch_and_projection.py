# requires: cpu
"""Real tiny transformers: per-image Gray lifecycle and compact projection."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from pi_ideogram_lora.first_step import FirstStepAdapters
from pi_ideogram_lora.runtime_lora import RuntimeLoraAdapter
from pi_ideogram_lib.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer


def model():
    return Ideogram4Transformer(Ideogram4Config(emb_dim=64, num_layers=2, num_heads=8,
        intermediate_size=128, adanln_dim=32, in_channels=8,
        llm_features_dim=32, mrope_section=(2, 1, 1))).eval()


def inputs(batch=1):
    text, image = 5, 12
    return dict(llm_features=torch.randn(batch, text, 32),
        x=torch.randn(batch, text + image, 8), t=torch.full((batch,), 0.5),
        position_ids=torch.zeros(batch, text + image, 3, dtype=torch.long),
        segment_ids=torch.ones(batch, text + image, dtype=torch.long),
        indicator=torch.tensor([[3] * text + [2] * image] * batch))


def state():
    return {'diffusion_model.layers.0.attention.o.lora_A.weight': torch.randn(4, 64) * .1,
            'diffusion_model.layers.0.attention.o.lora_B.weight': torch.randn(64, 4) * .1}


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    @torch.no_grad()
    def test_compact_equals_padded_for_batch_and_padding(self):
        m = model()
        kw = inputs(2)
        kw['indicator'][1, 3:5] = 0
        seen = []
        handle = m.llm_cond_proj.register_forward_pre_hook(lambda mod, args: seen.append(args[0].shape[1]))
        compact = m(**kw)
        padded = m(**dict(kw, llm_features=F.pad(kw['llm_features'], (0, 0, 0, 12))))
        handle.remove()
        torch.testing.assert_close(compact, padded, rtol=1e-5, atol=1e-6)
        self.assertEqual(seen, [5, 17])

    @torch.no_grad()
    def test_compact_with_projection_lora(self):
        m, kw = model(), inputs()
        ad = RuntimeLoraAdapter(m, {'diffusion_model.llm_cond_proj.lora_A.weight': torch.randn(4, 32),
            'diffusion_model.llm_cond_proj.lora_B.weight': torch.randn(64, 4)}, .5)
        self.assertGreater(ad.install(), 0)
        with ad.active():
            compact = m(**kw)
            padded = m(**dict(kw, llm_features=F.pad(kw['llm_features'], (0, 0, 0, 12))))
        ad.remove()
        torch.testing.assert_close(compact, padded, rtol=1e-5, atol=1e-6)

    @torch.no_grad()
    def test_gray_reapplies_each_image_with_uncond_at_one(self):
        m, kw = model(), inputs()
        original = {k: v.clone() for k, v in m.state_dict().items()}
        uncond = RuntimeLoraAdapter(m, state(), 1.0, 'uncond')
        uncond.install()
        baseline = m(**kw)
        gray = FirstStepAdapters([m], state(), -.25)
        for _ in range(3):
            gray.begin()
            first = m(**kw)
            self.assertFalse(torch.allclose(first, baseline))
            with uncond.active():
                combined = m(**kw)
            self.assertFalse(torch.allclose(first, combined))
            self.assertFalse(uncond.enabled)
            gray.finish()
            gray.finish()  # interrupt cleanup after normal step cleanup
            torch.testing.assert_close(m(**kw), baseline, rtol=0, atol=0)
        uncond.remove()
        self.assertTrue(all(not mod._forward_hooks for mod in m.modules()))
        for key, value in m.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)

    def test_partial_install_rolls_back(self):
        m = model()
        gray = FirstStepAdapters([m], state(), -.25)
        original_install = RuntimeLoraAdapter.install
        def failing(adapter):
            original_install(adapter)
            raise RuntimeError('simulated failure after hook registration')
        with patch.object(RuntimeLoraAdapter, 'install', failing):
            with self.assertRaisesRegex(ValueError, 'simulated failure'):
                gray.begin()
        self.assertFalse(gray.adapters)
        self.assertTrue(all(not mod._forward_hooks for mod in m.modules()))

    def test_zero_matches_refused(self):
        gray = FirstStepAdapters([model()], {}, -.25)
        with self.assertRaisesRegex(ValueError, 'matched no modules'):
            gray.begin()
        self.assertFalse(gray.adapters)

    def test_zero_strength_has_no_hooks(self):
        m = model()
        gray = FirstStepAdapters([m], state(), 0)
        gray.begin()
        self.assertTrue(all(not mod._forward_hooks for mod in m.modules()))

    @torch.no_grad()
    def test_projection_runs_once_across_timesteps(self):
        m, kw, cache = model(), inputs(), {}
        calls = []
        handle = m.llm_cond_proj.register_forward_pre_hook(lambda mod, args: calls.append(1))
        first = m(**kw, text_cache=cache)
        second_kw = dict(kw, t=torch.tensor([.3]))
        second = m(**second_kw, text_cache=cache)
        self.assertEqual(len(calls), 1)
        reference = m(**second_kw)
        handle.remove()
        torch.testing.assert_close(second, reference, rtol=0, atol=0)

    @torch.no_grad()
    def test_projection_cache_reset_after_gray_removal(self):
        m, kw, cache = model(), inputs(), {}
        sd = {'diffusion_model.llm_cond_proj.lora_A.weight': torch.randn(4, 32),
              'diffusion_model.llm_cond_proj.lora_B.weight': torch.randn(64, 4)}
        gray = FirstStepAdapters([m], sd, -.25)
        baseline = m(**kw)
        gray.begin()
        first = m(**kw, text_cache=cache)
        self.assertFalse(torch.allclose(first, baseline))
        gray.finish()
        cache.clear()  # same boundary as pipeline first_step_undo
        torch.testing.assert_close(m(**kw, text_cache=cache), baseline, rtol=0, atol=0)

    @torch.no_grad()
    def test_new_features_invalidate_projection(self):
        m, kw, cache = model(), inputs(), {}
        m(**kw, text_cache=cache)
        kw['llm_features'] = torch.randn_like(kw['llm_features'])
        torch.testing.assert_close(m(**kw, text_cache=cache), m(**kw), rtol=0, atol=0)

    @torch.no_grad()
    def test_real_sampling_loop_matches_uncached_reference(self):
        from pi_ideogram_lib.pipeline import Ideogram4Pipeline
        pipe = Ideogram4Pipeline(cond_path='unused', uncond_path=None,
            te_path='unused', vae_path='unused', device='cpu', dtype=torch.float32)
        pipe.conditional_transformer = model()
        pipe._verify_prompts = Mock()
        pipe.memory = Mock(stream_weights=False)
        pipe.memory.vram_gb.return_value = 32
        pipe._decode = lambda z, **kw: [z.clone()]
        kw = inputs()
        packed = {k: kw[k] for k in ('indicator', 'position_ids', 'segment_ids')}
        packed.update(max_text_tokens=5, num_image_tokens=12, grid_h=2, grid_w=6)
        pipe._encode_cache[(('scene',), 64, 192)] = (packed, kw['llm_features'])
        pipe.uncond_adapter = RuntimeLoraAdapter(pipe.conditional_transformer, state(), 1.0)
        pipe.uncond_adapter.install()
        sd = {'diffusion_model.llm_cond_proj.lora_A.weight': torch.randn(4, 32),
              'diffusion_model.llm_cond_proj.lora_B.weight': torch.randn(64, 4)}
        gray = FirstStepAdapters([pipe.conditional_transformer], sd, -.25)
        def sample():
            gray.begin()
            try:
                return pipe('scene', height=64, width=192, num_steps=8, seed=19,
                            first_step_undo=gray.finish)[0]
            finally:
                gray.finish()
        result = sample()
        forward = pipe.conditional_transformer.forward
        def reference(**kwargs):
            kwargs.pop('text_cache', None)
            if kwargs['llm_features'] is not None:
                kwargs['llm_features'] = F.pad(kwargs['llm_features'], (0, 0, 0, 12))
            return forward(**kwargs)
        with patch.object(pipe.conditional_transformer, 'forward', reference):
            expected = sample()
        pipe.uncond_adapter.remove()
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    failed = len(result.failures) + len(result.errors)
    print(f'RESULT: {result.testsRun - failed} passed, {failed} failed')
    raise SystemExit(not result.wasSuccessful())
