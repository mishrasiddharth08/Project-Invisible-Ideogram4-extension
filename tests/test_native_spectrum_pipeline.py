# requires: cpu
"""Exercise the real Ideogram loop and installed Forge Spectrum with tiny weights."""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FORGE = ROOT.parent.parent
for path in (ROOT, FORGE, FORGE / 'modules_forge/packages'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_gray_batch_and_projection import model, inputs, state
from pi_ideogram_lib.pipeline import Ideogram4Pipeline
from pi_ideogram_lora.runtime_lora import RuntimeLoraAdapter
from pi_ideogram_lora.first_step import FirstStepAdapters


def tiny_pipeline(device='cpu'):
    torch.manual_seed(23)
    torch.set_num_threads(2)
    pipe = Ideogram4Pipeline(cond_path='unused', uncond_path=None,
        te_path='unused', vae_path='unused', device=device, dtype=torch.float32)
    pipe.conditional_transformer = model().to(device)
    pipe._verify_prompts = Mock()
    pipe.memory = Mock(stream_weights=False)
    pipe.memory.vram_gb.return_value = 32
    pipe._decode = lambda z, **kw: [z.detach().clone()]
    kw = {key: value.to(device) for key, value in inputs().items()}
    packed = {key: kw[key] for key in ('indicator', 'position_ids', 'segment_ids')}
    packed.update(max_text_tokens=5, num_image_tokens=12, grid_h=2, grid_w=6)
    pipe._encode_cache[(('scene',), 64, 192)] = (packed, kw['llm_features'])
    pipe.uncond_adapter = RuntimeLoraAdapter(pipe.conditional_transformer, state(), 1.)
    pipe.uncond_adapter.install()
    return pipe


def sample(pipe, options=None, **kwargs):
    return pipe('scene', height=64, width=192, num_steps=16, seed=19,
                native_spectrum=options, **kwargs)[0]


@torch.no_grad()
def test_clean_previews_change_without_changing_final_latents():
    pipe = tiny_pipeline()
    try:
        baseline = sample(pipe)
        frames = []
        result = sample(pipe, preview_callback=lambda z: frames.append(z.clone()))
        torch.testing.assert_close(result, baseline, rtol=0, atol=0)
        assert len(frames) == 16
        assert all(torch.isfinite(frame).all() for frame in frames)
        assert not torch.equal(frames[0], frames[-1])
        z, v = torch.ones(2), torch.full((2,), 4.)
        clean = pipe._preview_endpoint(z, v, .25)
        assert torch.equal(clean, torch.full((2,), 4.))
        assert torch.equal(z, torch.ones(2))
        assert torch.equal(v, torch.full((2,), 4.))
    finally:
        pipe.uncond_adapter.remove()


def test_installed_native_rgb_preview_unpacks_without_vae_or_mutation(monkeypatch):
    monkeypatch.setattr(sys, 'argv', [sys.argv[0]])  # Forge parses its own CLI on first import
    from pi_ideogram_lib.native_preview import decode_native_preview, _native_factors
    factors, bias = _native_factors()
    assert factors.shape == (3, 32)
    z = torch.zeros(1, 6, 128)
    z[..., 0] = 1  # first spatial pixel's first channel in each packed token
    original = z.clone()
    image = decode_native_preview(z, (2, 3), preview_pixels=6)
    assert image.size == (6, 4)
    expected = ((factors[:, 0] + bias) * .5 + .5).clamp(0, 1).mul(255).round().to(torch.uint8)
    assert image.getpixel((0, 0)) == tuple(expected.tolist())
    assert image.getpixel((1, 0)) != image.getpixel((0, 0))
    assert torch.equal(z, original)
    pipe = tiny_pipeline()
    try:
        pipe.autoencoder = object()
        pipe._preview_grid = (2, 3)
        pipe.memory.reset_mock()
        assert pipe.decode_preview(z).size == (256, 171)
        pipe.memory.ensure_vae_on_gpu.assert_not_called()
        pipe.memory.offload_vae.assert_not_called()
    finally:
        pipe.uncond_adapter.remove()


@torch.no_grad()
def test_native_window_one_matches_spectrum_off_and_does_not_leak():
    pipe = tiny_pipeline()
    try:
        off = sample(pipe)
        exact = sample(pipe, dict(window_size=1, m=1, warmup_steps=0))
        torch.testing.assert_close(exact, off, rtol=0, atol=0)
        assert pipe._native_spectrum_stats['real_calls'] == 32
        assert pipe._native_spectrum_stats['forecast_calls'] == 0
        torch.testing.assert_close(sample(pipe), off, rtol=0, atol=0)
        assert pipe._native_spectrum_stats is None
    finally:
        pipe.uncond_adapter.remove()


@torch.no_grad()
def test_native_forecasting_runs_in_real_sampling_and_resets_per_image():
    pipe = tiny_pipeline()
    options = dict(m=1, w=0., warmup_steps=2, window_size=2, stop_caching_step=.875)
    try:
        first = sample(pipe, options)
        stats = pipe._native_spectrum_stats.copy()
        assert stats['forecast_calls'] > 0 and stats['real_calls'] < 32
        assert torch.isfinite(first).all()
        second = sample(pipe, options)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert pipe._native_spectrum_stats == stats
    finally:
        pipe.uncond_adapter.remove()


@torch.no_grad()
def test_native_gray_removal_and_interruption_leave_next_request_clean():
    pipe = tiny_pipeline()
    gray = FirstStepAdapters([pipe.conditional_transformer], state(), -.25)
    options = dict(m=1, w=0., warmup_steps=0, window_size=2)
    try:
        baseline = sample(pipe)
        gray.begin()
        try:
            def interrupt(i, total):
                if i == 4:
                    raise InterruptedError('test cancel')
            with pytest.raises(InterruptedError):
                sample(pipe, options, first_step_undo=gray.finish, step_callback=interrupt)
        finally:
            gray.finish()
        assert pipe._native_spectrum_stats['real_calls'] > 0
        assert not pipe.uncond_adapter.enabled
        torch.testing.assert_close(sample(pipe), baseline, rtol=0, atol=0)
    finally:
        pipe.uncond_adapter.remove()
