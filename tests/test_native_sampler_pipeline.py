"""Exercise the coordinate bridge independently of host registry imports."""
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from test_native_spectrum_pipeline import tiny_pipeline, sample


def _bridge(monkeypatch, build, run):
    module = ModuleType('pi_ideogram_lib.native_sampling')
    module.build_sigmas = build
    module.run_sampler = run
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.mark.parametrize('second_order', [False, True])
@torch.no_grad()
def test_solver_handoff_deterministic_preview_and_cleanup(monkeypatch, second_order):
    calls = []

    def build(request, official, **kwargs):
        assert bool((official[:-1] >= official[1:]).all())
        assert official[-1] == 0
        return official

    def run(request, model, x, sigmas, callback, seed):
        for i in range(len(sigmas) - 1):
            sigma, next_sigma = sigmas[i:i + 2]
            clean = model(x, sigma.expand(x.shape[0]))
            derivative = (x - clean) / sigma
            proposed = x + derivative * (next_sigma - sigma)
            if second_order and next_sigma > 0:
                clean2 = model(proposed, next_sigma.expand(x.shape[0]))
                derivative2 = (proposed - clean2) / next_sigma
                proposed = x + (derivative + derivative2) * .5 * (next_sigma - sigma)
            callback(dict(i=i, denoised=clean))
            calls.append(i)
            x = proposed
        return x

    _bridge(monkeypatch, build, run)
    pipe = tiny_pipeline()
    try:
        frames, progress, counts, undo = [], [], [], []
        options = dict(sampler_request={}, preview_callback=lambda z: frames.append(z.clone()),
                       step_callback=lambda i, n: progress.append((i, n)),
                       on_actual_steps=counts.append, first_step_undo=lambda: undo.append(True))
        result = sample(pipe, **options)
        repeat = sample(pipe, sampler_request={})
        torch.testing.assert_close(result, repeat, atol=0, rtol=0)
        assert torch.isfinite(result).all()
        assert len(frames) == 16 and len(progress) == 16
        assert counts[-1] == 16 and undo == [True]
        assert not torch.equal(frames[0], frames[-1])
        if not second_order:
            # Native terminal sigma is exactly zero; the official flow recipe
            # stops slightly short of t=1, so it is not pixel-identical.
            torch.testing.assert_close(result, frames[-1], atol=1e-6, rtol=1e-6)
    finally:
        pipe.uncond_adapter.remove()


@torch.no_grad()
def test_img2img_encode_entry_selected_sigmas_and_cancellation(monkeypatch):
    entries = []

    def build(request, official, **kwargs):
        return torch.tensor([request['entry_sigma'], 0.], device=official.device)

    def run(request, model, x, sigmas, callback, seed):
        entries.append(x.clone())
        denoised = model(x, sigmas[0].expand(x.shape[0]))
        callback(dict(i=0, denoised=denoised))
        return denoised

    _bridge(monkeypatch, build, run)
    pipe = tiny_pipeline()
    source = object()
    latent_dim = pipe.conditional_transformer.config.in_channels
    encoded = torch.full((1, 12, latent_dim), .25)
    encoded_bchw = encoded.transpose(1, 2).reshape(1, latent_dim, 2, 6)
    pipe._encode_img2img_image = Mock(return_value=encoded)
    try:
        common = dict(init_image=source, strength=.5)
        sample(pipe, sampler_request={'entry_sigma': 2.0}, **common)
        sample(pipe, sampler_request={'entry_sigma': .5}, **common)
        pipe._encode_img2img_image.assert_any_call(source, height=64, width=192)
        # Same encoded image + seed: changing the selected schedule's entry
        # sigma scales only the injected noise at the solver handoff.
        torch.testing.assert_close(entries[0] - encoded_bchw,
                                   (entries[1] - encoded_bchw) * 4., rtol=0, atol=1e-6)

        phases, counts, undo = [], [], []
        def cancel(index, total):
            assert (index, total) == (0, 1)
            raise InterruptedError('test cancel')
        with pytest.raises(InterruptedError, match='test cancel'):
            sample(pipe, sampler_request={'entry_sigma': 1.0},
                   phase_callback=phases.append, on_actual_steps=counts.append,
                   first_step_undo=lambda: undo.append(True),
                   step_callback=cancel, **common)
        assert phases[-1] == 'Sampling'
        assert counts[-1] == 1
        assert undo == [True]
    finally:
        pipe.uncond_adapter.remove()


@torch.no_grad()
def test_native_spectrum_multistage_reset_and_cancel_cleanup(monkeypatch):
    sessions = []

    class FakeSpectrum:
        def __init__(self, steps, device, options, model=None):
            self.real_calls = self.forecast_calls = 0
            self.real_items = self.forecast_items = 0
            self.resets = 0
            self.closed = 0
            sessions.append(self)

        def evaluate(self, branch, x, t, forward, step_index=None):
            self.real_calls += 1
            self.real_items += x.shape[0]
            return forward(x, t, torch.arange(x.shape[0], device=x.device))

        def reset(self):
            self.resets += 1

        def close(self):
            self.closed += 1

    spectrum = ModuleType('pi_ideogram_lib.native_spectrum')
    spectrum.ForgeSpectrum = FakeSpectrum
    monkeypatch.setitem(sys.modules, spectrum.__name__, spectrum)

    def build(request, official, **kwargs):
        return torch.tensor([2., 1., 0.], device=official.device)

    def run(request, model, x, sigmas, callback, seed):
        # Two evaluations at one solver step emulate predictor/corrector
        # stages. The pipeline must reset Spectrum between trajectories.
        sigma = sigmas[0].expand(x.shape[0])
        model(x, sigma)
        denoised = model(x, sigma)
        callback(dict(i=0, denoised=denoised))
        return denoised

    _bridge(monkeypatch, build, run)
    pipe = tiny_pipeline()
    try:
        def cancel(*_args):
            raise InterruptedError('cancel multistage')
        with pytest.raises(InterruptedError, match='cancel multistage'):
            sample(pipe, options={}, sampler_request={'function_name': 'sample_euler'},
                   step_callback=cancel)
        assert len(sessions) == 1
        assert sessions[0].resets >= 1
        assert sessions[0].closed == 1
        assert pipe._native_spectrum_stats['real_calls'] > 0
    finally:
        pipe.uncond_adapter.remove()


@pytest.mark.parametrize("function_name, forecasts", [("sample_ipndm", True), ("sample_ipndm_v", False)])
@torch.no_grad()
def test_spectrum_ipndm_single_evaluation_reuses_history(monkeypatch, function_name, forecasts):
    def build(request, official, **kwargs):
        return official

    def run(request, model, x, sigmas, callback, seed):
        for i in range(len(sigmas) - 1):
            sigma, next_sigma = sigmas[i:i + 2]
            clean = model(x, sigma.expand(x.shape[0]))
            x = x + ((x - clean) / sigma) * (next_sigma - sigma)
            callback(dict(i=i, denoised=clean))
        return x

    _bridge(monkeypatch, build, run)
    pipe = tiny_pipeline()
    try:
        result = sample(pipe, options={}, sampler_request={"function_name": function_name})
        assert torch.isfinite(result).all()
        assert (pipe._native_spectrum_stats["forecast_calls"] > 0) is forecasts
        if not forecasts:
            assert pipe._native_spectrum_stats["real_calls"] > 0
            assert "exact evaluations" in pipe._native_spectrum_stats["notice"]
    finally:
        pipe.uncond_adapter.remove()
