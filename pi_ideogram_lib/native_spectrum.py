"""Adapt packed Ideogram samples to Forge's installed Spectrum hook.

All prediction/caching math lives in extensions-builtin/sd_forge_spectrum.
This module only maps timesteps, batches and CFG branches to its real UnetPatcher
interface. The pipeline supplies denoised endpoints, matching KModel.apply_model.
It does not register a second engine or own model residency.
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import torch


DEFAULTS = dict(w=0.25, m=6, lam=0.5, window_size=2, flex_window=0.0,
                warmup_steps=6, stop_caching_step=0.9)


def _installed_node():
    directory = Path(__file__).resolve().parents[3] / 'extensions-builtin/sd_forge_spectrum'
    expected = (directory / 'lib_spectrum/__init__.py').resolve()
    if not expected.is_file():
        raise RuntimeError('Forge built-in Spectrum is missing. Install a compatible Forge build; no private fallback is used.')
    # Forge normally adds this directory during extension discovery. Standalone
    # component tests use the very same installed package, never a copied module.
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
    try:
        module = importlib.import_module('lib_spectrum')
        from backend.patcher.unet import UnetPatcher
    except Exception as error:
        raise RuntimeError(f'Cannot load Forge built-in Spectrum: {error}') from error
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError('Spectrum import resolved outside Forge extensions-builtin; refusing a replacement implementation.')
    return module.SpectrumNode, UnetPatcher, str(expected)


class ForgeSpectrum:
    def __init__(self, steps, device, options, model=None):
        self.steps = int(steps)
        if self.steps < 1:
            raise ValueError('Spectrum requires at least one sampling step')
        self.options = dict(DEFAULTS)
        if not isinstance(options, dict) or set(options) - DEFAULTS.keys():
            raise ValueError('Unknown native Spectrum options')
        self.options.update(options)
        limits = dict(w=(0, 1), m=(1, 8), lam=(0, 2), window_size=(1, 10),
                      flex_window=(0, 2), warmup_steps=(0, 20), stop_caching_step=(0, 1))
        for key, (low, high) in limits.items():
            value = self.options[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'Invalid native Spectrum option: {key}')
            if key in ('m', 'window_size', 'warmup_steps'):
                if int(value) != value:
                    raise ValueError(f'Native Spectrum {key} must be an integer')
                self.options[key] = int(value)
        self._node, patcher, self.source_path = _installed_node()
        device = torch.device(device)
        # A genuine Forge patcher carries the native function hook; existing
        # Ideogram memory management continues to own the actual weight tensors.
        self._anchor = patcher(model if model is not None else torch.nn.Identity(),
                               load_device=device, offload_device=torch.device('cpu'),
                               current_device=device)
        self._patchers = {}
        self._last_steps = {}
        self.real_calls = 0
        self.forecast_calls = 0
        self.real_items = 0
        self.forecast_items = 0

    def reset(self):
        """Invalidate histories after temporary model/adapter weights change."""
        self._patchers.clear()
        self._last_steps.clear()

    def evaluate(self, branch, x, t, forward, step_index=None):
        if self._anchor is None:
            raise RuntimeError('Native Spectrum session is closed')
        if branch not in (0, 1) or t.shape != (x.shape[0],):
            raise ValueError('Invalid native Spectrum CFG branch or timestep batch')
        if step_index is not None:
            previous = self._last_steps.get(branch)
            if previous is not None and step_index != previous + 1:
                # A CFG=1 interval may omit unconditional evaluations. Do not
                # forecast across that gap as if those model steps occurred.
                self._patchers.pop(branch, None)
            self._last_steps[branch] = step_index
        if branch not in self._patchers:
            self._patchers[branch] = self._node.patch(self._anchor, self.steps, **self.options)
        patcher = self._patchers[branch]
        wrapper = patcher.model_options.get('model_function_wrapper')
        if not callable(wrapper):
            raise RuntimeError('Forge Spectrum did not install its function wrapper')
        actual_items = 0

        def real(x_real, _coordinate, **conditioning):
            nonlocal actual_items
            indices = conditioning['i4_indices']
            result = forward(x_real, conditioning['i4_t'], indices)
            if result.shape != x_real.shape:
                raise ValueError('Ideogram velocity shape does not match native Spectrum input')
            actual_items += x_real.shape[0]
            self.real_calls += 1
            return result

        result = wrapper(real, {
            'input': x,
            # Forge detects a new pass when its coordinate increases; Ideogram
            # integrates t upwards. Preserve original t separately, without a
            # lossy round-trip, and present descending 1-t to the native hook.
            'timestep': 1.0 - t,
            'c': {'transformer_options': {'cond_or_uncond': [branch]},
                  'i4_t': t, 'i4_indices': torch.arange(x.shape[0], device=x.device)},
        })
        self.real_items += actual_items
        self.forecast_items += x.shape[0] - actual_items
        self.forecast_calls += int(actual_items < x.shape[0])
        if not torch.isfinite(result).all():
            raise RuntimeError('Forge built-in Spectrum returned non-finite velocities; disable Spectrum or adjust its native settings.')
        return result

    def close(self):
        self.reset()
        self._anchor = None
