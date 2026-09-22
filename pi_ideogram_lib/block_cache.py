"""Opt-in First Block Cache, scoped to one Ideogram generation.

Independent implementation of residual-change gating described by Comfy-WaveSpeed.
Approximate, never stacked with Spectrum. Full warmup/tail and at most one
consecutive reused evaluation; both guidance branches own separate histories.
"""
from contextvars import ContextVar
from functools import wraps
import math
import torch

_session = ContextVar('ideogram_first_block_cache', default=None)


def generation_cache(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        threshold = float(kwargs.get('first_block_cache', 0.0))
        if not math.isfinite(threshold) or not 0 <= threshold <= .2:
            raise ValueError('First Block Cache threshold must be between 0 and 0.2')
        spectrum = kwargs.get('spectrum')
        if threshold and getattr(spectrum, 'enabled', False):
            raise ValueError('Choose Spectrum OR First Block Cache, not both')
        state = dict(threshold=threshold, step=0, total=0, histories={}, hits=0, calls=0) if threshold else None
        token = _session.set(state)
        try:
            return fn(*args, **kwargs)
        finally:
            _session.reset(token)
            if state is not None:
                print(f"[Invisible-I4] First Block Cache: {state['hits']}/{state['calls']} branch evaluations reused (approximate)")
                state['histories'].clear()
    return wrapped


def set_step(step, total):
    state = _session.get()
    if state is not None:
        state['step'], state['total'] = step, total


def run_blocks(model, h, *, branch, attn_mask, cos, sin, adaln_input):
    state = _session.get()
    call = dict(attn_mask=attn_mask, cos=cos, sin=sin, adaln_input=adaln_input)
    if state is None or len(model.layers) < 2:
        for layer in model.layers:
            h = layer(h, **call)
        return h
    state['calls'] += 1
    key = (id(model), branch)
    step, total = state['step'], state['total']
    first = model.layers[0](h, **call)
    residual = first - h
    previous = state['histories'].get(key)
    eligible = 4 <= step < total - 2 and previous is not None
    if eligible:
        anchor, remainder, last_real, reused = previous
        if not reused and step == last_real + 1 and anchor.shape == residual.shape and anchor.device == residual.device and anchor.dtype == residual.dtype:
            # Float reduction, one scalar transfer; no giant float cache copy.
            diff = (residual - anchor).abs().mean(dtype=torch.float32)
            scale = anchor.abs().mean(dtype=torch.float32).clamp_min(1e-6)
            relative = float(diff / scale)
            if math.isfinite(relative) and relative < state['threshold']:
                state['histories'][key] = (anchor, remainder, last_real, True)
                state['hits'] += 1
                return first + remainder
    out = first
    for layer in model.layers[1:]:
        out = layer(out, **call)
    # Detach and bound ownership to two tensors per branch; never cache a
    # predicted residual, and never carry history to another image or seed.
    state['histories'][key] = (residual.detach(), (out - first).detach(), step, False)
    return out
