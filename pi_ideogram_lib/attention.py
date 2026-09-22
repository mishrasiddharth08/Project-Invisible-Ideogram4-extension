"""Engine-local bridge to Forge's already-selected attention implementation.

Never imports/boots Forge, monkey-patches another model, or installs kernels.
Masked batches use PyTorch directly so padding/segment semantics stay exact.
"""
from __future__ import annotations
import inspect
import sys
import torch
import torch.nn.functional as F

_checked = {}
_warned = set()


def _compatible(fn):
    if fn not in _checked:
        try:
            params = inspect.signature(fn).parameters
            _checked[fn] = all(k in params for k in ('heads', 'mask', 'skip_reshape', 'skip_output_reshape'))
        except (TypeError, ValueError):
            _checked[fn] = False
    return _checked[fn]


def selected_backend():
    host = sys.modules.get('backend.attention')
    return getattr(host, 'attention_function', None)


def attention_status():
    fn = selected_backend()
    if not callable(fn) or not _compatible(fn):
        return 'PyTorch SDPA (portable fallback)'
    name = getattr(fn, '__name__', type(fn).__name__)
    if 'sage' in name:
        return f'Forge {name}; this Forge build may fall back to SDPA for Ideogram\'s 256-wide heads'
    return f'Forge {name}; masked batches use PyTorch SDPA'


def ideogram_attention(q, k, v, attn_mask=None):
    fn = selected_backend()
    if attn_mask is None and callable(fn) and _compatible(fn):
        try:
            out = fn(q, k, v, heads=q.shape[1], mask=None,
                     skip_reshape=True, skip_output_reshape=True)
            if out.shape != q.shape or out.device != q.device or out.dtype != q.dtype:
                raise ValueError('host returned an incompatible attention tensor')
            return out
        except torch.cuda.OutOfMemoryError:
            raise
        except (TypeError, ValueError, NotImplementedError) as error:
            if fn not in _warned:
                print(f'[Invisible-I4] Forge attention incompatible ({error}); using PyTorch SDPA')
                _warned.add(fn)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
