"""Repair for non-persistent buffers lost by the meta-device fast load path.

WHY THIS EXISTS
---------------
Both big models here are constructed under `torch.device("meta")` and then
given real storage with `Module.to_empty(device=...)`. That is a large speed
win (torch never randomly initializes ~27 GB of weights just to overwrite
them), but `to_empty()` allocates *uninitialized* memory for EVERY tensor,
including buffers that are **not** in any checkpoint:

  - Ideogram4MRoPE.inv_freq            (our DiT, persistent=False)
  - Qwen3VL*RotaryEmbedding.inv_freq   (transformers, persistent=False)

`load_state_dict` never touches a non-persistent buffer, so without this
module both models run with garbage rotary frequencies - i.e. no usable
positional information at all, in the text encoder AND in the transformer.

Call `reinit_rope_buffers(model)` immediately after `to_empty()` (and again
after loading, which is free - the function is idempotent).
"""

from __future__ import annotations

import torch
import torch.nn as nn

TAG = "[Invisible-I4]"


def _real_device(t: torch.Tensor | None) -> torch.device:
  if t is None or t.is_meta:
    return torch.device("cpu")
  return t.device


def _reinit_plain_rope(mod: nn.Module, buffers: dict, label: str) -> bool:
  """Rebuild a textbook RoPE `inv_freq` for modules that keep no re-init hook.

  Some rotary modules (notably transformers' Qwen3VLVisionRotaryEmbedding)
  compute

      inv_freq = 1 / (theta ** (arange(0, dim, 2) / dim))

  in __init__ and keep NEITHER `dim` nor `theta` as attributes, so there is
  nothing to call again. Both are recoverable: the buffer's own length gives
  dim = 2 * numel, and theta comes from the module/config when exposed,
  otherwise the 10000.0 that every implementation of this form defaults to.

  Returns True when the buffer was rebuilt. The result is only accepted if it
  starts at 1.0, which every RoPE of this shape does - a cheap guard against
  silently writing a differently-parameterised curve.
  """
  buf = buffers.get("inv_freq")
  if not isinstance(buf, torch.Tensor) or buf.ndim != 1 or buf.numel() < 2:
    print(f"{TAG} WARNING: rotary module {label} has no known re-init hook; "
          "its inv_freq may be invalid")
    return False

  theta = None
  for attr in ("theta", "rope_theta", "base"):
    val = getattr(mod, attr, None)
    if isinstance(val, (int, float)):
      theta = float(val)
      break
  if theta is None:
    theta = 10000.0

  dim = 2 * buf.numel()
  device = _real_device(buf)
  exponent = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
  inv_freq = 1.0 / (theta ** exponent)
  if inv_freq.numel() != buf.numel() or abs(float(inv_freq[0]) - 1.0) > 1e-6:
    print(f"{TAG} WARNING: could not reconstruct inv_freq for {label}; "
          "leaving it as-is")
    return False
  mod.register_buffer("inv_freq", inv_freq.to(buf.dtype), persistent=False)
  return True


def reinit_rope_buffers(model: nn.Module, *, verbose: bool = False) -> list[str]:
  """Re-derive every rotary `inv_freq` buffer that meta-loading left invalid.

  Handles three shapes:
    1. modules exposing `reset_parameters()` alongside an `inv_freq` buffer
       (our Ideogram4MRoPE),
    2. transformers rotary embeddings, which keep `rope_init_fn(config, device)`,
    3. plain RoPE modules that keep no hook at all and must be reconstructed
       from the buffer itself (Qwen3VLVisionRotaryEmbedding) - see
       _reinit_plain_rope.

  Returns the list of module paths that were repaired. Never raises: a module
  it cannot understand is skipped and reported, because a partial repair is
  still strictly better than none.
  """
  fixed: list[str] = []
  for name, mod in model.named_modules():
    buffers = getattr(mod, "_buffers", None)
    if not isinstance(buffers, dict) or "inv_freq" not in buffers:
      continue
    label = name or "<root>"

    reset = getattr(mod, "reset_parameters", None)
    if callable(reset):
      try:
        reset()
        fixed.append(label)
        continue
      except Exception as e:
        print(f"{TAG} WARNING: could not reset rotary buffer on {label}: {e}")
        continue

    init_fn = getattr(mod, "rope_init_fn", None)
    config = getattr(mod, "config", None)
    if not callable(init_fn) or config is None:
      if _reinit_plain_rope(mod, buffers, label):
        fixed.append(label)
      continue
    try:
      device = _real_device(buffers.get("inv_freq"))
      inv_freq, attention_scaling = init_fn(config, device)
      mod.register_buffer("inv_freq", inv_freq, persistent=False)
      mod.attention_scaling = attention_scaling
      if hasattr(mod, "original_inv_freq"):
        mod.original_inv_freq = mod.inv_freq
      fixed.append(label)
    except Exception as e:
      print(f"{TAG} WARNING: could not re-derive rotary buffer on {label}: {e}")

  if verbose and fixed:
    print(f"{TAG} rotary buffers re-derived after meta load: {len(fixed)} module(s)")
  return fixed


def assert_rope_buffers_valid(model: nn.Module, *, what: str) -> None:
  """Fail loudly if any rotary buffer still looks like uninitialized memory.

  Cheap (a handful of tiny tensors) and worth it: a silently garbage inv_freq
  produces plausible-looking-but-wrong images, which is the worst failure mode
  there is.
  """
  for name, mod in model.named_modules():
    buffers = getattr(mod, "_buffers", None)
    if not isinstance(buffers, dict):
      continue
    buf = buffers.get("inv_freq")
    if not isinstance(buf, torch.Tensor) or buf.is_meta:
      continue
    b = buf.detach().float()
    if not torch.isfinite(b).all() or (b <= 0).any() or float(b.max()) > 1.5:
      raise RuntimeError(
        f"{TAG} {what}: rotary inv_freq on '{name or '<root>'}' is invalid "
        f"(min={float(b.min()):.3e}, max={float(b.max()):.3e}). This means the "
        "meta-device load path left an uninitialized buffer behind."
      )
