"""Quantized loading for Ideogram 4 single-file checkpoints (official, Apache-2.0, extended).

Supports:
1. NF4  - bitsandbytes pre-quantized state dicts (ideogram-ai/ideogram-4-nf4 HF layout)
2. FP8  - weight-only e4m3 with per-row .weight_scale + .comfy_quant JSON markers
          (the Comfy-Org/Ideogram-4 single-file layout, e.g. ideogram4_Fp8Scaled.safetensors)
3. BF16 - plain state dicts

The FP8 path stores weights as float8 with an Fp8Linear module that dequantizes
per-row at forward time (weight * weight_scale), so it runs on ANY CUDA GPU -
no FP8 tensor-core hardware needed (matches the official Fp8Linear.forward).
"""

from __future__ import annotations

import json
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

_BNB_SIBLING_SUFFIXES = (".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map")

FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"
FP8_TEXT_ENCODER_CONFIG_FLAG = "ideogram_fp8_weight_only"

_COMFY_QUANT_SUFFIX = ".comfy_quant"


def _parse_comfy_quant(tensor: torch.Tensor) -> dict | None:
  """The U8 .comfy_quant tensor is a UTF-8 JSON document, e.g. {"format": "float8_e4m3fn"}."""
  try:
    return json.loads(tensor.numpy().tobytes())
  except Exception:
    return None


def is_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if the checkpoint carries weight-only FP8 Linear weights (official or Comfy layout)."""
  return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or any(
    v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values()
  )


def is_bnb4bit_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  return any(".quant_state.bitsandbytes__" in k for k in state_dict)


def _load_bnb4bit():
  try:
    import bitsandbytes as bnb
    return bnb
  except ImportError as e:
    raise RuntimeError(
      "This checkpoint needs bitsandbytes (NF4), which is not installed in the Forge environment. "
      "Install it into the Forge venv, or use an FP8 checkpoint instead."
    ) from e


def swap_linears_to_bnb4bit(module: nn.Module, compute_dtype: torch.dtype, *, quant_type: str = "nf4", compress_statistics: bool = False, state_dict=None, prefix: str = '') -> None:
  bnb = _load_bnb4bit()
  for name, child in list(module.named_children()):
    child_prefix = f'{prefix}{name}'
    packed = state_dict is None or any(k.startswith(child_prefix + '.weight.quant_state.bitsandbytes__') for k in state_dict)
    if isinstance(child, nn.Linear) and packed:
      new_linear = bnb.nn.Linear4bit(
        child.in_features, child.out_features, bias=child.bias is not None,
        compute_dtype=compute_dtype, compress_statistics=compress_statistics, quant_type=quant_type,
      )
      setattr(module, name, new_linear)
    else:
      swap_linears_to_bnb4bit(child, compute_dtype, quant_type=quant_type, compress_statistics=compress_statistics,
                             state_dict=state_dict, prefix=child_prefix + '.')


def load_bnb4bit_state_dict(model, state_dict, device, dtype) -> None:
  bnb = _load_bnb4bit()
  consumed: set[str] = set()
  for full_name, tensor in state_dict.items():
    if ".quant_state." in full_name or full_name.endswith(_BNB_SIBLING_SUFFIXES):
      continue
    parent_path, _, param_name = full_name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    current = parent._parameters.get(param_name)
    if not isinstance(current, bnb.nn.Params4bit):
      continue
    prefix = full_name + "."
    quantized_stats = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
    consumed.add(full_name)
    consumed.update(quantized_stats.keys())
    parent._parameters[param_name] = bnb.nn.Params4bit.from_prequantized(
      data=tensor, quantized_stats=quantized_stats, requires_grad=False, device=device,
    )
  remaining = {k: v for k, v in state_dict.items() if k not in consumed}
  for k in list(remaining):
    if remaining[k].is_floating_point():
      remaining[k] = remaining[k].to(device=device, dtype=dtype)
    else:
      remaining[k] = remaining[k].to(device=device)
  missing, unexpected = model.load_state_dict(remaining, strict=False)
  real_missing = [m for m in missing if m not in consumed]
  if real_missing:
    raise RuntimeError(f"missing keys after quantized load: {real_missing[:10]}")
  if unexpected:
    raise RuntimeError(f"unexpected keys after quantized load: {unexpected[:10]}")
  for p in model.parameters():
    if isinstance(p, bnb.nn.Params4bit):
      continue
    if p.is_floating_point() and p.dtype != dtype:
      p.data = p.data.to(dtype=dtype)
    if p.device != device:
      p.data = p.data.to(device=device)


def quantize_weight_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  w = weight.detach().to(torch.float32)
  amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
  scale = amax / FP8_E4M3_MAX
  q = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
  return q, scale.squeeze(1).to(torch.float32)


class Fp8Linear(nn.Module):
  """Linear with e4m3 float8 weight + per-row float32 scale.

  Two compute paths, chosen automatically:
  - FAST (CUDA SM >= 9.0, probed once): native fp8 tensor-core matmul via
    torch._scaled_mm - the weight stays fp8 in VRAM, activations are
    quantized per-tensor, per-row weight scales feed the GEMM directly.
    No dequantized weight copy is ever materialized (max speed, max memory
    efficiency). Validated by a numerical probe before enabling.
  - PORTABLE (default fallback): dequantize weight per forward (weight * scale)
    and run a bf16 matmul - runs on ANY device, bit-identical to the official
    implementation.
  """

  weight: torch.Tensor
  weight_scale: torch.Tensor
  bias: torch.Tensor | None

  def __init__(self, in_features: int, out_features: int, bias: bool, compute_dtype: torch.dtype) -> None:
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.compute_dtype = compute_dtype
    self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE))
    self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
    if bias:
      self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
    else:
      self.bias = None
    self._fast: bool | None = None  # None = probe not done for this module

  def _forward_portable(self, x: torch.Tensor) -> torch.Tensor:
    w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
    bias = self.bias.to(x.dtype) if self.bias is not None else None
    return F.linear(x, w, bias)

  def _forward_fast(self, x: torch.Tensor) -> torch.Tensor:
    """TENSOR-WISE fp8 GEMM with an exact per-row correction:
    w_row_i = wq_i * ws_i, so  x @ W.t() = (xq*xs) @ (wq*ws).t()
    Using tensor-wise scale s_b = mean(ws): the GEMM computes xq @ wq.t() * xs * mean(ws);
    we then multiply output column i by (ws_i / mean(ws)) - cheap broadcast, exact math."""
    orig_dtype = x.dtype
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    amax = x2.detach().abs().amax().clamp(min=1e-12)
    xs = (amax / FP8_E4M3_MAX).float().reshape(())
    xq = (x2.float() / xs).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
    out = torch._scaled_mm(
      xq, self.weight.t(),
      scale_a=xs, scale_b=self._ws_tensorwise,
      bias=None, out_dtype=torch.float32, use_fast_accum=True,
    ) * self._ws_correction  # (1, out_features) f32 broadcast
    if self.bias is not None:
      out = out + self.bias.float()
    return out.reshape(*x.shape[:-1], self.out_features).to(orig_dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self._fast is None:
      self._prepare_fast()
    if self._fast:
      try:
        return self._forward_fast(x)
      except Exception:
        self._fast = False  # permanent per-module fallback
    return self._forward_portable(x)

  def _prepare_fast(self):
    """Decide the compute path ONCE per module. Default: the portable dequant
    matmul - measured fastest on this stack (bf16 dequant + cublas GEMM beats
    per-call f32 activation quantize + tensor-wise _scaled_mm at DiT layer
    sizes). The _scaled_mm fast path stays available behind config.json
    ('fp8_fast_gemm': true) for future torch builds where rowwise fp8 lands
    (it will then skip the f32 activation copy entirely)."""
    try:
      import json as _json
      from pathlib import Path as _Path
      cfg = _Path(__file__).resolve().parent.parent / "config.json"
      enabled = False
      try:
        enabled = bool(_json.loads(cfg.read_text(encoding="utf-8")).get("fp8_fast_gemm", False))
      except Exception:
        pass
      if not enabled:
        self._fast = False
        return
      if not _fp8_fast_mm_available(self.weight.device):
        self._fast = False
        return
      ws = self.weight_scale.float()
      ws_mean = ws.mean().reshape(()).to(self.weight.device)
      self._ws_tensorwise = ws_mean
      self._ws_correction = (ws / ws_mean).reshape(1, -1).to(self.weight.device)
      self._fast = True
    except Exception:
      self._fast = False


_FP8_FAST_STATE: bool | None = None  # None = not probed on this process


def _fp8_fast_mm_available(device: torch.device) -> bool:
  """Probe once per process whether TENSOR-WISE torch._scaled_mm works and is
  accurate here (fp8 tensor cores: SM >= 90). Rowwise fp8 is unsupported on
  this torch/driver stack, so the fast path folds per-row scales into a
  tensor-wise scale + exact per-row output correction (see _forward_fast)."""
  global _FP8_FAST_STATE
  # This optional kernel probe is NVIDIA-specific. ROCm uses the portable
  # dequantize + linear route, not NVIDIA SM capability numbers.
  if getattr(torch.version, 'hip', None):
    return False
  try:
    if device.type != "cuda" or not hasattr(torch, "_scaled_mm"):
      return False
    if torch.cuda.get_device_capability(device) < (9, 0):
      return False
    if _FP8_FAST_STATE is not None:
      return _FP8_FAST_STATE
    a = torch.randn(64, 128, device=device, dtype=torch.bfloat16) * 0.5
    w = torch.randn(32, 128, device=device, dtype=torch.bfloat16) * 0.5
    xs = (a.abs().amax().clamp(min=1e-12) / FP8_E4M3_MAX).float().reshape(())
    aq = (a.float() / xs).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
    ws_row = (w.abs().amax(dim=1).clamp(min=1e-12) / FP8_E4M3_MAX).float()
    wq = (w / ws_row.unsqueeze(1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
    ws_mean = ws_row.mean().reshape(())
    out = torch._scaled_mm(aq, wq.t(), scale_a=xs, scale_b=ws_mean,
                           out_dtype=torch.float32, use_fast_accum=True)
    out = out * (ws_row / ws_mean).reshape(1, -1)
    ref = a.float() @ w.float().t()
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    # e4m3 quantization noise floor is ~4-5%; allow it, forbid systematic error
    _FP8_FAST_STATE = bool(err < 0.06)
    print(f"[Invisible-I4] fp8 tensor-core GEMM {'enabled' if _FP8_FAST_STATE else f'disabled (probe rel-err {err:.3f})'}")
  except Exception as e:
    _FP8_FAST_STATE = False
    print(f"[Invisible-I4] fp8 tensor-core GEMM unavailable ({type(e).__name__}); using portable dequant path")
  return _FP8_FAST_STATE


def swap_linears_to_fp8(module: nn.Module, state_dict, compute_dtype: torch.dtype, *, prefix: str = "") -> None:
  """Replace each nn.Linear that has a saved FP8 scale (or comfy_quant marker) with an Fp8Linear."""
  for name, child in list(module.named_children()):
    child_prefix = f"{prefix}{name}"
    has_scale = f"{child_prefix}{FP8_SCALE_SUFFIX}" in state_dict
    has_marker = f"{child_prefix}{_COMFY_QUANT_SUFFIX}" in state_dict
    if isinstance(child, nn.Linear) and (has_scale or has_marker):
      setattr(module, name, Fp8Linear(child.in_features, child.out_features, bias=child.bias is not None, compute_dtype=compute_dtype))
    else:
      swap_linears_to_fp8(child, state_dict, compute_dtype, prefix=f"{child_prefix}.")


def load_fp8_state_dict(model, state_dict, device, dtype, *, assign: bool = False, strict: bool = True, required_prefix: str | None = None) -> None:
  """Load a weight-only FP8 checkpoint (official or Comfy layout) into a swapped model.

  Scale normalization: checkpoints use [out_features] (HF), [out_features, 1]
  (Forge/Comfy exports), or [] (per-tensor) representations. Fp8Linear keeps a
  flat per-row buffer, so normalize these equivalent forms at the load boundary.
  """
  prepared: dict[str, torch.Tensor] = {}
  for k, v in state_dict.items():
    if k.endswith(_COMFY_QUANT_SUFFIX):
      continue  # metadata marker; the presence of weight_scale drives the swap
    if v.dtype == FP8_WEIGHT_DTYPE:
      prepared[k] = v.to(device=device)
    elif k.endswith(FP8_SCALE_SUFFIX):
      v = v.to(device=device, dtype=torch.float32)
      weight_key = k[: -len(FP8_SCALE_SUFFIX)] + ".weight"
      sibling = state_dict.get(weight_key)
      if sibling is not None and sibling.ndim == 2:
        out_features = int(sibling.shape[0])
        if v.numel() == 1:
          v = v.reshape(()).expand(out_features)
        elif v.numel() == out_features:
          v = v.reshape(out_features)
        else:
          raise RuntimeError(
            f"invalid FP8 scale shape for {weight_key}: {tuple(v.shape)}; "
            f"expected scalar or {out_features} row scales"
          )
      elif v.ndim == 0:
        v = v.reshape(1)
      prepared[k] = v.contiguous()
    elif v.is_floating_point():
      prepared[k] = v.to(device=device, dtype=dtype)
    else:
      prepared[k] = v.to(device=device)
  missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
  if required_prefix:
    critical = [k for k in missing if k.startswith(required_prefix)]
    foreign = [k for k in unexpected if k.startswith(required_prefix)]
    if critical or foreign:
      raise RuntimeError(f'Incompatible text encoder weights: missing={critical[:5]}, unexpected={foreign[:5]}')
  if unexpected:
    warnings.warn(f"extra keys ignored after fp8 load: {unexpected[:5]}", stacklevel=2)
  if missing:
    if strict:
      raise RuntimeError(f"missing keys after fp8 load: {missing[:10]}")
    warnings.warn(f"missing keys after fp8 load: {missing[:10]}", stacklevel=2)
  # move to device WITHOUT touching any residual meta tensors (e.g. rotary
  # caches derived lazily): .to() would raise on them. Replace-with-real
  # storage first, then move only what is real.
  with torch.no_grad():
    for name, mod in model.named_modules():
      for pname, p in list(mod._parameters.items()):
        if isinstance(p, torch.Tensor) and p.is_meta:
          mod._parameters[pname] = None  # drop meta placeholder; lazy re-derive
      for bname, b in list(mod._buffers.items()):
        if isinstance(b, torch.Tensor) and b.is_meta:
          mod._buffers[bname] = None
  model.to(device)
