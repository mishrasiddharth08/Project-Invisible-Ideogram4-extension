"""Ideogram 4 transformer backbone (official implementation, Apache-2.0).

Modified by Project Invisible contributors (2026): scoped host attention,
optional generation-local caching and an explicit unconditional fast path.

9.3B single-stream DiT: text tokens (Qwen3-VL features) and image latent tokens
share one 34-layer transformer with 3D MRoPE, QK-RMSNorm, SwiGLU MLP, AdaLN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import ideogram_attention
from .block_cache import run_blocks

from .constants import (
  LLM_TOKEN_INDICATOR,
  OUTPUT_IMAGE_INDICATOR,
  QWEN3_VL_ACTIVATION_LAYERS,
)


@dataclass
class Ideogram4Config:
  emb_dim: int = 4608
  num_layers: int = 34
  num_heads: int = 18
  intermediate_size: int = 12288
  adanln_dim: int = 512
  in_channels: int = 128
  llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)
  rope_theta: int = 5_000_000
  mrope_section: tuple[int, ...] = (24, 20, 20)
  norm_eps: float = 1e-5


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
  half = x.shape[-1] // 2
  x1 = x[..., :half]
  x2 = x[..., half:]
  return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
  cos = cos.unsqueeze(1)
  sin = sin.unsqueeze(1)
  q_embed = (q * cos) + (_rotate_half(q) * sin)
  k_embed = (k * cos) + (_rotate_half(k) * sin)
  return q_embed, k_embed


class Ideogram4MRoPE(nn.Module):
  inv_freq: torch.Tensor

  def __init__(self, head_dim: int, base: int, mrope_section: tuple[int, ...]) -> None:
    super().__init__()
    self.mrope_section = tuple(mrope_section)
    self.head_dim = head_dim
    self.base = base
    self.register_buffer("inv_freq", self._compute_inv_freq(torch.device("cpu")), persistent=False)

  def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
    exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim
    return 1.0 / (self.base ** exponent)

  def reset_parameters(self) -> None:
    """Re-derive inv_freq in place.

    REQUIRED after a meta-device construction: inv_freq is a NON-PERSISTENT
    buffer (it is not in any checkpoint), so `to_empty()` leaves it holding
    uninitialized memory and `load_state_dict` never fills it. Callers that
    build this module under `torch.device("meta")` MUST call this (or
    pi_ideogram_lib.meta_init.reinit_rope_buffers) once storage is real,
    otherwise every rotary position embedding is garbage.
    """
    buf = self._buffers.get("inv_freq")
    device = torch.device("cpu") if buf is None or buf.is_meta else buf.device
    self.register_buffer("inv_freq", self._compute_inv_freq(device), persistent=False)

  @torch.no_grad()
  def forward(self, position_ids: torch.Tensor):
    assert position_ids.ndim == 3 and position_ids.shape[-1] == 3
    batch_size, seq_len, _ = position_ids.shape
    pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
    inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(3, batch_size, -1, 1)
    freqs = inv_freq @ pos.unsqueeze(2)
    freqs = freqs.transpose(2, 3)
    freqs_t = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
      length = self.mrope_section[axis] * 3
      idx = torch.arange(offset, length, 3, device=freqs_t.device)
      freqs_t[..., idx] = freqs[axis][..., idx]
    emb = torch.cat((freqs_t, freqs_t), dim=-1)
    return emb.cos(), emb.sin()


class Ideogram4RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class Ideogram4Attention(nn.Module):
  def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
    super().__init__()
    assert hidden_size % num_heads == 0
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
    self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
    self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
    self.o = nn.Linear(hidden_size, hidden_size, bias=False)

  def forward(self, x, attn_mask, cos, sin):
    batch_size, seq_len, _ = x.shape
    qkv = self.qkv(x)
    qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
    q, k, v = qkv.unbind(dim=2)

    q = self.norm_q(q)
    k = self.norm_k(k)

    # SDPA expects (B, num_heads, L, head_dim).
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q, k = _apply_rotary_pos_emb(q, k, cos, sin)

    # `attn_mask` is built ONCE per forward by the top-level module (None when
    # every token is in one segment). It used to be rebuilt here, in every
    # layer: 34 layers x 2 models x 20 steps = 1360 constructions of an
    # (1, 1, L, L) bool tensor - 27 MB each at L=5200 - for a value that never
    # changes during a generation.
    out = ideogram_attention(q, k, v, attn_mask=attn_mask)
    out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
    return self.o(out)


def _segment_attn_mask(segment_ids: torch.Tensor) -> torch.Tensor | None:
  """Block-diagonal attention mask from segment ids, or None when it is a no-op.

  segment_ids marks LEFT PADDING (batched prompts are padded to the longest
  one). With a single prompt there is no padding, so every token shares one
  segment and the mask is entirely True - it forbids nothing.

  That matters for speed, not just tidiness: passing ANY attn_mask to
  scaled_dot_product_attention rules out the FlashAttention kernel, so an
  all-True mask bought nothing and cost the fused path. Measured at the real
  shape (1 x 18 heads x 5200 x 256, bf16) on an RTX 5090:

      all-True mask   10.04 ms
      attn_mask=None   7.42 ms      -> 1.35x

  Returning None whenever the mask is vacuous is exact, not an approximation:
  attending everywhere is what an all-True mask means.
  """
  if segment_ids is None:
    return None
  first = segment_ids[:, :1]
  if not bool((segment_ids != first).any()):
    return None  # one segment: nothing is masked out
  return (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)


class Ideogram4MLP(nn.Module):
  def __init__(self, dim: int, hidden_dim: int) -> None:
    super().__init__()
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
  def __init__(self, hidden_size, intermediate_size, num_heads, norm_eps, adanln_dim) -> None:
    super().__init__()
    self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
    self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)
    self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
    self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
    self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
    self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
    self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)

  def forward(self, x, attn_mask, cos, sin, adaln_input):
    mod = self.adaln_modulation(adaln_input)
    scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
    gate_msa = torch.tanh(gate_msa)
    gate_mlp = torch.tanh(gate_mlp)
    scale_msa = 1.0 + scale_msa
    scale_mlp = 1.0 + scale_mlp
    attn_out = self.attention(self.attention_norm1(x) * scale_msa, attn_mask=attn_mask, cos=cos, sin=sin)
    x = x + gate_msa * self.attention_norm2(attn_out)
    x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
    return x


def _sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1e4) -> torch.Tensor:
  t = t.to(torch.float32)
  half = dim // 2
  freq = math.log(scale) / (half - 1)
  freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
  emb = t.unsqueeze(-1) * freq
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
  if dim % 2 == 1:
    emb = F.pad(emb, (0, 1))
  return emb


class Ideogram4EmbedScalar(nn.Module):
  def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
    super().__init__()
    self.dim = dim
    self.range_min, self.range_max = input_range
    assert self.range_max > self.range_min
    self.mlp_in = nn.Linear(dim, dim, bias=True)
    self.mlp_out = nn.Linear(dim, dim, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
    emb = _sinusoidal_embedding(scaled, self.dim)
    emb = emb.to(getattr(self.mlp_in, "compute_dtype", None) or self.mlp_in.weight.dtype)
    emb = F.silu(self.mlp_in(emb))
    return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
  def __init__(self, hidden_size: int, out_channels: int, adanln_dim: int) -> None:
    super().__init__()
    self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
    self.linear = nn.Linear(hidden_size, out_channels, bias=True)
    self.adaln_modulation = nn.Linear(adanln_dim, hidden_size, bias=True)

  def forward(self, x, c):
    scale = 1.0 + self.adaln_modulation(F.silu(c))
    return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer(nn.Module):
  """Ideogram 4 flow-matching transformer."""

  def __init__(self, config: Ideogram4Config) -> None:
    super().__init__()
    self.config = config
    head_dim = config.emb_dim // config.num_heads
    self.input_proj = nn.Linear(config.in_channels, config.emb_dim, bias=True)
    self.llm_cond_norm = Ideogram4RMSNorm(config.llm_features_dim, eps=1e-6)
    self.llm_cond_proj = nn.Linear(config.llm_features_dim, config.emb_dim, bias=True)
    self.t_embedding = Ideogram4EmbedScalar(config.emb_dim, input_range=(0.0, 1.0))
    self.adaln_proj = nn.Linear(config.emb_dim, config.adanln_dim, bias=True)
    self.embed_image_indicator = nn.Embedding(2, config.emb_dim)
    self.rotary_emb = Ideogram4MRoPE(head_dim=head_dim, base=config.rope_theta, mrope_section=config.mrope_section)
    self.layers = nn.ModuleList(
      [Ideogram4TransformerBlock(config.emb_dim, config.intermediate_size, config.num_heads, config.norm_eps, config.adanln_dim) for _ in range(config.num_layers)]
    )
    self.final_layer = Ideogram4FinalLayer(config.emb_dim, config.in_channels, config.adanln_dim)

  @property
  def device(self) -> torch.device:
    return next(self.parameters()).device

  def forward(self, *, llm_features, x, t, position_ids, segment_ids, indicator, text_cache=None):
    batch_size, seq_len, in_channels = x.shape
    assert in_channels == self.config.in_channels
    # param_dtype: bf16 for plain checkpoints; for Fp8Linear swaps the class
    # sets .compute_dtype so we stay in the correct compute dtype either way
    first = self.input_proj
    param_dtype = getattr(first, "compute_dtype", None) or first.weight.dtype
    x = x.to(param_dtype)
    t = t.to(param_dtype)
    indicator = indicator.to(torch.long)
    llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
    output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)
    x = x * output_image_mask
    x = self.input_proj(x) * output_image_mask
    t_cond = self.t_embedding(t)
    if t.dim() == 1:
      t_cond = t_cond.unsqueeze(1)
    adaln_input = F.silu(self.adaln_proj(t_cond))
    if llm_features is None:
      # Explicit image-only/unconditional call. There are no text tokens:
      # projecting a giant zero tensor and masking the result to zero buys
      # nothing. Keep this opt-in so ordinary conditioned calls are unchanged.
      h = x
    else:
      # The encoder supplies a compact text prefix. Do not run the very large
      # text projection on thousands of all-zero image slots at every step.
      # Full-length callers remain supported for compatibility.
      # Cache is owned by ONE sampling call and cleared when temporary weights
      # change. Never put this on the model/pipeline or share it between images.
      can_cache = text_cache is not None and not torch.is_grad_enabled()
      if (can_cache and text_cache.get('source') is llm_features
          and text_cache.get('indicator') is indicator):
        llm_features = text_cache['projected']
      else:
        source = llm_features
        text_len = llm_features.shape[1]
        if text_len > seq_len:
          raise ValueError('Text features exceed the latent sequence length')
        text_mask = llm_token_mask[:, :text_len]
        llm_features = llm_features.to(param_dtype) * text_mask
        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * text_mask
        llm_features = F.pad(llm_features, (0, 0, 0, seq_len - text_len))
        if can_cache:
          text_cache.clear()
          text_cache.update(source=source, indicator=indicator, projected=llm_features)
      h = x + llm_features
    image_indicator_embedding = self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
    h = h + image_indicator_embedding
    cos, sin = self.rotary_emb(position_ids)
    cos = cos.to(h.dtype)
    sin = sin.to(h.dtype)
    attn_mask = _segment_attn_mask(segment_ids)
    h = run_blocks(self, h, branch='uncond' if llm_features is None else 'cond',
                   attn_mask=attn_mask, cos=cos, sin=sin, adaln_input=adaln_input)
    out = self.final_layer(h, c=adaln_input)
    return out.to(torch.float32)
