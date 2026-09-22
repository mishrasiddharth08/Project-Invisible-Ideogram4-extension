"""Ideogram 4 dual-model pipeline (cond + uncond), loading from LOCAL single-file
safetensors only - no downloads, no HF cache dependency, no HF token required.

Modified by Project Invisible contributors (2026): compact exact text capture,
quality-preserving memory streaming, scoped acceleration and guarded loading.

Mirrors the official pipeline_ideogram4.py sampling loop:
- asymmetric CFG: cond pass = text+image tokens; uncond pass = image-only tokens
  through the SEPARATE unconditional transformer (official quality recipe)
- per-step guidance schedule (loop-index order: index 0 = final polish step)
- logit-normal schedule, resolution-shifted; Euler flow-matching integration
- Qwen3-VL 13-layer tap -> 53248-dim features
- Flux2 VAE decode with official latent shift/scale

Files consumed (already on the user's disk):
- models/Stable-diffusion/**: ideogram4 cond + unconditional checkpoints
  (BF16 / NF4 / FP8 single-file, Comfy-Org layout or official HF layout)
- models/text_encoder/qwen3vl_8b_*.safetensors
- models/VAE/flux2-vae.safetensors
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import threading
import torch
from PIL import Image
from safetensors.torch import load_file

from .autoencoder import AutoEncoder, AutoEncoderParams, convert_diffusers_state_dict
from .caption_verifier import CaptionVerifier
from .constants import (
  IMAGE_POSITION_OFFSET,
  LLM_TOKEN_INDICATOR,
  OUTPUT_IMAGE_INDICATOR,
  SEQUENCE_PADDING_INDICATOR,
  QWEN3_VL_ACTIVATION_LAYERS,
)
from .debanner import DebannerHooks, debanner_target_step, load_debanner_tensor
from .latent_norm import get_latent_norm
from .modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from .qwen3vl_text_encoder import Qwen3VLTextEncoder
from .quantized_loading import (
  is_bnb4bit_state_dict,
  is_fp8_state_dict,
  load_bnb4bit_state_dict,
  load_fp8_state_dict,
  swap_linears_to_bnb4bit,
  swap_linears_to_fp8,
)
from .memory import HIGH_VRAM_THRESHOLD_GB, Ideogram4MemoryManager
from .meta_init import assert_rope_buffers_valid, reinit_rope_buffers
from .scheduler import get_schedule_for_resolution, make_step_intervals
from .spectrum import SpectrumConfig, VelocityForecaster, plan, summarise
from .block_cache import generation_cache, set_step


@dataclass
class Ideogram4PipelineConfig:
  patch_size: int = 2
  ae_scale_factor: int = 8
  max_text_tokens: int = 2048


def layout_mismatch_message(unexpected_keys: list[str], filename: str = "") -> str:
  """Actionable guidance when a single-file load reports keys this model does
  not own. The extension loads the Comfy-Org single-file layout (keys like
  layers.N.attention.qkv.weight). Keys like 'transformer_blocks.' /
  '.attn.to_q.' / 'model.diffusion_model.' come from diffusers-format or
  legacy checkpoints and CANNOT be consumed here - say so plainly, name the
  two layouts, and point at the ungated in-process file. Pure string helper
  (no Forge deps) so tests can pin the wording."""
  keys = ", ".join(list(unexpected_keys)[:4])
  name = f" ({filename})" if filename else ""
  return (
    f"The checkpoint{name} does not match the Ideogram 4 single-file layout this "
    f"extension loads (Comfy-Org keys like layers.N.attention.qkv.weight). "
    f"Unexpected keys: {keys}. The official ideogram-ai repos are diffusers-layout "
    "folders - their key names are UNVERIFIED against this loader (file content is "
    "gated), so do not treat a raw repo file as loadable. Options: download the "
    "ungated Comfy-Org single-file fp8_scaled checkpoint into models/Ideogram4/ "
    "(in-process, recommended), or convert this file to the Comfy single-file layout "
    "via ComfyUI's export and drop it there. Nothing was loaded."
  )


# Guards every model construction that can see a globally patched
# torch.nn.Linear. Shared with the text encoder builder.
_MODEL_BUILD_LOCK = threading.RLock()


def _load_file_with_metadata(path: Path):
  """State dict plus the file's __metadata__ (safetensors header)."""
  from safetensors import safe_open
  meta = {}
  try:
    with safe_open(str(path), framework="pt") as f:
      meta = dict(f.metadata() or {})
  except Exception:
    meta = {}
  return load_file(str(path)), meta


def _forge_quant_ops(sd, file_meta, device, dtype, *, is_unet=True):
  """See pi_ideogram_lib.forge_quant - shared with the text encoder loader."""
  from .forge_quant import forge_quant_ops
  return forge_quant_ops(sd, file_meta, device, dtype, is_unet=is_unet)


def _align_unquantised_dtype(model, dtype: torch.dtype) -> int:
  """Put the model's plain half-precision tensors in the working dtype.

  A quantised checkpoint only quantises its Linear weights. Everything else -
  norms, biases, embeddings - is stored at whatever half precision the
  converter chose, and Forge Neo's converter writes fp16. Our activations are
  bf16, and torch will not fuse a norm whose weight dtype differs from its
  input:

      Mismatch dtype between input and weight: input dtype = BFloat16,
      weight dtype = Half, Cannot dispatch to fused implementation

  Measured on the fp8 checkpoint, where the same mismatch arose a different
  way (fp32 norms), the fallback cost 83 ms per forward - 1.0 s per image at
  12 steps, 4.0 s at 48.

  Only float16/bfloat16 tensors are touched. float32 is left alone because
  that is where the quantisation scales and the rotary inv_freq live, and a
  QuantizedTensor is never a plain half tensor, so the quantised weights are
  not reachable from here.
  """
  n = 0
  with torch.no_grad():
    for mod in model.modules():
      for name, p in list(mod._parameters.items()):
        if isinstance(p, torch.nn.Parameter) and p.dtype in (torch.float16, torch.bfloat16)            and p.dtype is not dtype and type(p.data) is torch.Tensor:
          mod._parameters[name] = torch.nn.Parameter(p.data.to(dtype), requires_grad=False)
          n += 1
      for name, b in list(mod._buffers.items()):
        if isinstance(b, torch.Tensor) and b.dtype in (torch.float16, torch.bfloat16)            and b.dtype is not dtype and type(b) is torch.Tensor:
          mod._buffers[name] = b.to(dtype)
          n += 1
  if n:
    print(f"[Invisible-I4] aligned {n} unquantised tensors to {dtype} "
          "(fused norm kernels need matching dtypes)")
  return n


def _quant_formats_present(sd: dict) -> set:
  from .forge_quant import quant_formats_present
  return quant_formats_present(sd)


def _load_transformer_from_file(path: Path, device: torch.device, dtype: torch.dtype) -> Ideogram4Transformer:
  """Build the official DiT and load a single-file checkpoint (BF16 / NF4 / FP8).

  SPEED: the model skeleton is constructed on a META device so torch never
  randomly initializes ~36 GB of parameters just to overwrite them (the old
  path wasted the majority of load time on CPU randn/kaiming). Real storage is
  allocated via to_empty() and weights are assigned directly from the
  safetensors mmap - no init, no double copies.

  CORRECTNESS: to_empty() also hands uninitialized memory to NON-PERSISTENT
  buffers, which no checkpoint contains - here that is the rotary inv_freq.
  reinit_rope_buffers() re-derives it on every branch below, and
  assert_rope_buffers_valid() refuses to hand back a model whose positional
  encoding is still junk.
  """
  if Path(path).suffix.lower() == '.gguf':
    raise RuntimeError('This extension has no verified Ideogram GGUF layout loader. Select a compatible safetensors checkpoint.')
  sd, file_meta = _load_file_with_metadata(path)
  config = Ideogram4Config()
  if "input_proj.weight" in sd:
    config.in_channels = int(sd["input_proj.weight"].shape[1])
  config.llm_features_dim = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)

  # MODERN QUANT FORMATS - int8_tensorwise (+convrot), convrot_w4a4, nvfp4,
  # mxfp8, asym_w4a8_int8 - are handled by FORGE, not by us. Every one of them
  # carries a per-layer `comfy_quant` descriptor, e.g.
  #     {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
  # and the convrot variants apply a Hadamard-style rotation before quantising,
  # so the weights cannot be recovered by multiplying a scale. Reimplementing
  # that here would mean guessing at a rotation convention; Forge Neo already
  # ships the real implementation (backend/quant_ops.py + Comfy-Kitchen
  # kernels), so we build the model with ITS Linear and let it load the file.
  #
  # This is also why these formats were previously refused with "no in-process
  # loader" - accurate at the time, but the loader existed one import away.
  forge_ops = _forge_quant_ops(sd, file_meta, device, dtype)
  if forge_ops is not None:
    ctx, label = forge_ops
    print(f"[Invisible-I4] {label} -> loading through Forge's quantised ops")
    # using_forge_operations() swaps torch.nn.Linear PROCESS-WIDE. The pipeline
    # builds cond, uncond, te and vae in parallel threads, so an unguarded swap
    # was visible to the text encoder mid-construction: transformers called
    # module.weight.data.normal_() on a Forge Linear, which has no .weight, and
    # the whole load died with
    #     AttributeError: 'Linear' object has no attribute 'weight'
    # Construction is milliseconds; serialise it and the threads stop colliding.
    with _MODEL_BUILD_LOCK:
      with ctx:
        model = Ideogram4Transformer(config)
    model.to_empty(device="cpu")
    reinit_rope_buffers(model)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    unexpected = [k for k in unexpected if not k.endswith(".comfy_quant")]
    if unexpected:
      raise RuntimeError(layout_mismatch_message(unexpected, path.name))
    critical = [k for k in missing if 'rotary' not in k]
    if critical:
      raise RuntimeError(f'Missing weights in quantized transformer: {critical[:10]}')
    _align_unquantised_dtype(model, dtype)
    model.to(device=device)
    del sd
    reinit_rope_buffers(model)
    assert_rope_buffers_valid(model, what=f"transformer {path.name}")
    model.eval()
    return model

  with _MODEL_BUILD_LOCK:
    with torch.device("meta"):
      model = Ideogram4Transformer(config)
  # DTYPE, before any storage exists. Ideogram4RMSNorm does torch.ones(dim) and
  # nn.Linear defaults likewise, so a freshly built skeleton is FLOAT32 - and
  # the fp8 branch below loads with copy_ (assign=False), which keeps the
  # destination dtype and silently upcast all 205 norm weights that the file
  # stores as bf16. torch then refused to fuse them:
  #     Mismatch dtype between input and weight: input dtype = BFloat16,
  #     weight dtype = float, Cannot dispatch to fused implementation
  # Measured at the real shapes (4337 tokens, 34 layers), one forward:
  #     205 norms   fp32 weight 91.8 ms  ->  bf16 weight 7.9 ms
  # which is ~1.0 s/image at V4_TURBO_12 and ~4.0 s at V4_QUALITY_48.
  # This is a dtype-only cast on meta tensors: no memory is touched.
  model.to(dtype)

  if is_bnb4bit_state_dict(sd):
    if device.type != "cuda" and not torch.cuda.is_available():
      raise ValueError("NF4 inference requires a CUDA GPU. Use an FP8 checkpoint instead.")
    model.to_empty(device="cpu")
    reinit_rope_buffers(model)
    swap_linears_to_bnb4bit(model, compute_dtype=dtype)
    load_bnb4bit_state_dict(model, sd, device=device, dtype=dtype)
  elif is_fp8_state_dict(sd):
    # 1) swap nn.Linear -> Fp8Linear ON THE META MODEL (the swap only reads
    #    shapes and replaces modules - no storage touched),
    # 2) to_empty() allocates REAL (uninitialized) storage for everything,
    #    including the new Fp8Linear buffers and the still-meta bf16 params,
    # 3) assign-load fills every tensor straight from the safetensors mmap.
    swap_linears_to_fp8(model, sd, compute_dtype=dtype)
    model.to_empty(device="cpu")
    reinit_rope_buffers(model)
    load_fp8_state_dict(model, sd, device=device, dtype=dtype, assign=True)
  else:
    model.to_empty(device="cpu")
    reinit_rope_buffers(model)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    unexpected = [k for k in unexpected if not k.endswith(".comfy_quant")]
    if unexpected:
      raise RuntimeError(layout_mismatch_message(unexpected, path.name))
    critical = [k for k in missing if "rotary" not in k]
    if critical:
      raise RuntimeError(f"missing keys after load: {critical[:10]} - the file may be a "
                         "diffusers-layout checkpoint (see the two-layout note above); "
                         "use the ungated Comfy-Org fp8_scaled single file instead")
    model.to(device=device, dtype=dtype)
  del sd
  # inv_freq lives on whatever device the branch above ended on; re-derive once
  # more so it follows the weights, then verify it is real (not meta garbage).
  reinit_rope_buffers(model)
  assert_rope_buffers_valid(model, what=f"transformer {path.name}")
  model.eval()
  return model


def _load_vae_from_file(path: Path, device: torch.device, dtype: torch.dtype) -> AutoEncoder:
  with _MODEL_BUILD_LOCK:
    ae = AutoEncoder(AutoEncoderParams())
  raw = load_file(str(path))
  # flux2-vae.safetensors is already in native KL-ae layout; convert_diffusers_state_dict
  # passes known keys through unchanged and handles diffusers-layout files too.
  sd = convert_diffusers_state_dict(raw)
  ae.load_state_dict(sd)
  ae._img2img_ready = True
  ae.to(device=device, dtype=dtype)
  ae.eval()
  return ae





class _NullContext:
  """`with` target for the disabled-debanner path; keeps the loop branch-free."""

  def __enter__(self):
    return None

  def __exit__(self, *_exc):
    return False


class Ideogram4Pipeline:
  """Dual-model Ideogram 4 pipeline over local files."""

  def __init__(self, *, cond_path, uncond_path, te_path, vae_path, device="cuda", dtype=torch.bfloat16):
    self.config = Ideogram4PipelineConfig()
    from .device_policy import resolve_device, compute_dtype
    self.device = resolve_device(device)
    self.dtype = compute_dtype(self.device, dtype)
    if self.dtype != dtype:
      print(f"[Invisible-I4] {self.device}: BF16 unavailable; using FP32 (more RAM and slower computation)")
    self.cond_path = Path(cond_path)
    self.uncond_path = Path(uncond_path) if uncond_path else None
    self.te_path = Path(te_path)
    self.vae_path = Path(vae_path)
    self.caption_verifier = CaptionVerifier()
    self.conditional_transformer = None
    self.unconditional_transformer = None
    self.text_encoder = None
    self.autoencoder = None
    shift, scale = get_latent_norm()
    self.latent_shift = shift.to(self.device)
    self.latent_scale = scale.to(self.device)
    self._loaded = False
    self._encode_cache: dict[tuple, tuple[dict, torch.Tensor]] = {}
    # Set by generate.py when an uncond-REPLACEMENT LoRA is in play. It is a
    # RuntimeLoraAdapter, deliberately not a merge: the adapter must be live
    # for the unconditional forward and dormant for the conditional one, in
    # the same step. See pi_ideogram_lora/runtime_lora.py.
    self.uncond_adapter = None
    self._preview_grid: tuple[int, int] | None = None
    self._preview_low_vram = False
    self.memory = Ideogram4MemoryManager(self)
    # Forge owns process-global CUDA/cuDNN policy. Do not change it here:
    # enabling cuDNN benchmarking also creates a large first-decode peak.

  # ------------------------------------------------------------------ loading

  def load(self, progress_cb=None) -> None:
    def _report(msg):
      print(f"[Invisible-I4] {msg}")
      if progress_cb:
        progress_cb(msg)

    if self._loaded:
      return

    # Take the card before allocating 27 GB on top of whatever Forge is
    # holding. Inside Forge this is the difference between a 12.7s load and a
    # 260.6s one - see Ideogram4MemoryManager.release_forge_models.
    self.memory.release_forge_models()

    t0 = time.time()

    # PARALLEL LOAD: cond + uncond transformers are two independent 8.6 GB
    # files; the TE and the VAE are independent too. Loading them on separate
    # threads overlaps disk read + GPU copy instead of paying each in series.
    # (safetensors loads are mmap-read + tensor-assign: no shared state between
    # the four; threads only race on the GPU allocator, which torch serializes.)
    from concurrent.futures import ThreadPoolExecutor

    # Low-memory cards must never allocate whole trunks on CUDA even during
    # loading: offloading them AFTER a parallel load is already too late.
    load_device = torch.device('cpu') if self.memory.stream_weights else self.device
    jobs: dict[str, callable] = {"cond": lambda: _load_transformer_from_file(self.cond_path, load_device, self.dtype)}
    if self.uncond_path and Path(self.uncond_path).exists():
      jobs["uncond"] = lambda: _load_transformer_from_file(Path(self.uncond_path), load_device, self.dtype)
    elif getattr(self, "_i4_intentional_single", False):
      pass  # turbo / uncond-LoRA profile: single-transformer mode is intentional
    else:
      _report("WARNING: no unconditional model paired - base-quality CFG will be approximated with zeroed-text cond pass on the conditional model only")
    jobs["te"] = self._load_te  # CPU-resident load; see _load_te for why
    jobs["vae"] = lambda: _load_vae_from_file(self.vae_path, load_device, self.dtype)

    _report(f"Loading in parallel: {', '.join(jobs.keys())}")
    results: dict[str, object] = {}
    errors: list[Exception] = []

    def _run(name, fn):
      try:
        results[name] = fn()
      except Exception as e:  # surface in join order below
        errors.append(e)
        results[name] = None

    threads = []
    if "uncond" in jobs:
      threads.append(("uncond", jobs["uncond"]))
    for name in ("te", "vae", "cond"):
      threads.append((name, jobs[name]))
    with ThreadPoolExecutor(max_workers=4) as ex:
      futs = [ex.submit(_run, n, f) for n, f in threads]
      for f in futs:
        f.result()
    if errors:
      raise errors[0]

    self.conditional_transformer = results.get("cond")
    self.unconditional_transformer = results.get("uncond")
    self.autoencoder = results.get("vae")
    te_obj = results.get("te")
    if isinstance(te_obj, Qwen3VLTextEncoder):
      self.text_encoder = te_obj
    self.memory.install_streaming()
    self._loaded = True
    _report(f"Pipeline ready in {time.time() - t0:.1f}s (parallel load)")
    from .attention import attention_status
    _report('Attention: ' + attention_status())
    # MEMORY (ABSOLUTE-inspired): only the cond transformer needs to stay
    # resident after load. Park TE + VAE + uncond on CPU immediately - they
    # re-upload on demand (encode, decode, first CFG step). This collapses the
    # ~31 GB load peak to ~11 GB right away instead of carrying 4 models idle.
    self.memory.offload_text_encoder()
    self.memory.offload_vae()
    self.memory.offload_uncond()

  def _load_te(self) -> "Qwen3VLTextEncoder":
    # Load on CPU, not GPU: the ~8 GB Qwen3-VL encoder only needs the GPU
    # during prompt encoding (ensure_text_encoder_on_device uploads it on
    # first use). Loading it on CUDA would push the parallel-load peak past
    # ~31 GB; on CPU it never contributes to the spike.
    te = Qwen3VLTextEncoder(self.te_path, torch.device("cpu"), self.dtype)
    te.load()
    return te

  def unload(self) -> None:
    self.memory.remove_streaming()
    # Runtime adapters own strong model references and CUDA hook closures.
    # Drop those owners before clearing the trunks or cached pipes keep VRAM.
    adapters = list(getattr(self, '_i4_runtime_adapters', None) or [])
    for name in ('_i4_turbo_adapter', 'uncond_adapter'):
      adapter = getattr(self, name, None)
      if adapter is not None:
        adapters.append(adapter)
      setattr(self, name, None)
    for adapter in adapters:
      adapter.remove()
    self._i4_runtime_adapters = []
    adapters.clear()
    adapter = None
    for name in ('_i4_applied_lora_key', '_i4_turbo_key', '_i4_turbo_lora',
                 '_i4_uncond_lora', '_i4_bypass_lora'):
      setattr(self, name, None)
    self._i4_turbo_merged = False
    self._i4_lora_merged = False
    # Encoded features and packed inputs own CUDA tensors too. Clearing only
    # the models leaves those allocations alive in a cached pipeline object.
    self._encode_cache.clear()
    self._preview_grid = None
    self.conditional_transformer = None
    self.unconditional_transformer = None
    if self.text_encoder is not None:
      self.text_encoder.unload()
    self.autoencoder = None
    self._loaded = False
    self.memory._cleanup()

  # Memory management delegated to self.memory (Ideogram4MemoryManager)

  @property
  def loaded(self) -> bool:
    return self._loaded

  # ------------------------------------------------------------------ sampling

  def _build_inputs(self, prompts: list[str], height: int, width: int) -> dict:
    tokenized = []
    for p in prompts:
      enc = self.text_encoder.tokenize(p)
      token_ids = enc["input_ids"][0]
      num_text = int(token_ids.shape[0])
      if num_text > self.config.max_text_tokens:
        # REFUSE, do not truncate. This is the only place that knows the true
        # token count, and silently keeping the first N tokens produced an
        # image built from part of the prompt with nothing to explain why -
        # the single hardest failure for a user to diagnose. Words are a poor
        # proxy here: plain prose measures ~1.16 tokens/word but heavily
        # punctuated text reaches ~2.08, so json_prompt's word ceiling is a
        # friendly early check and THIS is the authority.
        cap = self.config.max_text_tokens
        over = num_text - cap
        raise ValueError(
          f"This caption is {num_text} tokens; Ideogram 4's text window is {cap}. "
          f"It is over by {over}. Nothing was generated, because cutting the tail "
          "would silently drop part of your prompt. Shorten the prompt (roughly "
          f"{max(1, int(over / 1.6))} words for typical prose - dense punctuation "
          "costs more per word), or move detail into raw JSON with bounding boxes."
        )
      tokenized.append((token_ids, num_text))

    patch = self.config.patch_size * self.config.ae_scale_factor
    if height % patch != 0 or width % patch != 0:
      raise ValueError(f"height/width must be divisible by {patch}")
    grid_h = height // patch
    grid_w = width // patch
    num_image_tokens = grid_h * grid_w

    max_text_tokens = max(num_text for _, num_text in tokenized)
    total_seq_len = max_text_tokens + num_image_tokens
    batch_size = len(prompts)

    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    token_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
    text_position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
    position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
    segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
    indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

    for b, (toks, num_text) in enumerate(tokenized):
      pad_len = max_text_tokens - num_text
      offset = pad_len
      token_ids[b, offset:offset + num_text] = toks
      text_pos = torch.arange(num_text)
      text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
      text_position_ids[b, offset:offset + num_text] = text_pos_3d
      position_ids[b, offset:offset + num_text] = text_pos_3d
      position_ids[b, offset + num_text:] = image_pos
      indicator[b, offset:offset + num_text] = LLM_TOKEN_INDICATOR
      indicator[b, offset + num_text:] = OUTPUT_IMAGE_INDICATOR
      segment_ids[b, offset:offset + num_text + num_image_tokens] = 1

    return {
      "token_ids": token_ids.to(self.device),
      "text_position_ids": text_position_ids.to(self.device),
      "position_ids": position_ids.to(self.device),
      "segment_ids": segment_ids.to(self.device),
      "indicator": indicator.to(self.device),
      "num_image_tokens": num_image_tokens,
      "grid_h": grid_h,
      "grid_w": grid_w,
      "max_text_tokens": max_text_tokens,
    }



  def _encode_text_batch(self, inputs: dict, batch_size: int, *, text_only: bool = False) -> torch.Tensor:
    """Capture only text features without changing the encoder computation.

    Keep the original full-sequence forward shapes: shortening those shapes
    measurably changes real FP8 Qwen outputs even though causal attention
    makes the text prefix mathematically independent of later image slots.
    Clone only the text prefix at each tap, freeing unused image activations
    instead of retaining them through all 36 layers and stacking them.
    The default preserves the original padded float32 result for callers.
    The pipeline caches only the compact prefix in the encoder's native dtype.
    """
    from transformers.masking_utils import create_causal_mask

    text_tokens = inputs["max_text_tokens"]
    token_ids = inputs["token_ids"]
    attn_mask = (inputs["indicator"] == LLM_TOKEN_INDICATOR).to(torch.long)
    pos_2d = inputs["text_position_ids"][..., 0].contiguous()
    lm = self.text_encoder.model.language_model

    inputs_embeds = lm.embed_tokens(token_ids)
    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]
    seq_len = token_ids.shape[1]
    cache_position = torch.arange(seq_len, device=self.device)
    causal_mask = create_causal_mask(
      config=lm.config, input_embeds=inputs_embeds, attention_mask=attn_mask,
      cache_position=cache_position, past_key_values=None, position_ids=text_position_ids,
    )
    position_embeddings = lm.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    with torch.no_grad():
      for layer_idx, decoder_layer in enumerate(lm.layers):
        hidden_states = decoder_layer(
          hidden_states,
          attention_mask=causal_mask,
          position_ids=text_position_ids,
          past_key_values=None,
          position_embeddings=position_embeddings,
        )
        if layer_idx in tap_set:
          captured[layer_idx] = hidden_states[:, :text_tokens].clone()
    selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
    stacked = torch.stack(selected, dim=0)
    stacked = torch.permute(stacked, (1, 2, 3, 0))
    stacked = stacked.reshape(batch_size, text_tokens, -1)
    text_mask = attn_mask[:, :text_tokens].to(stacked.dtype).unsqueeze(-1)
    stacked = stacked * text_mask
    if text_only:
      return stacked
    return torch.nn.functional.pad(
      stacked.to(torch.float32),
      (0, 0, 0, inputs["token_ids"].shape[1] - text_tokens),
    )

  def _verify_prompts(self, prompts: list[str], *, raise_on_issues: bool = True) -> None:
    messages = []
    for i, prompt in enumerate(prompts):
      issues = self.caption_verifier.verify_raw(prompt)
      if issues:
        messages.append(f"caption verifier flagged prompt[{i}]:\n" + "\n".join(issues))
    if not messages:
      return
    combined = "\n".join(messages)
    if raise_on_issues:
      raise ValueError(combined)
    # Not raising is no reason to hide them: a schema slip is exactly what pushes
    # the model into its gray safety-filter attractor.
    print("[Invisible-I4] caption schema warnings (generation continues):"
          + '\n' + combined)

  @torch.no_grad()
  @generation_cache
  def __call__(
    self,
    prompts: str | list[str],
    *,
    height: int = 1024,
    width: int = 1024,
    num_steps: int = 20,
    guidance_scale: float = 7.0,
    guidance_schedule=None,
    mu: float = 0.0,
    std: float = 1.75,
    seed=None,
    raise_on_caption_issues: bool = True,
    step_callback=None,
    preview_callback=None,
    on_actual_steps=None,
    filter_bypass: bool = False,
    bypass_extra_steps: int = 0,
    sigma_smooth_steps: int = 0,
    first_step_undo=None,
    debanner_enabled: bool = False,
    debanner_strength: float = 0.6,
    spectrum: "SpectrumConfig | None" = None,
    first_block_cache: float = 0.0,
    init_image: Image.Image | None = None,
    strength: float = 1.0,
  ) -> list[Image.Image]:
    """Run the full Ideogram 4 generation pipeline.

    Args:
      prompts: Scene description(s) — plain text or official JSON.
      height, width: Output resolution (must be divisible by 64).
      num_steps: Base denoising steps (extended by bypass_extra_steps).
      guidance_scale: CFG scale (ignored when guidance_schedule given).
      guidance_schedule: Per-step CFG values in loop-index order.
      mu, std: Logit-normal schedule parameters.
      seed: RNG seed for reproducibility.
      raise_on_caption_issues: If False, warnings are logged but not raised.
      step_callback: Called with (step, total_steps) after each denoising step.
      on_actual_steps: Called once with the real step count (may exceed
        num_steps when bypass_extra_steps > 0).  Used by the Forge
        progress bar to show the correct total.
      filter_bypass: If True, shift the first sigma up by +0.005 to
        bypass the model's baked-in safety filter.
      bypass_extra_steps: Extra interpolated steps inserted into the FIRST
        sigma interval (ExtendIntermediateSigmas-style smoothing). 0 = off.
      sigma_smooth_steps: Same smoothing, driven by the bypass recipe
        (P2); used when bypass_extra_steps is 0.
      first_step_undo: Callable invoked ONCE right after the first step's
        forwards (before any offload), so first-step-only adapters (gray
        bypass LoRA) can be removed from the live weights in time.
      debanner_enabled: If True, apply MrJackSpade correction tensor
        on the first step to remove the Ideogram watermark.
      debanner_strength: Correction magnitude (0.0–1.5, default 0.6).
      spectrum: Opt-in step forecasting (see spectrum.py). None or a config
        with enabled=False runs the official recipe unchanged. When on,
        some steps are extrapolated instead of computed, which is faster
        and CHANGES the image for a given seed.

    Returns:
      List of decoded PIL Images.
    """
    strength = float(strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
      raise ValueError("img2img strength must be between 0.0 and 1.0")
    if init_image is not None and strength == 0.0:
      if on_actual_steps is not None:
        on_actual_steps(0)
      return [init_image.copy()]

    partial_img2img = init_image is not None and 0.0 < strength < 1.0
    if partial_img2img:
      active_steps = max(1, int(np.ceil(num_steps * strength)))
    else:
      active_steps = num_steps

    if isinstance(prompts, str):
      prompts = [prompts]
    self._verify_prompts(prompts, raise_on_issues=raise_on_caption_issues)

    schedule = get_schedule_for_resolution((height, width), known_mean=mu, std=std)
    original_num_steps = num_steps
    original_intervals = make_step_intervals(original_num_steps).to(self.device)
    step_intervals = (
      original_intervals[: active_steps + 1]
      if partial_img2img else original_intervals
    )
    if partial_img2img:
      num_steps = active_steps

    # BYPASS (P2): smooth the FIRST sigma interval - the first denoising step
    # spans [intervals[-2], intervals[-1]], and subdividing that interval is
    # the community ExtendIntermediateSigmas lever that keeps the gray-screen
    # attractor from triggering on the initial sigma jump.
    extra_n = 0 if partial_img2img else (int(bypass_extra_steps) if bypass_extra_steps and bypass_extra_steps > 0 else int(sigma_smooth_steps))
    if extra_n > 0:
      from .bypass import smooth_first_sigmas
      step_intervals = smooth_first_sigmas(step_intervals, extra=extra_n)
      num_steps = num_steps + extra_n

    if guidance_schedule is not None:
      gw_per_step = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=self.device)
      # BYPASS (P2): the smoothing above extends the step count - the guidance
      # schedule must be padded to the SAME length for either lever
      # (bypass_extra_steps OR sigma_smooth_steps; the smooth substeps reuse the
      # schedule's last entry, matching the first-interval subdivision).
      if partial_img2img:
        gw_per_step = gw_per_step[:active_steps]
      if extra_n > 0 and gw_per_step.shape[0] < num_steps:
        pad = gw_per_step[-1:].expand(extra_n)
        gw_per_step = torch.cat([gw_per_step, pad])
      if gw_per_step.shape != (num_steps,):
        raise ValueError(f"guidance_schedule must have shape ({num_steps},)")
    else:
      gw_per_step = torch.full((num_steps,), float(guidance_scale), dtype=torch.float32, device=self.device)

    # SPEED + VRAM: identical prompts + size reuses packed inputs and features.
    # Store only TEXT features in bf16, not thousands of zero image slots.
    # Two-entry LRU: keep the other prompt when one entry needs eviction.
    # PHASE TIMING. Two speed hypotheses were tested against this pipeline and
    # both were wrong (fp8 _scaled_mm would cost MORE memory; live previews are
    # 1.5 s of a 125 s run). Measured block cost accounts for ~38 s of that
    # run, so ~88 s was unexplained - and unexplained is not the same as
    # attributable to whatever is easiest to blame. These timers make the next
    # real generation say where the time actually goes.
    import time as _time
    # ACCUMULATE ACROSS THE BATCH. This dict was recreated on every __call__,
    # so a 2-image run reported the timings of the LAST image against the
    # elapsed time of BOTH - which is how a healthy run printed
    #     sampling 87.5s | ... | elsewhere 95.2s
    # and made 95 seconds of phantom overhead appear out of arithmetic.
    # generate() clears it once per REQUEST; the pipeline only ever adds.
    _ph = getattr(self, "_phase_ms", None)
    if not isinstance(_ph, dict) or getattr(self, "_phase_reset", False):
        _ph = {"encode": 0.0, "steps": 0.0, "preview": 0.0, "decode": 0.0,
               "fwd_cond": 0.0, "fwd_uncond": 0.0}
        self._phase_reset = False
    self._phase_ms = _ph
    _t_enc = _time.perf_counter()

    cache_key = (tuple(prompts), height, width)
    if cache_key in self._encode_cache:
      inputs, text_features = self._encode_cache.pop(cache_key)
      self._encode_cache[cache_key] = (inputs, text_features)
    else:
      if len(self._encode_cache) >= 2:
        del self._encode_cache[next(iter(self._encode_cache))]
      # A previous run can leave both the uncond model and preview VAE on
      # CUDA. Park them BEFORE uploading Qwen, not after encoding has already
      # exceeded VRAM. These models are idle throughout text encoding.
      self.memory.offload_uncond()
      self.memory.offload_vae()
      self.memory.offload_cond()
      self.memory.reclaim_if_contended(width=width, height=height)
      self.memory.ensure_text_encoder_on_device()
      inputs = self._build_inputs(prompts, height=height, width=width)
      batch_size = len(prompts)
      try:
        text_features = self._encode_text_batch(inputs, batch_size, text_only=True)
        text_features = text_features.to(torch.bfloat16)
      finally:
        self.memory.offload_text_encoder()
      self._encode_cache[cache_key] = (inputs, text_features)
      # MEMORY (ABSOLUTE-inspired): the ~10 GB Qwen3-VL text encoder is idle
      # during all diffusion steps; with the encode cache above, seed sweeps
      # never need it again. Park it on CPU - frees ~10 GB VRAM for the
      # sampler. It re-uploads automatically only for a genuinely new prompt.

    llm_features = text_features

    _ph["encode"] += (_time.perf_counter() - _t_enc) * 1000.0

    batch_size = len(prompts)
    num_image_tokens = inputs["num_image_tokens"]
    grid_h, grid_w = inputs["grid_h"], inputs["grid_w"]
    max_text_tokens = inputs["max_text_tokens"]
    latent_dim = self.conditional_transformer.config.in_channels

    neg_position_ids = inputs["position_ids"][:, max_text_tokens:]
    neg_segment_ids = inputs["segment_ids"][:, max_text_tokens:]
    neg_indicator = inputs["indicator"][:, max_text_tokens:]

    generator = torch.Generator(device=self.device)
    if seed is not None:
      generator.manual_seed(int(seed))
    if partial_img2img:
      self.memory.offload_cond()
      self.memory.offload_uncond()
      try:
        z_image = self._encode_img2img_image(init_image, height=height, width=width)
      finally:
        self.memory.offload_vae()
      noise = torch.randn(z_image.shape, dtype=torch.float32, device=self.device, generator=generator)
      t_entry = schedule(step_intervals[-1:].to(self.device)).to(dtype=z_image.dtype)
      z = self._blend_img2img_latent(z_image, noise, t_entry.reshape(1, 1, 1))
    else:
      z = torch.randn(
        batch_size, num_image_tokens, latent_dim,
        dtype=torch.float32, device=self.device, generator=generator,
      )

    text_z_padding = torch.zeros(
      batch_size, max_text_tokens, latent_dim,
      dtype=torch.float32, device=self.device,
    )

    has_uncond = self.unconditional_transformer is not None
    # Load debanner tensor once (cached globally)
    debanner_dirs = load_debanner_tensor() if debanner_enabled else None
    if debanner_enabled:
      if debanner_dirs:
        print(f"[Invisible-I4] Debanner tensor loaded: {len(debanner_dirs)} blocks ({sorted(debanner_dirs.keys())})")
      else:
        print("[Invisible-I4] Debanner tensor NOT loaded (None) - debanner disabled")
    # WHICH step the correction belongs on. The bundle's own metadata says
    # target_step 0, and this loop counts DOWN (index 0 = final step), so that
    # is the LAST step. Applying it on the FIRST step steers the whole
    # trajectory instead of cleaning the finished image - measured: even at
    # strength 0.05 it repainted 63% of the pixels.
    debanner_step = debanner_target_step(0) if debanner_dirs else -1

    # MEMORY (hardware-aware, ABSOLUTE-inspired):
    #  - >= 28 GB VRAM: park VAE (1 GB, only needed at decode) and keep both
    #    transformers resident - per-step offload would re-upload the 9.3 GB
    #    uncond CPU->GPU on every step (that was the "very slow generation").
    #  - < 28 GB VRAM: park uncond per step as before so cond + activations fit.
    # Reclaim the card EVERY generation, not only on load. The pipeline is
    # cached, so a user who generates with Ideogram 4, switches to SDXL,
    # generates, and switches back would otherwise re-enter here with Forge's
    # SDXL weights still resident and load() short-circuited - straight back
    # into the spill that cost 260s of load and 199s of decode.
    self.memory.reclaim_if_contended(width=width, height=height)

    vram_gb = self.memory.vram_gb()
    high_vram = (vram_gb >= HIGH_VRAM_THRESHOLD_GB and not self.memory.stream_weights)
    self.memory.offload_vae()
    if has_uncond and not high_vram:
      self.memory.offload_uncond()
    print(f"[Invisible-I4] VRAM: {vram_gb:.1f}GB -> "
          + ("RAM-backed block streaming" if self.memory.stream_weights else
             "resident path (transformers on GPU, no per-step offload)" if high_vram
             else "low-VRAM path (uncond per-step offload)"))

    # decode_preview needs the true grid to rebuild the image; without it a
    # preview can only reinterpret a horizontal strip of tokens as a square.
    self._preview_grid = (grid_h, grid_w)
    # A preview decoder is idle between previews even on a large GPU.
    self._preview_low_vram = not high_vram or width * height > 1024 * 1024
    self.memory.ensure_cond_on_gpu()

    if on_actual_steps is not None:
      on_actual_steps(num_steps)

    # SPECTRUM: see pi_ideogram_lib/spectrum.py. Two forecasters, one per CFG
    # branch, so the official 7->3 guidance schedule keeps being applied at
    # each step's own weight instead of being baked into the history. Both
    # live and die with this call, so a new image starts with no anchors.
    _spectrum = spectrum if (spectrum is not None and spectrum.enabled) else None
    _steps_plan = plan(num_steps, _spectrum) if _spectrum else [False] * num_steps
    _fc_pos = VelocityForecaster(_spectrum) if _spectrum else None
    _fc_neg = VelocityForecaster(_spectrum) if _spectrum else None
    _spectrum_used = 0
    _spectrum_refused = 0
    if _spectrum:
      print("[Invisible-I4] " + summarise(_steps_plan, num_steps))

    # Prompt projection is independent of timestep. Reuse it within this image;
    # temporary first-step adapters invalidate it immediately after removal.
    text_projection_cache = {}
    _t_steps = _time.perf_counter()
    for i in range(num_steps - 1, -1, -1):
      t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
      s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
      # BYPASS: shift first sigma up by +0.005 to bypass the model's baked-in safety filter
      if filter_bypass and not partial_img2img and i == num_steps - 1:
        t_val = t_val + 0.005
      t = torch.full((batch_size,), t_val, dtype=torch.float32, device=self.device)

      pos_z = torch.cat([text_z_padding, z], dim=1)

      gw_i = float(gw_per_step[i])
      _step_idx = num_steps - 1 - i
      set_step(_step_idx, num_steps)

      # SPECTRUM (opt-in, off by default). On a planned step the velocity is
      # extrapolated from previous REAL steps and both forwards are skipped.
      #
      # A forecast is used only when every branch it needs came back with a
      # trustworthy answer; the forecaster refuses (returns None) on a
      # degenerate fit, a non-finite history, or a prediction that would jump
      # the velocity magnitude. Any refusal falls through to the real network,
      # so the worst case is the unaccelerated run, never a wrong image.
      #
      # The debanner step is never forecast: it exists to perturb the residual
      # stream inside the transformer, and there is no transformer pass to
      # perturb on a skipped step.
      _do_forecast = (_spectrum is not None and _steps_plan[_step_idx]
                      and not (bool(debanner_dirs) and i == debanner_step))
      _need_neg = gw_i != 1.0
      if _do_forecast:
        _cand_pos = _fc_pos.predict(t_val)
        _cand_neg = _fc_neg.predict(t_val) if _need_neg else None
        if _cand_pos is not None and (_cand_neg is not None or not _need_neg):
          _spectrum_used += 1
          v = (gw_i * _cand_pos + (1.0 - gw_i) * _cand_neg) if _need_neg else _cand_pos
        else:
          _spectrum_refused += 1
          _do_forecast = False
      if not _do_forecast:
        # DEBANNER (first step only). The correction tensor is emb_dim-wide, so
        # it belongs on the residual stream INSIDE blocks 25-28 - hooks, not a
        # post-hoc edit of the 128-channel velocity the model returns. Applying
        # it to the output silently no-ops (shape mismatch), which is exactly
        # what this used to do while reporting success.
        use_debanner = bool(debanner_dirs) and i == debanner_step
        if use_debanner:
          hooks = DebannerHooks(self.conditional_transformer, debanner_dirs,
                                debanner_strength, max_text_tokens, grid_h, grid_w)
        else:
          hooks = _NullContext()
        with hooks as dbg:
          _t_fwd = _time.perf_counter()
          pos_out = self.conditional_transformer(
            llm_features=llm_features,
            text_cache=text_projection_cache,
            x=pos_z,
            t=t,
            position_ids=inputs["position_ids"],
            segment_ids=inputs["segment_ids"],
            indicator=inputs["indicator"],
          )
        if use_debanner:
          fired = sorted(set(getattr(dbg, "applied", [])))
          if fired:
            print(f"[Invisible-I4] Debanner: blocks {fired} corrected on step 1 "
                  f"(strength={debanner_strength})")
          else:
            print("[Invisible-I4] Debanner: correction did NOT apply on any block "
                  "(see the reason above); generation continues unchanged.")
        if z.is_cuda:
          torch.cuda.synchronize()
        _ph["fwd_cond"] += (_time.perf_counter() - _t_fwd) * 1000.0
        pos_v = pos_out[:, max_text_tokens:]

        if has_uncond and gw_i != 1.0:
          # MEMORY: on high-VRAM the uncond stays resident (ensure is a cheap
          # device check); on low-VRAM this is the one re-upload before its pass.
          self.memory.ensure_uncond_on_gpu()
          _t_unc = _time.perf_counter()
          neg_v = self.unconditional_transformer(
            llm_features=None,
            x=z,
            t=t,
            position_ids=neg_position_ids,
            segment_ids=neg_segment_ids,
            indicator=neg_indicator,
          )
          if z.is_cuda:
            torch.cuda.synchronize()
          _ph["fwd_uncond"] += (_time.perf_counter() - _t_unc) * 1000.0
          v = gw_i * pos_v + (1.0 - gw_i) * neg_v
        elif not has_uncond and gw_i != 1.0:
          # No separate uncond transformer. Two sub-cases:
          #  - an uncond-REPLACEMENT LoRA is loaded: switch it on for THIS pass
          #    only, so the conditional pass above ran on clean weights. That
          #    per-pass split is the whole point of the adapter; leaving it
          #    merged for both passes makes CFG subtract two near-identical
          #    outputs and prompt adherence collapses.
          #  - nothing loaded: fall back to the documented zeroed-text
          #    approximation on the cond model.
          # The indicator removes every text slot in this branch. Explicit
          # None conditioning skips the large zero allocation and projection.
          adapter = self.uncond_adapter
          ctx = adapter.active() if adapter is not None else _NullContext()
          with ctx:
            neg_out = self.conditional_transformer(
              llm_features=None,
              x=pos_z,
              t=t,
              position_ids=inputs["position_ids"],
              segment_ids=inputs["segment_ids"],
              indicator=neg_indicator_full(inputs),
            )
          neg_v = neg_out[:, max_text_tokens:]
          v = gw_i * pos_v + (1.0 - gw_i) * neg_v
        else:
          v = pos_v
        # Only REAL velocities become anchors. Feeding a forecast back in
        # would let its error compound step over step, which is the failure
        # mode that turns "slightly different image" into "different image".
        if _spectrum:
          _fc_pos.update(t_val, pos_v)
          if _need_neg:
            _fc_neg.update(t_val, neg_v)

      delta = s_val - t_val
      z = z + v * delta
      # FIRST-STEP-ONLY ADAPTERS (gray bypass LoRA): remove the temporary
      # merge right after step 1's forwards, BEFORE any offload moves the
      # weights CPU-side (the undo needs them on GPU).
      if first_step_undo is not None:
        try:
          first_step_undo()
        except Exception as e:
          raise RuntimeError(f"First-step adapter cleanup failed: {e}") from e
        first_step_undo = None
        text_projection_cache.clear()
        if _spectrum:
          # The first velocity used different weights. Never extrapolate the
          # remaining trajectory from an adapter that has now been removed.
          _fc_pos.reset()
          _fc_neg.reset()
      # MEMORY: low-VRAM path parks uncond after each step; high-VRAM keeps it
      # resident to avoid a 9.3 GB CPU<->GPU round trip on every step.
      if has_uncond and gw_i != 1.0 and not high_vram:
        self.memory.offload_uncond()
      if step_callback is not None:
        step_callback(num_steps - 1 - i, num_steps)
      # Live previews show the actual noisy latent developing step by step.
      # Forge publishes the full-resolution decoded result after sampling.
      if preview_callback is not None:
        # Sync BEFORE the timer: decode_preview ends in a .cpu() copy, so
        # without this the preview phase is charged for every sampler kernel
        # still queued from this step. That is what made the console claim
        # "previews 15.7s" when the real cost, measured with syncs on both
        # sides, was 0.71s for a whole 12-step generation.
        if self.device.type == "cuda":
          torch.cuda.synchronize()
        _t_pv = _time.perf_counter()
        # Show actual denoising: noise gradually develops into the final image.
        # Display only; this never changes the sampling trajectory.
        preview_callback(z)
        _ph["preview"] += (_time.perf_counter() - _t_pv) * 1000.0

    _ph["steps"] += (_time.perf_counter() - _t_steps) * 1000.0
    if _spectrum:
      # Report what HAPPENED, not what was planned. A refusal means the
      # forecaster declined and the real network ran, so the count of
      # forwards actually skipped is the only number worth printing.
      note = (f", {_spectrum_refused} declined -> computed normally"
              if _spectrum_refused else "")
      print(f"[Invisible-I4] Spectrum: {_spectrum_used}/{num_steps} steps "
            f"forecast{note}")

    # MEMORY: reload VAE for decode (offloaded on both paths before sampling)
    _t_dec = _time.perf_counter()
    # Decoder activations grow with canvas size. Sampling models are idle now.
    text_projection_cache.clear()
    self.memory.offload_uncond()
    self.memory.offload_cond()
    self.memory.ensure_vae_on_gpu()
    out = self._decode(z, grid_h=grid_h, grid_w=grid_w)
    _ph["decode"] += (_time.perf_counter() - _t_dec) * 1000.0
    return out

  def _encode_img2img_image(self, image: Image.Image, *, height: int, width: int) -> torch.Tensor:
    """Encode an already-resized Forge img2img image into model latents."""
    if self.autoencoder is None or not hasattr(self.autoencoder, "encoder"):
      raise RuntimeError("img2img requires the complete VAE encoder")
    if not getattr(self.autoencoder, "_img2img_ready", True):
      raise RuntimeError("img2img VAE encoder weights are unavailable; txt2img remains supported")
    image = image.convert("RGB")
    if image.size != (width, height):
      raise ValueError(f"img2img image size {image.size} does not match {(width, height)}")
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1.0)
    pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype)
    self.memory.ensure_vae_on_gpu()
    posterior = self.autoencoder.encoder(pixels)
    z_ae = posterior[:, : self.autoencoder.params.z_channels]
    patch = self.config.patch_size
    grid_h = z_ae.shape[-2] // patch
    grid_w = z_ae.shape[-1] // patch
    if z_ae.shape[-2:] != (grid_h * patch, grid_w * patch):
      raise ValueError("encoded img2img latent is not divisible by the VAE patch size")
    z = z_ae.reshape(z_ae.shape[0], z_ae.shape[1], grid_h, patch, grid_w, patch)
    z = z.permute(0, 2, 4, 3, 5, 1).reshape(z_ae.shape[0], grid_h * grid_w, -1)
    z = (z.float() - self.latent_shift) / self.latent_scale
    expected_tokens = (height // (self.config.patch_size * self.config.ae_scale_factor)) * (
      width // (self.config.patch_size * self.config.ae_scale_factor)
    )
    expected_channels = self.autoencoder.params.z_channels * self.config.patch_size * self.config.patch_size
    if z.shape != (1, expected_tokens, expected_channels):
      raise RuntimeError(f"unexpected packed img2img latent shape {tuple(z.shape)}")
    return z

  @staticmethod
  def _blend_img2img_latent(z_image: torch.Tensor, noise: torch.Tensor, t_entry: torch.Tensor) -> torch.Tensor:
    return t_entry * z_image + (1.0 - t_entry) * noise

  def _decode(self, z: torch.Tensor, *, grid_h: int, grid_w: int) -> list[Image.Image]:
    batch_size = z.shape[0]
    patch = self.config.patch_size
    z = z * self.latent_scale + self.latent_shift
    ae_channels = z.shape[-1] // (patch * patch)
    z = z.view(batch_size, grid_h, grid_w, patch, patch, ae_channels)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    z = z.view(batch_size, ae_channels, grid_h * patch, grid_w * patch)
    z = z.to(self.dtype)
    decoded = self.autoencoder.decoder(z)
    decoded = decoded.float().clamp(-1.0, 1.0)
    decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
    decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(arr) for arr in decoded]

  def decode_preview(self, z: torch.Tensor, preview_pixels: int = 256) -> Image.Image | None:
    """Decode the in-flight latent into a small preview image.

    Two things the previous implementation got wrong, both fixed here:

    1. DEVICE. The VAE is parked on CPU for the whole sampling loop (see the
       memory policy in __call__), so decoding immediately raised a device
       mismatch on every call and the blanket except returned None - the live
       preview never once rendered. We now bring the VAE up (it is by far the
       smallest of the four models), decode, and park it again on the
       low-VRAM path.
    2. FRAMING. The first grid*grid tokens are NOT the top-left corner: tokens
       are row-major across a grid_w-wide grid, so that slice is the first few
       full rows of the image squashed into a square. We decode the real grid
       and thumbnail it instead.

    Args:
        z: Latent tensor, shape (B, num_tokens, channels).
        preview_pixels: Longest edge of the returned image (default 256).

    Returns:
        PIL.Image, or None if no preview can be produced right now.
    """
    grid = self._preview_grid
    if grid is None or self.autoencoder is None:
      return None
    grid_h, grid_w = grid
    if z.shape[1] < grid_h * grid_w:
      return None
    try:
      self.memory.ensure_vae_on_gpu()
      z_one = z[:1, : grid_h * grid_w, :].detach()

      # DECODE SMALL. This used to decode the FULL image and then throw ~93% of
      # it away in thumbnail(). At 1024x1024 that was 143 ms and looked free;
      # at 1280x1728, every step, it was measured at 35-39 s per generation -
      # about 27% of the whole run, spent on pictures nobody keeps.
      #
      # Average-pooling the token grid first costs one cheap reduction and
      # shrinks the decode quadratically. A preview is a thumbnail either way;
      # the only difference is that the shrinking now happens BEFORE the
      # expensive part instead of after it.
      gh, gw = grid_h, grid_w
      px_per_cell = 16  # patch_size(2) * VAE upsample(8)
      f = max(1, int(max(gh, gw) * px_per_cell // max(64, preview_pixels)))
      if f > 1 and gh // f >= 2 and gw // f >= 2:
        gh2, gw2 = gh // f, gw // f
        c = z_one.shape[-1]
        z_one = (z_one.reshape(1, gh, gw, c)[:, : gh2 * f, : gw2 * f, :]
                 .reshape(1, gh2, f, gw2, f, c)
                 .mean(dim=(2, 4))
                 .reshape(1, gh2 * gw2, c))
        gh, gw = gh2, gw2

      decoded = self._decode(z_one, grid_h=gh, grid_w=gw)
      if not decoded:
        return None
      img = decoded[0]
      if max(img.size) > preview_pixels:
        img.thumbnail((preview_pixels, preview_pixels), Image.BILINEAR)
      return img
    except Exception:
      return None
    finally:
      if self._preview_low_vram:
        try:
          self.memory.offload_vae()
        except Exception:
          pass


def neg_indicator_full(inputs: dict) -> torch.Tensor:
  """Indicator for the degraded no-uncond path: text slots become image-indicator zeros."""
  ind = inputs["indicator"].clone()
  max_text = inputs["max_text_tokens"]
  ind[:, :max_text] = 0
  return ind
