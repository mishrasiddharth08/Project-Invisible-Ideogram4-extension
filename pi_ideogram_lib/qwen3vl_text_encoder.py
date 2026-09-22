"""Qwen3-VL-8B text encoder with the official Ideogram 4 13-layer tap.

Uses the installed `transformers` library (already in the Forge venv, v4.57.6)
with the model architecture built from the vendored config, then loads weights
from a local single-file safetensors (fp8_scaled Comfy layout or plain bf16 HF layout).

Tap: hidden states from layers (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35),
concatenated -> (B, seq, 4096*13 = 53248). Tokenization uses the Qwen chat
template from the vendored tokenizer (identical to the official pipeline).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .constants import QWEN3_VL_ACTIVATION_LAYERS
from .forge_quant import forge_quant_ops
from .meta_init import assert_rope_buffers_valid, reinit_rope_buffers
from .quantized_loading import (
  FP8_SCALE_SUFFIX,
  _COMFY_QUANT_SUFFIX,
  swap_linears_to_fp8,
  load_fp8_state_dict,
  is_bnb4bit_state_dict,
  swap_linears_to_bnb4bit,
  load_bnb4bit_state_dict,
)

_THIS_DIR = Path(__file__).resolve().parent
TE_CONFIG_DIR = _THIS_DIR / "qwen3vl_8b_config"


def _build_lock():
    """The lock that guards a globally patched torch.nn.Linear.

    The DiT loader may be building an int8/nvfp4 checkpoint through Forge's
    quantised ops on another thread, and that swaps torch.nn.Linear for the
    whole process. transformers constructs its own Linears here and then calls
    module.weight.data.normal_() on them - which a Forge Linear does not have:

        AttributeError: 'Linear' object has no attribute 'weight'

    Sharing one lock keeps the two builders out of each other's way. Falls
    back to a no-op only if the pipeline module cannot be imported, which
    cannot happen in the running extension.
    """
    try:
        from pi_ideogram_lib.pipeline import _MODEL_BUILD_LOCK
        return _MODEL_BUILD_LOCK
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def _build_model(device: torch.device, dtype: torch.dtype):
  """Build Qwen3VLModel on CPU exactly like the official loader: NO meta device,
  because transformers computes non-persistent buffers (rotary caches) at init -
  meta tensors would later break .to(device). torch_dtype keeps the init small."""
  from transformers import AutoConfig, AutoModel

  cfg = AutoConfig.from_pretrained(str(TE_CONFIG_DIR), trust_remote_code=True)
  try:
    with _build_lock():
        model = AutoModel.from_config(cfg, trust_remote_code=True, torch_dtype=dtype)
  except TypeError:  # older transformers without torch_dtype kwarg
    with _build_lock():
        model = AutoModel.from_config(cfg, trust_remote_code=True)
  return model


def _build_model_meta(device: torch.device, dtype: torch.dtype):
  """SPEED variant: construct on meta (no random init of ~16 GB of weights),
  then allocate real storage WITHOUT initialization via to_empty(device='cpu').

  Every persistent buffer the model needs comes from the safetensors file
  (assigned via load_state_dict). The rotary caches do NOT: transformers marks
  inv_freq non-persistent, so no checkpoint contains it and to_empty() hands it
  uninitialized memory - it is NOT re-derived lazily at forward time. We
  therefore re-derive it explicitly here, and the caller verifies it before
  using the model. The untrusted path stays guarded: if anything is still meta
  after loading, we rebuild via _build_model (correctness over speed)."""
  from transformers import AutoConfig, AutoModel

  cfg = AutoConfig.from_pretrained(str(TE_CONFIG_DIR), trust_remote_code=True)
  with torch.device("meta"):
    with _build_lock():
        model = AutoModel.from_config(cfg, trust_remote_code=True)
  model.to_empty(device="cpu")
  reinit_rope_buffers(model)
  return model


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
  return load_file(str(path))


def _load_state_dict_with_meta(path: Path):
  """Weights plus the safetensors header metadata.

  Accepts EITHER a single file or a Hugging Face encoder FOLDER. The folder
  case exists because Forge's module dropdown lists single .safetensors files
  only, while every abliterated / uncensored Qwen3-VL-8B is published as a
  sharded repo (config.json + model.safetensors.index.json +
  model-0000N-of-0000M.safetensors). Point `text_encoder_override` in
  config.json at the folder and it loads like any other encoder.

  Some quantised files (the Star converter's int8_convrot output among them)
  describe themselves ONLY in a `_quantization_metadata` header blob rather
  than per-layer tensors, and Forge's convert_quantization() needs it.
  """
  p = Path(path)
  if p.is_dir():
    from .detect import te_shard_files
    shards = te_shard_files(p)
    if not shards:
      raise RuntimeError(
        f"{p} holds no .safetensors shards - a Hugging Face encoder folder needs "
        "model.safetensors.index.json and its model-0000N-of-0000M shards")
    sd: dict[str, torch.Tensor] = {}
    meta: dict = {}
    for i, shard in enumerate(shards, 1):
      print(f"[Invisible-I4]   shard {i}/{len(shards)}: {shard.name}")
      try:
        from safetensors import safe_open
        with safe_open(str(shard), framework="pt") as f:
          meta.update(dict(f.metadata() or {}))
      except Exception:
        pass
      sd.update(load_file(str(shard)))
    return sd, meta
  try:
    from safetensors import safe_open
    with safe_open(str(p), framework="pt") as f:
      meta = dict(f.metadata() or {})
  except Exception:
    meta = {}
  return load_file(str(p)), meta


def _strip_comfy_markers(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  return {k: v for k, v in sd.items() if not k.endswith(_COMFY_QUANT_SUFFIX)}


def _is_fp8_te(sd: dict[str, torch.Tensor]) -> bool:
  """True only for a genuinely FP8 file.

  This used to answer "does any key end in .weight_scale?", which is also true
  of int8_tensorwise, nvfp4, mxfp8 and the whole convrot family. An
  int8_convrot encoder was therefore swapped to Fp8Linear and died on the
  first buffer:

      size mismatch for language_model.layers.0.self_attn.q_proj.weight_scale:
      copying a param with shape torch.Size([4096, 1]) ... is torch.Size([4096])

  - and had it not, dequantising a Hadamard-rotated int8 weight as if it were
  fp8 would have produced conditioning for a prompt nobody typed. Those
  formats are recognised by their `comfy_quant` descriptor and go to Forge
  (see forge_quant.py) before this is ever asked.
  """
  if any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd.values()):
    return True
  return (any(k.endswith(FP8_SCALE_SUFFIX) for k in sd)
          and not any(k.endswith(_COMFY_QUANT_SUFFIX) for k in sd))


# llama.cpp -> HF, per decoder block. Verified against
# Qwen3VL-8B-Uncensored-HauhauCS-Aggressive-int8_convrot.safetensors and the
# shapes transformers' Qwen3VLModel expects - every pair below was checked,
# not recalled:
#     blk.0.attn_q      [4096, 4096]  -> self_attn.q_proj      (4096, 4096)
#     blk.0.attn_k      [1024, 4096]  -> self_attn.k_proj      (1024, 4096)
#     blk.0.attn_k_norm [128]         -> self_attn.k_norm      (128,)
#     blk.0.attn_norm   [4096]        -> input_layernorm       (4096,)
#     blk.0.ffn_down    [4096, 12288] -> mlp.down_proj         (4096, 12288)
_GGUF_BLOCK_MAP = {
  "attn_q": "self_attn.q_proj",
  "attn_k": "self_attn.k_proj",
  "attn_v": "self_attn.v_proj",
  "attn_output": "self_attn.o_proj",
  "attn_q_norm": "self_attn.q_norm",
  "attn_k_norm": "self_attn.k_norm",
  "attn_norm": "input_layernorm",
  "ffn_norm": "post_attention_layernorm",
  "ffn_gate": "mlp.gate_proj",
  "ffn_up": "mlp.up_proj",
  "ffn_down": "mlp.down_proj",
}
_GGUF_GLOBAL_MAP = {
  "token_embd": "language_model.embed_tokens",
  "output_norm": "language_model.norm",
}


def _is_gguf_layout(sd: dict) -> bool:
  """Does this state dict use llama.cpp tensor names?

  Converting a GGUF encoder to safetensors (Forge Neo's converter does exactly
  this, and its int8_convrot output is a perfectly good encoder) keeps the
  llama.cpp names. Nothing downstream recognised them, so the file was
  rejected as 'NOT Qwen3-VL-8B' despite being precisely that.
  """
  return "token_embd.weight" in sd or any(k.startswith("blk.") for k in sd)


def _remap_gguf_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  """llama.cpp names -> the names transformers' Qwen3VLModel expects.

  `output.weight` (the LM head) is dropped: Ideogram 4 taps hidden states and
  never decodes a token, so carrying a 151936x4096 matrix would cost ~600 MB
  to compute nothing.

  Quantisation siblings (.weight_scale, .comfy_quant) travel with their weight
  - they are what tells Forge's Linear its format, and a scale left behind
  under the old name would load as an unexpected key.
  """
  out: dict[str, torch.Tensor] = {}
  for k, v in sd.items():
    # Preserve EVERY sibling, including NVFP4 secondary scales, activation
    # scales and packed bitsandbytes quant_state. A fixed suffix list loses
    # information as formats evolve. The layer name, not its suffix, maps.
    parts = k.split(".")
    stem_parts = 3 if k.startswith("blk.") else 1
    stem = ".".join(parts[:stem_parts])
    tail_kept = k[len(stem):]
    if stem == "output":
      continue  # LM head: never used, see docstring
    if stem in _GGUF_GLOBAL_MAP:
      out[_GGUF_GLOBAL_MAP[stem] + tail_kept] = v
      continue
    if stem.startswith("blk."):
      parts = stem.split(".", 2)
      if len(parts) == 3 and parts[1].isdigit() and parts[2] in _GGUF_BLOCK_MAP:
        out[f"language_model.layers.{parts[1]}.{_GGUF_BLOCK_MAP[parts[2]]}{tail_kept}"] = v
        continue
    out[k] = v  # unknown: pass through so it shows up as an unexpected key
  return out


def _remap_hf_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  """Normalize key layouts to what transformers' Qwen3VLModel expects:
  language_model.*, visual.* (+ optional lm_head which we drop - not needed for taps).

  Order matters: 'model.visual.' must be remapped BEFORE the generic 'model.'
  prefix, otherwise visual keys would land under language_model.
  """
  if _is_gguf_layout(sd):
    return _remap_gguf_keys(sd)
  has_lm = any(k.startswith("language_model.") for k in sd)
  if has_lm:
    out = {k: v for k, v in sd.items() if not k.startswith("lm_head.")}
    return out
  out: dict[str, torch.Tensor] = {}
  for k, v in sd.items():
    if k == "lm_head.weight" or k.startswith("lm_head."):
      continue  # not needed: we tap hidden states, never decode tokens
    if k.startswith("model.visual."):
      out["visual." + k[len("model.visual."):]] = v
    elif k.startswith("model."):
      out["language_model." + k[len("model."):]] = v
    elif k.startswith("visual."):
      out[k] = v
    else:
      out[k] = v
  return out


class Qwen3VLTextEncoder:
  """Wraps transformers' Qwen3VLModel; returns official Ideogram 4 conditioning features."""

  def __init__(self, weights_path: str | Path, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    self.device = device
    self.dtype = dtype
    self.model = None
    self.tokenizer = None
    self._weights_path = Path(weights_path)

  def load(self) -> None:
    from transformers import AutoTokenizer

    print(f"[Invisible-I4] Loading text encoder: {self._weights_path.name}")
    sd, file_meta = _load_state_dict_with_meta(self._weights_path)

    # MODERN QUANT FORMATS FIRST. int8_tensorwise (+convrot), nvfp4, mxfp8 and
    # the rest carry per-layer `comfy_quant` descriptors that only Forge can
    # read; none of them survive our fp8 path. See forge_quant.py.
    forge = forge_quant_ops(sd, file_meta, self.device, self.dtype, is_unet=False)
    # Header-only descriptors name the original weight keys. Materialize
    # their marker tensors first, then remap weights AND markers together.
    sd = _remap_hf_keys(sd)
    if forge is not None:
      ctx, label = forge
      print(f"[Invisible-I4] text encoder {label} -> loading through Forge's quantised ops")
      self.model = self._load_through_forge(sd, ctx)
      self.tokenizer = AutoTokenizer.from_pretrained(str(TE_CONFIG_DIR))
      print("[Invisible-I4] Text encoder ready")
      return

    sd = _strip_comfy_markers(sd)
    if is_bnb4bit_state_dict(sd):
      if not torch.cuda.is_available():
        raise RuntimeError('NF4 text encoding requires compatible CUDA/bitsandbytes; select portable FP8.')
      model = _build_model_meta(self.device, self.dtype)
      # Ideogram consumes the language tower only. Do not demand or allocate
      # an unused vision tower in language-only quantized exports.
      model.visual = None
      sd = {k: v for k, v in sd.items() if not k.startswith('visual.')}
      swap_linears_to_bnb4bit(model, self.dtype, state_dict=sd)
      load_bnb4bit_state_dict(model, sd, device=self.device, dtype=self.dtype)
      reinit_rope_buffers(model)
      assert_rope_buffers_valid(model, what='text encoder (NF4)')
      self.model = model.eval()
      self.tokenizer = AutoTokenizer.from_pretrained(str(TE_CONFIG_DIR))
      print('[Invisible-I4] NF4 text encoder ready')
      return
    is_fp8 = _is_fp8_te(sd)

    model = None
    try:
      # fast path: meta construction, no random init, direct assignment
      model = _build_model_meta(self.device, self.dtype)
      if is_fp8:
        swap_linears_to_fp8(model, sd, self.dtype)
        load_fp8_state_dict(model, sd, device=self.device, dtype=self.dtype, assign=True, strict=False,
                            required_prefix='language_model.')
      else:
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        critical_missing = [m for m in missing if m.startswith("language_model.")]
        if critical_missing:
          raise RuntimeError(f"Text encoder file is missing language-model weights: {critical_missing[:5]}")
      # correctness guard: if any persistent tensor is still on meta, the fast
      # path was invalid for this transformers build - fall back to the slow,
      # official loader (weights win over speed, always)
      bad = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
             if t.is_meta and "rotary" not in n]
      if bad:
        raise RuntimeError(f"meta remained after load ({bad[:3]}); falling back")
      # 'rotary' is excluded above because those buffers are non-persistent and
      # never in the file - so they need their own, stricter check: garbage here
      # silently destroys every position embedding.
      reinit_rope_buffers(model)
      assert_rope_buffers_valid(model, what="text encoder")
      if is_fp8:
        model.to(device=self.device)  # fp8 buffers + scales already placed; just ensure device
      else:
        model.to(device=self.device, dtype=self.dtype)
    except Exception as e:
      print(f"[Invisible-I4] fast TE load unavailable ({type(e).__name__}); using official loader")
      model = _build_model(self.device, self.dtype)
      if is_fp8:
        swap_linears_to_fp8(model, sd, self.dtype)
        load_fp8_state_dict(model, sd, device=self.device, dtype=self.dtype, assign=True, strict=False,
                            required_prefix='language_model.')
      else:
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        critical_missing = [m for m in missing if m.startswith("language_model.")]
        if critical_missing:
          raise RuntimeError(f"Text encoder file is missing language-model weights: {critical_missing[:5]}")
        model.to(device=self.device, dtype=self.dtype)
      assert_rope_buffers_valid(model, what="text encoder (official loader)")

    model.eval()
    self.model = model
    self.tokenizer = AutoTokenizer.from_pretrained(str(TE_CONFIG_DIR))
    print("[Invisible-I4] Text encoder ready")

  def _load_through_forge(self, sd: dict[str, torch.Tensor], ctx):
    """Build Qwen3VLModel with Forge's quantised Linear and load `sd` into it.

    Two things make this different from the fp8 path:

    * `no_init_weights()` is mandatory. Forge's Linear creates no `.weight` at
      construction - the weight is materialised from the file - so
      transformers' `_init_weights`, which does `module.weight.data.normal_()`,
      raises `AttributeError: 'Linear' object has no attribute 'weight'`.
    * the `comfy_quant` markers must stay IN the state dict. They are what
      tells each Linear its format, groupsize and scale layout.

    _build_lock() is shared with the DiT loader because `using_forge_operations`
    replaces torch.nn.Linear for the whole process, and these two models are
    built on parallel threads.
    """
    from transformers import AutoConfig, AutoModel
    from transformers.modeling_utils import no_init_weights

    cfg = AutoConfig.from_pretrained(str(TE_CONFIG_DIR), trust_remote_code=True)
    with _build_lock():
      with no_init_weights():
        with ctx:
          with torch.device("meta"):
            model = AutoModel.from_config(cfg, trust_remote_code=True)

    model.to_empty(device="cpu")
    reinit_rope_buffers(model)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    critical = [m for m in missing if m.startswith("language_model.")]
    if critical:
      raise RuntimeError(f"Text encoder file is missing language-model weights: {critical[:5]}")
    unexpected = [k for k in unexpected if not k.endswith(_COMFY_QUANT_SUFFIX)]
    if unexpected:
      raise RuntimeError(
        f"{self._weights_path.name} has keys this Qwen3-VL build does not: {unexpected[:5]}")

    # A GGUF-derived encoder holds the LANGUAGE tower only - llama.cpp keeps
    # the vision tower in a separate mmproj file. Ideogram 4 taps
    # language_model hidden states and never calls visual.*, so those staying
    # unloaded is correct, not a failure. The language tower is still required
    # in full: that is what actually encodes the prompt.
    _no_vision = not any(k.startswith("visual.") for k in sd)
    still_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                  if t.is_meta and "rotary" not in n
                  and not (_no_vision and n.startswith("visual."))]
    if still_meta:
      raise RuntimeError(f"text encoder weights did not load ({still_meta[:3]})")
    if _no_vision:
      # Drop the unloaded vision tower rather than carry ~1 GB of meta/garbage
      # parameters onto the GPU behind a module that is never called.
      try:
        model.visual = None
        print("[Invisible-I4] text encoder has no vision tower (GGUF layout) - "
              "dropped it; Ideogram 4 taps the language model only")
      except Exception:
        pass
    # non-persistent, never in any file, and to_empty() left them as garbage
    reinit_rope_buffers(model)
    assert_rope_buffers_valid(model, what="text encoder (Forge quant ops)")

    model.to(device=self.device)  # device only: casting would undo the quantisation
    model.eval()
    return model

  def tokenize(self, prompt: str):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = self.tokenizer.apply_chat_template(
      messages, add_generation_prompt=True, tokenize=False
    )
    return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)

  @torch.no_grad()
  def encode_features(self, prompt: str, max_text_tokens: int = 2048) -> torch.Tensor:
    """Return (1, num_text_tokens, 53248) float32 features (no left-padding here;
    the pipeline packs batches)."""
    enc = self.tokenize(prompt)
    token_ids = enc["input_ids"][0]
    num_text = int(token_ids.shape[0])
    if num_text > max_text_tokens:
      raise ValueError(
        f"Prompt is {num_text} tokens, exceeds the model's 2048-token text window. "
        "Shorten the prompt or trim optional JSON fields."
      )
    lm = self.model.language_model
    input_ids = token_ids.unsqueeze(0).to(self.device)
    attn_mask = torch.ones_like(input_ids)
    # (B, L). The three MRoPE axes are produced by the expand() below, not by
    # stacking here - stacking gave (1, L, 3), whose expand made cos/sin of
    # width 3 and blew up in apply_rotary_pos_emb ("size of tensor a (31) must
    # match tensor b (3)"). pipeline._encode_text_batch has always had this
    # right; this helper drifted because nothing calls it.
    pos_2d = torch.arange(num_text, device=self.device).unsqueeze(0)

    from transformers.masking_utils import create_causal_mask

    inputs_embeds = lm.embed_tokens(input_ids)
    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]
    causal_mask = create_causal_mask(
      config=lm.config, input_embeds=inputs_embeds, attention_mask=attn_mask,
      cache_position=torch.arange(num_text, device=self.device),
      past_key_values=None, position_ids=text_position_ids,
    )
    position_embeddings = lm.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(lm.layers):
      hidden_states = decoder_layer(
        hidden_states,
        attention_mask=causal_mask,
        position_ids=text_position_ids,
        past_key_values=None,
        position_embeddings=position_embeddings,
      )
      if layer_idx in tap_set:
        captured[layer_idx] = hidden_states
    selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
    stacked = torch.stack(selected, dim=0)  # (num_taps, B, L, H)
    stacked = torch.permute(stacked, (1, 2, 3, 0))  # (B, L, H, num_taps)
    feats = stacked.reshape(1, num_text, -1)  # (B, L, 4096*13)
    return feats.to(torch.float32)

  def unload(self) -> None:
    self.model = None
    self.tokenizer = None
    import gc

    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
