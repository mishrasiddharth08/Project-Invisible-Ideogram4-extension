"""Routing modern `comfy_quant` checkpoints through Forge's quantised ops.

int8_tensorwise (+convrot), convrot_w4a4, nvfp4, mxfp8 and asym_w4a8_int8 all
carry a per-layer descriptor tensor:

    model.layers.0.self_attn.q_proj.comfy_quant  U8[72]
    -> {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}

The convrot variants rotate the weight by a block-diagonal Hadamard before
quantising, so `weight * scale` does NOT recover the original matrix - the
activation has to be rotated by the same transform. Guessing at that
convention is how a checkpoint ends up loading "successfully" and generating
noise. Forge Neo already ships the real implementation (backend/quant_ops.py
plus the Comfy-Kitchen kernels), so the model is built with ITS Linear and the
file is handed straight to it.

Both the DiT and the Qwen3-VL text encoder use this - the second is why it
lives here rather than in pipeline.py: the text encoder is built by
`transformers`, and importing the pipeline from it would be circular.

Both transformer and text encoder are covered. See BENCHMARKS.md for the
release measurements; format support is not a claim of identical quality.
"""

from __future__ import annotations

import json as _json

_COMFY_SUFFIX = ".comfy_quant"

# Formats our own loaders already read correctly. Routing these through Forge
# regressed them once: `weight_scale` came back as unexpected keys on some
# layers and the load failed outright. Forge's path is the answer for the
# convrot / nvfp4 / mxfp8 family, not a blanket replacement.
_OURS = {"float8_e4m3fn"}


def quant_formats_present(sd: dict) -> set:
  """The distinct `format` values named by the file's own layer descriptors."""
  out = set()
  for k, v in sd.items():
    if not k.endswith(_COMFY_SUFFIX):
      continue
    try:
      conf = _json.loads(bytes(v.tolist()).decode("utf-8").rstrip(chr(0)))
      f = conf.get("format")
      if f:
        out.add(f + ("+convrot" if conf.get("convrot") else ""))
    except Exception as error:
      raise RuntimeError(f'Invalid quantization descriptor: {k}') from error
    if not isinstance(f, str) or not f:
      raise RuntimeError(f'Quantization descriptor has no valid format: {k}')
  return out


def forge_quant_ops(sd: dict, file_meta: dict, device, dtype, *, is_unet: bool = True):
  """(context_manager, label) for a Forge-quantised checkpoint, else None.

  A file may carry per-layer `.comfy_quant` tensors, or only a
  `_quantization_metadata` blob in the header; convert_quantization()
  normalises the second into the first, so both load.

  `is_unet=False` marks a text encoder, which Forge runs with
  full-precision matrix multiply (weights stay quantised in memory, the GEMM
  is done in bf16) - conditioning features are worth the extra milliseconds.

  Returns None - leaving our own fp8 / nf4 / bf16 paths untouched - when Forge
  is absent, when the file is not one of these, or when the format is one we
  already handle.
  """
  # Only formats OUR loaders cannot read justify a warning when Forge is
  # absent. An fp8 file also carries comfy_quant markers, and telling a user
  # running fp8 in a plain script that their checkpoint "may be refused" would
  # be a false alarm.
  _fmts_seen = quant_formats_present(sd)
  looks_quantised = ((any(k.endswith(_COMFY_SUFFIX) for k in sd)
                      or "_quantization_metadata" in (file_meta or {}))
                     and not (_fmts_seen and _fmts_seen.issubset(_OURS)))
  try:
    from backend.operations import using_forge_operations
    from backend.state_dict import convert_quantization, detect_quantization
  except Exception as e:
    # Outside Forge (tests, standalone scripts) this is expected and our own
    # loaders take over. INSIDE Forge it means a file we can read is about to
    # be refused, so say why - a silent fallback here is what made these
    # formats look permanently unsupported.
    if looks_quantised:
      print(f"[Invisible-I4] WARNING: this checkpoint uses a Forge quant format "
            f"but backend.operations could not be imported ({type(e).__name__}: {e}); "
            "this format cannot safely fall back to the portable FP8 loader")
      raise RuntimeError(
        'This quantized checkpoint requires Forge Neo\'s quantization backend. '
        'It could not be imported; update/restart Forge or select portable FP8.'
      ) from e
    return None

  try:
    if not any(k.endswith(_COMFY_SUFFIX) for k in sd):
      if "_quantization_metadata" not in (file_meta or {}):
        return None
      new_sd, _meta = convert_quantization(sd, dict(file_meta))
      # Forge's header-metadata converter commonly updates the SAME dict.
      # Clearing that object here erased every weight it had just converted.
      if new_sd is not sd:
        sd.clear()
        sd.update(new_sd)
    qc = detect_quantization(sd, is_unet=is_unet)
    if not qc:
      if looks_quantised:
        raise RuntimeError('Quantization markers were present but Forge returned no compatible loader.')
      return None
  except Exception as e:
    if looks_quantised:
      raise RuntimeError('Forge could not interpret this checkpoint\'s quantization metadata.') from e
    return None

  fmts = quant_formats_present(sd)
  if fmts and fmts.issubset(_OURS):
    return None
  try:
    from backend.quant_ops import QUANT_ALGOS
  except ImportError as error:
    raise RuntimeError('This Forge build lacks the quantization format registry; update Forge Neo.') from error
  unsupported = {f.split('+')[0] for f in fmts} - set(QUANT_ALGOS)
  if unsupported:
    raise RuntimeError('This Forge build does not support quantization format(s): '
                       + ', '.join(sorted(unsupported)) + '. Update Forge Neo or select portable FP8.')
  label = "quant: " + ", ".join(sorted(fmts)) if fmts else "quant: comfy_quant"
  return using_forge_operations(extra_dtype=dict(qc), device=device, dtype=dtype), label
