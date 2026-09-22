"""PROJECT INVISIBLE - native Ideogram 4 LoRA / LoKr handling.

Mandate: LoRAs and LoKrs trained on Ideogram 4 must work seamlessly with ANY
quantization of the checkpoint. LoKr must stay real LoKr (Kronecker
factorization, lokr_factor / alpha honored) - never flattened into a fake LoRA.

Adapters are CLASSIFIED before touching any weight (spec section H):

  kind                 | trunk policy         | recipe
  ---------------------|----------------------|----------------------------------
  style / character    | DUAL (cond + uncond) | merged at Extra Networks weight
  turbo_time           | cond only            | distill: 2-8 steps, CFG 1.0
  uncond_replacement   | uncond pass only     | replaces the 9.3B uncond transformer
  gray_bypass          | first-step only      | negative strength, removed after step 1
  incompatible         | refused              | never touches Ideogram 4 weights

The merge is quant-aware - for every LoRA-touched layer we
  - bf16 nn.Linear      -> delta added directly
  - Fp8Linear (fp8 + per-row scale) -> dequantize -> add delta -> requantize
    (new per-row scale computed from the merged weights)
  - bnb Linear4bit / other exotic quant modules -> NOT merged in-process today;
    those layers log a 'skip' (load the fp8/bf16 file of the same trunk to
    apply LoRAs on an NF4 profile)

merge_lora_undoable() returns an undo closure so first-step-only adapters
(gray bypass) can be removed after step 1 without reloading the model.
"""

from __future__ import annotations

import math
import json
import re
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class LoraRefused(Exception):
  pass


FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn


# --------------------------------------------------------------------------- #
# file inspection (header-only, cheap)
# --------------------------------------------------------------------------- #

def read_safetensors_header(path: Path) -> dict:
  with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n))


_IDEOGRAM4_KEY_PATTERNS = [
  r"^layers\.\d+\.attention\.",
  r"^layers\.\d+\.feed_forward\.",
  r"^layers\.\d+\.attention_norm[12]\.",
  r"^layers\.\d+\.ffn_norm[12]\.",
  r"^layers\.\d+\.adaln_modulation\.",
  r"^input_proj\.",
  r"^llm_cond_(norm|proj)\.",
  r"^t_embedding\.",
  r"^adaln_proj\.",
  r"^embed_image_indicator\.",
  r"^final_layer\.",
]

_FOREIGN_MARKERS = {
  "sd15": (r"^lora_te\.", r"^te\."),
  "sdxl": (r"^lora_te1\.", r"^lora_te2\.", r"^transformer_blocks\."),
  "flux": (r"^double_blocks\.", r"^single_blocks\.", r"^transformer_blocks\."),
  "flux2/klein": (r"^blocks\.\d+\.attn\.", r"^blocks\.\d+\.mlp\.", r"^txt_attn\.", r"^txt_mlp\."),
  "wan": (r"^blocks\.\d+\.", r"^text_blocks\."),
  "qwen-image": (r"^transformer\.single_transformer_blocks\."),
}

_IDEOGRAM4_HINTS = ("ideogram4", "ideogram-4", "ideogram_4", "ideogram 4", "flowmatch", "attention_mlp")

# key fragments that identify a module we can merge onto (post-normalization)
_MERGEABLE_HINTS = ("layers.", "input_proj", "llm_cond_", "t_embedding", "adaln_proj", "embed_image_indicator", "final_layer")


def inspect_lora(path: str | Path) -> dict:
  """Header-based inspection: is this an Ideogram 4 LoRA/LoKr? Why / why not?"""
  path = Path(path)
  header = read_safetensors_header(path)
  keys = [k for k in header.keys() if k != "__metadata__"]
  meta = " ".join(str(v) for v in (header.get("__metadata__") or {}).values()).lower() if isinstance(header.get("__metadata__"), dict) else ""
  name = path.stem.lower()

  if any(h in meta for h in _IDEOGRAM4_HINTS) or "ideogram" in name:
    return {"is_ideogram4": True, "reason": "name/metadata mentions Ideogram 4", "path": str(path)}

  hits = 0
  for k in keys:
    for pat in _IDEOGRAM4_KEY_PATTERNS:
      if re.match(pat, k):
        hits += 1
        break
  if hits >= 4:
    return {"is_ideogram4": True, "reason": f"{hits} keys match the official Ideogram 4 DiT layout", "path": str(path)}

  # LoKr keys still reference the same modules (w1/w2 hang off DiT module paths)
  kron_hits = sum(1 for k in keys if re.search(r"\.w[12](_[ab])?$|lora_kronecker", k))
  if kron_hits and any(h in k for k in keys for h in _MERGEABLE_HINTS):
    return {"is_ideogram4": True, "reason": f"LoKr keys target official Ideogram 4 modules ({kron_hits} kron keys)", "path": str(path)}

  for family, pats in _FOREIGN_MARKERS.items():
    for k in keys[:200]:
      for pat in pats:
        if re.match(pat, k):
          return {"is_ideogram4": False, "reason": f"keys match the {family} architecture, not Ideogram 4", "path": str(path)}

  return {"is_ideogram4": False, "reason": "no Ideogram 4 signature found in keys or metadata", "path": str(path)}


# --------------------------------------------------------------------------- #
# ADAPTER CLASSIFICATION (spec section H)
# --------------------------------------------------------------------------- #

def _adapter_config(path: Path) -> dict:
  """adapter_config.json / <stem>.json next to the file. Never raises."""
  for cfg_name in ("adapter_config.json", f"{path.stem}.json"):
    cfg = path.parent / cfg_name
    try:
      data = json.loads(cfg.read_text(encoding="utf-8"))
      if isinstance(data, dict):
        return data
    except Exception:
      continue
  return {}


def _tokens(path: Path, header: dict) -> set[str]:
  toks = set(re.sub(r"[^a-z0-9]", " ", path.stem.lower()).split())
  meta = header.get("__metadata__") or {}
  if isinstance(meta, dict):
    for k, v in meta.items():
      if k.lower() in ("ss_base_model", "ss_title", "ss_network_module", "title", "models_name"):
        toks.update(re.sub(r"[^a-z0-9]", " ", str(v).lower()).split())
  return toks


def classify_adapter(path: str | Path) -> dict:
  """-> {kind, trunk, reason}. kind: style | turbo_time | uncond_replacement |
  gray_bypass | incompatible. Never raises."""
  path = Path(path)
  try:
    header = read_safetensors_header(path)
  except Exception:
    return {"kind": "incompatible", "trunk": "none",
            "reason": "unreadable safetensors header", "path": str(path)}
  toks = _tokens(path, header)
  cfg = _adapter_config(path)

  # explicit override wins (adapter_config.json or metadata)
  override = str(cfg.get("i4_adapter_kind", "") or "").strip().lower()
  if not override:
    meta = header.get("__metadata__") or {}
    if isinstance(meta, dict):
      override = str(meta.get("i4_adapter_kind", "")).strip().lower()

  if override == "gray_bypass" or override == "gray":
    return {"kind": "gray_bypass", "trunk": "gray_first_step", "reason": "explicit i4_adapter_kind=gray_bypass", "path": str(path)}
  if override == "turbo_time":
    return {"kind": "turbo_time", "trunk": "cond", "reason": "explicit i4_adapter_kind=turbo_time", "path": str(path)}
  if override == "uncond_replacement":
    return {"kind": "uncond_replacement", "trunk": "uncond_replace", "reason": "explicit i4_adapter_kind=uncond_replacement", "path": str(path)}
  if override == "style":
    return {"kind": "style", "trunk": "dual", "reason": "explicit i4_adapter_kind=style", "path": str(path)}

  if not inspect_lora(path)["is_ideogram4"]:
    return {"kind": "incompatible", "trunk": "none",
            "reason": "not an Ideogram 4 LoRA/LoKr (foreign base model)", "path": str(path)}

  name_l = path.stem.lower()
  joined = re.sub(r"[^a-z0-9]", "", name_l)
  if "gray" in toks or "bypass" in toks or "gray" in name_l or "bypass" in name_l or "000002000" in name_l:
    return {"kind": "gray_bypass", "trunk": "gray_first_step",
            "reason": "filename marks the gray-screen bypass LoRA", "path": str(path)}
  if "turbotime" in toks or "turbo_time" in name_l or "turbotime" in name_l:
    return {"kind": "turbo_time", "trunk": "cond",
            "reason": "filename/metadata marks the Ostris TurboTime distill LoRA", "path": str(path)}
  # `joined` has every separator stripped, so match the stripped spelling -
  # "unconditional_lora" could never occur in it and the check silently fell
  # through to the narrower filename clause below.
  if "unconditionallora" in joined or ({"unconditional", "lora"} <= toks and "ideogram_4_unconditional" in name_l):
    return {"kind": "uncond_replacement", "trunk": "uncond_replace",
            "reason": "filename marks the Ostris unconditional LoRA (replaces the 9.3B uncond transformer)",
            "path": str(path)}

  return {"kind": "style", "trunk": "dual", "reason": "native Ideogram 4 style/character LoRA", "path": str(path)}


def trunk_policy(path: str | Path) -> str:
  """'dual' | 'cond' | 'uncond_replace' | 'gray_first_step'. Classification is
  authoritative; explicit metadata overrides filename guessing."""
  return classify_adapter(path)["trunk"]


# Step counts a distill adapter is trained for, read from the file rather than
# assumed. ostris/TurboTime ships ss_output_name "ideogram_turbo_8_v1" and its
# card says "as few as 2 steps", samples shown at 8.
_STEP_HINT_RE = re.compile(r"(?:^|[^0-9])(\d{1,2})\s*step|turbo[_-]?(\d{1,2})|[_-](\d{1,2})steps?")

DISTILL_DEFAULT_STEPS = 8
DISTILL_STEP_BOUNDS = (2, 12)


def adapter_step_hint(path: str | Path) -> int | None:
    """How many steps this distill adapter expects, or None.

    Looked up in the adapter's own metadata and filename - never guessed - so
    selecting a turbo LoRA can set the step count the way its author intended
    instead of leaving the user on the 20-step default it was never meant for.
    """
    path = Path(path)
    blobs: list[str] = [path.stem.lower()]
    try:
        meta = read_safetensors_header(path).get("__metadata__") or {}
        if isinstance(meta, dict):
            for key in ("ss_output_name", "ss_base_model_version", "training_info", "version"):
                val = meta.get(key)
                if val:
                    blobs.append(str(val).lower())
    except Exception:
        pass
    for blob in blobs:
        m = _STEP_HINT_RE.search(blob)
        if not m:
            continue
        for grp in m.groups():
            if not grp:
                continue
            try:
                n = int(grp)
            except ValueError:
                continue
            lo, hi = DISTILL_STEP_BOUNDS
            if lo <= n <= hi:
                return n
    return None


def lora_wants_uncond(path: str | Path) -> bool:
  """Back-compat shim: True when the adapter asks for the uncond trunk as well
  (dual policy). style -> True (dual by default), uncond_replace/gray -> True
  handled separately, turbo -> False."""
  return trunk_policy(path) in ("dual", "uncond_replace", "gray_first_step")


def load_lora_state_dict(path: str | Path) -> dict:
  return load_file(str(path))


# --------------------------------------------------------------------------- #
# key normalization + module grouping (standard LoRA + LoKr)
# --------------------------------------------------------------------------- #

# Longest-first: the loop below breaks on the first match, so a longer prefix
# that starts with a shorter one MUST come first.
#
# "diffusion_model." is the prefix ai-toolkit writes, and ai-toolkit is what
# every published Ideogram 4 adapter is trained with - ostris/TurboTime,
# ostris/Unconditional and the Civitai gray-screen bypass all ship keys like
#   diffusion_model.layers.0.attention.qkv.lora_A.weight
# while the DiT itself calls that parameter
#   layers.0.attention.qkv.weight
# Without this entry every real Ideogram 4 LoRA normalizes to a module path
# that matches nothing and is refused as "trained for a different model".
_PREFIXES = (
  "base_model.model.diffusion_model.",
  "model.diffusion_model.",
  "base_model.model.",
  "base_model.",
  "diffusion_model.",
  "transformer.",
  "lora_unet_",
  "lora.",
  "lokr.",
  "model.",
  "module.",
  "unet.",
)

# Same list, used by the run-time resolver to retry a path that did not match
# on the first pass (an exporter may stack prefixes we did not enumerate).
_STRIPPABLE = tuple(p for p in _PREFIXES if p.endswith("."))

_SIDE_TOKENS = [
  ("lora_a", "A"), ("lora_a.weight", "A"),
  ("lora_b", "B"), ("lora_b.weight", "B"),
  ("lora_down", "A"), ("lora_down.weight", "A"), ("down", "A"), ("down.weight", "A"),
  ("lora_up", "B"), ("lora_up.weight", "B"), ("up", "B"), ("up.weight", "B"),
  # LoKr, LyCORIS / ai-toolkit spelling. These carry a "lokr_" prefix on the
  # LEAF name, which the bare "w1"/"w2" entries below cannot match: the check
  # is endswith("." + token), and "...qkv.lokr_w1" does not end in ".w1".
  # Without these every LyCORIS LoKr file was refused outright as "no
  # recognizable LoRA/LoKr weight pairs", even though the Kronecker math
  # underneath was correct. Longest spellings first.
  ("lokr_w1_a", "W1A"), ("lokr_w1_b", "W1B"),
  ("lokr_w2_a", "W2A"), ("lokr_w2_b", "W2B"),
  ("lokr_w1", "W1"), ("lokr_w2", "W2"),
  ("lokr_t1", "T1"), ("lokr_t2", "T2"),
  ("w1_a", "W1A"), ("w1_a.weight", "W1A"),
  ("w1_b", "W1B"), ("w1_b.weight", "W1B"),
  ("w1", "W1"), ("w1.weight", "W1"),
  ("w2_a", "W2A"), ("w2_a.weight", "W2A"),
  ("w2_b", "W2B"), ("w2_b.weight", "W2B"),
  ("w2", "W2"), ("w2.weight", "W2"),
  ("alpha", "ALPHA"), ("alpha.weight", "ALPHA"),
]


def _normalize_key(k: str) -> tuple[str, str] | None:
  """-> (module_path, side) or None if not a LoRA/LoKr tensor key."""
  nk = k
  for p in _PREFIXES:
    if nk.startswith(p):
      nk = nk[len(p):]
      break
  nk = nk[:-len(".weight")] if nk.endswith(".weight") else nk
  lowered = nk.lower()
  for token, side in _SIDE_TOKENS:
    if lowered.endswith("." + token) or lowered == token:
      module = nk[: len(nk) - len(token) - 1] if len(nk) > len(token) + 1 else ""
      return module, side
  return None


def resolve_module_path(module_path: str, known: set) -> str | None:
  """Map a LoRA's module path onto a real module path of the target model.

  Tried in order: the path as-is, then with each known wrapper prefix peeled
  off (repeatedly, so stacked prefixes resolve), then a unique suffix match.
  The suffix pass is what makes this exporter-agnostic: any wrapper naming we
  have never seen still lands as long as the tail is the model's own path.
  """
  if module_path in known:
    return module_path
  cur = module_path
  for _ in range(4):  # bounded: stacked prefixes in the wild are 1-2 deep
    peeled = None
    for pref in _STRIPPABLE:
      if cur.startswith(pref):
        peeled = cur[len(pref):]
        break
    if peeled is None:
      break
    cur = peeled
    if cur in known:
      return cur
  # last resort: unique suffix match ('<anything>.layers.0.attention.qkv')
  tail = cur or module_path
  matches = [k for k in known if k == tail or k.endswith("." + tail)]
  if len(matches) == 1:
    return matches[0]
  return None


def _group_modules(sd: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
  groups: dict[str, dict[str, torch.Tensor]] = {}
  for k, v in sd.items():
    parsed = _normalize_key(k)
    if parsed is None:
      continue
    module, side = parsed
    groups.setdefault(module, {})[side] = v
  return groups


# A LoRA alpha is a RANK-LIKE quantity: conventionally <= the adapter's own
# dim, so alpha/rank lands somewhere around 1/64 .. 64. LyCORIS trainers that
# want no alpha scaling at all write a sentinel instead of a real value - the
# Ideogram 4 character LoKr in the wild carries alpha = 9,999,220,736 on every
# one of its 204 modules.
#
# Taken literally that is a scale of 6.25e8, and the measured result is exactly
# what it sounds like:
#
#   unscaled kron delta   absmax 0.0071
#   alpha-scaled delta    absmax 4,455,100
#   base qkv weights      ~0.02 - 0.05
#
# The adapter did not "fail to show up"; it destroyed every layer it touched,
# then fp8 saturation at 448 finished the job. A scale this size is never a
# real training choice, so treat it as the sentinel it is.
MAX_PLAUSIBLE_ALPHA_SCALE = 100.0

_ALPHA_WARNED: set[int] = set()


def _alpha_scale(sides: dict[str, torch.Tensor], rank: int) -> float:
  alpha = sides.get("ALPHA")
  if alpha is None:
    return 1.0
  try:
    a = float(alpha.reshape(-1)[0])
  except Exception:
    return 1.0
  if not math.isfinite(a) or a <= 0.0:
    return 1.0
  scale = a / max(1, rank)
  if scale > MAX_PLAUSIBLE_ALPHA_SCALE:
    # Sentinel, not a scale. Say so once per adapter rather than per module -
    # 204 identical lines would bury it.
    key = int(a)
    if key not in _ALPHA_WARNED:
      _ALPHA_WARNED.add(key)
      print(f"[Invisible-I4] adapter alpha {a:.6g} implies a x{scale:.3g} scale, "
            "which no real adapter uses - treating it as LyCORIS's "
            "'no alpha scaling' sentinel and applying at x1.0")
    return 1.0
  return scale


def _module_delta(module: str, sides: dict[str, torch.Tensor], factor: float | None = None) -> torch.Tensor | None:
  """Compute the un-scaled LoRA or LoKr delta for one module. Returns None if
  the format is unrecognizable. `factor` is the adapter_config lokr_factor
  applied on top of the tensor-level alpha scaling."""
  A, B = sides.get("A"), sides.get("B")

  # ---- standard LoRA pair ----
  if A is not None and B is not None:
    if A.ndim != 2 or B.ndim != 2:
      return None
    rank = min(A.shape[0], B.shape[0])
    scale = _alpha_scale(sides, rank)
    return (B.float() @ A.float()) * scale

  # ---- LoKr: kron(w1, w2), with optional factorizations ----
  def _factor(a, b):
    if a is not None and b is not None:
      return b.float() @ a.float()
    return a.float() if a is not None else (b.float() if b is not None else None)

  # CP decomposition (lokr_t1 / lokr_t2) reconstructs w through a third,
  # tucker-style core tensor. Folding it into a plain kron(w1, w2) would
  # produce a confidently WRONG delta, so refuse and let the caller report it.
  if sides.get("T1") is not None or sides.get("T2") is not None:
    return None

  w1 = _factor(sides.get("W1A"), sides.get("W1B")) if (sides.get("W1A") is not None or sides.get("W1B") is not None) else _factor(sides.get("W1"), None)
  w2 = _factor(sides.get("W2A"), sides.get("W2B")) if (sides.get("W2A") is not None or sides.get("W2B") is not None) else _factor(sides.get("W2"), None)
  if w1 is None or w2 is None or w1.ndim != 2 or w2.ndim != 2:
    return None

  scale = _alpha_scale(sides, max(1, min(w1.shape[-1], w2.shape[-1])))
  if factor:
    scale *= float(factor)
  kron = torch.kron(w1, w2) * scale
  if kron.ndim == 2 and kron.numel() > 0:
    return kron
  return None


# --------------------------------------------------------------------------- #
# quant-aware weight merge
# --------------------------------------------------------------------------- #

def _merge_fp8_linear(mod: nn.Module, delta: torch.Tensor) -> bool:
  """Dequantize -> merge -> requantize an Fp8Linear (fp8 weight + per-row scale)."""
  if not (hasattr(mod, "weight_scale") and mod.weight.dtype == FP8_WEIGHT_DTYPE):
    return False
  w = mod.weight.to(torch.float32) * mod.weight_scale.to(torch.float32).unsqueeze(1)
  w = w + delta.to(torch.float32).to(w.device)
  new_scale = w.abs().amax(dim=1).clamp(min=1e-12) / FP8_E4M3_MAX
  q = (w / new_scale.unsqueeze(1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
  with torch.no_grad():
    mod.weight.copy_(q)
    mod.weight_scale.copy_(new_scale.to(torch.float32))
  return True


# --------------------------------------------------------------------------- #
# apply + undo (the undo closure subtracts the same delta from the CURRENT
# weights, so it works after FP8 re-quantization drift too)
# --------------------------------------------------------------------------- #

def _collect_merge_actions(model: nn.Module, lora_sd: dict[str, torch.Tensor], strength: float,
                          lokr_factor: float | None = None,
                          ) -> tuple[int, list[str], list]:
  """(applied_count, log_lines, undo_closures). Raises LoraRefused when nothing
  matched. Does NOT mutate the model yet - actions are applied by the caller
  (apply_lora_to_model / merge_lora_undoable)."""
  logs: list[str] = []
  actions: list = []
  groups = _group_modules(lora_sd)
  if not groups:
    raise LoraRefused(
      "This LoRA file did not contain any recognizable LoRA/LoKr weight pairs. It has NOT been applied."
    )

  named = dict(model.named_parameters())
  fp8_modules: dict[str, nn.Module] = {}
  for name, mod in model.named_modules():
    if not name:
      continue
    if (hasattr(mod, "weight_scale") and getattr(mod, "weight", None) is not None
        and getattr(mod.weight, "dtype", None) == FP8_WEIGHT_DTYPE):
      fp8_modules[name] = mod

  # Every module path the target model can actually accept a delta on.
  known_paths = {n[: -len(".weight")] for n in named if n.endswith(".weight")}
  known_paths |= set(fp8_modules)

  unresolved: list[str] = []
  unsupported: list[str] = []
  for raw_path, sides in groups.items():
    if raw_path == "":
      continue
    delta = _module_delta(raw_path, sides, lokr_factor)
    if delta is None:
      # Tell the difference between "we could not build a delta" and "this
      # module simply is not in the model" - they need different advice.
      if sides.get("T1") is not None or sides.get("T2") is not None:
        unsupported.append(raw_path)
      continue

    module_path = resolve_module_path(raw_path, known_paths)
    if module_path is None:
      unresolved.append(raw_path)
      continue

    param = named.get(module_path + ".weight")
    if param is not None and param.ndim == 2 and param.shape == delta.shape:
      actions.append(("param", param, raw_path, sides, lokr_factor, float(strength)))
      continue
    mod = fp8_modules.get(module_path)
    if mod is not None and mod.weight.shape == delta.shape:
      actions.append(("fp8", mod, raw_path, sides, lokr_factor, float(strength)))
      continue
    if param is not None and param.ndim == 2 and delta.T.shape == param.shape:
      actions.append(("param", param, raw_path, sides, lokr_factor, float(strength), True))
      continue
    logs.append(f"skip {module_path}: shape mismatch (delta {tuple(delta.shape)})")

  if unsupported and not actions:
    raise LoraRefused(
      "This is a CP-decomposed LoKr (lokr_t1/lokr_t2 tensors). Reconstructing it "
      "needs the tucker-style core factorization, which this build does not "
      f"implement ({len(unsupported)} modules affected). Nothing was changed - "
      "a wrong reconstruction would look like a working LoRA while quietly "
      "corrupting the weights. Re-export the adapter without CP decomposition, "
      "or use a plain LoKr / LoRA."
    )
  if unsupported and actions:
    logs.append(f"{len(unsupported)} CP-decomposed LoKr module(s) skipped "
                "(lokr_t1/lokr_t2 not supported); the rest merged normally")
  if unresolved and actions:
    logs.append(f"{len(unresolved)} adapter module(s) had no counterpart in this "
                f"trunk (e.g. {unresolved[0]}) - the rest merged normally")
  if not actions:
    raise LoraRefused(
      "This LoRA/LoKr did not match any module of the Ideogram 4 model "
      f"({len(groups)} modules found in the file, none applied). It was probably trained "
      "for a different model. Nothing was changed."
    )
  return len(actions), logs, actions


def _apply_actions(model: nn.Module, actions: list) -> list:
  """Execute merge actions, returning one undo closure per action.

  'param' (bf16) deltas are added directly; 'fp8' uses dequant-add-requant.
  The undo re-applies the NEGATED delta to the current weights (exact for
  bf16, tiny e4m3 drift only on the fp8 path)."""
  def _delta_of(act, strength_mult: float) -> torch.Tensor:
    module_path, sides, lokr_factor, strength = act[2], act[3], act[4], act[5]
    d = _module_delta(module_path, sides, lokr_factor)
    if d is None:
      return None
    if len(act) > 6 and act[6]:  # transposed variant
      d = d.T
    return d * (strength * strength_mult)

  def _param_undo(act):
    """Undo = add the NEGATED original delta (recomputed from the small LoRA
    factors; holds no big tensor clones). 1-ulp bf16 drift is inherent to an
    in-place merge and far below the fp8-requant noise floor."""
    def _u():
      d = _delta_of(act, -1.0)  # == -(unscaled delta * strength)
      if d is None:
        return
      with torch.no_grad():
        act[1].add_(d.to(act[1].dtype).to(act[1].device))
    return _u

  def _fp8_undo(act):
    def _u():
      d = _delta_of(act, -1.0)
      if d is not None:
        _merge_fp8_linear(act[1], d)
    return _u

  undo: list = []
  for act in actions:
    kind = act[0]
    if kind == "param":
      d = _delta_of(act, 1.0)
      if d is None:
        continue
      with torch.no_grad():
        act[1].add_(d.to(act[1].dtype).to(act[1].device))
      undo.append(_param_undo(act))
    elif kind == "fp8":
      d = _delta_of(act, 1.0)
      if d is None:
        continue
      if _merge_fp8_linear(act[1], d):
        undo.append(_fp8_undo(act))
  return undo


def apply_lora_to_model(model: nn.Module, lora_sd: dict[str, torch.Tensor], strength: float,
                        alpha_hint: float | None = None) -> tuple[int, list[str]]:
  """Merge an Ideogram 4 LoRA/LoKr into the model.

  In-process quant support: bf16/bf32 weights merge directly; FP8 modules
  (fp8 weight + per-row scale) dequant->merge->requant. LoRAs still apply in
  the model's own dtype. bnb Linear4bit weights are NOT merged in-process
  today - those modules are logged as 'skip' (load the fp8/bf16 file of the
  same trunk to apply LoRAs on an NF4 profile).

  Returns (num_applied_modules, log_lines). Raises LoraRefused if nothing matched.
  """
  lokr_factor = None
  if alpha_hint is not None and alpha_hint:
    f = float(alpha_hint)
    if f not in (0.0, 1.0):
      lokr_factor = f
  applied, logs, actions = _collect_merge_actions(model, lora_sd, strength, lokr_factor=lokr_factor)
  _apply_actions(model, actions)
  fmt = "LoKr (Kronecker)" if any(k in ("W1", "W1A", "W2", "W2A") for k in _kinds_of(_group_modules(lora_sd))) else "LoRA"
  logs.append(f"{fmt} merged into {applied} modules at strength {strength} (cond model; quant-aware merge)")
  return applied, logs


def merge_lora_undoable(model: nn.Module, lora_sd: dict[str, torch.Tensor], strength: float,
                        lokr_factor: float | None = None) -> tuple[int, list[str], callable]:
  """Merge and return an undo closure. Used by first-step-only adapters
  (gray bypass) and uncond-replacement LoRAs that must not linger."""
  applied, logs, actions = _collect_merge_actions(model, lora_sd, strength, lokr_factor=lokr_factor)
  undo_list = _apply_actions(model, actions)

  def _undo():
    for u in undo_list:
      try:
        u()
      except Exception:
        pass

  return applied, logs, _undo


def unload_loras(model) -> None:
  """LoRA deltas are merged additively into the live weights; unloading is
  handled at pipeline-cache level. Kept for API symmetry."""
  return None


def _kinds_of(groups: dict) -> list[str]:
  return list(next(iter(groups.values())).keys()) if groups else []