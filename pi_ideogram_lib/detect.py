"""Ideogram 4 checkpoint detection + cond/uncond pairing.

A file is an Ideogram 4 diffusion checkpoint if:
- its safetensors metadata carries model_type=ideogram4_cond / ideogram4_uncond, OR
- its tensor keys contain the unique marker 'embed_image_indicator.weight' (official DiT).

Everything is header-only (cheap): we never load weights to detect.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path


def read_safetensors_header(path: str | Path) -> dict:
  """Read only the JSON header of a .safetensors file (no weights touched)."""
  path = Path(path)
  with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n))


def _key_set(header: dict) -> set[str]:
  return {k for k in header.keys() if k != "__metadata__"}


def is_ideogram4_header(header: dict) -> bool:
  meta = header.get("__metadata__") or {}
  mt = str(meta.get("model_type", "")).lower()
  if mt in ("ideogram4_cond", "ideogram4_uncond", "ideogram4"):
    return True
  return "embed_image_indicator.weight" in _key_set(header)


def is_ideogram4_checkpoint_file(path: str | Path) -> bool:
  """Cheap header-only check. Never raises (unreadable files -> False)."""
  try:
    p = Path(path)
    if p.suffix.lower() not in (".safetensors", ".sft"):
      return False
    return is_ideogram4_header(read_safetensors_header(p))
  except Exception:
    return False


def is_unconditional_file(path: str | Path) -> bool:
  try:
    meta = read_safetensors_header(path).get("__metadata__") or {}
    if str(meta.get("model_type", "")).lower() == "ideogram4_uncond":
      return True
  except Exception:
    pass
  name = Path(path).stem.lower()
  return "uncond" in name


def _quant_signature(stem: str) -> str:
  """Normalize a filename stem to its quant fingerprint (lowercase alphanumerics only)."""
  s = re.sub(r"[^a-z0-9]", "", stem.lower())
  s = s.replace("ideogram", "").replace("unconditional", "").replace("uncond", "")
  return s


def is_ideogram4_uncond_file(path: str | Path) -> bool:
  """The Ideogram 4 UNCONDITIONAL half specifically - never a file that merely
  carries the 'uncond' token. Header metadata (model_type=ideogram4_uncond)
  when readable; otherwise an Ideogram 4 file whose name says 'uncond'."""
  p = Path(path)
  low = p.stem.lower()
  if "uncond" not in low:
    # header-only case (unconventional filenames)
    try:
      meta = (read_safetensors_header(p).get("__metadata__") or {})
      return str(meta.get("model_type", "")).lower() == "ideogram4_uncond"
    except Exception:
      return False
  try:
    if is_ideogram4_checkpoint_file(p):
      return True
  except Exception:
    pass
  return "ideogram" in low


def find_matching_uncond(cond_path: str | Path, search_dirs: list[str | Path]) -> Path | None:
  """Auto-pair a cond checkpoint with its matching unconditional checkpoint.

  Match rule: an Ideogram 4 file whose name says 'unconditional' AND whose quant
  fingerprint equals the cond file's (fp8scaled==fp8scaled, nvfp4mixed==nvfp4mixed...).
  """
  cond_path = Path(cond_path)
  cond_sig = _quant_signature(cond_path.stem)
  seen: set[Path] = set()
  for d in search_dirs:
    d = Path(d)
    if not d.is_dir():
      continue
    for cand in d.rglob("*.safetensors"):
      if cand.resolve() in seen or cand.resolve() == cond_path.resolve():
        continue
      seen.add(cand.resolve())
      if not is_unconditional_file(cand):
        continue
      if not is_ideogram4_checkpoint_file(cand):
        continue
      if _quant_signature(cand.stem) == cond_sig:
        return cand
  return None


def find_matching_cond(uncond_path: str | Path, search_dirs: list[str | Path]) -> Path | None:
  """REVERSE pairing: the user picked the unconditional half in the dropdown -
  find its conditional twin so the official dual-model pair can run. Same match
  rule as find_matching_uncond, mirrored."""
  uncond_path = Path(uncond_path)
  if not is_unconditional_file(uncond_path):
    return None
  uncond_sig = _quant_signature(uncond_path.stem)
  seen: set[Path] = set()
  for d in search_dirs:
    d = Path(d)
    if not d.is_dir():
      continue
    for cand in d.rglob("*.safetensors"):
      if cand.resolve() in seen or cand.resolve() == uncond_path.resolve():
        continue
      seen.add(cand.resolve())
      if is_unconditional_file(cand):
        continue
      if not is_ideogram4_checkpoint_file(cand):
        continue
      if _quant_signature(cand.stem) == uncond_sig:
        return cand
  return None


KNOWN_QUANTS = ("fp8_scaled", "fp8scaled", "int8_convrot", "int8convrot", "nvfp4_mixed", "nvfp4mixed", "nf4", "gguf")


def list_ideogram4_checkpoints(search_dirs: list[str | Path]) -> list[Path]:
  """All Ideogram 4 cond/uncond diffusion files found under the given directories."""
  out: list[Path] = []
  seen: set[Path] = set()
  for d in search_dirs:
    d = Path(d)
    if not d.is_dir():
      continue
    for cand in d.rglob("*.safetensors"):
      r = cand.resolve()
      if r in seen:
        continue
      seen.add(r)
      if is_ideogram4_checkpoint_file(cand):
        out.append(cand)
  return sorted(out)


# --- text encoder / VAE discovery -------------------------------------------------

TE_NAME_PATTERNS = ("qwen3vl_8b", "qwen3_vl_8b", "qwen3vl-8b")


def find_text_encoder(search_dirs: list[str | Path]) -> Path | None:
  best: tuple[int, Path] | None = None
  for d in search_dirs:
    d = Path(d)
    if not d.is_dir():
      continue
    for cand in d.rglob("*.safetensors"):
      name = cand.stem.lower()
      if not any(p in name for p in TE_NAME_PATTERNS):
        continue
      # rank: fp8_scaled > bf16 > anything else (prefer smaller/faster when equal)
      rank = 0
      if "fp8" in name:
        rank += 2
      if "8b" in name:
        rank += 1
      if best is None or rank > best[0]:
        best = (rank, cand)
  if best:
    return best[1]
  # NAME-INDEPENDENT FALLBACK. The patterns above only recognise files still
  # called qwen3vl_8b*; a renamed or re-quantised encoder was invisible. When
  # nothing matches by name, ask the FILES what they are - the Qwen3-VL text
  # tower is unmistakable in a safetensors header.
  return _find_by_header(search_dirs, _looks_like_qwen3vl_te)


def _find_by_header(search_dirs, predicate) -> Path | None:
  """First file under `search_dirs` whose safetensors header satisfies
  `predicate`. Header-only: no weights are read."""
  for d in search_dirs:
    d = Path(d)
    if not d.is_dir():
      continue
    for cand in sorted(d.rglob("*.safetensors")):
      try:
        if predicate(read_safetensors_header(cand)):
          return cand
      except Exception:
        continue
  return None


def _shape(header: dict, *suffixes) -> tuple | None:
  for k, v in header.items():
    if k == "__metadata__":
      continue
    if any(k.endswith(sfx) for sfx in suffixes):
      try:
        return tuple(v["shape"])
      except Exception:
        continue
  return None


# The DIMENSIONS matter, not just the key names. A first version matched any
# file with a language tower plus visual mergers, and happily picked
# qwen3vl_32b_minimax_h3_nvfp4_awq - right architecture family, wrong model.
# Silently loading a 32B encoder in place of the 8B is worse than finding
# nothing, so the width is part of the identity.
QWEN3VL_8B_HIDDEN = 4096      # the 32B is 5120
FLUX2_VAE_Z_CHANNELS = 32     # Flux1's ae.safetensors is 16


def _looks_like_qwen3vl_te(header: dict) -> bool:
  """Qwen3-VL-8B at the width Ideogram 4 taps (hidden 4096).

  Two layouts qualify:

  * HF naming - a language tower plus the visual deepstack mergers.
  * llama.cpp naming - `blk.N.*` and `token_embd.weight`. A GGUF encoder
    converted to safetensors (Forge Neo's own converter produces exactly this)
    keeps those names AND has no vision tower, because llama.cpp keeps that in
    a separate mmproj file. Demanding `deepstack_merger` rejected such a file
    with "is NOT Qwen3-VL-8B" when it was precisely that; the vision tower is
    irrelevant here anyway, since Ideogram 4 taps only the language model.
  """
  keys = _key_set(header)
  if "token_embd.weight" in keys or any(k.startswith("blk.") for k in keys):
    emb = _shape(header, "token_embd.weight")
    # llama.cpp stores this already in torch order (vocab, hidden).
    return bool(emb and len(emb) == 2 and emb[1] == QWEN3VL_8B_HIDDEN)
  has_lm = any(k.startswith(("model.layers.", "language_model.layers.")) for k in keys)
  has_vis = any("deepstack_merger" in k or k.startswith(("visual.", "model.visual.")) for k in keys)
  if not (has_lm and has_vis):
    return False
  emb = _shape(header, "embed_tokens.weight")
  return bool(emb and len(emb) == 2 and emb[1] == QWEN3VL_8B_HIDDEN)


def _looks_like_flux2_vae(header: dict) -> bool:
  """Flux2 KL autoencoder: encoder+decoder towers, latent width 32."""
  keys = _key_set(header)
  if not (any(k.startswith("encoder.") for k in keys)
          and any(k.startswith("decoder.") for k in keys)):
    return False
  if any(k.startswith(("model.layers.", "layers.")) for k in keys):
    return False
  conv = _shape(header, "decoder.conv_in.weight")
  return bool(conv and len(conv) == 4 and conv[1] == FLUX2_VAE_Z_CHANNELS)


def find_vae(search_dirs: list[str | Path]) -> Path | None:
  for d in search_dirs:
    d = Path(d)
    if not d.is_dir():
      continue
    for cand in d.rglob("*.safetensors"):
      if cand.stem.lower() in ("flux2-vae", "flux2_vae", "flux2vae"):
        return cand
  # Same reasoning as the text encoder: fall back to what the file IS, so a
  # renamed VAE still resolves.
  return _find_by_header(search_dirs, _looks_like_flux2_vae)


VAE_NAME_PATTERNS = ("flux2-vae", "flux2_vae", "flux2vae")


# --- Forge's own module selection (the DEFAULT mechanism) -------------------
# Forge Neo remembers the VAE/TE a user picked for a checkpoint and re-selects
# it on the next launch, per UI preset:
#
#   forge_checkpoint_ideogram4         -> the checkpoint
#   forge_additional_modules_ideogram4 -> [VAE path, TE path]
#
# and exposes the live list as shared.opts.forge_additional_modules. Scanning
# disk instead of reading that list silently overrides the user: the picker
# below ranks "fp8" highest, so a user who deliberately selected
# qwen3vl_8b-int8_convrot got the fp8 file anyway. Selection is an instruction,
# not a hint - honour it, and fall back to the scan only when nothing is
# selected.

def forge_selected_modules() -> list[Path]:
  """Absolute paths of the VAE/TE modules currently selected in Forge.

  Empty when Forge is absent (standalone tests) or nothing is selected.
  """
  try:
    from modules import shared
    mods = getattr(shared.opts, "forge_additional_modules", None) or []
  except Exception:
    return []
  out: list[Path] = []
  for m in mods:
    try:
      q = Path(str(m))
      if q.is_file():
        out.append(q)
    except Exception:
      continue
  return out


def _te_size_ok(path) -> bool:
  """Is this Qwen3-VL at the width Ideogram 4 taps (8B, hidden 4096)?

  Unreadable -> True: a header we cannot parse is not evidence of the wrong
  model, and the loader will still report a real problem if there is one.
  """
  try:
    return _looks_like_qwen3vl_te(read_safetensors_header(path))
  except Exception:
    return True


def te_shard_files(folder) -> list[Path]:
  """The safetensors shards of a Hugging Face encoder folder, in index order.

  Every abliterated Qwen3-VL-8B is published this way - `config.json`, a
  `model.safetensors.index.json` and model-0000N-of-0000M.safetensors - rather
  than as the single file Forge's module dropdown can list. Returns [] for
  anything that is not such a folder.
  """
  d = Path(folder)
  if not d.is_dir():
    return []
  idx = d / "model.safetensors.index.json"
  if idx.is_file():
    try:
      wm = (json.loads(idx.read_text(encoding="utf-8")) or {}).get("weight_map", {})
      names = sorted(set(wm.values()))
      shards = [d / n for n in names]
      if shards and all(p.is_file() for p in shards):
        return shards
    except Exception:
      pass
  shards = sorted(d.glob("*.safetensors"))
  return shards


def _te_dir_ok(folder) -> bool:
  """Is this folder a Qwen3-VL-8B encoder at the width Ideogram 4 taps?

  Judged from config.json when present (authoritative and free), otherwise
  from the shards' own headers - same shape test as a single file.
  """
  d = Path(folder)
  shards = te_shard_files(d)
  if not shards:
    return False
  cfg = d / "config.json"
  if cfg.is_file():
    try:
      conf = json.loads(cfg.read_text(encoding="utf-8")) or {}
      text = conf.get("text_config") or conf
      hidden = int(text.get("hidden_size", 0) or 0)
      layers = int(text.get("num_hidden_layers", 0) or 0)
      if hidden and hidden != QWEN3VL_8B_HIDDEN:
        return False
      # Ideogram 4 taps layer 35, so anything shallower cannot serve it.
      from .constants import QWEN3_VL_ACTIVATION_LAYERS
      if layers and layers <= max(QWEN3_VL_ACTIVATION_LAYERS):
        return False
      if hidden:
        return True
    except Exception:
      pass
  for p in shards:
    try:
      if _looks_like_qwen3vl_te(read_safetensors_header(p)):
        return True
    except Exception:
      continue
  return False


def classify_module(path: str | Path) -> str | None:
  """'te' | 'vae' | None for one selected module file.

  Name first (free), header second (cheap, and the authority when a user has
  renamed a file). Anything we cannot place returns None and is left alone -
  Forge module lists legitimately carry files that are neither.
  """
  q = Path(path)
  name = q.stem.lower()
  if any(t in name for t in TE_NAME_PATTERNS):
    return "te"
  if name in VAE_NAME_PATTERNS or "flux2-vae" in name or "flux2_vae" in name:
    return "vae"
  try:
    keys = _key_set(read_safetensors_header(q))
  except Exception:
    return None
  if any(k.startswith(("encoder.", "decoder.")) for k in keys):
    return "vae"
  if any("language_model" in k or k.startswith("model.layers.") for k in keys):
    return "te"
  return None


def resolve_te_vae(models_dirs) -> tuple[Path | None, Path | None, list[str]]:
  """(text_encoder, vae, notes) - Forge's selection wins, the disk scan fills gaps.

  This is the single owner of "which TE and which VAE does this run use".
  `notes` are human-readable lines for the console so the choice is never
  silent: the user can see whether their pick was used or a fallback was.
  """
  notes: list[str] = []
  te = vae = None

  # EXPLICIT OVERRIDE, ahead of everything. Forge's module dropdown lists
  # single .safetensors FILES, so a Hugging Face encoder folder - which is how
  # every abliterated Qwen3-VL-8B is published, sharded across
  # model-0000N-of-0000M.safetensors plus an index - cannot be selected there
  # at all. This is the way to point at one. Set `text_encoder_override` in
  # config.json to a file OR a folder.
  try:
    import pi_ideogram_lib.config as pi_config
    ov = str(pi_config.get("text_encoder_override", "") or "").strip()
    if ov:
      q = Path(ov)
      if _te_dir_ok(q) if q.is_dir() else (q.is_file() and _te_size_ok(q)):
        te = q
        notes.append(f"text encoder: using your override ({q.name})")
      elif q.exists():
        notes.append(f"text encoder: override {q.name} is not a Qwen3-VL-8B "
                     "encoder (Ideogram 4 needs hidden 4096, 36 layers) - ignoring it")
      else:
        notes.append(f"text encoder: override path does not exist ({ov}) - ignoring it")
  except Exception:
    pass

  # NOTE: this loop resolves the VAE too, so it must always run. The `te is
  # None` guards inside already leave an override in place.
  for m in forge_selected_modules():
    kind = classify_module(m)
    if kind == "te" and te is None:
      # VALIDATE THE SIZE. Ideogram 4 taps 13 layers of Qwen3-VL-**8B**
      # (hidden 4096). Selecting the 4B (hidden 2560) or the 32B (5120) is an
      # easy mistake - the filenames differ by two characters - and the failure
      # was ~2000 lines of shape mismatches ending in
      #     size mismatch for language_model.embed_tokens.weight:
      #       copying a param with shape [151936, 2560] ... current [151936, 4096]
      # which says everything and explains nothing. One line, up front, beats
      # that: refuse the wrong model and keep looking for the right one.
      if _te_size_ok(m):
        te = m
        notes.append(f"text encoder: using your Forge selection ({m.name})")
      else:
        notes.append(f"text encoder: {m.name} is NOT Qwen3-VL-8B "
                     "(Ideogram 4 needs the 8B: hidden 4096) - ignoring it "
                     "and looking for the right one")
    elif kind == "vae" and vae is None:
      vae = m
      notes.append(f"VAE: using your Forge selection ({m.name})")

  dirs = te_vae_search_dirs(models_dirs)
  picked = []
  if te is None:
    te = find_text_encoder(dirs)
    if te is not None:
      picked.append(te)
      notes.append(f"text encoder: nothing selected in Forge, auto-picked {te.name}")
  if vae is None:
    vae = find_vae(dirs)
    if vae is not None:
      picked.append(vae)
      notes.append(f"VAE: nothing selected in Forge, auto-picked {vae.name}")

  # MAKE IT STICK. Every other model in Forge is chosen once and stays chosen;
  # an auto-pick that lives only for one generation is not the same thing, and
  # left the modules row looking empty next launch. Write what we resolved into
  # Forge's own per-preset memory so it behaves like any other selection - the
  # user can still change it, and we never overwrite a choice they made.
  if picked:
    notes.extend(remember_modules(picked))
  return te, vae, notes


def remember_modules(paths) -> list[str]:
  """Record resolved TE/VAE in Forge's module selection, REPLACING by role.

  This used to append. The module list is shared with every other UI preset,
  so appending accumulated one entry per role per visit - a real list from
  this install read:

      ["Qwen3VL-8B-Uncensored-...-int8_convrot.safetensors",
       "flux2-klein-9b-uncensored-f16.safetensors",
       "qwen3vl_4b_fp8_scaled.safetensors"]

  three text encoders and no VAE. Whichever one classify_module() happened to
  reach first then won, so the encoder in use could change without anyone
  choosing it.

  A slot can hold one TE and one VAE. Writing a resolved TE therefore drops
  any other TE, and likewise for the VAE; entries that are neither (another
  model's modules) are left exactly where they are.
  """
  try:
    from modules import shared
    current = list(getattr(shared.opts, "forge_additional_modules", None) or [])
    incoming = {}
    for q in paths:
      kind = classify_module(q)
      if kind in ("te", "vae"):
        incoming[kind] = str(q)
    if not incoming:
      return []
    kept = []
    dropped = []
    for x in current:
      k = classify_module(x)
      if k in incoming and str(Path(x)).lower() != str(Path(incoming[k])).lower():
        dropped.append(Path(str(x)).name)
        continue
      kept.append(x)
    have = {str(Path(x)).lower() for x in kept}
    added = [v for v in incoming.values() if str(Path(v)).lower() not in have]
    if not added and not dropped:
      return []
    new_list = kept + added
    shared.opts.set("forge_additional_modules", new_list)
    try:
      preset = str(getattr(shared.opts, "forge_preset", "") or "")
      if preset:
        # new_list, NOT new_list + added: `added` is already in it, and adding
        # it twice is how a list grows a duplicate on every single generation.
        shared.opts.set(f"forge_additional_modules_{preset}", new_list)
    except Exception:
      pass
    try:
      shared.opts.save(shared.config_filename)
    except Exception:
      pass
    msg = f"remembered {len(added)} module selection(s) for next launch"
    if dropped:
      msg += f" (replaced {', '.join(dropped)})"
    return [msg]
  except Exception:
    return []


def remember_checkpoint(path) -> list[str]:
  """Record the Ideogram 4 checkpoint that just ran, BY ABSOLUTE PATH.

  Forge remembers a checkpoint by the string in `sd_model_checkpoint`, and for
  our files that string is the DISPLAY LABEL our dropdown hook installs:

      forge_checkpoint_ideogram4  'Ideogram4 — ideogram4-int8_convrot'

  (that separator is an em dash, U+2014). Restoring it therefore depends on
  the rename having already run, on the label being byte-identical, and on the
  em dash surviving a JSON round trip - three ways to lose a selection that a
  path has none of. Forge's other presets store a real path
  ('FLUX2 KLEIN\\flux-2-klein-9b-int8_convrot.safetensors') and never have the
  problem.

  This is our own copy, in the extension's config, used only as a fallback by
  restore_checkpoint(). It never overrides a selection that already resolves.
  """
  try:
    p = Path(str(path))
    if not p.is_file():
      return []
    import pi_ideogram_lib.config as pi_config
    if str(pi_config.get("last_checkpoint", "") or "") == str(p):
      return []
    pi_config.save({"last_checkpoint": str(p)})
    return [f"remembered checkpoint for next launch: {p.name}"]
  except Exception:
    return []


def restore_checkpoint() -> list[str]:
  """Re-select the remembered checkpoint when Forge's own value is unusable.

  Only acts when BOTH are true:
    * Forge's current selection does not resolve to an Ideogram 4 file, and
    * the remembered path is still on disk.

  So a deliberate switch to any other model is left alone - this fires for a
  selection that was LOST, never one that was changed.
  """
  try:
    import pi_ideogram_lib.config as pi_config
    remembered = str(pi_config.get("last_checkpoint", "") or "")
    if not remembered:
      return []
    q = Path(remembered)
    if not q.is_file():
      return []
    from modules import shared, sd_models
    current = str(getattr(shared.opts, "sd_model_checkpoint", "") or "")
    if current:
      ci = sd_models.get_closet_checkpoint_match(current)
      if ci is not None and Path(str(getattr(ci, "filename", ""))).is_file():
        return []  # Forge's own value resolves; nothing to repair
    target = None
    for cand in getattr(sd_models, "checkpoints_list", {}).values():
      try:
        if Path(str(getattr(cand, "filename", ""))) == q:
          target = cand
          break
      except Exception:
        continue
    if target is None:
      return []
    name = str(getattr(target, "name", "") or getattr(target, "title", ""))
    shared.opts.set("sd_model_checkpoint", name)
    try:
      preset = str(getattr(shared.opts, "forge_preset", "") or "")
      if preset:
        shared.opts.set(f"forge_checkpoint_{preset}", name)
      shared.opts.save(shared.config_filename)
    except Exception:
      pass
    return [f"restored your last Ideogram 4 checkpoint ({q.name}) - "
            "Forge's saved selection no longer resolved"]
  except Exception:
    return []


# --- on-disk layout helpers -------------------------------------------------
# The Ideogram 4 folder spellings are defined ONCE here. Every consumer
# (engine discovery + dropdown registration, LoRA scan, bypass-LoRA scan,
# downloader) iterates this tuple instead of re-listing the spellings.

IDE04_DIR_NAMES = ("Ideogram4", "Ideogram 4", "Ideogram-4")


def _existing(dirs) -> list[Path]:
  return [d for d in dirs if d.is_dir()]


def model_search_dirs(models_dirs) -> list[Path]:
  """Checkpoint search dirs: models/Stable-diffusion + every Ideogram 4
  spelling + models/diffusion_models, each with a nested /checkpoints subdir
  when present (mirrors the old engine._sd_search_dirs behaviour)."""
  bases: list[Path] = []
  for d in models_dirs or []:
    d = Path(d)
    bases += [d / "Stable-diffusion", *(d / s for s in IDE04_DIR_NAMES), d / "diffusion_models"]
  out: list[Path] = []
  for b in bases:
    if b.is_dir():
      out.append(b)
      ck = b / "checkpoints"
      if ck.is_dir():
        out.append(ck)
  return out


def te_vae_search_dirs(models_dirs) -> list[Path]:
  """text encoder / VAE search dirs (old engine._te_vae_search_dirs)."""
  dirs: list[Path] = []
  for d in models_dirs or []:
    d = Path(d)
    dirs += [d / "text_encoder", d / "text_encoders", d / "VAE",
             *(d / s for s in IDE04_DIR_NAMES)]
  return _existing(dirs)


def ideogram4_ckpt_dirs(models_dirs) -> list[Path]:
  """Directories the checkpoint dropdown registers as 'Ideogram4 - <name>'.
  Every spelling, plus its nested /checkpoints when present."""
  out: list[Path] = []
  for d in models_dirs or []:
    d = Path(d)
    for s in IDE04_DIR_NAMES:
      base = d / s
      if base.is_dir():
        out.append(base)
        ck = base / "checkpoints"
        if ck.is_dir():
          out.append(ck)
  return out


# The one place Ideogram adapters are written. Everything else in the list
# below is a legacy location we still READ, so an existing install keeps
# working, but nothing creates them any more.
IDE04_LORA_SUBDIR = "IDEOGRAM LORA"


def lora_search_dirs(models_dirs) -> list[Path]:
  """LoRA dirs, canonical first.

  models/Lora/IDEOGRAM LORA is where adapters now live - inside Forge's own
  Lora tree, so they appear wherever LoRAs are expected. The old private
  spellings (models/Ideogram4/loras and friends) are still searched so nobody
  loses files, but _existing() filters them out when absent and no code
  creates them: that side effect is what left this install with three separate
  Ideogram folders holding pieces of the same set.
  """
  dirs: list[Path] = []
  for d in models_dirs or []:
    d = Path(d)
    dirs.append(d / "Lora" / IDE04_LORA_SUBDIR)
    dirs.append(d / "Lora")
    for s in IDE04_DIR_NAMES:                       # legacy, read-only
      dirs += [d / s / "loras", d / s / "lora"]
  return _existing(dirs)


ADAPTER_NONE_LABEL = "None"

# How each classified kind is described in the dropdown. The role matters: the
# three adapters this project ships links for do THREE different jobs, and a
# single unlabelled list would invite picking TurboTime as a "filter bypass"
# and wondering why nothing changed.
_ADAPTER_ROLE_LABEL = {
  "gray_bypass": "filter bypass",
  "turbo_time": "turbo schedule",
  "uncond_replacement": "uncond replacement",
  "style": "style / character",
}


def adapter_choices(models_dirs) -> list[str]:
  """Every Ideogram 4 adapter on disk, labelled with what it DOES.

  Searched recursively (rglob) through models/Lora/IDEOGRAM LORA, models/Lora
  and the legacy folders, so adapters filed in subfolders are found.

  Entries read 'name.safetensors  -  turbo schedule'. The name before the
  separator is what gets applied, so parse it back with adapter_name_of().
  """
  from pi_ideogram_lora.ideogram4_lora import classify_adapter
  out: list[str] = []
  seen: set[str] = set()
  for d in lora_search_dirs(models_dirs):
    d = Path(d)
    if not d.is_dir():
      continue
    in_i4_dir = any(tok in d.as_posix().lower() for tok in ("ideogram", "ideo4"))
    for cand in sorted(d.rglob("*.safetensors")):
      if cand.name in seen:
        continue
      low = cand.stem.lower()
      looks_i4 = (in_i4_dir
                  or any(t in low for t in ("ideogram", "ideo4", "ido4", "i4_"))
                  or "gray" in low or "bypass" in low or "000002000" in low)
      if not looks_i4:
        continue  # models/Lora holds hundreds of adapters for other models
      seen.add(cand.name)
      try:
        kind = str(classify_adapter(cand).get("kind") or "")
      except Exception:
        kind = ""
      if kind == "incompatible":
        continue
      role = _ADAPTER_ROLE_LABEL.get(kind, "adapter")
      out.append(f"{cand.name}  -  {role}")
  return out


def adapter_name_of(choice: str) -> str:
  """The filename from a dropdown entry made by adapter_choices()."""
  s = str(choice or "").strip()
  if not s or s == ADAPTER_NONE_LABEL:
    return ""
  return s.split("  -  ", 1)[0].strip()


def bypass_lora_choices(models_dirs) -> list[str]:
  """LoRA filenames worth offering as a filter bypass.

  Filtered by NAME only - no header reads. models/Lora on a real install holds
  hundreds of adapters for every other model (521 here), which is useless in a
  dropdown and, if we classified them properly, would mean opening every file.
  A bypass LoRA is either the known gray-screen one or an Ideogram 4 adapter,
  and both are identifiable from the filename:

    * gray / bypass / 000002000   -> the Civitai gray-screen bypass, listed FIRST
    * ideogram / ideo4 / ido4 / i4_ -> an Ideogram 4 adapter
    * anything inside an Ideogram LoRA folder -> the user put it there on purpose

  Anything else is for a different model and cannot apply to this one.
  """
  known: list[str] = []
  others: list[str] = []
  seen: set[str] = set()
  for d in lora_search_dirs(models_dirs):
    d = Path(d)
    if not d.is_dir():
      continue
    # A dedicated Ideogram LoRA folder is itself a statement of intent.
    in_i4_dir = any(tok in d.as_posix().lower() for tok in ("ideogram", "ideo4"))
    for cand in sorted(d.rglob("*.safetensors")):
      nm = cand.name
      if nm in seen:
        continue
      low = cand.stem.lower()
      is_gray = ("gray" in low or "bypass" in low or "000002000" in low)
      is_i4 = any(t in low for t in ("ideogram", "ideo4", "ido4", "i4_"))
      if not (is_gray or is_i4 or in_i4_dir):
        continue
      seen.add(nm)
      (known if is_gray else others).append((nm, cand))

  # The name gate leaves a handful of files (521 -> 2 on this install), so the
  # survivors CAN be classified properly - and must be. Listing every Ideogram
  # adapter meant the uncond-replacement and TurboTime LoRAs appeared as
  # "filter bypass" options; picking one applied a distill/uncond adapter to
  # the conditional trunk at -0.25, which is not a bypass and not something
  # either adapter was trained for. Those roles have their own machinery.
  return [nm for nm, path in known + others if _usable_as_bypass(path)]


_NON_BYPASS_KINDS = frozenset({"uncond_replacement", "turbo_time", "incompatible"})


def _usable_as_bypass(path: Path) -> bool:
  """Can this adapter serve as a filter bypass? Unclassifiable -> yes.

  Only adapters with a DIFFERENT documented role are excluded. Anything we
  cannot read stays listed: a user who dropped a file into the Ideogram LoRA
  folder meant it to be available, and a failed header read is not evidence
  against them.
  """
  try:
    from pi_ideogram_lora.ideogram4_lora import classify_adapter
    return classify_adapter(path).get("kind") not in _NON_BYPASS_KINDS
  except Exception:
    return True


def bypass_lora_path(name, models_dirs) -> Path | None:
  """Resolve a dropdown filename back to a path on disk.

  Returns None for an adapter that is not usable as a bypass, so a selection
  persisted from an older build (when the dropdown wrongly offered the uncond
  and TurboTime adapters) cannot still be applied. config.json outlives any
  change to the choice list; the resolver has to be the one that says no.
  """
  if not name:
    return None
  want = str(name).strip().lower()
  for d in lora_search_dirs(models_dirs):
    if not Path(d).is_dir():
      continue
    for cand in Path(d).rglob("*.safetensors"):
      if cand.name.lower() == want or cand.stem.lower() == want:
        return cand if _usable_as_bypass(cand) else None
  return None


def bypass_lora_file(models_dirs) -> Path | None:
  """Locate the gray-screen bypass LoRA (Civitai 2750357) on disk. Name
  tokens: 'gray', 'bypass', or the model id '000002000'."""
  for d in lora_search_dirs(models_dirs):
    for cand in d.rglob("*.safetensors"):
      name = cand.stem.lower()
      if "gray" in name or "bypass" in name or "000002000" in name:
        return cand
  return None


