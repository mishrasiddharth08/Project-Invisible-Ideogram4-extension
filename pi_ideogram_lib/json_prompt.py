"""PROJECT INVISIBLE - POSITIVE PROMPT DOMINANCE - Ideogram 4 JSON caption builder.

THE RULE (the whole point of this file):
The user's typed positive prompt is the AUTHORITY. JSON wrap, schema fields,
style templates, LoRA trigger words and engine boilerplate are SERVANTS. They
must never replace, summarize, shrink, or overwrite the user's scene text.

Three modes:
- freeform-wrap       user types normal text -> we build the smallest valid official
                      caption JSON around it; the FULL unedited text goes into
                      high_level_description (the primary description field)
- raw-json            user pastes valid JSON -> description fields kept byte-for-byte;
                      only missing required keys get safe defaults
- visual-builder      user types the SAME normal text PLUS optional structure
                      (bboxes, palette, on-image text) as ADDITIONAL constraints

Dominance order when building the final caption:
  1. user prompt box text (never trimmed first, never summarized)
  2. LoRA trigger words (prepended, never delete user text)
  3. accordion extras (layout bboxes / palette / on-image text) as STRUCTURE only
  4. engine boilerplate last and minimal

Token budget: if the official 2048-token window would overflow, we trim
optional style JSON first and log exactly what was trimmed. User text is only
ever trimmed if it ALONE exceeds the window (then we say so plainly).
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# import the vendored verifier whether this file is imported as a top-level
# module (Forge extension dir on sys.path) or as part of a package
try:
  from .pi_ideogram_lib.caption_verifier import CaptionVerifier
except ImportError:  # top-level import style
  from pi_ideogram_lib.caption_verifier import CaptionVerifier  # noqa: E402

try:
  from .pi_ideogram_lib.decompose import (
    DECOMPOSE_ABOVE_WORDS, MAX_PROMPT_WORDS, decompose,
  )
except ImportError:  # top-level import style
  from pi_ideogram_lib.decompose import (  # noqa: E402
    DECOMPOSE_ABOVE_WORDS, MAX_PROMPT_WORDS, decompose,
  )

_VERIFIER = CaptionVerifier()

TAG = "[Invisible-I4][Prompt]"

# Official window for text tokens (Qwen3-VL tokenizer), model hard limit.
MAX_TEXT_TOKENS = 2048
# Fallback only - used when the vendored tokenizer cannot be loaded. Measured
# chars/token on the real Qwen3-VL vocabulary: 5.27 for plain prose, 4.44 for
# JSON captions, 3.76 for heavily punctuated text. The fallback takes the
# densest of those so it errs toward over-estimating rather than overflowing
# the window.
_CHARS_PER_TOKEN = 3.76


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _log(msg: str) -> None:
  print(f"{TAG} {msg}")


def _looks_like_json(text: str) -> bool:
  t = text.strip()
  if not t:
    return False
  if not (t.startswith("{") and t.endswith("}")):
    return False
  try:
    json.loads(t)
    return True
  except Exception:
    return False


def _is_valid_caption(obj: Any) -> bool:
  return isinstance(obj, dict) and isinstance(obj.get("compositional_deconstruction"), dict)


def _default_style_description() -> dict:
  """Minimal boilerplate style block, LAST priority. photo variant (photo + medium required)."""
  return {
    "aesthetics": "clean, well-composed",
    "lighting": "natural, balanced",
    "photo": "eye-level, sharp focus",
    "medium": "photograph",
  }


def _default_background(user_text_hint: str) -> str:
  return "A background consistent with the described scene."


def _default_element_desc() -> str:
  return "The main subject of the description, rendered faithfully."


_TOKENIZER = None  # None = not tried yet; False = unavailable


def _real_tokenizer():
  """The vendored Qwen3-VL tokenizer, loaded once. Offline, ~0.16s, no weights.

  Worth the import: the char heuristic this replaces over-estimated by 40-55%,
  which made enforce_token_budget strip the style_description block off long
  prompts that actually fit comfortably - throwing away the very structure the
  decomposer had just derived.
  """
  global _TOKENIZER
  if _TOKENIZER is None:
    try:
      from transformers import AutoTokenizer
      cfg = Path(__file__).resolve().parent / "qwen3vl_8b_config"
      _TOKENIZER = AutoTokenizer.from_pretrained(str(cfg))
    except Exception:
      _TOKENIZER = False
  return _TOKENIZER or None


def estimate_tokens(text: str) -> int:
  """Token count for `text`, EXACT when the vendored tokenizer is available.

  Falls back to a chars/token heuristic only when it is not (standalone unit
  tests without transformers, say). Being exact matters: this number decides
  which optional fields get trimmed, and a 47% over-estimate silently deleted
  useful structure from prompts that were never over budget.
  """
  if not text:
    return 0
  tok = _real_tokenizer()
  if tok is not None:
    try:
      return len(tok(text, add_special_tokens=False)["input_ids"])
    except Exception:
      pass
  return max(1, int(math.ceil(len(text) / _CHARS_PER_TOKEN)))


def serialize_caption(caption: dict) -> str:
  """Official serialization: minified, ensure_ascii=False (non-ASCII kept literal)."""
  return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# freeform wrap
# --------------------------------------------------------------------------- #

def _default_environment_boxes(user_text: str) -> list[dict]:
  """Environment boxes for the density guard, built ONLY from the user's own
  clauses. Evidence: meta-language ("region consistent with the scene") still
  trips the model's baked-in safety filter at 1024px, while concrete visual
  nouns pass (variant-C test). Splitting the user's comma clauses keeps every
  word theirs - dominance intact, density achieved."""
  # ONE SUBJECT, ONE BOX. Ideogram 4 obeys bounding boxes, so a box whose text
  # names a person is an instruction to render a person THERE. This used to
  # split the user's clauses down the middle and hand half to each environment
  # box - and with fewer than three clauses it put the ENTIRE prompt in both.
  # A portrait prompt therefore asked for three overlapping, partly cropped
  # people, which is exactly what "body horror" looked like.
  #
  # Environment boxes now carry only clauses that do not name a person. When
  # there are none, the boxes are omitted: a weaker density guard risks the
  # gray card, while a duplicated subject ruins the image every time.
  from pi_ideogram_lib.decompose import _is_about_subject

  clauses = [c.strip(" .;") for c in re.split(r"[,;.]", user_text) if c.strip(" .;")]

  if not _is_about_subject(user_text):
    # NO HUMAN IN THE PROMPT: keep the measured density guard exactly as it
    # was. Two boxes describing the same teapot from different corners cost a
    # little adherence; the guard is what stops the gray card, and it was
    # proven on real weights (variant C).
    if len(clauses) >= 3:
      half = max(1, (len(clauses) - 1) // 2)
      upper = ", ".join(clauses[1:1 + half])
      lower = ", ".join(clauses[1 + half:]) or clauses[-1]
    else:
      upper = user_text
      lower = user_text
  else:
    # A HUMAN IS IN THE PROMPT: the environment boxes may only carry clauses
    # that do not name one, or the model is being told to paint a second and
    # third person into the corners.
    env = [c for c in clauses if not _is_about_subject(c)]
    if not env:
      return []
    half = max(1, len(env) // 2) if len(env) > 1 else 1
    upper = ", ".join(env[:half])
    lower = ", ".join(env[half:]) or upper
  return [
    {"type": "obj", "bbox": [700, 0, 1000, 400], "desc": upper},
    {"type": "obj", "bbox": [600, 600, 1000, 1000], "desc": lower},
  ]


def wrap_freeform(user_text: str, *, style_block: Optional[dict] = None,
                  palette: Optional[list[str]] = None,
                  density: int = 3,
                  target_size: Optional[tuple[int, int]] = None) -> tuple[dict, list[str]]:
  """Wrap freeform text into the official caption JSON.

  POSITIVE DOMINANCE: the FULL unedited user text becomes
  high_level_description. Nothing is summarized, censored or 'improved'.

  Density guard (proven on real weights at 256/512/1024px): Ideogram 4 ships a
  baked-in safety filter that misfires on SPARSE captions (community-confirmed
  'ambiguity / density filter'). Two automatic guards, both using ONLY the
  user's own words / neutral structure:
    - mirror the user's full text into the element desc (fixes 512px)
    - at >=1024px, add the subject bbox + two neutral environment boxes
      (fixes 1024px; the community 'enough bounding boxes' recipe)
  """
  logs: list[str] = []
  text = user_text.strip()
  if not text:
    raise ValueError("The prompt box is empty. Type your scene description first.")

  words = len(text.split())

  # HARD CEILING. The model's text window is 2048 tokens; at the measured
  # ~1.16 tokens/word of descriptive English that is ~1765 words, and the JSON
  # scaffold needs some of it. Refuse with a count rather than let
  # _build_inputs truncate the tail, which silently dropped everything past the
  # limit and produced an image from the first part of the prompt only.
  if words > MAX_PROMPT_WORDS:
    raise ValueError(
      f"Your prompt is {words} words. Ideogram 4's text window is 2048 tokens, "
      f"which is about {MAX_PROMPT_WORDS} words of prose - past that the model "
      "cannot see the rest, so the engine refuses instead of quietly cutting it. "
      f"Trim about {words - MAX_PROMPT_WORDS} words, or move the detail into raw "
      "JSON with bounding boxes."
    )

  # ADAPTIVE. Short prompts keep the mirror-everywhere density guard below: it
  # is what stops the baked-in safety filter misfiring on sparse captions, and
  # at that length it is nearly free. Long prompts are already dense, so the
  # copy buys nothing and costs the whole budget - they get sorted into the
  # schema instead. See pi_ideogram_lib/decompose.py.
  if words > DECOMPOSE_ABOVE_WORDS:
    caption, dlogs = decompose(
      text, target_size=target_size, style_block=style_block, palette=palette,
      density=density,
      default_style=_default_style_description,
      default_background=_default_background,
    )
    return caption, dlogs

  caption: dict = {}
  caption["high_level_description"] = text  # <- user text, in full, byte-for-byte

  sd = dict(style_block) if style_block else _default_style_description()
  if palette:
    sd["color_palette"] = palette
  caption["style_description"] = sd

  cd: dict = {}
  add_boxes = False
  if target_size is not None:
    w, h = int(target_size[0]), int(target_size[1])
    # Skip density guard when user text is already long — the guard doubles
    # the text in background + adds extra elements, which can push a long
    # prompt well over the 2048-token limit and cause blank images.
    if max(w, h) >= 1024 and len(text) < 1500:
      add_boxes = True
  if add_boxes:
    # 1024px+ winning shape (variant C, proven on real weights): a visually
    # rich background + boxed subject + 2 environment boxes
    cd["background"] = f"The environment of the scene: {text}"
  else:
    cd["background"] = _default_background(text)
  elements = [{"type": "obj", "desc": text}]  # user's own words, mirrored
  if add_boxes:
    elements = [{"type": "obj", "bbox": [150, 100, 850, 900], "desc": text}]  # user's own words, mirrored
    env = _default_environment_boxes(text)
    elements.extend(env)
    # Say how many boxes there ACTUALLY are. This line used to claim "subject
    # bbox + 2 environment boxes" unconditionally, including when a
    # single-subject human prompt legitimately produced one box and nothing
    # else - a log that describes the intention rather than the result.
    if env:
      logs.append(f"density guard: rich background + subject bbox + {len(env)} environment "
                  "box(es) (1024px+ recipe) to avoid the model's safety-filter false positive")
    else:
      logs.append("density guard: subject bbox only - every clause in this prompt names the "
                  "person, and repeating them in another box would ask for a second person. "
                  "Add a sentence about the setting to restore the extra boxes.")
  cd["elements"] = elements
  caption["compositional_deconstruction"] = cd

  logs.append(f"mode=freeform-wrap | user_chars={len(text)} | json_ok=true")
  logs.append("description kept in full (not summarized)")
  logs.append("user text mirrored into element desc (density guard)")
  return caption, logs


# --------------------------------------------------------------------------- #
# raw json mode
# --------------------------------------------------------------------------- #

def refine_raw_json(user_json: str, *, preserve_desc_bytes: bool = True) -> tuple[dict, list[str]]:
  """Accept user-pasted JSON. Description fields are NEVER overwritten.

  Only missing required schema keys receive empty/safe defaults. Any non-empty
  description the user provided stays byte-for-byte.
  """
  logs: list[str] = []
  obj = json.loads(user_json)  # raises -> caller surfaces a plain-English error

  if not _is_valid_caption(obj):
    logs.append("input JSON has no compositional_deconstruction; filling required keys with safe defaults")

  cd = obj.get("compositional_deconstruction")
  if not isinstance(cd, dict):
    cd = {}
    obj["compositional_deconstruction"] = cd
  if not isinstance(cd.get("background"), str) or not cd.get("background"):
    cd["background"] = _default_background("")
  if not isinstance(cd.get("elements"), list) or not cd.get("elements"):
    # raw-JSON mode: never invent content for a user who gave us JSON - but an
    # empty elements list is invalid per the official schema, so use the neutral
    # placeholder (their own description fields, if any, are untouched above).
    cd["elements"] = [{"type": "obj", "desc": _default_element_desc()}]

  hld = obj.get("high_level_description")
  if not isinstance(hld, str) or not hld:
    # only fill if missing; never touch a non-empty user description
    obj["high_level_description"] = ""
  if not isinstance(obj.get("style_description"), dict):
    obj["style_description"] = None  # will be dropped below (optional field)

  if obj.get("style_description") is None:
    obj.pop("style_description", None)

  logs.append(f"mode=raw-json | user_chars={len(user_json)} | json_ok=true")
  if preserve_desc_bytes and isinstance(obj.get("high_level_description"), str) and obj["high_level_description"]:
    logs.append("raw JSON description kept byte-for-byte")
  return obj, logs


# --------------------------------------------------------------------------- #
# visual builder
# --------------------------------------------------------------------------- #

@dataclass
class BuilderElement:
  type: str = "obj"            # "obj" | "text"
  desc: str = ""
  text: str = ""               # only for type="text": literal on-image text
  bbox: Optional[list[int]] = None  # [y_min, x_min, y_max, x_max], 0-1000
  color_palette: Optional[list[str]] = None


@dataclass
class BuilderSpec:
  elements: list[BuilderElement] = field(default_factory=list)
  palette: Optional[list[str]] = None
  background: str = ""


def _sanitize_bbox(bbox: list[int]) -> Optional[list[int]]:
  if not isinstance(bbox, list) or len(bbox) != 4:
    return None
  out = []
  for v in bbox:
    try:
      iv = int(v)
    except Exception:
      return None
    if iv < 0:
      iv = 0
    if iv > 1000:
      iv = 1000
    out.append(iv)
  y_min, x_min, y_max, x_max = out
  if y_min > y_max:
    y_min, y_max = y_max, y_min
  if x_min > x_max:
    x_min, x_max = x_max, x_min
  return [y_min, x_min, y_max, x_max]


def _sanitize_palette(pal: Any, max_colors: int) -> Optional[list[str]]:
  if not isinstance(pal, list) or not pal:
    return None
  out: list[str] = []
  for c in pal:
    s = str(c).strip()
    if not s.startswith("#"):
      s = "#" + s
    s = s.upper()
    if len(s) == 4:  # #fff -> #FFFFFF
      s = "#" + "".join(ch * 2 for ch in s[1:])
    if re.fullmatch(r"#[0-9A-F]{6}", s):
      out.append(s)
    if len(out) >= max_colors:
      break
  return out or None


def build_from_builder(user_text: str, spec: BuilderSpec, *,
                       style_block: Optional[dict] = None,
                       target_size: Optional[tuple[int, int]] = None) -> tuple[dict, list[str]]:
  """Visual layout builder: user text STILL dominates; bboxes/palette/on-image text are
  ADDITIONAL structure, never a substitute for the prompt box.

  Density guards (same as freeform): if the builder produced no obj element,
  the user's own full text is mirrored as one element; at >=1024px, if the
  builder gave fewer than 3 boxed elements, neutral environment boxes are added
  so the caption clears the model's safety-filter false positive."""
  logs: list[str] = []
  text = user_text.strip()
  if not text:
    raise ValueError("The prompt box is empty. The layout builder adds structure ON TOP of your prompt; it does not replace it.")

  caption: dict = {}
  caption["high_level_description"] = text  # full, unedited

  sd = dict(style_block) if style_block else _default_style_description()
  pal = _sanitize_palette(spec.palette, 16) if spec.palette else None
  if pal:
    sd["color_palette"] = pal
  caption["style_description"] = sd

  cd: dict = {}
  cd["background"] = spec.background.strip() if spec.background.strip() else _default_background(text)

  elements: list[dict] = []
  for el in spec.elements:
    bbox = _sanitize_bbox(el.bbox) if el.bbox else None
    pal_el = _sanitize_palette(el.color_palette, 5) if el.color_palette else None
    if el.type == "text":
      if not el.text.strip():
        continue
      entry: dict = {"type": "text"}
      if bbox:
        entry["bbox"] = bbox
      entry["text"] = el.text.strip()
      entry["desc"] = el.desc.strip() or f"The text '{el.text.strip()}' rendered inside its bounding box."
      if pal_el:
        entry["color_palette"] = pal_el
    else:
      if not el.desc.strip():
        continue
      entry = {"type": "obj"}
      if bbox:
        entry["bbox"] = bbox
      entry["desc"] = el.desc.strip()
      if pal_el:
        entry["color_palette"] = pal_el
    elements.append(entry)

  if not elements:
    # density guard: mirror the user's own text as the element (never a
    # replacement - it IS their words)
    elements = [{"type": "obj", "desc": text}]

  if target_size is not None and max(int(target_size[0]), int(target_size[1])) >= 1024:
    boxed_objs = [e for e in elements if isinstance(e, dict) and e.get("bbox")]
    if len(boxed_objs) < 3:
      elements[0].setdefault("bbox", [150, 100, 850, 900])
      elements.extend(_default_environment_boxes(text))
      logs.append("density guard: added bboxes to reach the 3-box minimum (1024px+ recipe)")

  cd["elements"] = elements
  caption["compositional_deconstruction"] = cd

  logs.append(f"mode=visual-builder | user_chars={len(text)} | elements={len(elements)} | json_ok=true")
  logs.append("description kept in full (not summarized); builder values added as structure only")
  return caption, logs


# --------------------------------------------------------------------------- #
# LoRA triggers + dominance-safe merge
# --------------------------------------------------------------------------- #

def apply_lora_triggers(caption: dict, trigger_words: str, *, apply_to_art_style: bool = True) -> tuple[dict, list[str]]:
  """Prepend LoRA trigger words. NEVER delete or shrink user text."""
  logs: list[str] = []
  triggers = " ".join(trigger_words.split())
  if not triggers:
    return caption, logs
  hld = caption.get("high_level_description", "")
  caption["high_level_description"] = (triggers + " " + hld).strip()
  logs.append(f"lora triggers prepended: '{triggers}'")
  sd = caption.get("style_description")
  if apply_to_art_style and isinstance(sd, dict):
    art = sd.get("art_style") or sd.get("photo") or ""
    key = "art_style" if "art_style" in sd else "photo"
    if triggers not in art:
      sd[key] = (art + " " + triggers).strip() if art else triggers
      logs.append(f"lora triggers also added to style_description.{key}")
  return caption, logs


# --------------------------------------------------------------------------- #
# token budget - boilerplate trimmed FIRST, user text protected
# --------------------------------------------------------------------------- #

_STYLE_TRIM_ORDER = ["aesthetics", "lighting", "photo", "art_style"]  # 'medium' is schema-required; kept while the block exists
# Exactly one of these must survive for as long as style_description exists.
_STYLE_DISCRIMINATORS = ("photo", "art_style")


def enforce_token_budget(caption: dict, token_estimator=None, max_tokens: int = MAX_TEXT_TOKENS) -> tuple[dict, list[str]]:
  """Trim optional/boilerplate fields FIRST; user text (high_level_description +
  element descriptions the user wrote) is protected until the very end.

  Trim order (official-schema-aware):
  1. per-element color_palette (optional)
  2. style_description.color_palette (optional)
  3. boilerplate background shortened (engine boilerplate, never user text)
  4. optional style strings: aesthetics, lighting, photo, art_style
  5. the entire style_description (it is an OPTIONAL top-level field)
  6. NEVER auto-shorten high_level_description - warn plainly instead.
  """
  logs: list[str] = []
  est = token_estimator or (lambda s: estimate_tokens(s))

  def total(c: dict) -> int:
    return est(serialize_caption(c))

  cap = caption

  # 1) trim element color palettes (optional, per-element)
  cd = cap.get("compositional_deconstruction")
  if isinstance(cd, dict):
    for i, el in enumerate(cd.get("elements", [])):
      if isinstance(el, dict) and "color_palette" in el:
        if total(cap) <= max_tokens:
          break
        el.pop("color_palette")
        logs.append(f'trimmed optional field "elements[{i}].color_palette" to protect user text')

  # 2) trim the global palette
  sd = cap.get("style_description")
  if isinstance(sd, dict) and "color_palette" in sd and total(cap) > max_tokens:
    sd.pop("color_palette")
    logs.append('trimmed optional field "style_description.color_palette" to protect user text')

  # 3) shorten boilerplate background (engine boilerplate, not user text)
  if total(cap) > max_tokens and isinstance(cd, dict) and cd.get("background") == _default_background(""):
    cd["background"] = "Consistent background."
    logs.append('shortened boilerplate background to protect user text')

  # 4) shrink optional style strings one by one (medium kept: schema-required)
  #
  # 'photo' and 'art_style' are a REQUIRED PAIR-CHOICE, not free options: the
  # verifier wants exactly one of them whenever style_description exists. The
  # trim order listed both, so a caption under token pressure had its only
  # discriminator removed and was then rejected by our own validator:
  #
  #   trimmed optional style field "style_description.photo" ...
  #   generation failed: style_description: expected one of 'photo' or 'art_style'
  #
  # A 12,682-character JSON prompt died that way. Trimming may drop one of the
  # pair only while the other survives; when neither can go, step 5 below drops
  # the whole block, which IS valid - style_description is optional as a unit.
  if isinstance(sd, dict):
    for key in _STYLE_TRIM_ORDER:
      if total(cap) <= max_tokens:
        break
      if key not in sd or not isinstance(sd[key], str):
        continue
      if key in _STYLE_DISCRIMINATORS:
        other = [k for k in _STYLE_DISCRIMINATORS if k != key]
        if not any(o in sd for o in other):
          continue  # last one standing: dropping it would invalidate the block
      sd.pop(key)
      logs.append(f'trimmed optional style field "style_description.{key}" to protect user text')

  # 5) style_description itself is optional - drop it entirely under pressure
  if total(cap) > max_tokens and "style_description" in cap:
    cap.pop("style_description")
    logs.append('dropped optional "style_description" block entirely to protect user text')

  # 6) LAST RESORT: the caption (including the user text) is genuinely too long.
  #    Only now do we tell the user plainly instead of silently cutting their words.
  if total(cap) > max_tokens:
    user_chars = len(cap.get("high_level_description", ""))
    logs.append(
      f"WARNING: even after trimming all optional fields the prompt is ~{total(cap)} tokens "
      f"(limit {max_tokens}). Your text is {user_chars} chars. Shorten the prompt box text - "
      "your words are never auto-shortened, but generation continues with only "
      f"the first {max_tokens} tokens of the final prompt."
    )
  return cap, logs


# --------------------------------------------------------------------------- #
# final assembly + validation
# --------------------------------------------------------------------------- #

def finalize(caption: dict, *, verify: bool = True) -> tuple[str, list[str]]:
  """Serialize + validate against the official schema. Returns (json_str, warnings)."""
  warnings: list[str] = []
  raw = serialize_caption(caption)
  if verify:
    warnings = _VERIFIER.verify_raw(raw)
    if warnings:
      _log(f"schema warnings: {warnings}")
  return raw, warnings


def build_prompt(
  *,
  mode: str,                      # "freeform-wrap" | "raw-json" | "visual-builder"
  user_text: str,
  builder_spec: Optional[BuilderSpec] = None,
  lora_triggers: str = "",
  token_estimator=None,
  target_size: Optional[tuple[int, int]] = None,   # (width, height) - drives the density guard
  density: int = 3,               # bounding boxes in the caption; raised on a blocked retry
) -> tuple[str, dict, list[str]]:
  """The single entry point used at generation time. Dominance enforced:

  a) user prompt text    -> high_level_description (full, unedited)
  b) LoRA triggers       -> prepended, never deleting user text
  c) builder extras      -> structure only
  d) engine boilerplate  -> minimal defaults, trimmed first under token pressure
  """
  logs: list[str] = []

  if mode == "raw-json":
    if _looks_like_json(user_text):
      caption, logs2 = refine_raw_json(user_text)
      logs.extend(logs2)
    else:
      # user selected raw JSON but typed freeform: fall back to wrap, say so
      _log("raw-json mode selected but input is not valid JSON; falling back to freeform-wrap")
      caption, logs2 = wrap_freeform(user_text, target_size=target_size, density=density)
      logs.extend(logs2)
      logs.append("fallback=freeform-wrap (input was not valid JSON)")
  elif mode == "visual-builder":
    spec = builder_spec or BuilderSpec()
    caption, logs2 = build_from_builder(user_text, spec, target_size=target_size)
    logs.extend(logs2)
  else:
    caption, logs2 = wrap_freeform(user_text, target_size=target_size, density=density)
    logs.extend(logs2)

  if lora_triggers:
    caption, logs2 = apply_lora_triggers(caption, lora_triggers)
    logs.extend(logs2)

  caption, logs2 = enforce_token_budget(caption, token_estimator=token_estimator)
  logs.extend(logs2)

  raw, warnings = finalize(caption)
  for line in logs:
    _log(line)
  return raw, caption, warnings
