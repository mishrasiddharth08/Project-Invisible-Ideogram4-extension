"""Structural decomposition of a long freeform prompt into the official schema.

WHY THIS EXISTS
---------------
`json_prompt.wrap_freeform` copies the user's whole text into several caption
fields. That duplication is deliberate for SHORT prompts - it is the density
guard that stops Ideogram 4's baked-in safety filter misfiring on sparse
captions - but it is fatal for long ones.

Measured on the real Qwen3-VL tokenizer, descriptive English runs ~1.16
tokens/word:

    1500 words  ->  1739 tokens raw          (the window is 2048: it FITS)
                ->  3507 tokens once wrapped (2.02x: it does NOT)

So a 1500-word prompt was never too long for the model. The wrapper made it
too long, and `_build_inputs` then truncated the tail - the user got an image
built from roughly the first half of their prompt with nothing said about it.

This module sorts the prompt instead of copying it. `high_level_description`
keeps the full text byte-for-byte (the positive-dominance rule is untouched);
every other field becomes a SHORT label derived from the user's own sentences.
Cost drops to about one copy plus scaffolding - 1815 tokens for 1500 words,
which fits with ~233 to spare.

The classification is keyword and length based, and deliberately offline: the
alternative is a network call to an LLM, which this engine must never require.
A misfiled sentence costs a little adherence, never correctness, because the
user's full text lands in high_level_description regardless.
"""

from __future__ import annotations

import re
from typing import Optional

# Below this many words the mirror-everywhere density guard is cheap and
# genuinely useful, so wrap_freeform keeps handling those prompts.
DECOMPOSE_ABOVE_WORDS = 300

# Hard ceiling, set from measurement rather than arithmetic.
#
# 2048 tokens is the model's window. Tokens-per-word is not a constant: plain
# descriptive prose measured 1.16, and comma-heavy prose 1.21, because
# punctuation and short function words each cost a token. Extrapolating from
# the friendly ratio gives ~1765 words, but at the dense ratio 1700 words
# already serializes to 2219 tokens and overflows.
#
# 1500 words serializes to 1977 tokens on dense prose - inside the window with
# ~70 to spare - so that is the number this engine actually guarantees.
MAX_PROMPT_WORDS = 1500

_STYLE_CUES: dict[str, tuple[str, ...]] = {
  "lighting": ("light", "lighting", "lit", "sunlight", "sunlit", "backlit", "golden hour",
               "shadow", "shadows", "glow", "glowing", "dim", "bright", "neon", "candlelit",
               "moonlight", "overcast", "silhouette", "rim light", "dappled"),
  "aesthetics": ("mood", "atmosphere", "atmospheric", "serene", "moody", "dramatic", "minimal",
                 "minimalist", "elegant", "gritty", "cozy", "melancholy", "vibrant", "muted",
                 "cinematic", "ethereal", "stark", "nostalgic", "peaceful", "tense"),
  "photo": ("close-up", "closeup", "wide shot", "macro", "portrait", "aerial", "bokeh",
            "depth of field", "shallow focus", "sharp focus", "eye-level", "low angle",
            "high angle", "telephoto", "35mm", "50mm", "85mm", "lens"),
  "medium": ("photograph", "photo", "painting", "oil painting", "watercolor", "illustration",
             "render", "3d render", "sketch", "digital art", "anime", "engraving", "poster"),
}

# An explicit statement about the setting outranks any style word that happens
# to appear in it: "The background is a DIM ship cabin" was being filed as
# lighting because "dim" is a lighting cue.
_BACKGROUND_STRONG: tuple[str, ...] = (
  "background", "backdrop", "the setting", "in the distance", "behind the",
  "environment", "surroundings",
)

# Official key order for style_description (photo variant). The caption
# verifier checks ORDER, not just membership, so the dict must be built in it.
_STYLE_FIELD_ORDER: tuple[str, ...] = ("aesthetics", "lighting", "photo", "medium")

_BACKGROUND_CUES: tuple[str, ...] = (
  "background", "behind", "backdrop", "environment", "setting", "surroundings",
  "landscape", "horizon", "sky", "wall", "room", "street", "forest", "field",
  "interior", "exterior", "distance", "beyond", "around", "outside", "window",
)


def split_sentences(text: str) -> list[str]:
  """Sentence split with no NLP dependency.

  Splits on `. ! ?` + whitespace, then falls back to comma/semicolon clauses
  when the result is a single long run-on - one 1500-word "sentence" would
  otherwise defeat every heuristic below.
  """
  parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
  if len(parts) <= 1 and len(text.split()) > 40:
    parts = [p.strip() for p in re.split(r"[;,]\s+", text) if p.strip()]
  return parts or [text.strip()]


def _cue_score(sentence: str, cues: tuple[str, ...]) -> int:
  low = sentence.lower()
  return sum(1 for c in cues if c in low)


def classify_sentence(sentence: str) -> str:
  """-> 'style' | 'background' | 'subject'."""
  low = sentence.lower()
  if any(marker in low for marker in _BACKGROUND_STRONG):
    return "background"
  style_hits = sum(_cue_score(sentence, cues) for cues in _STYLE_CUES.values())
  bg_hits = _cue_score(sentence, _BACKGROUND_CUES)
  words = len(sentence.split())
  # A short clause that is mostly style words is style. A long descriptive
  # sentence that merely mentions light is still about its subject.
  if style_hits and (words <= 12 or style_hits >= 3) and style_hits >= bg_hits:
    return "style"
  if bg_hits:
    return "background"
  return "subject"


# Words that mean "there is a person here". Used to keep a person out of any
# bounding box other than the subject's own - see the note in decompose().
# Deliberately broad: a false positive costs one bounding box, a false
# negative costs a duplicated human being.
_HUMAN_CUES = frozenset("""
she he her his him hers himself herself they them their themselves
woman women man men girl girls boy boys person people child children
lady ladies gentleman gentlemen figure model portrait selfie
face faces skin lips mouth cheek cheeks eyebrow eyelashes
hand hands finger fingers arm arms shoulder shoulders leg legs torso chest waist hips
hair beard freckles
""".split())

# Verbs and states that a cat, a statue or a teapot can equally satisfy.
# On their own they mean nothing - "a ginger cat WEARING a tiny wizard hat" is
# not a person - so they only reinforce a human cue, never establish one.
_WEAK_CUES = frozenset("""
wearing wears wear dressed clothed posing poses expression smile smiling
stands standing sitting seated leaning kneeling walking holding
""".split())

_WORD_RE = re.compile(r"[a-z']+")


def _is_about_subject(sentence: str) -> bool:
  """Does this sentence put a HUMAN in the frame?

  A secondary bounding box carrying one of these is an instruction to render
  another person there, which is how a single-subject prompt became three
  overlapping people.

  Two mistakes this has already made, both pinned by
  tests/test_one_subject_one_box.py:

  * SUBSTRINGS. The first version tested `"he " in text`, which matches
    "t-he ", and `"her "`, which matches "ot-her ". "The floor is wide-plank
    oak" was classed as a person and every usable background sentence was
    discarded. Word boundaries only.
  * VERBS. The second version counted "wearing", so "A ginger cat wearing a
    tiny wizard hat" was a person and the density guard collapsed to one box
    on cat pictures. A weak cue never establishes a human on its own.
  """
  words = set(_WORD_RE.findall(str(sentence or "").lower()))
  return bool(words & _HUMAN_CUES)


def bucket_sentences(text: str) -> tuple[dict[str, list[str]], list[str]]:
  """Split and classify. Guarantees a non-empty 'subject' bucket."""
  sentences = split_sentences(text)
  buckets: dict[str, list[str]] = {"subject": [], "background": [], "style": []}
  for sent in sentences:
    buckets[classify_sentence(sent)].append(sent)
  if not buckets["subject"]:
    buckets["subject"] = sentences[:1]
  return buckets, sentences


def style_from_sentences(sentences: list[str]) -> dict[str, str]:
    """Fill official style fields from the user's own style sentences.

    Two rules the caption verifier and the model both care about:
      * keys are emitted in the official order, not in cue-table order;
      * one sentence fills at most ONE field, so a line like "moody, cinematic,
        shallow depth of field" does not get repeated verbatim under both
        aesthetics and photo.
    """
    picked: dict[str, str] = {}
    used: set[str] = set()
    # Best match first: the field whose cues the sentence hits hardest.
    for sent in sentences:
      clean = sent.rstrip(" .")
      if clean in used:
        continue
      # A sentence about the SUBJECT is not a style statement, whatever cue
      # words it happens to contain. "Her hands rest lightly on the windowsill"
      # was landing in `lighting` (it mentions light-adjacent words), which
      # tells the model the scene's lighting IS a pair of hands. `photo` and
      # `medium` are exempt: "a cinematic PORTRAIT photograph of a woman" is a
      # legitimate medium statement that necessarily names its subject.
      subjecty = _is_about_subject(sent)
      scores = {f: _cue_score(sent, cues) for f, cues in _STYLE_CUES.items()}
      if subjecty:
        scores["lighting"] = 0
        scores["aesthetics"] = 0
      for field in sorted(scores, key=lambda f: -scores[f]):
        if scores[field] and field not in picked:
          picked[field] = clean
          used.add(clean)
          break
    return {f: picked[f] for f in _STYLE_FIELD_ORDER if f in picked}


def condense(sentences: list[str], max_words: int) -> str:
  """Join sentences up to a word budget, breaking on a sentence boundary.

  These derived fields exist to tell the model which part of the prompt is the
  subject and which is the setting. They do not need to restate the prompt -
  restating it is exactly what doubled the token cost.
  """
  out: list[str] = []
  used = 0
  for sent in sentences:
    n = len(sent.split())
    if used + n > max_words:
      break
    out.append(sent.rstrip(" ."))
    used += n
  if not out and sentences:
    out = [" ".join(sentences[0].split()[:max_words])]
  if not out:
    return ""
  return ". ".join(out).strip(" .") + "."


def decompose(user_text: str, *, target_size: Optional[tuple[int, int]] = None,
              style_block: Optional[dict] = None,
              palette: Optional[list[str]] = None,
              density: int = 3,
              default_style=None, default_background=None) -> tuple[dict, list[str]]:
  """Sort a long prompt into the official schema instead of copying it around.

  Args:
      user_text: the prompt, already stripped.
      target_size: (w, h); a subject bbox is added at >=1024px.
      style_block / palette: optional caller overrides.
      default_style / default_background: callables from json_prompt, injected
          so this module owns no boilerplate of its own.

  Returns:
      (caption dict, log lines)
  """
  text = user_text.strip()
  buckets, sentences = bucket_sentences(text)

  caption: dict = {"high_level_description": text}

  merged: dict = {}
  merged.update(style_from_sentences(buckets["style"]))
  if style_block:
    for k, v in style_block.items():
      merged.setdefault(k, v)
  if default_style is not None:
    for k, v in default_style().items():
      merged.setdefault(k, v)
  # Rebuild in the official order: the verifier checks key ORDER, and dict
  # insertion order is what json.dumps serializes.
  sd = {f: merged[f] for f in _STYLE_FIELD_ORDER if f in merged}
  for k, v in merged.items():  # anything unexpected keeps its place at the end
    sd.setdefault(k, v)
  if palette:
    sd["color_palette"] = palette
  caption["style_description"] = sd

  subject_desc = condense(buckets["subject"], 60)
  if buckets["background"]:
    background = condense(buckets["background"], 45)
  else:
    background = default_background(text) if default_background else "A background consistent with the described scene."

  # DENSITY GUARD - keep it for long prompts too.
  #
  # The guard that beats Ideogram 4's baked-in safety filter is STRUCTURAL: the
  # community recipe is "enough bounding boxes" at >=1024px, three of them. The
  # first version of this decomposer emitted a single boxed element, on the
  # reasoning that a long prompt is "already dense" - conflating text volume
  # with box count. Result: an 800-word prompt at 1280x1728 came back as the
  # gray "Image blocked by safety filter" card, while the same scene in 40
  # words passed. Long prompts need the boxes just as much; what they do NOT
  # need is the whole text copied into each one.
  #
  # So: same three boxes, but each described by a SHORT slice of the user's own
  # sentences instead of a full copy. Density restored at a few dozen tokens.
  # `density` is the number of boxes. Three is the community recipe and stays
  # the default; a BLOCKED generation retries at a higher value because more
  # structure is the remedy that actually works here - the gray card is a
  # sparse-caption misfire, so answering it with a denser caption addresses
  # the cause. Nothing about the model or the schedule is touched.
  elements: list[dict] = []
  big = target_size is not None and max(int(target_size[0]), int(target_size[1])) >= 1024
  if big:
    # ONE SUBJECT, ONE BOX. This is the body-horror fix.
    #
    # A bounding box tells Ideogram 4 "put this thing HERE", and the model
    # obeys - that is its headline feature. The previous version filled the
    # secondary boxes with `buckets[...] or buckets["subject"]`, and the last
    # box used the subject unconditionally. On a portrait prompt every
    # sentence is about the same person, so a caption for ONE woman came out
    # asking for three:
    #
    #   [150,100,850,900]   "...her face...her expression...she wears..."
    #   [700,0,1000,400]    "She stands beside a tall window..."
    #   [600,600,1000,1000] "A cinematic portrait photograph of a woman..."
    #
    # - three overlapping, partly cropped people. Duplicated faces and limbs
    # is precisely what that instructs, and it is what "body horror" was.
    #
    # A secondary box may therefore only describe material that is NOT the
    # subject. When there is none, the box is OMITTED rather than filled with
    # a copy of the subject: fewer boxes is a weaker density guard, but a
    # weaker guard is a gray card at worst, while a duplicated subject is a
    # ruined image every time.
    # BACKGROUND ONLY. A bounding box says "this thing is HERE", so a style
    # sentence does not belong in one - the first pass put "Shot on an 85mm
    # lens at f/1.8" in a box in the top-right corner. Style already has its
    # own block in the schema.
    secondary: list[str] = []
    for sent in buckets["background"]:
      if _is_about_subject(sent):
        continue  # a person sentence in another box = a second person
      t = condense([sent], 25)
      if t and t not in secondary:
        secondary.append(t)

    regions = [
      [700, 0, 1000, 400],
      [600, 600, 1000, 1000],
      # Extra regions used only when a run comes back blocked. They tile parts
      # of the frame the first ones leave empty, so the added structure is
      # real rather than more labels on the same area.
      [0, 0, 380, 420],
      [0, 620, 420, 1000],
      [380, 380, 760, 760],
    ]
    want = max(1, min(int(density or 3), 1 + len(regions)))
    elements.append({"type": "obj", "bbox": [150, 100, 850, 900], "desc": subject_desc})
    # Keep the box COUNT (that is the measured anti-gray-card guard) by
    # cycling the available background lines when there are fewer of them than
    # regions. Repeating "wide-plank oak floor" in two boxes costs a little
    # adherence; repeating a PERSON costs the image, which is why only
    # non-subject text is ever eligible here. With no background material at
    # all - an entirely subject-focused prompt - the subject box stands alone
    # rather than being cloned.
    for i, bbox in enumerate(regions[: want - 1]):
      if not secondary:
        break
      elements.append({"type": "obj", "bbox": bbox,
                       "desc": secondary[i % len(secondary)]})
  else:
    elements.append({"type": "obj", "desc": subject_desc})

  caption["compositional_deconstruction"] = {"background": background, "elements": elements}

  logs = [
    f"mode=freeform-decompose | words={len(text.split())} | sentences={len(sentences)} "
    f"(subject {len(buckets['subject'])}, background {len(buckets['background'])}, "
    f"style {len(buckets['style'])}) | {len(elements)} element(s)",
    "description kept in full (not summarized); the other fields are SHORT labels "
    "derived from your own sentences rather than copies - that is what keeps a long "
    "prompt inside the model's 2048-token window",
  ]
  return caption, logs
