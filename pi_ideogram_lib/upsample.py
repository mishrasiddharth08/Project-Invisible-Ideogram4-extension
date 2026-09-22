"""PROJECT INVISIBLE - LOCAL prompt upsampling for Ideogram 4 (spec P4 / F).

Ideogram 4 is not intended for short natural-language prompts: sparse captions
drift and can trip the baked-in gray-screen attractor. The community recipe
(KJ-node approach) is structured captions over short text.

This module expands a short prompt into a denser, scene-grounded caption using
lightweight TEMPLATES only - no LLM, no cloud API, no network. It is the local
prompt_upsampling path (config.json "prompt_upsampling": true, default off on
<=12 GB). When bypass strength > 0 the template also adds neutral situational
context (P4's situation-bias lever): the scene is described with environment
framing instead of a bare noun list, which the community found pushes the
attractor off.

The official JSON caption builder (json_prompt.build_prompt) remains the
primary path; this module only enriches the freeform text it receives.
"""

from __future__ import annotations

import re

_ENVIRONMENT_FRAMES = (
    "in a softly lit {setting}",
    "during {time}, in {place}",
    "with {detail} in the background",
    "framed by {detail}",
)

_SETTINGS = ("studio", "outdoor", "interior", "urban street", "natural landscape")
_TIMES = ("golden hour", "overcast midday", "evening", "clear morning")
_PLACES = ("a calm setting", "a tasteful venue", "an open space", "a neutral environment")
_DETAILS = ("soft depth of field", "balanced composition", "neutral tones", "gentle rim light")

_WORD_SPLIT = re.compile(r"\s+")


def _density(text: str) -> int:
    """0 = very sparse (<= 4 words), 1 = short, 2 = already dense."""
    words = [w for w in _WORD_SPLIT.split(text.strip()) if w]
    if len(words) <= 4:
        return 0
    if len(words) <= 12:
        return 1
    return 2


def local_upsample(prompt: str, *, bypass_active: bool = False) -> str:
    """Expand `prompt` with neutral scene context when it is sparse.

    Never drops a single user word - context is only appended. Returns the
    original text unchanged when it is already dense (>= 13 words).
    """
    text = (prompt or "").strip()
    if not text:
        return text
    density = _density(text)
    if density >= 2:
        return text

    import random
    rng = random.Random(42)  # deterministic: same prompt -> same expansion
    parts = [text]
    if density == 0:
        setting = rng.choice(_SETTINGS)
        tm = rng.choice(_TIMES)
        place = rng.choice(_PLACES)
        detail = rng.choice(_DETAILS)
        parts.append(_ENVIRONMENT_FRAMES[0].format(setting=setting))
        parts.append(_ENVIRONMENT_FRAMES[1].format(time=tm, place=place))
        parts.append(_ENVIRONMENT_FRAMES[3].format(detail=detail))
    else:
        detail = rng.choice(_DETAILS)
        place = rng.choice(_PLACES)
        parts.append(_ENVIRONMENT_FRAMES[2].format(detail=detail))
        parts.append(place)
    return ". ".join(parts) + "."


def situation_bias(prompt: str, *, strength: float) -> str:
    """P4 situation-bias lever: when bypass strength > 0, re-frame the scene
    with neutral situational context (never a banned-word list - the model's
    attractor responds to caption STRUCTURE, and framing is the documented
    community lever). No-op at strength <= 0.
    """
    try:
        s = float(strength)
    except (TypeError, ValueError):
        s = 0.0
    if s <= 0.0:
        return prompt
    return local_upsample(prompt, bypass_active=True)