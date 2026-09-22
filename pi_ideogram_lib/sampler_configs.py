"""Named sampler presets for Ideogram 4 (official registry, Apache-2.0).

The CFG schedule is stored in LOOP-INDEX order (index 0 = final polish step).
Run order: main steps at gw=7 first, then polish steps at gw=3 - this is the
official "CFGOverride tail" baked into the preset, exactly as the ComfyUI
template expresses it with CFGOverride(3.0, 0.7, 1.0) + DualModelGuider(7.0).
"""

from __future__ import annotations

from .scheduler import SamplerParameters

PRESETS: dict[str, SamplerParameters] = {
  "V4_QUALITY_48": SamplerParameters(
    num_steps=48,
    guidance_schedule=(3.0,) * 3 + (7.0,) * 45,
    mu=0.0,
    std=1.5,
  ),
  "V4_DEFAULT_20": SamplerParameters(
    num_steps=20,
    guidance_schedule=(3.0,) * 2 + (7.0,) * 18,
    mu=0.0,
    std=1.75,
  ),
  "V4_TURBO_12": SamplerParameters(
    num_steps=12,
    guidance_schedule=(3.0,) * 1 + (7.0,) * 11,
    mu=0.5,
    std=1.75,
  ),
  # Distilled Fast packs (cond-only, CFG=1): 20 steps / 8 steps
  "V4_FAST_20": SamplerParameters(
    num_steps=20,
    guidance_schedule=(1.0,) * 20,
    mu=0.0,
    std=1.75,
  ),
  "V4_INSTANT_8": SamplerParameters(
    num_steps=8,
    guidance_schedule=(1.0,) * 8,
    mu=0.5,
    std=1.75,
  ),
}

PRESET_LABELS = {
  "V4_QUALITY_48": "Quality 48 (best)",
  "V4_DEFAULT_20": "Default 20 (balanced)",
  "V4_TURBO_12": "Turbo 12 (fast)",
  "V4_FAST_20": "Fast 20 (distilled, CFG 1)",
  "V4_INSTANT_8": "Instant 8 (distilled, CFG 1)",
}

DISTILL_PRESETS = {"V4_FAST_20", "V4_INSTANT_8"}


def build_custom_schedule(num_steps: int, main_cfg: float, tail_cfg: float, tail_fraction: float) -> tuple[float, ...]:
  """Per-step CFG in loop-index order (index 0 = last step). Main steps at main_cfg, tail steps at tail_cfg."""
  if num_steps < 2:
    return (main_cfg,) * num_steps
  tail_steps = max(1, int(round(num_steps * tail_fraction)))
  schedule = [tail_cfg] * tail_steps + [main_cfg] * (num_steps - tail_steps)
  return tuple(schedule)
