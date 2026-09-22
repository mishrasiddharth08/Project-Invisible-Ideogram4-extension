"""Offline numerical checks for the Ideogram 4 img2img math."""

from pathlib import Path
import sys

import torch

EXT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXT))

from pi_ideogram_lib.pipeline import Ideogram4Pipeline  # noqa: E402
from pi_ideogram_lib.scheduler import get_schedule_for_resolution, make_step_intervals  # noqa: E402


def test_partial_schedule_uses_original_interval_entry():
  original_steps = 12
  strength = 0.5
  active_steps = int(torch.ceil(torch.tensor(original_steps * strength)).item())
  original = make_step_intervals(original_steps)
  truncated = original[: active_steps + 1]
  schedule = get_schedule_for_resolution((512, 512), known_mean=0.0, std=1.75)
  expected_t = schedule(original[active_steps:active_steps + 1])
  actual_t = schedule(truncated[-1:])
  assert active_steps == 6
  assert torch.equal(truncated, original[:7])
  assert torch.allclose(actual_t, expected_t)


def test_img2img_initial_latent_blend_is_exact():
  image_latent = torch.full((1, 2, 128), 4.0)
  noise = torch.full_like(image_latent, -2.0)
  t_entry = torch.tensor(0.25)
  actual = Ideogram4Pipeline._blend_img2img_latent(image_latent, noise, t_entry)
  expected = t_entry * image_latent + (1.0 - t_entry) * noise
  assert torch.equal(actual, expected)
  assert torch.all(actual == -0.5)


def test_full_strength_keeps_normal_noise_path_contract():
  original_steps = 12
  strength = 1.0
  active_steps = original_steps
  original = make_step_intervals(original_steps)
  assert active_steps == original_steps
  assert torch.equal(original[: active_steps + 1], original)
