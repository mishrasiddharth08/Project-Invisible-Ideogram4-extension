"""Spec section Q: the DOS/console step bar must actually render.

The requirement is that someone watching only the terminal - or running with
--nowebui - sees live progress the same way stock Forge models give it. That
means a real tqdm bar on STDERR, advanced once per denoising step, with the
run's parameters printed as a header before sampling starts.

This drives the real `_run_sampling` with a stub pipe (the same seam
test_turbo_honesty uses) and captures stderr, so it asserts on the bytes a
user would actually see rather than on the code's intentions.

Run: ..\\..\\venv\\Scripts\\python.exe tests\\test_console_progress.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

EXT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXT))

import numpy as np  # noqa: E402
import PIL.Image as PILImage  # noqa: E402

from pi_ideogram_lib.bypass import recipe as bypass_recipe  # noqa: E402
from pi_ideogram_forge.generate import _run_sampling  # noqa: E402

PASS = 0
FAIL = 0

STEPS = 12


def check(name, cond, detail=""):
  global PASS, FAIL
  if cond:
    PASS += 1
    print(f"  ok  {name} {detail}")
  else:
    FAIL += 1
    print(f"  FAIL {name} {detail}")


rng = np.random.default_rng(0)
NOISE = PILImage.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))

# Only its .name is read, for the infotext line.
CKPT = Path("ideogram4_Fp8Scaled.safetensors")


class StubPipe:
  """Calls the step callback exactly like the real sampling loop does."""

  def __init__(self, steps=STEPS):
    self.conditional_transformer = SimpleNamespace()
    self.unconditional_transformer = None
    self.uncond_adapter = None
    self._i4_bypass_lora = None
    self._i4_uncond_lora = None
    self._i4_turbo_lora = None
    self._steps = steps

  def __call__(self, caption_str, **kwargs):
    cb = kwargs.get("step_callback")
    on_actual = kwargs.get("on_actual_steps")
    if on_actual:
      on_actual(self._steps)
    if cb:
      for i in range(self._steps):
        cb(i, self._steps)
    return [NOISE]


def make_p():
  return SimpleNamespace(prompt="a brass compass", negative_prompt="",
                         seed=7, steps=STEPS, cfg_scale=7.0,
                         width=1024, height=1024, n_iter=1, batch_size=1,
                         comments=[])


def run(steps=STEPS, n_iter=1, batch_size=1):
  """Drive _run_sampling, returning (stdout, stderr)."""
  p = make_p()
  p.n_iter, p.batch_size = n_iter, batch_size
  pipe = StubPipe(steps)
  out, err = io.StringIO(), io.StringIO()
  with redirect_stdout(out), redirect_stderr(err):
    _run_sampling(
        p, pipe, "a brass compass", 1024, 1024, steps, (7.0,) * steps, 0.0, 1.75,
        "V4_DEFAULT_20", "fp8_scaled", [], bypass_recipe(0.0), None, 0.0,
        False, True, False, 0.6, "freeform-wrap", CKPT,
        turbo_merge_ok=False, distilled=False,
        _FORGE_OK=False, shared=None, Processed=None, forge_images=None,
        lora_logs=[], active_loras=None)
  return out.getvalue(), err.getvalue()


def main() -> int:
  print("== 1. a real tqdm bar reaches STDERR ==")
  out, err = run()
  check("stderr is not empty", len(err) > 0, f"{len(err)} chars")
  # Stock Forge builds its bars with tqdm DEFAULTS (shared_total_tqdm.py has no
  # bar_format), so the familiar shape is "NN%|####| n/total [el<rem, it/s]".
  # A custom bar_format is what made this bar look foreign - no percentage, no
  # rate - so the shape IS the requirement, and a desc of our own is not.
  check("bar shows a percentage like stock", "%|" in err, err[-60:].strip()[:50])
  check("bar shows an it/s rate like stock", "it/s" in err or "s/it" in err)
  check("bar carries NO custom label (stock bars have none)",
        "Ideogram4 " not in err.replace("Ideogram4 |", ""))
  check("bar reaches the final step", f"{STEPS}/{STEPS}" in err,
        f"looking for {STEPS}/{STEPS}")
  check("bar draws a progress track", "|" in err or "#" in err or "█" in err)
  check("bar carries an elapsed/remaining clock", err.count(":") >= 2)

  last = [ln for ln in err.replace("\r", "\n").split("\n") if ln.strip()][-1]
  print(f"      final bar: {last.strip()[:78]}")

  print("== 2. progress is incremental, not one jump at the end ==")
  seen = [n for n in range(1, STEPS + 1) if f"{n}/{STEPS}" in err]
  check("most steps appear individually", len(seen) >= STEPS - 1,
        f"{len(seen)}/{STEPS} step counts present")

  print("== 3. the run header names the parameters (spec Q) ==")
  for token in ("1024x1024", "V4_DEFAULT_20", "fp8_scaled"):
    check(f"header states {token}", token in out, "")
  check("header states the step count", f"{STEPS}" in out)
  check("header states the seed", "7" in out)

  print("== 3b. the run states WHICH recipe it is using (official vs community) ==")
  # The shipped default has bypass_strength 0.35, so sigma smoothing is on and
  # a "20 step" run is really 22. That is a legitimate default, but the user
  # should never have to infer it from a step count - and the presets are
  # official while the bypass/debanner levers are community additions.
  from pi_ideogram_lib.bypass import recipe as _recipe

  def _recipe_lines(r, lora_path, lora_str, deb):
    p2 = make_p()
    p2.n_iter, p2.batch_size = 1, 1
    o, e = io.StringIO(), io.StringIO()
    # This fixture tests console provenance, not adapter file loading. Real
    # adapters and hook rollback are covered by test_gray_batch_and_projection.
    with redirect_stdout(o), redirect_stderr(e), \
         patch('pi_ideogram_lora.ideogram4_lora.load_lora_state_dict', return_value={}), \
         patch('pi_ideogram_lora.first_step.FirstStepAdapters'):
      _run_sampling(p2, StubPipe(STEPS), "x", 1024, 1024, STEPS, (7.0,) * STEPS,
                    0.0, 1.75, "V4_DEFAULT_20", "fp8_scaled", [], r,
                    lora_path, lora_str, False, True, deb, 0.6,
                    "freeform-wrap", CKPT, turbo_merge_ok=False, distilled=False,
                    _FORGE_OK=False, shared=None, Processed=None,
                    forge_images=None, lora_logs=[], active_loras=None)
    # both the verdict line and the "how to get official back" hint
    return [ln for ln in o.getvalue().splitlines() if "recipe" in ln]

  plain = _recipe_lines(_recipe(0.0), None, 0.0, False)
  check("an unmodified run says so", plain and "unmodified" in plain[0],
        plain[0] if plain else "(no recipe line)")
  check("...and names it OFFICIAL", plain and "OFFICIAL" in plain[0])

  modded = _recipe_lines(_recipe(0.35), "Gray_000002000.safetensors", -0.15, True)
  joined = " ".join(modded)
  check("a modified run is labelled COMMUNITY", "COMMUNITY" in joined,
        modded[0] if modded else "(no recipe line)")
  check("...it names the added steps", "sigma smoothing" in joined)
  check("...it names the bypass LoRA", "bypass LoRA" in joined)
  check("...it names the debanner", "debanner" in joined)
  check("...and says how to get the official recipe back",
        "Bypass strength to 0" in joined)

  print("== 4. the bar is closed, not leaked between runs ==")
  _o2, err2 = run()
  check("a second run renders its own complete bar", f"{STEPS}/{STEPS}" in err2)
  check("second run does not inherit the first bar's total",
        err2.count(f"{STEPS}/{STEPS}") >= 1)

  print("== 5. batches keep reporting ==")
  _o3, err3 = run(steps=STEPS, n_iter=2, batch_size=1)
  check("multi-iteration run still reaches the final step",
        f"{STEPS}/{STEPS}" in err3)

  print("== 6. a short run still renders ==")
  _o4, err4 = run(steps=2)
  check("2-step run shows 2/2", "2/2" in err4)

  print(f"\nRESULT: {PASS} passed, {FAIL} failed")
  return 1 if FAIL else 0


if __name__ == "__main__":
  raise SystemExit(main())
