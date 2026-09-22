"""PROJECT INVISIBLE - filter-bypass levers for Ideogram 4 (spec section P).

Community-proven levers, all local, independently togglable:

P1  Gray-screen bypass LoRA (Civitai 2750357, Gray_000002000.safetensors):
    negative strength (default -0.25), applied ONLY on the first denoising
    step, then removed. Applied to the cond trunk; when the dual transformer
    is loaded the same first-step recipe also hits the uncond trunk
    (documented choice: symmetric first-step removal is community-default and
    keeps both branches consistent for the rest of the run).
P2  Smooth early sigmas: subdivide the first sigma interval (Comfy's
    ExtendIntermediateSigmas behaviour) so the first step is not a giant jump.
P3  Layer-weight dampening: forward-hook multipliers on cond layers ~10-13 &
    16-22 (~0.4) and uncond layers ~13-17 (~0.1). Soft-fails when the model
    has no addressable blocks.
P4  Caption/JSON lever lives in json_prompt.py (structured captions + density
    guard); nothing cloud-based is ever called.
P5  One master "Bypass strength" slider 0.0-1.0 -> recipe() below.
P6  log_line() prints the exact DOS line required by the spec.
P7  We never edit weights to "disable the safety filter" - only the levers
    above, applied transiently around the official sampling loop.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
# P2 - smooth first sigmas (ExtendIntermediateSigmas-style)
# --------------------------------------------------------------------------- #

def smooth_first_sigmas(step_intervals: torch.Tensor, extra: int = 2) -> torch.Tensor:
    """Insert `extra` intermediate boundaries in the FIRST sigma interval.

    step_intervals is the ascending boundary tensor from make_step_intervals
    (0.0 ... 1.0). The first denoising step spans [intervals[-2], intervals[-1]];
    subdividing that interval turns the big first jump into `extra` smaller
    ones, which is the community "smooth first sigmas" lever (Comfy
    ExtendIntermediateSigmas). Returns a NEW tensor; input untouched.
    """
    if extra <= 0 or step_intervals.numel() < 2:
        return step_intervals.clone()
    a = float(step_intervals[-2].item())
    b = float(step_intervals[-1].item())
    if b - a <= 1e-9:
        return step_intervals.clone()
    inserts = torch.linspace(a, b, extra + 2, dtype=step_intervals.dtype, device=step_intervals.device)[1:-1]
    return torch.cat([step_intervals[:-1], inserts, step_intervals[-1:]])


# --------------------------------------------------------------------------- #
# P3 - layer dampening via forward hooks (soft-fail)
# --------------------------------------------------------------------------- #

def install_layer_dampen(model, ranges: list[tuple[int, int]], mult: float) -> list:
    """Scale the CONTRIBUTION of DiT blocks in `ranges` (inclusive) by `mult`.

    A transformer block computes `x_out = x_in + f(x_in)`. Dampening means
    weakening f, the block's contribution:

        out = x_in + mult * (x_out - x_in)

    NOT `out = mult * x_out`. Scaling the whole residual stream compounds
    across every hooked block - with the community recipe's 11 cond layers at
    0.4 the stream lands at 5-13% of its normal norm well before the stack
    ends, which is far outside the activation scale the AdaLN gates, RMSNorms
    and attention were trained on. That does not dampen the layers, it
    destroys the signal; the final LayerNorm then rescales the wreckage so the
    output *looks* plausible while carrying no usable structure.

    With the residual form, mult=1.0 is EXACTLY the identity and mult=0.0
    skips the block cleanly - both useful properties for a user-facing slider.

    Returns a list of hook handles. Empty list = nothing addressable (soft
    fail). Ranges are the community Layer Weight Multiplier recipe:
      cond layers ~10-13 and 16-22 -> ~0.4 ; uncond layers ~13-17 -> ~0.1
    """
    hooks: list = []
    layers = getattr(model, "layers", None)
    if layers is None or not hasattr(layers, "__len__") or len(layers) == 0:
        return hooks
    if abs(float(mult) - 1.0) < 1e-9:
        return hooks  # identity: do not pay for hooks that change nothing
    wanted: set[int] = set()
    for lo, hi in ranges:
        for i in range(max(0, int(lo)), min(int(hi) + 1, len(layers))):
            wanted.add(i)

    def _make_hook(factor: float):
        def _hook(_module, args, output):
            x_in = args[0] if args else None
            if not isinstance(x_in, torch.Tensor):
                return output  # unknown signature: leave the block alone
            if isinstance(output, (tuple, list)):
                head = output[0]
                if not isinstance(head, torch.Tensor) or head.shape != x_in.shape:
                    return output
                return (x_in + factor * (head - x_in),) + tuple(output[1:])
            if isinstance(output, torch.Tensor) and output.shape == x_in.shape:
                return x_in + factor * (output - x_in)
            return output
        return _hook

    try:
        for i in sorted(wanted):
            hooks.append(layers[i].register_forward_hook(_make_hook(float(mult)),
                                                         with_kwargs=False))
    except Exception:
        hooks = [h for h in hooks if h is not None]
    return hooks


def remove_layer_dampen(hooks: list) -> None:
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# P5 - strength master -> concrete recipe
# --------------------------------------------------------------------------- #

DEFAULT_DAMPEN_CFG = {
    "cond": [(10, 13), (16, 22)],
    "cond_mult": 0.4,
    "uncond": [(13, 17)],
    "uncond_mult": 0.1,
}


def dampen_cfg(cond_mult: float | None = None, uncond_mult: float | None = None) -> dict:
    """SINGLE owner of the layer-dampen recipe (P3). Layer RANGES are fixed by
    the community recipe; only the multipliers are user-tunable sliders. Both
    caller (generate.py) and tests build the cfg here, never inline."""
    cfg = {
        "cond": list(DEFAULT_DAMPEN_CFG["cond"]),
        "cond_mult": float(cond_mult) if cond_mult is not None else DEFAULT_DAMPEN_CFG["cond_mult"],
        "uncond": list(DEFAULT_DAMPEN_CFG["uncond"]),
        "uncond_mult": float(uncond_mult) if uncond_mult is not None else DEFAULT_DAMPEN_CFG["uncond_mult"],
    }
    return cfg


def recipe(strength: float, *, layer_dampen_enabled: bool = False,
           dampen_cfg: dict | None = None, sigma_smooth_steps: int = 2) -> dict:
    """Map the master bypass strength (0.0-1.0) to concrete lever settings.

    Anchors from the spec (P5):
      0.0   -> everything off
      0.35  -> LoRA at -0.15 + smooth sigmas
      1.0   -> LoRA at -0.25 + smooth sigmas + optional layer dampen
    Interpolation is linear between anchors (0 < s < 0.35 ramps the LoRA up
    from 0; 0.35 < s < 1.0 ramps LoRA from -0.15 to -0.25). Layer dampen only
    engages when the user enabled it AND s >= 0.7. The +0.005 first-sigma
    shift (community lever, already proven in this engine) rides along with
    sigma smoothing.
    """
    try:
        s = float(strength)
    except (TypeError, ValueError):
        s = 0.0
    s = max(0.0, min(1.0, s))
    on = s > 0.0
    if s <= 0.0:
        lora = 0.0
    elif s <= 0.35:
        lora = -0.15 * (s / 0.35)
    else:
        lora = -0.15 - 0.10 * ((s - 0.35) / 0.65)
    lora = round(max(-0.25, lora), 3)
    dampen_on = bool(layer_dampen_enabled) and s >= 0.7 and on
    dampen_cfg = dampen_cfg or DEFAULT_DAMPEN_CFG
    return {
        "enabled": on,
        "strength": round(s, 2),
        "lora_strength": lora,                 # negative; applied first-step-only
        "sigma_smooth": on,
        "sigma_smooth_steps": sigma_smooth_steps if on else 0,
        "sigma_shift": 0.005 if on else 0.0,   # first-sigma bump
        "layer_dampen": dampen_on,
        "dampen_cfg": dampen_cfg if dampen_on else None,
    }


# --------------------------------------------------------------------------- #
# P6 - DOS log line
# --------------------------------------------------------------------------- #

def log_line(r: dict, *, lora_file: str | None = None) -> str:
    """Exact DOS line the spec requires (P6)."""
    return (
        f"bypass={'on' if r['enabled'] else 'off'} strength={r['strength']:.2f} "
        f"lora={r['lora_strength']:+.2f}{f' ({lora_file})' if lora_file else ''} "
        f"first_step_only=yes sigma_smooth={'yes' if r['sigma_smooth'] else 'no'} "
        f"layer_dampen={'yes' if r['layer_dampen'] else 'no'}"
    )