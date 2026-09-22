"""Spectrum for Ideogram 4: predict some steps instead of computing them.

WHAT IT DOES
------------
Every sampling step asks the 9.3B DiT for a velocity `v`. Those answers move
smoothly along the trajectory, so on some steps `v` can be EXTRAPOLATED from
the last few real ones instead of running the network. A skipped step costs a
weighted sum of a handful of small tensors rather than 34 transformer layers.

This is an APPROXIMATION. A forecast step changes the trajectory, so the same
seed produces a slightly different image. That is the trade, stated up front.

WHERE IT COMES FROM
-------------------
The design is taken from `ComfyUI-Spectrum-MiniMax-H3` (read-only, never
modified), which forecasts MiniMax H3's packed hidden state. Three of its
choices are the reason this is safe enough to ship, and all three are kept:

  * the first `warmup` steps are always real - that is where composition is
    decided and the trajectory is least polynomial;
  * the last `tail` steps are always real - that is the detail pass;
  * a forecast only happens with enough real anchors behind it
    (>= degree + 1), never on a guess.

WHAT IS DIFFERENT HERE, AND WHY IT IS SIMPLER
---------------------------------------------
H3 forecasts a large packed hidden state, so that project keeps history in
system RAM and works hard to avoid full-precision coefficients. Ideogram 4's
velocity is (B, image_tokens, 128) - **2.2 MB** at 1280x1728, measured. Eight
anchors is ~18 MB, so history stays on the GPU and none of that machinery is
needed. Copying it would have been cargo-culting.

The two CFG branches are forecast SEPARATELY and blended with the current
step's guidance weight, so the official 7->3 guidance schedule is applied
exactly as written rather than baked into the history.

THE MATH
--------
For anchors at coordinates c_k with values v_k, fit a degree-M Chebyshev
polynomial by ridge regression and evaluate it at the target coordinate. The
per-element solve is avoided entirely: the prediction is linear in the
history, so one K-vector of weights is computed from the K x (M+1) design and
the answer is a weighted sum of the stored tensors.

    w = t(c) . (T^T T + lambda I)^-1 T^T          # shape (K,)
    v_hat = sum_k w_k * v_k

T is K x (M+1) and the inverse is (M+1) x (M+1), so the cost does not depend
on how big the tensors are.

WHAT IT ACTUALLY DOES, MEASURED
-------------------------------
RTX 5090, ideogram4-int8_convrot, 1024x1024, V4_DEFAULT_20, fixed seed. The
baseline is deterministic (two runs were bit-identical, PSNR inf), so every
difference below is Spectrum and nothing else.

  setting                     forecast   time    speedup   verdict
  off                            0/20    14.7s    1.00x    reference
  stride 3, degree 1             5/20    11.1s    1.32x    safe
  stride 2, degree 1             8/20     9.2s    1.62x    unreliable
  stride 2, degree 2             8/20     9.0s    1.66x    unreliable

The defaults above are the "safe" row, deliberately not the fastest one.

Pixel metrics turned out to be the wrong instrument and are recorded here so
nobody re-derives them: mean|d| runs 9-70/255 and PSNR 13-22 dB across all
settings, which reads like catastrophe. It is not. A forecast step moves the
trajectory, so the run lands on a DIFFERENT SAMPLE of the same prompt - the
way two samplers differ - and a different sample scores terribly against
per-pixel difference while looking perfectly good.

What separates the rows is inspection, not the metric:

  * stride 3 / degree 1 keeps the same picture. Same composition, same face,
    same lighting; slightly softer fine texture (Laplacian variance 0.66-0.88x
    on portraits). Across a still life, two portraits and two hand studies at
    two seeds, nothing broke.
  * stride 2 (either degree) is prompt-dependent. It produced the best-looking
    portrait of the whole set AND smeared the potter's-hands study into
    merged, structureless fingers at the same settings. A lever that improves
    one prompt and wrecks another is not a default.

So: 1.32x for a slightly softer, still-correct image. That is the honest
offer. It is off unless asked for, and there is no setting here that makes
the model faster for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpectrumConfig:
    """Settings for one generation. `enabled=False` makes every call a no-op."""

    enabled: bool = False
    degree: int = 1
    ridge: float = 0.1
    history: int = 6
    warmup: int = 4
    tail: int = 2
    stride: int = 3
    blend: float = 1.0
    max_growth: float = 2.0

    def sanitised(self) -> "SpectrumConfig":
        """Clamp to a workable range instead of raising mid-generation."""
        degree = max(1, min(int(self.degree), 6))
        history = max(degree + 1, min(int(self.history), 16))
        return SpectrumConfig(
            enabled=bool(self.enabled),
            degree=degree,
            ridge=max(0.0, float(self.ridge)),
            history=history,
            warmup=max(1, int(self.warmup)),
            tail=max(1, int(self.tail)),
            stride=max(2, int(self.stride)),
            blend=min(1.0, max(0.0, float(self.blend))),
            max_growth=max(1.0, float(self.max_growth)),
        )


def chebyshev_design(coords: torch.Tensor, degree: int) -> torch.Tensor:
    """K x (degree+1) Chebyshev design matrix, by the standard recurrence."""
    x = coords.reshape(-1, 1).to(dtype=torch.float64)
    cols = [torch.ones_like(x)]
    if degree >= 1:
        cols.append(x)
    for _ in range(2, degree + 1):
        cols.append(2.0 * x * cols[-1] - cols[-2])
    return torch.cat(cols[: degree + 1], dim=1)


class VelocityForecaster:
    """Ring buffer of recent (coordinate, velocity) anchors + a ridge fit.

    One instance per CFG branch. Tensors are kept on whatever device they
    arrive on; see the module docstring for why that is affordable here.
    """

    def __init__(self, cfg: SpectrumConfig) -> None:
        self.cfg = cfg.sanitised()
        self.reset()

    def reset(self) -> None:
        self._coords: list[float] = []
        self._values: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self._values)

    @property
    def ready(self) -> bool:
        """Enough real anchors to fit the requested polynomial."""
        return len(self._values) >= self.cfg.degree + 1

    def update(self, coord: float, value: torch.Tensor) -> None:
        """Record a REAL step. Forecast outputs must never be fed back in:
        doing so compounds its own error and the trajectory drifts."""
        self._coords.append(float(coord))
        self._values.append(value.detach())
        if len(self._values) > self.cfg.history:
            self._coords.pop(0)
            self._values.pop(0)

    def weights(self, coord: float) -> torch.Tensor | None:
        """The K-vector w with v_hat = sum_k w_k * v_k, or None if not ready.

        Coordinates are mapped onto [-1, 1] over the history window first.
        Chebyshev polynomials are DEFINED on that interval, and raw sigma
        values sit in a narrow band that makes the Gram matrix badly
        conditioned at higher degrees. The rescaling is affine, so with
        ridge=0 the fitted polynomial is unchanged - it only stops the
        arithmetic from falling apart.
        """
        if not self.ready:
            return None
        raw = torch.tensor(self._coords, dtype=torch.float64)
        lo, hi = float(raw.min()), float(raw.max())
        span = hi - lo
        if not (span > 1e-9):
            # Every anchor sits at the same coordinate: there is no slope to
            # measure. torch.linalg.solve does NOT raise here - it hands back
            # the mean, which looks like an answer and is not one.
            return None
        def _norm(t):
            return 2.0 * (t - lo) / span - 1.0

        design = chebyshev_design(_norm(raw), self.cfg.degree)       # K x (M+1)
        gram = design.T @ design                                     # (M+1)^2
        gram = gram + self.cfg.ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
        # Symmetric PSD, so eigenvalues are the condition number directly.
        # A near-singular Gram makes the fit arbitrarily sensitive to noise in
        # the anchors, which shows up as a wildly overshooting velocity.
        try:
            ev = torch.linalg.eigvalsh(gram)
            if float(ev.min()) <= 1e-10 * max(float(ev.max()), 1.0):
                return None
            solved = torch.linalg.solve(gram, design.T)               # (M+1) x K
        except Exception:
            return None
        target = chebyshev_design(
            _norm(torch.tensor([float(coord)], dtype=torch.float64)),
            self.cfg.degree)                                          # 1 x (M+1)
        w = (target @ solved).reshape(-1)                             # K
        if not torch.isfinite(w).all():
            return None
        return w

    def predict(self, coord: float) -> torch.Tensor | None:
        """Extrapolated value at `coord`, or None when it cannot be trusted."""
        w = self.weights(coord)
        if w is None:
            return None
        ref = self._values[-1]
        w = w.to(device=ref.device, dtype=torch.float32)
        out = torch.zeros_like(ref, dtype=torch.float32)
        for k, v in enumerate(self._values):
            out += w[k] * v.to(torch.float32)
        blend = self.cfg.blend
        if blend < 1.0:
            out = blend * out + (1.0 - blend) * ref.to(torch.float32)
        if not torch.isfinite(out).all():
            return None
        # OVERSHOOT GUARD. Polynomial extrapolation fails by exploding, and a
        # velocity several times too large wrecks the latent with no error
        # message. The premise of forecasting at all is that the trajectory is
        # smooth, so a step that multiplies the magnitude contradicts the
        # premise - refuse and let the caller run the real network.
        ref_rms = float(ref.to(torch.float32).pow(2).mean().sqrt())
        out_rms = float(out.pow(2).mean().sqrt())
        if ref_rms > 0.0 and out_rms > self.cfg.max_growth * ref_rms:
            return None
        return out.to(ref.dtype)


def plan(num_steps: int, cfg: SpectrumConfig) -> list[bool]:
    """Which steps are FORECAST. `plan[i] is True` -> skip the network.

    Index 0 is the first step executed. The guarantees, in order of
    importance:

      * the first `warmup` steps are real,
      * the last `tail` steps are real,
      * at most one forecast between two real steps (stride >= 2), so an
        anchor is never more than one skipped step away,
      * with too few steps to satisfy all of that, nothing is forecast and the
        run is exactly the official recipe.
    """
    cfg = cfg.sanitised()
    n = int(num_steps)
    out = [False] * n
    if not cfg.enabled or n <= 0:
        return out
    first = max(cfg.warmup, cfg.degree + 1)
    last_allowed = n - cfg.tail - 1
    if first > last_allowed:
        return out
    i = first
    while i <= last_allowed:
        out[i] = True
        i += cfg.stride
    return out


def summarise(steps_plan: list[bool], num_steps: int) -> str:
    """One honest line for the console: what will actually be skipped."""
    skipped = sum(1 for f in steps_plan if f)
    if not skipped:
        return "Spectrum: no step qualifies - running the official recipe unchanged"
    real = num_steps - skipped
    pct = 100.0 * skipped / max(1, num_steps)
    return (f"Spectrum: {real}/{num_steps} steps computed, {skipped} forecast "
            f"({pct:.0f}% of forwards skipped) - this is an APPROXIMATION, the "
            "image will differ from the same seed with Spectrum off")
