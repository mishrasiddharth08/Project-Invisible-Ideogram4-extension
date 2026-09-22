"""Ideogram 4 watermark removal (MrJackSpade debanner).

Extracted from pipeline.py (Phase 5 of architecture decomposition).
Owns the correction tensor loading, caching, and application logic.

Obtain the optional correction tensor manually from:
  https://github.com/MrJackSpade/Ideogram4-Debanner

The correction lives in the DiT's 4608-wide residual stream, so it has to be
injected INSIDE blocks 25-28 - not applied to the model's 128-channel output,
where the shapes cannot meet. Use the context manager:

    dirs = load_debanner_tensor() if debanner_enabled else None
    if dirs and is_first_step:
        with DebannerHooks(transformer, dirs, strength, n_ctx, grid_h, grid_w) as dbg:
            out = transformer(...)
        print(dbg.applied)   # blocks that genuinely fired - trust this, not a count
    else:
        out = transformer(...)

The hooks are removed on exit, so a first-step-only recipe cannot leak into
later steps or later generations.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

TAG = "[Invisible-I4]"

# Transformer blocks where the watermark correction is applied
DEBANNER_BLOCKS = list(range(25, 29))

TENSOR_NAME = "ideogram4_correction_v1.safetensors"

# Module-level cache: dict {block_idx: tensor}, loaded once, reused forever
_tensor_cache: dict[int, torch.Tensor] | None = None


def load_debanner_tensor() -> dict[int, torch.Tensor] | None:
    """Load the local MrJackSpade correction tensor. Never downloads.

    Returns:
        Dict mapping block index (25-28) to correction tensor, or None on failure.
        Cached after first successful load — subsequent calls are free.
    """
    global _tensor_cache
    if _tensor_cache is not None:
        return _tensor_cache

    # Locate the extension's pretrained directory
    ext_dir = Path(__file__).resolve().parent.parent  # project-invisible-ideogram-4/
    tdir = ext_dir / "pretrained"
    tfile = "ideogram4_correction_v1.safetensors"
    tpath = tdir / tfile

    # No hidden network traffic during generation; keep existing local assets.
    if not tpath.is_file() or tpath.stat().st_size < 1000:
        print(f'{TAG} Debanner skipped: place {tfile} in {tdir}. '
              'Source: https://github.com/MrJackSpade/Ideogram4-Debanner (check its license). '
              'No download was started.')
        return None

    # Load and extract per-block correction tensors
    try:
        from safetensors.torch import load_file as _lf
        data = _lf(str(tpath), device="cpu")
        dirs: dict[int, torch.Tensor] = {}

        # Try named keys first (block_25, block_26, etc.)
        for key in data:
            lo = key.lower()
            for b in DEBANNER_BLOCKS:
                if f"block_{b:02d}" in lo and b not in dirs:
                    dirs[b] = data[key].float().contiguous()
                    break

        # Fallback: positional keys (first 4 tensors → blocks 25-28)
        if not dirs:
            keys = list(data.keys())
            for i, b in enumerate(DEBANNER_BLOCKS):
                if i < len(keys):
                    dirs[b] = data[keys[i]].float().contiguous()

        _tensor_cache = dirs
        return dirs or None
    except Exception as e:
        print(f"{TAG} Debanner tensor load failed: {e}")
        return None


_meta_cache: dict | None = None


def debanner_metadata() -> dict:
    """__metadata__ from the correction bundle: target_step, strength, schema."""
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache
    import json as _json
    import struct as _struct

    tpath = Path(__file__).resolve().parent.parent / "pretrained" / TENSOR_NAME
    meta: dict = {}
    try:
        with open(tpath, "rb") as f:
            n = _struct.unpack("<Q", f.read(8))[0]
            meta = _json.loads(f.read(n)).get("__metadata__") or {}
    except Exception:
        meta = {}
    _meta_cache = meta
    return meta


def debanner_target_step(default: int = 0) -> int:
    """Which LOOP INDEX the correction belongs on.

    The bundle says `target_step: '0'`, and this engine's loop index counts
    DOWN - `for i in range(num_steps - 1, -1, -1)` - with index 0 as the final
    step, the same convention sampler_configs.py documents for the guidance
    schedule. So target_step 0 is the LAST step, not the first.

    It was being applied on the first step, and measurement showed why that
    cannot work: perturbing the residual stream at step 1 of 22 steers the
    whole trajectory. Even at strength 0.05 - a twelfth of the author's 0.6 -
    63% of pixels changed and the image became a different picture, which is
    trajectory steering, not watermark removal.
    """
    try:
        return int(str(debanner_metadata().get("target_step", default)).strip())
    except Exception:
        return default


def apply_debanner_correction(
    h: torch.Tensor,
    correction: torch.Tensor,
    strength: float,
    n_ctx: int,
    grid_h: int | None = None,
    grid_w: int | None = None,
) -> torch.Tensor:
    """Apply one block's correction direction to a transformer HIDDEN STATE.

    The correction tensor is (1, 8, 8, emb_dim) - emb_dim, i.e. the 4608-wide
    residual stream INSIDE the stack. It is meaningless applied to anything
    else, which is why this must run as a hook on blocks 25-28 rather than on
    the model's 128-channel velocity output. Shape mismatches now return `h`
    with a warning instead of being swallowed silently.

    Args:
        h: Hidden state, shape (batch, seq_len, emb_dim).
        correction: Correction tensor, (1, gh, gw, emb_dim) or (gh, gw, emb_dim).
        strength: Correction magnitude (0.0-1.5; author default 0.6).
        n_ctx: Number of text context tokens (image tokens start here).
        grid_h, grid_w: True token grid. Required for non-square images; when
            omitted a square grid is inferred (and non-square input is refused).

    Returns:
        Corrected hidden state, same shape as the input.
    """
    C = h.shape[-1]
    if correction.dim() == 3:
        correction = correction.unsqueeze(0)
    if correction.dim() != 4:
        print(f"{TAG} Debanner: correction has {correction.dim()} dims, expected 3 or 4 - skipped")
        return h
    if correction.shape[-1] != C:
        print(f"{TAG} Debanner: correction width {correction.shape[-1]} != hidden size {C}. "
              "This tensor belongs on the transformer's residual stream; skipped.")
        return h

    n_img = h.shape[1] - n_ctx
    if n_img <= 0:
        return h
    if grid_h is None or grid_w is None:
        side = int(round(n_img ** 0.5))
        if side * side != n_img:
            print(f"{TAG} Debanner: non-square token grid ({n_img} tokens) and no grid given - skipped")
            return h
        grid_h = grid_w = side
    if grid_h * grid_w != n_img:
        print(f"{TAG} Debanner: grid {grid_h}x{grid_w} does not cover {n_img} image tokens - skipped")
        return h

    # Layout is (1, gh, gw, C): the grid axes are dims 1 and 2, NOT 2 and 3.
    if correction.shape[1] != grid_h or correction.shape[2] != grid_w:
        correction = F.interpolate(
            correction.permute(0, 3, 1, 2).float(),
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

    correction = correction.reshape(1, n_img, C).to(device=h.device, dtype=h.dtype)
    img_tokens = h[:, n_ctx:n_ctx + n_img]
    corrected = img_tokens - strength * correction

    # Preserve each token's L2 norm so the residual stream keeps its scale.
    norms = torch.linalg.vector_norm(img_tokens, dim=-1, keepdim=True)
    corrected = F.normalize(corrected, p=2, dim=-1) * norms

    out = h.clone()
    out[:, n_ctx:n_ctx + n_img] = corrected
    return out


class DebannerHooks:
    """Install the correction on blocks 25-28 for the duration of one forward.

    Context-manager so the hooks cannot leak into later steps or later
    generations - the recipe is first-step-only, and a stuck hook would quietly
    corrupt every subsequent image.

    Usage:
        with DebannerHooks(transformer, dirs, strength, n_ctx, gh, gw) as dbg:
            out = transformer(...)
        print(dbg.applied)   # blocks that actually fired
    """

    def __init__(self, transformer, dirs: dict, strength: float, n_ctx: int,
                 grid_h: int | None = None, grid_w: int | None = None) -> None:
        self._transformer = transformer
        self._dirs = dirs or {}
        self._strength = float(strength)
        self._n_ctx = int(n_ctx)
        self._grid = (grid_h, grid_w)
        self._handles: list = []
        self.applied: list[int] = []

    def __enter__(self) -> "DebannerHooks":
        layers = getattr(self._transformer, "layers", None)
        if layers is None or not self._dirs or self._strength == 0.0:
            return self
        gh, gw = self._grid
        for bi in DEBANNER_BLOCKS:
            if bi not in self._dirs or bi >= len(layers):
                continue
            corr = self._dirs[bi]

            def _hook(_module, _args, output, _corr=corr, _bi=bi):
                # A block returns the residual stream; a tuple-returning variant
                # is handled so this survives a refactor of the block signature.
                hs = output[0] if isinstance(output, tuple) else output
                new_hs = apply_debanner_correction(
                    hs, _corr, self._strength, self._n_ctx, gh, gw
                )
                if new_hs is not hs:
                    self.applied.append(_bi)
                if isinstance(output, tuple):
                    return (new_hs,) + tuple(output[1:])
                return new_hs

            self._handles.append(layers[bi].register_forward_hook(_hook))
        return self

    def __exit__(self, *_exc) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
