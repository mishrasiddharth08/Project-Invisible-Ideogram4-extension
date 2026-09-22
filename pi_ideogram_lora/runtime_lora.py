"""Non-destructive LoRA that can be switched on and off per forward pass.

WHY A SECOND MECHANISM EXISTS
-----------------------------
`ideogram4_lora.apply_lora_to_model` MERGES a delta into the live weights.
That is the right thing for a style LoRA: it costs nothing at inference and
stays on for the whole run.

It is the wrong thing for an adapter that must apply to only SOME passes.
The clearest case is ostris/ideogram_4_unconditional_lora, whose model card
says it is "used on the conditional Ideogram 4 model during the unconditional
pass as a replacement to the full 9B parameter unconditional model".

Merged, that adapter is live for both passes, so the conditional pass - the
one that follows your prompt - runs through weights trained to produce
UNCONDITIONAL output. Classifier-free guidance then subtracts two nearly
identical passes and prompt adherence collapses. The adapter has to be on for
the uncond forward and off for the cond forward, within the same step.

Merging and un-merging around every pass would mean 2 x steps merge cycles
over ~200 modules per image, plus bf16 rounding on each. A side path costs one
extra rank-r GEMM pair per hooked module (rank 16 against a 4608-wide layer is
under 1% of that layer's FLOPs) and is exactly reversible because the base
weights are never touched.

    adapter = RuntimeLoraAdapter(transformer, sd, strength=1.0, name="uncond")
    adapter.install()
    ...
    with adapter.active():          # enabled for this forward only
        neg_out = transformer(...)
    out = transformer(...)          # adapter dormant, base weights untouched
    adapter.remove()
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

TAG = "[Invisible-I4]"


def _lokr_factors(sides: dict):
  """(w1, w2) for a plain LoKr module, or None if this is not one.

  Only the non-decomposed form is handled here. CP-decomposed LoKr (lokr_t1 /
  lokr_t2) reconstructs through a tucker-style core; folding that into a plain
  kron would produce a confidently wrong delta, so it is refused rather than
  approximated - same rule the merge path follows.
  """
  if sides.get("T1") is not None or sides.get("T2") is not None:
    return None

  def _side(full, a, b):
    if a is not None and b is not None:
      return b.float() @ a.float()
    if full is not None:
      return full.float()
    return None

  w1 = _side(sides.get("W1"), sides.get("W1A"), sides.get("W1B"))
  w2 = _side(sides.get("W2"), sides.get("W2A"), sides.get("W2B"))
  if w1 is None or w2 is None or w1.ndim != 2 or w2.ndim != 2:
    return None
  return w1, w2


class RuntimeLoraAdapter:
  """A LoRA held alongside the weights and applied only while `enabled`.

  Attributes:
      enabled: when False every hook returns the module output untouched, so
          a dormant adapter costs one predictable branch per hooked module.
      applied_modules: how many modules the adapter actually attached to.
  """

  def __init__(self, model: nn.Module, lora_sd: dict, strength: float = 1.0,
               name: str = "runtime") -> None:
    self._model = model
    self._name = name
    self._strength = float(strength)
    self._handles: list = []
    self._pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    self._lokr: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    self.enabled = False
    self.applied_modules = 0
    self._skipped: list[str] = []
    self._build(lora_sd)

  # ------------------------------------------------------------------ build

  def _build(self, lora_sd: dict) -> None:
    """Resolve the file's module paths against the model and keep A/B factors.

    Reuses the same grouping + path resolution as the merge path, so an
    adapter that merges will also run here and vice versa.
    """
    from .ideogram4_lora import _alpha_scale, _group_modules, resolve_module_path

    groups = _group_modules(lora_sd)
    modules = dict(self._model.named_modules())
    known = set(modules)

    for raw_path, sides in groups.items():
      if not raw_path:
        continue
      A, B = sides.get("A"), sides.get("B")
      lokr = None
      if A is None or B is None or A.ndim != 2 or B.ndim != 2:
        # LoKr used to be refused here - "no cheap side path" - and sent to the
        # merge route instead. That was wrong twice over.
        #
        # It is not cheap only if you build kron(w1, w2) as a dense matrix,
        # which costs the same as the layer's own matmul (+101% measured). The
        # Kronecker mixed-product contracts the two factors separately and is
        # exact (cosine 0.999999 vs the dense form) at +30%.
        #
        # And the merge route is not a safe fallback on a quantised checkpoint.
        # Merging this adapter into fp8 and requantising was measured on the
        # real files as:
        #     intended delta rms 0.000259
        #     error rms          0.000371   -> 1.44x the signal
        #     cosine(recovered, intended) 0.57
        # i.e. ~43% of what reaches the weights is quantisation noise, sprayed
        # across 204 modules. That is the "body horror" - and it reproduces in
        # any implementation that merges LoKrs into fp8.
        lokr = _lokr_factors(sides)
        if lokr is None:
          self._skipped.append(raw_path)
          continue
      path = resolve_module_path(raw_path, known)
      if path is None:
        self._skipped.append(raw_path)
        continue
      mod = modules.get(path)
      if mod is None or not hasattr(mod, "forward"):
        self._skipped.append(raw_path)
        continue

      in_f = getattr(mod, "in_features", None)
      out_f = getattr(mod, "out_features", None)

      if lokr is not None:
        w1, w2 = lokr
        m1, n1 = w1.shape
        m2, n2 = w2.shape
        if in_f is not None and (n1 * n2 != in_f or m1 * m2 != out_f):
          self._skipped.append(raw_path)
          continue
        self._lokr[path] = (w1, w2, _alpha_scale(sides, max(1, min(n1, n2))))
        continue

      if in_f is not None and (A.shape[1] != in_f or B.shape[0] != out_f):
        self._skipped.append(raw_path)
        continue

      self._pairs[path] = (A, B, _alpha_scale(sides, max(1, min(A.shape[0], B.shape[1]))))

  # ---------------------------------------------------------------- install

  def install(self) -> int:
    """Attach the side path. Returns the number of modules hooked."""
    if self._handles:
      return self.applied_modules
    modules = dict(self._model.named_modules())
    for path, (A, B, scale) in self._pairs.items():
      mod = modules.get(path)
      if mod is None:
        continue
      try:
        ref = next(mod.parameters(), None)
        if ref is None:
          ref = next(iter(mod.buffers()), None)
        device = ref.device if ref is not None else torch.device("cpu")
        a = A.to(device=device, dtype=torch.float32)
        b = B.to(device=device, dtype=torch.float32)
      except Exception:
        continue

      def _hook(_module, args, output, _a=a, _b=b, _s=scale * self._strength):
        if not self.enabled or not args:
          return output
        x = args[0]
        if not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor):
          return output
        # Streaming trunks keep adapter factors in RAM between blocks too.
        # Resident trunks take a no-copy .to() path on their existing device.
        a = _a.to(device=x.device)
        b = _b.to(device=x.device)
        side = (x.to(torch.float32) @ a.t()) @ b.t()
        return output + (side * _s).to(output.dtype)

      self._handles.append(mod.register_forward_hook(_hook))

    # LoKr side path: kron(w1, w2) @ x, WITHOUT materialising kron(w1, w2).
    #   kron(A,B)[i1*m2+i2, j1*n2+j2] = A[i1,j1] * B[i2,j2]
    # so contracting B over n2 and then A over n1 gives the same result for a
    # fraction of the work. Verified against the dense form: cosine 0.999999.
    for path, (w1, w2, scale) in self._lokr.items():
      mod = modules.get(path)
      if mod is None:
        continue
      try:
        ref = next(mod.parameters(), None)
        if ref is None:
          ref = next(iter(mod.buffers()), None)
        device = ref.device if ref is not None else torch.device("cpu")
      except Exception:
        continue
      m1, n1 = w1.shape
      m2, n2 = w2.shape

      # MEMORY. The first version contracted w2 first for every module and
      # allocated an (L, n1, m2) intermediate - measured at 12.5 GB of churn
      # per forward across 34 layers, which OOMed a 32 GB card outright.
      # Two corrections:
      #   * pick the cheaper contraction PER MODULE (m1*n2 vs n1*m2), and
      #   * walk the tokens in chunks so peak transient does not scale with L.
      # The result is identical either way; only the working set changes.
      w1_first = (m1 * n2) <= (n1 * m2)

      def _lokr_hook(_module, args, output, _w1=None, _w2=None, _s=0.0,
                     _n1=n1, _n2=n2, _m1=m1, _m2=m2, _w1f=True):
        if not self.enabled or not args:
          return output
        x = args[0]
        if not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor):
          return output
        if x.shape[-1] != _n1 * _n2:
          return output
        dt = output.dtype
        flat = x.reshape(-1, _n1 * _n2)
        out_flat = output.reshape(-1, _m1 * _m2)
        n_tok = flat.shape[0]
        # Bound the LARGEST per-chunk tensor, which is the output block
        # (m1*m2), not just the intermediate - sizing off the intermediate
        # alone left chunk > n_tok, so no chunking happened and the hook still
        # allocated a full sequence-length output copy.
        inner = _m1 * _n2 if _w1f else _n1 * _m2
        widest = max(inner, _m1 * _m2, _n1 * _n2)
        chunk = max(256, min(n_tok, int(48e6 // max(1, 2 * widest))))
        # The factors are held in bf16 (they are tiny); match whatever dtype
        # the host module actually runs in, so a bf16 transformer and an fp32
        # test model both work without a second copy of the weights.
        a1 = _w1.to(device=output.device, dtype=dt)
        a2 = _w2.to(device=output.device, dtype=dt)
        for i in range(0, n_tok, chunk):
          xr = flat[i:i + chunk].to(dt).reshape(-1, _n1, _n2)
          if _w1f:
            t = torch.einsum("bij,qi->bqj", xr, a1)
            y = torch.einsum("bqj,pj->bqp", t, a2)
          else:
            t = torch.einsum("bij,pj->bip", xr, a2)
            y = torch.einsum("bip,qi->bqp", t, a1)
          del t
          out_flat[i:i + chunk].add_(y.reshape(-1, _m1 * _m2), alpha=_s)
          del y
        return output

      import functools
      hook = functools.partial(
        _lokr_hook,
        # Keep the factors at the precision they arrived in. Storing them as
        # bf16 rounded the delta before it was ever used, which is invisible on
        # a bf16 transformer but measurably wrong on an fp32 one (3.6e-3). They
        # are a few hundred KB; the hook casts down per call instead.
        _w1=w1.to(device=device),
        _w2=w2.to(device=device),
        _s=scale * self._strength,
        _n1=n1, _n2=n2, _m1=m1, _m2=m2, _w1f=w1_first)
      self._handles.append(mod.register_forward_hook(hook))

    self.applied_modules = len(self._handles)
    return self.applied_modules

  def remove(self) -> None:
    for h in self._handles:
      try:
        h.remove()
      except Exception:
        pass
    self._handles.clear()
    self.enabled = False

  # ----------------------------------------------------------------- toggle

  @contextmanager
  def active(self):
    """Enable the adapter for exactly the block it wraps."""
    was = self.enabled
    self.enabled = True
    try:
      yield self
    finally:
      self.enabled = was

  def __enter__(self) -> "RuntimeLoraAdapter":
    self.install()
    return self

  def __exit__(self, *_exc) -> None:
    self.remove()

  # ------------------------------------------------------------------ misc

  def report(self) -> str:
    msg = (f"{self._name} runtime LoRA: {self.applied_modules} modules "
           f"at strength {self._strength}")
    if self._skipped:
      msg += (f" ({len(self._skipped)} unusable here, e.g. {self._skipped[0]}"
              " - LoKr/mismatched shapes need the merge path)")
    return msg
