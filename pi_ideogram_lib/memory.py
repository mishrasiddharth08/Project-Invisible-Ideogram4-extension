"""VRAM management for Ideogram 4 pipeline (Phase 4 extraction from pipeline.py).

Owns all model offloading/reloading logic and VRAM detection.
Composed into Ideogram4Pipeline via self.memory = Ideogram4MemoryManager(self).

The memory manager holds a weak-style reference to the pipeline (via the
`pipe` argument) to access its models and device. It never imports or
depends on pipeline.py — the dependency is one-directional.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .pipeline import Ideogram4Pipeline

TAG = "[Invisible-I4]"

# 24-GB cards have room for the FP8 pair at the tested 1024px canvas.
# Smaller cards stream blocks; unusually large canvases still need headroom.
HIGH_VRAM_THRESHOLD_GB = 23.5


class Ideogram4MemoryManager:
    """Manages VRAM allocation for the Ideogram 4 pipeline models.

    Composed into the pipeline via self.memory = Ideogram4MemoryManager(self).
    All methods are no-ops when the referenced model is None or not on the
    expected device, so they're safe to call in any state.
    """

    def __init__(self, pipe: Ideogram4Pipeline) -> None:
        self._pipe = pipe
        self._MM = None  # cached Forge memory_management module (or False)
        self._streamers = []
        self._streamed_models = set()
        self._streamers_by_model = {}

    @property
    def stream_weights(self) -> bool:
        """Keep full checkpoints in RAM on sub-24-GiB cards.

        Budget is based on the actual device, not a 5090-specific kernel.
        A private override allows testing constrained budgets on a larger GPU.
        """
        budget = getattr(self, '_test_vram_budget_gb', None)
        if self._pipe.device.type != 'cuda':
            return False
        if budget is not None:
            return 0 < budget < 23.5
        decision = getattr(self, '_stream_decision', None)
        if decision is None:
            budget = self.vram_gb()
            try:
                free, _ = torch.cuda.mem_get_info(self._pipe.device)
                budget = min(budget, free / 1024**3)
            except Exception:
                pass
            decision = budget < 23.5 or getattr(self._pipe, 'dtype', None) == torch.float32
            # Freeze before loading. Our own allocations must not change the
            # policy halfway through model loading or installation of hooks.
            self._stream_decision = decision
        return decision

    def install_streaming(self) -> None:
        if not self.stream_weights or self._streamers:
            return
        from .streaming import BlockStreamer, transformer_modules, text_encoder_modules
        for model in (self._pipe.conditional_transformer, self._pipe.unconditional_transformer):
            if model is not None:
                # Match resident initialization numerics: CPU and CUDA pow
                # can round the rotary frequencies differently. Only this
                # tiny buffer visits CUDA here, never the entire transformer.
                rotary = getattr(model, 'rotary_emb', None)
                if rotary is not None and callable(getattr(rotary, 'reset_parameters', None)):
                    try:
                        rotary.to(device=self._pipe.device)
                        rotary.reset_parameters()
                    finally:
                        rotary.to(device='cpu')
                streamer = BlockStreamer(transformer_modules(model), self._pipe.device)
                streamer.install()
                self._streamers.append(streamer)
                self._streamed_models.add(id(model))
                self._streamers_by_model[id(model)] = streamer
        te = self._pipe.text_encoder
        if te is not None and te.model is not None:
            streamer = BlockStreamer(text_encoder_modules(te.model), self._pipe.device)
            streamer.install()
            self._streamers.append(streamer)
            self._streamed_models.add(id(te.model))
            self._streamers_by_model[id(te.model)] = streamer
        print(f'{TAG} RAM-backed block streaming enabled: model precision and sampling steps are unchanged')

    def remove_streaming(self) -> None:
        for streamer in self._streamers:
            streamer.remove()
        self._streamers.clear()
        self._streamed_models.clear()
        self._streamers_by_model.clear()

    def forget_model(self, model) -> None:
        """Release hooks AND block references when a cached model is dropped."""
        streamer = self._streamers_by_model.pop(id(model), None)
        if streamer is not None:
            streamer.remove()
            self._streamers.remove(streamer)
        self._streamed_models.discard(id(model))

    # ------------------------------------------------------------------
    # Forge's own VRAM
    # ------------------------------------------------------------------

    def release_forge_models(self) -> float:
        """Ask Forge to evict every model IT manages before we take the GPU.

        Measured on a 32 GB card, 1024x1024, fp8 cond+uncond:

            standalone script   pipeline load  12.7s   VAE decode ~1s
            inside Forge        pipeline load 260.6s   VAE decode 198.8s
                                peak VRAM 34.70 GB on a 32 GB card

        Forge arrives holding its own weights (its model manager, ADetailer,
        ControlNet preprocessors...). Our ~27 GB of Ideogram 4 then lands on
        top, the allocator spills into system memory, and every step crawls -
        a 20x load penalty and a 200x decode penalty that never shows up in a
        standalone test.

        soft_empty_cache() cannot fix this: it releases cached blocks, not the
        models Forge still has loaded. unload_all_models() is the API that
        does, and it is safe - Forge reloads whatever it needs on the next
        non-Ideogram generation, exactly as it does when you switch checkpoint.

        Returns GB freed (0.0 when unavailable). Never raises.
        """
        from .memory_handoff import release_idle_qwen_worker
        release_idle_qwen_worker()
        if not torch.cuda.is_available():
            return 0.0
        try:
            free_before, _ = torch.cuda.mem_get_info(self._pipe.device)
        except Exception:
            return 0.0

        mm = self._forge_mm()
        if mm is None:
            return 0.0
        try:
            mm.unload_all_models()
        except Exception as e:
            print(f"{TAG} note: could not ask Forge to unload its models ({e})")
            return 0.0

        self._cleanup()
        try:
            free_after, total = torch.cuda.mem_get_info(self._pipe.device)
        except Exception:
            return 0.0
        freed = (free_after - free_before) / 1024**3
        if freed > 0.05:
            print(f"{TAG} released {freed:.1f} GB held by Forge's own models "
                  f"({free_after / 1024**3:.1f} GB now free of {total / 1024**3:.1f} GB) - "
                  "the Ideogram 4 pipeline needs the card to itself")
        return max(0.0, freed)

    def transient_gb(self, width: int = 1024, height: int = 1024) -> float:
        """Rough VRAM this pipeline needs ON TOP of its resident weights.

        Measured on this stack, int8 checkpoint, V4_DEFAULT_20:

            1280x1728 (2.21 MP)   peak 22.54 GB, live 11.11  ->  11.4 GB transient
            1024x1024 (1.05 MP)   peak 19.87 GB, live  8.65  ->  11.2 GB transient

        Transient cost is dominated by per-layer activations over the token
        sequence, which grows with pixel count, plus a roughly fixed part. The
        linear fit through those two points is deliberately crude - it exists
        to decide "ask Forge for the card back or not", not to predict bytes.
        """
        mp = max(0.25, (int(width) * int(height)) / 1_000_000.0)
        return 11.0 + 0.2 * mp

    def reclaim_if_contended(self, need_free_gb: float | None = None,
                             width: int = 1024, height: int = 1024) -> float:
        """Release Forge's models only when the card is actually contended.

        Called once per generation. When our pipeline already owns the GPU
        this is a cheap mem_get_info and returns immediately; it only pays the
        unload cost when something else has taken VRAM back (the user switched
        to another checkpoint and returned, so our cached pipeline skipped
        load()).

        The threshold used to be a flat 6 GB, which is far below what a large
        canvas actually needs: a 1280x1728 run found ~19 GB free, declined to
        reclaim, and then peaked at 32.63 GB on a 31.8 GB card - our 22.5 GB
        plus the ~10 GB Forge was still holding. cudaMallocAsync absorbed the
        overshoot into system RAM rather than failing, which is why it showed
        up as a slow run instead of an OOM.
        """
        if need_free_gb is None:
            need_free_gb = self.transient_gb(width, height)
        if not torch.cuda.is_available():
            return 0.0
        try:
            free, _total = torch.cuda.mem_get_info(self._pipe.device)
        except Exception:
            return 0.0
        if free / 1024**3 >= need_free_gb:
            return 0.0
        print(f"{TAG} only {free / 1024**3:.1f} GB VRAM free - reclaiming from Forge "
              "before sampling")
        return self.release_forge_models()

    def _forge_mm(self):
        """Forge's memory_management module, cached. None when unavailable."""
        if self._MM is None:
            try:
                from backend import memory_management
                self._MM = memory_management
            except Exception:
                self._MM = False
        return self._MM or None

    # ------------------------------------------------------------------
    # VRAM detection
    # ------------------------------------------------------------------

    def vram_gb(self) -> float:
        """Return total VRAM in GB, or 0.0 if CUDA is unavailable."""
        try:
            if torch.cuda.is_available():
                _, total = torch.cuda.mem_get_info(self._pipe.device)
                return total / 1024**3
        except Exception:
            pass
        return 0.0

    @property
    def high_vram(self) -> bool:
        """True when VRAM >= threshold — keep models resident, no offload."""
        return self.vram_gb() >= HIGH_VRAM_THRESHOLD_GB

    # ------------------------------------------------------------------
    # Text encoder offload / reload
    # ------------------------------------------------------------------

    @staticmethod
    def _on_cpu(model) -> bool:
        """True only when every parameter AND buffer is already on CPU.

        Quantised layers may store weights entirely in buffers. Looking only
        at the first parameter would miss a mixed-device quantised model.
        """
        from itertools import chain
        return all(t.device.type == "cpu"
                   for t in chain(model.parameters(), model.buffers()))

    def offload_text_encoder(self) -> None:
        """Park the Qwen3-VL text encoder on CPU. Frees ~10 GB VRAM."""
        te = getattr(self._pipe, "text_encoder", None)
        if te is None or te.model is None:
            return
        try:
            if self._on_cpu(te.model):
                return
            te.model.to("cpu")
            self._cleanup()
            free, total = torch.cuda.mem_get_info(self._pipe.device)
            print(f"{TAG} text encoder offloaded to CPU RAM "
                  f"(VRAM now {(total - free) / 1024**3:.1f} GB in use)")
        except Exception as e:
            print(f"{TAG} note: TE offload skipped ({e})")

    def ensure_text_encoder_on_device(self) -> None:
        """Re-upload text encoder to GPU if it was parked on CPU."""
        te = getattr(self._pipe, "text_encoder", None)
        if te is None or te.model is None:
            return
        if id(te.model) in self._streamed_models:
            return
        try:
            p0 = next(te.model.parameters())
            if p0.device != self._pipe.device:
                print(f"{TAG} re-uploading text encoder")
                te.model.to(self._pipe.device)
        except StopIteration:
            pass
        except Exception as e:
            print(f"{TAG} note: TE re-upload check skipped ({e})")

    def offload_cond(self) -> None:
        """Park the idle image model during text encoding and final decode."""
        model = getattr(self._pipe, 'conditional_transformer', None)
        if model is not None and not self._on_cpu(model):
            model.to('cpu')
            self._cleanup()

    def ensure_cond_on_gpu(self) -> None:
        model = getattr(self._pipe, 'conditional_transformer', None)
        if model is not None and id(model) not in self._streamed_models:
            model.to(self._pipe.device)

    # ------------------------------------------------------------------
    # VAE offload / reload
    # ------------------------------------------------------------------

    def offload_vae(self) -> None:
        """Move VAE to CPU. Only needed for final decode on low-VRAM."""
        ae = getattr(self._pipe, "autoencoder", None)
        if ae is None or not hasattr(ae, "parameters"):
            return
        try:
            if self._on_cpu(ae):
                return
            ae.to("cpu")
            self._cleanup()
        except Exception:
            pass

    def ensure_vae_on_gpu(self) -> None:
        """Re-upload VAE to GPU if it was offloaded."""
        ae = getattr(self._pipe, "autoencoder", None)
        if ae is None or not hasattr(ae, "parameters"):
            return
        try:
            p0 = next(ae.parameters())
            if p0.device != self._pipe.device:
                ae.to(self._pipe.device)
        except StopIteration:
            pass

    # ------------------------------------------------------------------
    # Unconditional transformer offload / reload
    # ------------------------------------------------------------------

    def ensure_uncond_on_gpu(self) -> None:
        """Move the unconditional transformer to GPU."""
        te = getattr(self._pipe, "unconditional_transformer", None)
        if te is None or not hasattr(te, "parameters"):
            return
        if id(te) in self._streamed_models:
            return
        try:
            p0 = next(te.parameters())
            if p0.device != self._pipe.device:
                te.to(self._pipe.device)
        except StopIteration:
            pass

    def offload_uncond(self) -> None:
        """Move the unconditional transformer to CPU."""
        te = getattr(self._pipe, "unconditional_transformer", None)
        if te is None or not hasattr(te, "parameters"):
            return
        try:
            if self._on_cpu(te):
                return
            te.to("cpu")
            self._cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Memory cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Force PyTorch to release cached memory back to the CUDA driver.

        Three layers, each unconditional:
          1. gc.collect() so dead tensors drop their Python refs.
          2. Forge's soft_empty_cache() when importable (respects the
             memory_management offload streams).
          3. torch.cuda.empty_cache() ALWAYS as the final guarantee -
             under cudaMallocAsync this is what actually returns freed
             blocks to the driver pool (verified empirically).

        Layer 2 must not be able to swallow layer 3: the empty_cache call
        sits outside the try/except so a backend failure can never turn
        the whole cleanup into a silent no-op.
        """
        gc.collect()
        if torch.cuda.is_available():
            mm = self._forge_mm()
            if mm is not None:
                try:
                    mm.soft_empty_cache()
                except Exception:
                    pass
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
