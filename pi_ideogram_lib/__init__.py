"""PROJECT INVISIBLE - Ideogram 4 engine for Forge Neo.

Vendored, self-contained inference stack for the Ideogram 4 model family,
following the official ideogram-oss/ideogram4 implementation (Apache-2.0):

- modeling_ideogram4.py   9.3B single-stream DiT (34 layers, MRoPE, AdaLN)
- autoencoder.py          Flux2 KL VAE (128-ch latent, 16x downscale, BN latent)
- latent_norm.py          official per-channel latent shift/scale
- scheduler.py           logit-normal timestep schedule + Euler flow matching
- quantized_loading.py    NF4 (bitsandbytes) + weight-only FP8 single-file load
- caption_verifier.py     official JSON caption schema checks
- magic_prompt.py         optional magic prompt (Ideogram API / OpenRouter)
- pipeline.py             dual-model (cond + uncond) pipeline, local-file loader
- qwen3vl_text_encoder.py Qwen3-VL-8B tap (13 layers -> 53248-dim features)

Everything is feature-detected and soft-fails; nothing here imports Forge core
internals at import time (only inside functions, guarded by try/except).
"""

from .pipeline import Ideogram4Pipeline
from .detect import is_ideogram4_checkpoint_file, list_ideogram4_checkpoints

__all__ = ["Ideogram4Pipeline", "list_ideogram4_checkpoints", "is_ideogram4_checkpoint_file"]
