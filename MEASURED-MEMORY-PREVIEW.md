# Measured memory and preview check

Local RTX 5090, INT8 ConvRot checkpoint and text encoder, full Flux2 VAE, unconditional adapter at 1.0. Standalone pipeline with Forge closed, 12-step V4_TURBO_12 recipe, no Spectrum or Gray adapter, previews enabled. Model loading took 16.2 seconds separately.

| Canvas | Generation including text encoding and previews | Peak tensor allocation | Peak PyTorch reserved memory | Previews |
|---|---:|---:|---:|---:|
| 1024 × 1024 | 26.80 s | 10.65 GiB | 11.16 GiB | 6 |
| 1280 × 1920 | 46.25 s | 11.80 GiB | 12.92 GiB | 12 |

The larger image was visually checked and contains the requested still life, not a gray card. Fresh previews appeared about every 1.7 seconds at 1024 and every 3.1 seconds at the larger size. Browser progress polls every second, but generation cannot supply a new image between completed sampling steps.

These are standalone measurements, not a controlled before/after comparison or a measurement of the complete Forge process. Driver/context memory and other applications are not included in PyTorch reserved memory. The earlier user log used 25 steps and a different prompt. Total desktop VRAM and generation time can therefore differ.
