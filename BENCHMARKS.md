# Release-candidate measurements — 2026-09-10

These are local engineering checks, not a broad quality evaluation. No downloads or Forge core edits were needed. Pipelines and isolated Gradio components were tested separately. Normal Forge UI/API generation still needs a post-install restart and smoke test. The historical V2 measurements below are not new V3 speedup claims.

## Machine and scene

RTX 5090, 32 GB advertised VRAM, approximately 96 GiB system RAM; driver 616.92; PyTorch 2.13.0+cu130, Transformers 4.57.6, Gradio 4.40.0. Forge Neo's installed quantization/attention backend was used in the INT8 run. Other standalone runs used the portable attention fallback.

1024×1024, batch 1, seed 1234, Default-20 full guidance; a brass compass on a nautical chart. No style/bypass adapters. Conditional/unconditional FP8 E4M3 scaled, Qwen3-VL-8B FP8 scaled and Flux2 VAE, except where INT8 is specified.

## Quality-preserving work

| Check | Before | Candidate | Evidence |
| --- | ---: | ---: | --- |
| Warm full generation | 26.59 s | 26.33 s | Same weights; output pixels identical |
| Warm peak PyTorch allocation | 20.47 GiB | 20.09 GiB | Same run pair; cold initialization excluded |
| Text-encoding peak allocation | 10.95 GiB | 9.22 GiB | Full sequence retained; captured text features identical |
| Cached feature tensor size | 461,021,184 bytes | 24,813,568 bytes | Approximately 94.6% smaller |
| Warm text encoding | 0.581 s | 0.577 s | Main improvement is memory, not a large latency gain |

The before/after generation comparison replaced the old pipeline call/text capture while keeping the same loaded candidate weights. It is not a fresh installation benchmark of every old component. An attempted text-prefix-only forward changed conditioning and was discarded.

## Final capacity test

The final sampler/memory code ran both cases in one process with cuDNN benchmarking disabled, full precision-preserving guidance, unchanged weights, and the same seed. Capacity was enforced using PyTorch's allocator limit. No reduced resolution, step count or replacement guidance model was used.

| Simulated allocation budget | Generation | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: |
| 24 GiB, resident diffusion pair | 39.45 s | 19.19 GiB | 20.56 GiB |
| 4 GiB, RAM-backed block streaming | 100.05 s | 2.79 GiB | 3.92 GiB |

**The two final output images were pixel-identical.** This supports this scene/configuration, not every possible image. Streaming is slower because weights cross PCIe. The first generation includes cold work; do not compare these times directly with warm acceleration tests. Allocated/reserved values exclude some driver/context and non-PyTorch memory. A 4 GiB allocator cap on a 5090 does not prove operation on a physical 4 GB card.

Earlier cross-process streaming comparisons differed by up to four 8-bit channel levels and failed a strict one-level comparison. The final paired test fixed the comparison conditions; no general bitwise guarantee across kernels/devices is claimed. Removing the extension's global cuDNN benchmark toggle and parking guidance before 24-GB decoding also avoided unnecessary cold workspace pressure.

## Optional approximations

| Same-run comparison | Full evaluation | Acceleration | Time reduction |
| --- | ---: | ---: | ---: |
| FP8 + Spectrum | 26.31 s | 19.83 s | 24.6% |
| FP8 + First Block Cache | 26.61 s | 19.48 s | 26.8% |
| INT8 convrot conditional + INT8 convrot text encoder, FP8 guidance | 20.17 s | 15.26 s with Spectrum | 24.3% |

Spectrum forecast 5 of 20 steps; First Block Cache reused 11 of 40 guidance-branch forwards. Repeated accelerated runs were repeatable in these tests. Fine details changed with Spectrum; First Block Cache softened texture. Neither is enabled by default. INT8 and FP8 are different weight representations, so their images are not expected to match.

These acceleration measurements preceded removal of the global cuDNN benchmarking toggle; they are warm runs and should be remeasured in the user's actual host configuration. No claim is made that all measured speedups combine.

## Regression coverage and remaining validation

- V2 baseline: 774 CPU checks across 28 suites. V3 adds 26 checks for preset registration, cache identity, multiple adapters, memory release and quantization boundaries. One optional suite requires local adapters; GPU scripts run separately.
- Three tiny real-CUDA checks passed: mixed NF4/dense text-encoder load and streaming, streamed LoRA, streamed LoKr. NF4 used authentic bitsandbytes packed statistics, with exact output agreement to its reference fixture.
- Real full-image FP8 and INT8 convrot text encoders passed. Modern metadata remapping, malformed metadata rejection and missing-language-weight refusal are covered by CPU tests.
- Gradio panel rendered in a standalone local preview. Selection visibility, control order, non-Ideogram passthrough, changed host signatures and unrelated module preservation have automated checks.
- Not validated: physical RTX 30/40 GPUs, all resolutions/batches, all model releases, full-model NF4 quality, every FP4/MXFP8 kernel, raw GGUF loading, every third-party extension combination or future Forge version.

Reproduce capacity tests with `tests/benchmark_gpu.py --help`; supply explicit local model paths and a disposable output folder. Small CUDA fixtures: `tests/test_quantized_components_gpu.py`. Do not run large GPU tests while Forge is generating.

## Acceleration choices and sources

We delegate existing attention and modern quantization operations to [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), with local contract checks and fallback. Its SageAttention implementation falls back for the tested 256-wide Ideogram heads; this is not advertised as a Sage speed gain.

The existing velocity Spectrum implementation was retained and compared with the model-specific work in [ComfyUI Spectrum Ideogram4](https://github.com/Nif00/ComfyUI-Spectrum-Ideogram4). The new First Block Cache independently adapts the residual-reuse idea documented by [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed), without importing ComfyUI or replacing global operators.

[TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache) uses model-specific behavior; no verified Ideogram calibration was established here, so arbitrary coefficients were not added. Compilation is not enabled automatically: cold compilation, changing shapes, quantized operators and CPU streaming require their own compatibility and performance measurements. Distilled Turbo/Instant recipes remain explicit choices rather than silent substitutes for full guidance.
