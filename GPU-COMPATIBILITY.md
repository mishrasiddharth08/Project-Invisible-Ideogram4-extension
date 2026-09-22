# GPU compatibility and memory policy

This extension is not validated on every GPU or at every VRAM size.

The supported device interfaces in the code are PyTorch CUDA (including AMD
through a compatible ROCm build) and CPU. DirectML and Vulkan are not implemented.
Driver, GPU, operating-system and host Forge support are prerequisites; an
extension cannot add those runtimes. AMD inference has not been hardware-tested
in this workspace.

## Portability changes

- Device probing and memory queries use the selected GPU rather than GPU 0.
- BF16 support is checked on that device. When it is unavailable, FP32 is used
  instead of assuming that FP16 can represent the model's values. This fallback
  increases memory use and can be substantially slower.
- RAM-backed block streaming is selected when available VRAM is below 23.5 GiB,
  even on a larger card shared with other workloads. FP32 GPU execution also
  streams. The decision is fixed before loading so allocations cannot change
  the policy midway through initialization.
- ROCm skips the optional NVIDIA-specific FP8 acceleration probe. Portable FP8
  storage and ordinary linear computation are the intended fallback. Specialized
  Forge quantization formats still require their own compatible host kernels;
  selecting such a file does not make those kernels portable.

## Limits

Streaming still requires space for an active block, activations, adapters and
VAE decoding, plus enough system RAM for model weights. Tiny VRAM configurations
can still run out of memory. Lower canvas size, a smaller compatible checkpoint
and the unconditional replacement LoRA can reduce demand. CPU fallback is
functional in small-model tests, not a promise of practical full-model speed.

Policy tests simulate 2–80 GiB tiers, AMD backend detection, BF16 capability,
multiple devices and shared VRAM. These are logic tests, not successful image
generation benchmarks for those GPUs. No minimum supported VRAM is claimed.
