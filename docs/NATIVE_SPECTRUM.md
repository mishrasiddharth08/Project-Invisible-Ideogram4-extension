# Forge built-in Spectrum integration

## Use

Restart Forge. Select an Ideogram checkpoint, enable Forge's existing **Spectrum
Integrated** control and use its normal seven settings. Press normal Generate.
No additional Ideogram panel or duplicate Spectrum toggle is added.

For the API, use the existing always-on script entry:

```json
{"alwayson_scripts":{"Spectrum Integrated":{"args":[true,0.25,6,0.5,2,0.0,6,0.9]}}}
```

Order: enable, prediction weighting, polynomial degree, regularization, cache
window, window growth, warmup steps, stop-caching fraction. A short recipe may
complete warmup without forecasting. The console and infotext record actual
real/forecast branch calls. Spectrum approximates predictions, so pixels can vary.

## Actual implementation

The bridge imports the installed `extensions-builtin/sd_forge_spectrum/lib_spectrum`
and calls its **SpectrumNode.patch** on real Forge **UnetPatcher** instances.
Native prediction, blending, cache cadence, warmup and termination stay inside
Forge's implementation. There is no private forecaster or copied algorithm.

The Ideogram adapter maps its increasing flow time to descending `1-t` for native
pass detection, while retaining exact original model timesteps. As with Forge's
KModel, the hook forecasts denoised endpoints `D = z + (1-t)*velocity`. Only
forecast items convert back to velocity; real results are retained bit-for-bit.
Near the endpoint, real inference avoids unstable division by tiny noise levels.

CFG branches remain separate. Fresh sessions are created per image; histories
reset after temporary adapter removal, correction hooks and gaps in unconditional
evaluation. Cancellation closes native sessions in a `finally` block. Existing
Ideogram code continues to own model residency; native hooks do not duplicate
checkpoint tensors or register a competing generation engine.

This is native Spectrum in a version-bound Ideogram compatibility bridge. It does
not make the entire Ideogram route a full native Forge engine. No host source
file or built-in Spectrum source was changed.

## Evidence — 22 October 2026

Installed native module SHA-256:
`b5b34d0503d314ba2e802319c92f87d76ef485cb5895f85a024fa34a7b72026a`.

Checks cover installed-node/patcher identity, actual forward skipping, branch
isolation, timestep direction, partial batch masks, history reset, all-real exact
parity, repeated sampling, cancellation and restoration of normal sampling.
The focused regression and installed-host lanes passed 72 tests plus 14 subtests;
native Settings, JavaScript progress and console-progress checks also passed.

A full-weight standalone A/B check used RTX 5090, INT8 ConvRot checkpoint and text
encoder, full Flux2 VAE, unconditional LoRA at 1.0, 1024x1024, seed 1234, and the
12-step recipe. Native controls were their installed defaults. A 4 GiB PyTorch
allocation cap and block streaming bounded this test without unloading other
processes. Both outputs were visually checked: red teapot, blue mug and oranges.

| Mode | Recorded request time | Peak tensor allocation | Peak reserved | Branch calls |
|---|---:|---:|---:|---|
| Off | 95.85 s | 2.351 GiB | 3.750 GiB | 24 real |
| Forge native | 64.49 s | 2.350 GiB | 3.816 GiB | 22 real, 2 forecast |

These are individual integration checks, **not a matched performance benchmark**:
the second run reuses prompt features, and other GPU activity was present. Do not
attribute the time difference to Spectrum alone. Device-wide VRAM includes other
processes and exceeds this process's tensor allocation.

An earlier 512px trial produced block cards with Spectrum both off and on. Those
outputs were retained as failures and did not count as visual acceptance. The
corrected check uses 1024px and rejects detected blocked outputs. This does not
establish quality for every prompt, seed, resolution or device.

Reproduction: `tests/check_native_spectrum_gpu.py <outside-extension-output-folder>`
using Forge's Python. This explicit GPU script requires 6 GiB free VRAM and
30 GiB available RAM, and enforces the per-process cap. Keep normal regression
tests separate from installed-host component tests, because Forge parses its own
command-line arguments when imported into a test process.
