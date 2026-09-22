# Project Invisible — Ideogram 4 for Forge Neo

An engine-scoped Ideogram 4 extension: the normal Forge checkpoint selector, prompt box, Generate button and output folders. No second WebUI or environment.

This is a **tested release candidate**, not a promise that every checkpoint, GPU or future Forge version will work. See [BENCHMARKS.md](../BENCHMARKS.md) and [ISOLATION.md](../ISOLATION.md).

## Start here

1. Close Forge completely before installing or updating this extension.
2. Place this folder under Forge's `extensions/project-invisible-ideogram-4`.
3. Keep your compatible Ideogram conditional/unconditional checkpoints in `models/Stable-diffusion`, Qwen3-VL-8B in `models/text_encoder`, and Flux2 VAE in `models/VAE`. Existing subfolders are searched.
4. Start Forge, choose **ideogram4** in the top-left **UI Preset**, select the Ideogram **conditional** checkpoint and expand **Ideogram 4**.
5. Choose **Balanced**, 1024×1024, 20 steps, batch 1. Write a prompt and use the normal Generate button.

The extension does not install packages or automatically download weights. Model Setup shows links and destinations. Accept model/adapter licenses at their source. A renamed file does not convert its internal model layout.

## Text-to-image and image-to-image

Use **txt2img** to create a new image. For changes to an existing image, use
Forge's **img2img** tab and its normal image upload, size, resize mode, and
**Denoising strength** controls. There is no separate editing panel.

Image-to-image is **experimental latent regeneration**, not a native instruction
editor. Lower strength preserves more of the source; higher strength changes more.
Zero returns the resized source unchanged; one starts from noise. Partial strength
needs a complete compatible Flux2 VAE, including encoder weights. Inpainting/masks
and latent-only resizing are not supported and produce an explanation.

In **Model Setup**, manual download is recommended and selected by default.
Use the supplied links and destination folders, or choose **Download selected files
for me**, select the specific files, and click **Download selected files**.
Only selected files are downloaded; nothing is fetched on startup or Generate.
Gray remains a manual creator download. Review each model's license before downloading.

## A simple speed/quality choice

Quality-preserving defaults retain all sampling steps, full guidance and checkpoint precision. Prompt features use compact caching, unnecessary unconditional text projection is skipped, and memory transfers avoid redundant cleanup.

**Advanced → Extra acceleration** offers one approximation at a time:

- **Off:** full computation; best starting point for precise details and comparisons.
- **Spectrum:** forecasts selected evaluations. The measured FP8 test fell from 26.3 to 19.8 seconds, with changed fine details.
- **First Block Cache — experimental:** reuses remaining-block residuals when the first block changes little. The measured test fell from 26.6 to 19.5 seconds, but texture softened.

These numbers are single-machine, single-prompt samples, not guarantees. Both approximations reset between images, separate guidance histories, preserve warmup/final steps, and cannot stack. Turbo/Instant recipes are separate quality/speed choices and require compatible distilled weights/adapters.

Attention follows **Forge's already-selected implementation**. Its own hardware fallbacks still apply; the tested Forge SageAttention implementation falls back to SDPA for Ideogram's 256-wide heads. Masked batches use SDPA to preserve padding behavior. The extension never enables global cuDNN benchmarking, TF32 or reduced-precision accumulation behind your back.

## Memory and supported GPUs

Target: NVIDIA RTX 30-series and newer, with compatible Forge/PyTorch/CUDA builds.

- Below approximately 24 GiB, transformer and text-encoder blocks stream from system RAM.
- Larger cards keep the diffusion models resident; 24-GB cards park the guidance model for final decoding headroom.
- Streaming retains weights and steps, but PCIe transfers cost time. More VRAM is faster.
- A 1024px full-pair FP8 image completed under an enforced 4 GiB allocation budget on an RTX 5090. That is **not** physical RTX 3050/3060/40-series validation.
- Low VRAM does not mean low system RAM: the full models still need tens of gigabytes of RAM. 48–64 GB is a sensible target for full-pair streaming; Windows, other applications and file cache need additional room.

An out-of-memory error does **not** silently lower resolution, replace guidance, or merge a runtime adapter into quantized weights. Advanced users can explicitly opt into the legacy quality-reducing recovery using `allow_quality_reducing_oom_recovery`; it defaults to false.

## Quantization: honest compatibility

| Format | Route / validation |
| --- | --- |
| FP8 E4M3 scaled | Portable weight-only loader; full-image tests |
| INT8 tensorwise / convrot | Forge quantization operations; full-image test |
| BF16 / FP16 | Dense loader; memory requirement is larger |
| bitsandbytes NF4 | Native-key single-file layout for diffusion and text encoder; tiny real-CUDA loading/streaming test, not a full NF4 model-quality benchmark |
| FP8 E5M2, MXFP8, NVFP4, convrot W4A4, asymmetric W4A8, mixed descriptors | Routed by actual metadata to the installed Forge backend; each GPU/kernel/layout combination still needs validation |
| GGUF, unrecognized W8A8 / foreign diffusers layouts | Refused with guidance; no verified Ideogram layout loader in this release |

“Listed” does not mean every checkpoint has been tested. New formats are not treated as ordinary FP8. Missing backends, unknown descriptors and missing weights fail with an explanation. No automatic conversion, kernel downloads or driver changes.

The **text encoder uses these routes too**: FP8 and INT8 convrot were tested in full-image runs. Converted llama.cpp-named safetensors preserve secondary/activation scales and packed NF4 metadata; this does not add a raw `.gguf` reader. Mixed dense/NF4 language layers stay in their saved formats. FP4/MXFP8 and newer kernels remain hardware/backend-dependent; they are not promised on every RTX card.

## Filter-bypass controls

The existing prompt-density and optional adapter mechanisms remain available. A named, installed compatible bypass adapter is required for the adapter path; selecting None is not an active adapter. Raw prompt pass-through disables JSON wrapping. No mechanism can promise success for every prompt or checkpoint, and no bypass weights are bundled.

The optional debanner also requires its separately licensed local correction file under `pretrained/`; missing assets produce guidance, never an automatic download.

## Isolation and updates

Only `scripts/engine.py` is a Forge script entry point. Integration helpers live in `pi_ideogram_forge`; model, configuration and prompt code are in `pi_ideogram_lib`; asset helpers are in `pi_ideogram_assets`. CSS is scoped to the Ideogram panel.

No Forge core files, built-in extensions, preset enum members/recipe tables or other extension configuration are edited. An additive wrapper around Forge's public preset choices restores **ideogram4** in the native top-left selector; only Ideogram-specific option keys are registered. Runtime hooks preserve non-Ideogram calls. Incompatible host interfaces fail closed. Disable/remove the extension and **restart Forge completely** to remove its runtime integration. Hot-reloading a running model is not supported.

If the preset is missing after updating from V2, close the Forge launcher/console and start it again; refreshing the browser cannot load changed Python code. The console should report `UI Preset 'ideogram4' served to browser: OK`. Check Extensions if that line is absent. This preset selects the UI recipe; you still need a compatible local checkpoint, text encoder and VAE.

## Tests

From this extension folder, using Forge's Python:

```text
<Forge Python> -B tests/run_cpu_suite.py
```

The runner skips GPU/integration investigations unless explicitly invoked. Optional model-file checks use `IDEOGRAM_TEST_MODELS` and `IDEOGRAM_TEST_LOKR`. Tests may update the extension's local configuration; run them in a disposable copy, not your live extension.

## License and credits

Extension code: [Apache License 2.0](LICENSE), approved by the project owner. See [NOTICE](NOTICE) for upstream components.

**Model weights and adapters retain their own licenses.** Apache licensing of this extension does not grant commercial rights to Ideogram weights. Read [Ideogram's licensing page](https://ideogram.ai/licensing/) and each asset's model card before use or redistribution. No weights, credentials or personal configuration are included.

Acceleration research: [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), [ComfyUI Spectrum for Ideogram](https://github.com/Nif00/ComfyUI-Spectrum-Ideogram4), [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed), [TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache). This extension does not import or install ComfyUI nodes.
