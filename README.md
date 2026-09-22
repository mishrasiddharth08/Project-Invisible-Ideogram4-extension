# Project Invisible — Ideogram 4 for Forge Neo

**Ideogram 4 in Forge's familiar workflow. No separate generation tab or extra Python environment.**

[Installation](#beginner-installation) · [Models](#model-files) · [Update safety](UPDATE_SAFETY.md) · [Technical notes](docs/TECHNICAL.md)

## The Project Invisible idea

Use Forge's normal checkpoint selector, prompt, Generate button and image folders.
Special controls stay inside a compact, collapsed **Ideogram 4** panel.
“Invisible” means familiar—not hiding downloads, errors or limitations.
This is an independent community extension, not an official Ideogram product.

## Features and testing status

- **Text-to-image:** tested with local weights on the author's NVIDIA system.
- **Image-to-image:** experimental latent regeneration, tested in a standalone GPU check. Editing stays in Forge's **img2img** tab.
- **Organized controls:** generation recipe first; Style, Filter bypass, Performance and Model Setup grouped separately.
- **Progress:** current-image and overall-batch bars; starting gradient, evolving previews, actual final image.
- **Memory:** streaming/offloading, cached prompt features and safe release of cached models when switching checkpoints.
- **Downloads:** manual installation recommended; optional downloads require selecting files and clicking Download.
- **Adapters and acceleration:** compatible LoRA controls and optional speed settings; approximate acceleration can change image details.

Automated tests do not prove every GPU, model file or Forge version works.
AMD/ROCm is **not hardware-validated here**. DirectML/Vulkan are not implemented.
No promise is made for every VRAM size.

## Beginner installation

1. Stop Forge completely.
2. Choose **Code → Download ZIP** on this repository.
3. Extract and rename the folder to `project-invisible-ideogram-4`.
4. Place it inside `sd-webui-forge-classic/extensions/`.
5. Avoid double nesting. The correct path ends with:
   `extensions/project-invisible-ideogram-4/scripts/engine.py`.
6. Start Forge normally. The extension does not install or upgrade shared packages.
7. Refresh your browser with **Ctrl+F5** after updates.

## Model files

**Weights are not included.** Open **Model Setup** for source links, compatible variants and optional selected-file downloads.

| Component | Forge folder |
|---|---|
| Ideogram 4 conditional checkpoint | `models/Stable-diffusion` |
| Unconditional checkpoint for separate-model guidance | `models/Stable-diffusion` |
| Compatible Qwen3-VL-8B text encoder for Ideogram 4 | `models/text_encoder` |
| Compatible complete Flux2 VAE | `models/VAE` |
| Optional compatible adapters | `models/Lora` |

Existing subfolders are searched. Do not substitute an unrelated encoder or VAE.
An explicitly selected compatible unconditional replacement adapter can replace
the separate guidance model. Gray requires a manual creator download.
Review original licenses before downloading. Generate never silently fetches weights.

## First generation

1. Select **ideogram4** in Forge's UI preset selector.
2. Select your compatible **conditional** checkpoint.
3. Open **txt2img**, enter a prompt and expand **Ideogram 4** if needed.
4. Start with **Balanced**, batch **1** and a modest image size.
5. Press Forge's normal **Generate** button.

For an existing image, use **img2img** and its normal upload/denoising controls.
Lower strength preserves more of the source; zero preserves the resized source;
one starts from noise. **Inpainting/masks, latent-only resizing and native
instruction editing are unsupported.**

## Progress and memory

Preview transitions affect display only, not saved pixels. A new sampling stage
requires completed model steps; one-second polling cannot guarantee a new image
every second. The final image appears after VAE decoding.

Offloading lowers VRAM pressure but uses system RAM and transfers, which can slow
generation. Active generation is not killed by a checkpoint change; cleanup occurs
at a safe boundary. Spectrum/First Block Cache trade exact output matching for
possible speed. See [measurements](MEASURED-MEMORY-PREVIEW.md) and
[GPU limitations](GPU-COMPATIBILITY.md).

## Isolation and Forge updates

Extension code stays in its own folder. It does not rewrite Forge core files,
install shared dependencies or modify other extensions. Runtime integration still
shares Forge's Python, PyTorch and UI APIs.

**Future compatibility cannot be guaranteed.** Keep a known-working backup outside
Forge. The release's `verify_integrity.py` detects changed, missing or added source
files; it does not freeze Forge or automatically restore anything.
Read [UPDATE_SAFETY.md](UPDATE_SAFETY.md) and [ISOLATION.md](ISOLATION.md).

## If something goes wrong

Restart Forge, verify the model files and try one small image. If the problem
follows an update, run the integrity check and consult your known-working backup.

Open a GitHub issue with Forge version/commit, operating system, GPU/VRAM/RAM,
model/encoder/VAE names, image size, recipe, steps, enabled adapters/acceleration,
reproduction steps and the complete relevant traceback.

**Remove private paths, prompts, tokens and personal images before posting.**

## License and credits

Extension code: [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for Ideogram
inference components, Qwen tokenizer assets and acknowledgments. Models and
upstream projects retain their own licenses.

Thanks to Forge Neo, Ideogram, Qwen and the wider open-source communities.
