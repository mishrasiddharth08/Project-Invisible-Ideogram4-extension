"""Ideogram asset sources, as links you click yourself.

WHY THERE IS NO DOWNLOADER ANY MORE
-----------------------------------
This extension used to fetch assets itself, with buttons in the accordion and
a Civitai API key stored in config.json. That is gone:

* the Civitai gray-bypass LoRA is creator-gated, so an unauthenticated request
  answers 401 and the button could not work at all without a key;
* asking someone to paste an API token into a text field, so a background
  thread can put it in an Authorization header, is a lot to ask for what is
  really "please fetch this file";
* the key had to be stored somewhere, and config.json is the file most likely
  to end up in a screenshot, a bug report or a git commit.

Your browser is already signed in to Hugging Face and Civitai, already knows
how to resume a 9 GB download, and already shows you the licence gate. So the
panel now prints links and the folder to drop the file into, and nothing here
touches the network.

Every path is relative to the Forge install root.
"""

from __future__ import annotations

HF = "https://huggingface.co"


def _hf(repo: str, path: str) -> str:
    """A direct download URL for one file in a Hugging Face repo."""
    return f"{HF}/{repo}/resolve/main/{path}?download=true"


def _hf_page(repo: str) -> str:
    return f"{HF}/{repo}"


# (label, filename, url, destination folder, note)
CHECKPOINTS: list[tuple[str, str, str, str, str]] = [
    ("Ideogram 4 - int8_convrot  (recommended)", "ideogram4_int8_convrot.safetensors",
     _hf("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_int8_convrot.safetensors"),
     "models/Stable-diffusion/IDEOGRAM 4",
     "~9.2 GB. Fastest and most accurate of the quantised builds on this stack."),
    ("Ideogram 4 - fp8_scaled", "ideogram4_fp8_scaled.safetensors",
     _hf("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_fp8_scaled.safetensors"),
     "models/Stable-diffusion/IDEOGRAM 4",
     "~9.3 GB. Widest compatibility; runs on any CUDA GPU."),
    ("Ideogram 4 - nvfp4_mixed", "ideogram4_nvfp4_mixed.safetensors",
     _hf("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_nvfp4_mixed.safetensors"),
     "models/Stable-diffusion/IDEOGRAM 4",
     "Blackwell (RTX 50xx) only - Forge disables the format on older GPUs."),
]

UNCOND: list[tuple[str, str, str, str, str]] = [
    ("Unconditional half - int8_convrot", "ideogram4_unconditional_int8_convrot.safetensors",
     _hf("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_unconditional_int8_convrot.safetensors"),
     "models/Stable-diffusion/IDEOGRAM 4",
     "Optional. Match the quant of your cond checkpoint, or use the "
     "uncond-replacement LoRA instead and save 9.3 GB of VRAM."),
    ("Unconditional half - fp8_scaled", "ideogram4_unconditional_fp8_scaled.safetensors",
     _hf("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"),
     "models/Stable-diffusion/IDEOGRAM 4", "Optional, as above."),
]

MODULES: list[tuple[str, str, str, str, str]] = [
    ("Text encoder - Qwen3-VL-8B fp8", "qwen3vl_8b_fp8_scaled.safetensors",
     _hf("Comfy-Org/Ideogram-4", "text_encoders/qwen3vl_8b_fp8_scaled.safetensors"),
     "models/text_encoder", "~10.6 GB. Required."),
    ("VAE - Flux2 KL", "flux2-vae.safetensors",
     _hf("Comfy-Org/Ideogram-4", "vae/flux2-vae.safetensors"),
     "models/VAE", "~330 MB. Required. Flux1's ae.safetensors will NOT work "
     "(16 latent channels vs the 32 this model needs)."),
]

ADAPTERS: list[tuple[str, str, str, str, str]] = [
    ("TurboTime LoRA (4-step distill)", "ideogram_4_turbotime_v1.safetensors",
     _hf("ostris/ideogram_4_turbotime_lora", "ideogram_4_turbotime_v1.safetensors"),
     "models/Lora/IDEOGRAM LORA", "Powers the TurboTime-4 preset."),
    ("Uncond-replacement LoRA", "ideogram_4_unconditional_lora_r16.safetensors",
     _hf("ostris/ideogram_4_unconditional_lora", "ideogram_4_unconditional_lora_r16.safetensors"),
     "models/Lora/IDEOGRAM LORA",
     "Stands in for the 9.3 GB unconditional transformer, which is then not loaded."),
    ("Gray-bypass LoRA (filter)", "Gray_000002000.safetensors",
     "https://civitai.com/models/2750357", "models/Lora/IDEOGRAM LORA",
     "Creator-gated: sign in to Civitai in your browser, then download. "
     "Applied at -0.25 on the first step only."),
]

# Uncensored / abliterated encoders. Sharded HF folders, so point
# `text_encoder_override` in config.json at the folder itself.
ALT_ENCODERS: list[tuple[str, str, str]] = [
    ("huihui-ai / Huihui-Qwen3-VL-8B-Instruct-abliterated",
     _hf_page("huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"),
     "Verified against this loader: hidden 4096, 36 layers, intermediate 12288."),
    ("prithivMLmods / Qwen3-VL-8B-Instruct-abliterated-v2",
     _hf_page("prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2"), ""),
    ("HauhauCS / Qwen3VL-8B-Uncensored-HauhauCS-Aggressive",
     _hf_page("HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive"),
     "GGUF only. Convert to safetensors first (Forge Neo's converter does "
     "this); llama.cpp key names are handled."),
]


def _rows(items) -> str:
    out = ["| file | get it | put it in |", "|---|---|---|"]
    for label, filename, url, dest, note in items:
        n = f"<br><sub>{note}</sub>" if note else ""
        out.append(f"| **{label}**<br><code>{filename}</code>{n} "
                   f"| [download]({url}) | `{dest}` |")
    return "\n".join(out)


def markdown() -> str:
    """The whole panel. Pure string - no disk access, no network."""
    alts = "\n".join(
        f"- [{name}]({url})" + (f" - <sub>{note}</sub>" if note else "")
        for name, url, note in ALT_ENCODERS)
    return f"""### Get the files

Manual download is recommended. Optional automatic downloads require selecting
files above and clicking **Download selected files**. Or click a link and save the file into the folder
named beside it, then press **Refresh** on Forge's checkpoint dropdown.
Downloads run in your browser, where you are already signed in and where a
9 GB transfer can resume.

**Required to generate:** one checkpoint + the text encoder + the VAE.

#### Checkpoint (pick one)
{_rows(CHECKPOINTS)}

#### Text encoder and VAE (both required)
{_rows(MODULES)}

#### Adapters (optional)
{_rows(ADAPTERS)}

<details><summary>Unconditional half (optional - better CFG, +9.3 GB VRAM)</summary>

{_rows(UNCOND)}

Skip these and use the **uncond-replacement LoRA** instead: it does the same
job as a side path, and the big half is never loaded.
</details>

<details><summary>Uncensored / abliterated text encoders</summary>

Drop-in replacements for the Qwen3-VL-8B encoder. These are sharded HF
folders, so download the whole folder and point `text_encoder_override` in
this extension's `config.json` at it:

```json
"text_encoder_override": "<Forge folder>/models/text_encoder/<folder>"
```

{alts}
</details>

**LoRAs** go in `models/Lora/IDEOGRAM LORA` or anywhere under `models/Lora` -
subfolders are searched, at any depth. Use them with `<lora:name:1.0>` in the
prompt or the **Adapter** dropdown above.
"""
