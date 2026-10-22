"""Use the installed Forge Flux2 RGB preview coefficients, without a VAE pass."""
from functools import lru_cache

import torch
from PIL import Image


@lru_cache(maxsize=1)
def _native_factors():
    from huggingface_guess.latent import SDXL_Flux2
    from modules.sd_vae_approx import latent_format
    return latent_format(SDXL_Flux2())[:2]


@torch.no_grad()
def decode_native_preview(z, grid, patch=2, preview_pixels=384):
    """Unpack Ideogram's pixel-major tokens, then apply native RGB preview.

    Forge Flux2's channel-major packing differs: feeding packed Ideogram
    tokens directly to its reshape would scramble the picture.
    """
    gh, gw = grid
    channels = z.shape[-1] // (patch * patch)
    if channels != 32 or z.shape[1] < gh * gw:
        raise ValueError('Forge Flux2 preview requires a complete 32-channel latent grid')
    factors, bias = _native_factors()
    pixels = (z[:1, :gh * gw].detach().float()
              .reshape(gh, gw, patch, patch, channels)
              .permute(0, 2, 1, 3, 4).reshape(gh * patch, gw * patch, channels))
    rgb = torch.nn.functional.linear(pixels, factors.to(pixels),
                                     bias.to(pixels) if bias is not None else None)
    rgb = ((rgb * .5 + .5).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    image = Image.fromarray(rgb)
    scale = preview_pixels / max(image.size)
    return image.resize((max(1, round(image.width * scale)),
                         max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
