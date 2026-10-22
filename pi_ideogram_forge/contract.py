"""Explicit limits of the current version-bound Ideogram compatibility route."""
CAPABILITIES = {
    'contract_version': 1,
    'route': 'version-bound Forge compatibility adapter',
    'tasks': ('txt2img', 'experimental-img2img'),
    'representation': '32-channel Flux2 latents, 2x2 packed to 128 channels',
    'prediction': 'flow velocity; Euler with Ideogram logit-normal schedule',
    'unsupported': ('inpainting', 'outpainting', 'reference editing', 'hires fix',
                    'negative text conditioning', 'variation seeds', 'prompt arrays'),
}


def validate_request(p):
    """Reject unsupported intent before loading weights; never silently ignore it."""
    for name in ('prompt', 'negative_prompt', 'seed', 'subseed'):
        if isinstance(getattr(p, name, None), (list, tuple)):
            raise ValueError(f'Ideogram compatibility mode does not support {name} arrays. Submit separate requests.')
    if str(getattr(p, 'negative_prompt', '') or '').strip():
        raise ValueError('Ideogram uses image-only unconditional guidance; negative text conditioning is unsupported. Clear Negative prompt.')
    if getattr(p, 'enable_hr', False):
        raise ValueError('Native Hires fix is unsupported by this Ideogram compatibility route.')
    if float(getattr(p, 'subseed_strength', 0) or 0) != 0:
        raise ValueError('Variation seeds are unsupported by this Ideogram compatibility route.')
    sampler = str(getattr(p, 'sampler_name', '') or '').strip()
    if sampler and sampler.lower() != 'euler':
        raise ValueError('Ideogram currently supports Euler only. Select Euler in the native sampler control.')
    for name in ('width', 'height'):
        value = int(getattr(p, name, 1024))
        if not 256 <= value <= 2048 or value % 16:
            raise ValueError(f'Ideogram {name} must be 256–2048 and divisible by 16; received {value}.')


def validate_native_spectrum(runner, p):
    """Read and validate the installed Spectrum script's native arguments."""
    from pi_ideogram_forge.spectrum_settings import read_native_spectrum
    return read_native_spectrum(runner, p)
