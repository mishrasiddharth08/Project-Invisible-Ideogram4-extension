"""Publish completed images without another VAE decode or preview throttle."""


def publish_start(state, width, height):
    """A lightweight display-only canvas; never passed to the sampler."""
    from PIL import Image, ImageOps
    scale = 256 / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    gradient = Image.linear_gradient('L').resize(size)
    state.current_image = ImageOps.colorize(gradient, '#202938', '#b7c9dc')
    state.sampling_step = 0
    state.id_live_preview += 1


def publish_final(state, image, steps):
    state.current_image = image
    state.sampling_step = int(steps)
    state.id_live_preview += 1
