"""Publish completed images without another VAE decode or preview throttle."""

from contextlib import contextmanager


@contextmanager
def native_preview_session(options):
    """Enable Forge's real preview transport only for this Ideogram job.

    The host progress endpoint itself refuses images when previews are off;
    publishing current_image alone cannot make them visible. Never save these
    temporary values or leave other models' preview preferences changed.
    """
    data = options.data
    missing = object()
    forced = {'live_previews_enable': True, 'show_progress_every_n_steps': 1}
    previous = {key: data.get(key, missing) for key in forced}
    data.update(forced)
    try:
        yield
    finally:
        for key, value in previous.items():
            # Keep an explicit user change made while the job was running.
            if data.get(key, missing) != forced[key]:
                continue
            if value is missing:
                data.pop(key, None)
            else:
                data[key] = value


def _publish(state, image):
    """Use Forge's image assignment contract when this host exposes it."""
    assign = getattr(state, 'assign_current_image', None)
    if callable(assign):
        assign(image)
        return
    state.current_image = image
    state.id_live_preview = int(getattr(state, 'id_live_preview', 0)) + 1


def publish_start(state, width, height):
    """A lightweight display-only canvas; never passed to the sampler."""
    from PIL import Image, ImageOps
    scale = 256 / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    gradient = Image.linear_gradient('L').resize(size)
    state.sampling_step = 0
    _publish(state, ImageOps.colorize(gradient, '#202938', '#b7c9dc'))


def publish_frame(state, image):
    """Publish an intermediate frame without changing sampling progress."""
    _publish(state, image)


def publish_final(state, image, steps):
    state.sampling_step = int(steps)
    _publish(state, image)
