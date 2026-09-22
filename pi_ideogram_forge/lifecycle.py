"""Release this extension's cached models on checkpoint changes."""
from threading import RLock

lock = RLock()
selection = None


def release_changed(cache, selected):
    global selection
    if selected == selection or not lock.acquire(blocking=False):
        return
    try:
        if selected == selection:
            return
        for pipeline in list({id(p): p for p in cache.values()}.values()):
            pipeline.unload()
        cache.clear()
        selection = selected
    finally:
        lock.release()


def install(cache):
    from modules import shared, script_callbacks

    def bind(*_args):
        option = shared.opts.data_labels['sd_model_checkpoint']
        previous = option.onchange
        if getattr(previous, '_i4_release', False):
            return

        def changed():
            release_changed(cache, shared.opts.sd_model_checkpoint)
            if previous:
                previous()

        changed._i4_release = True
        option.onchange = changed

    script_callbacks.on_app_started(bind)
