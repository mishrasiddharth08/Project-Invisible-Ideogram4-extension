"""Release only this extension's cached models on checkpoint changes."""
from threading import Lock, RLock, Thread

lock = RLock()
selection = None
_pending_lock = Lock()
_pending_selection = None
_release_worker = None


def _release_locked(cache, selected):
    """Unload every unique pipeline; one broken unload must not retain others."""
    global selection
    if selected == selection:
        return
    failures = []
    for pipeline in list({id(p): p for p in cache.values()}.values()):
        try:
            pipeline.unload()
        except Exception as error:
            failures.append(error)
    cache.clear()  # never reuse a partially unloaded pipeline
    selection = selected
    if failures:
        print(f"[Project Invisible Ideogram 4] cleanup warning: "
              f"{len(failures)} pipeline unload(s) failed; cache was discarded")


def _drain_pending(cache):
    global _pending_selection, _release_worker
    while True:
        with _pending_lock:
            selected = _pending_selection
            _pending_selection = None
            if selected is None:
                _release_worker = None
                return
        with lock:
            _release_locked(cache, selected)


def _defer_release(cache, selected):
    """Keep UI callbacks non-blocking, but never lose a change at lock release."""
    global _pending_selection, _release_worker
    with _pending_lock:
        _pending_selection = selected  # latest checkpoint is authoritative
        if _release_worker is None:
            _release_worker = Thread(
                target=_drain_pending, args=(cache,),
                name="i4-checkpoint-cleanup", daemon=True)
            _release_worker.start()


def _wait_for_pending_release(timeout=2):
    """Test helper: wait for the single deferred cleanup worker."""
    with _pending_lock:
        worker = _release_worker
    if worker is not None:
        worker.join(timeout)
    return worker is None or not worker.is_alive()


def release_changed(cache, selected):
    if not lock.acquire(blocking=False):
        # Record even a switch back to `selection`: it cancels an older queued
        # switch-away request while generation still owns the lock.
        _defer_release(cache, selected)
        return
    try:
        _release_locked(cache, selected)
    finally:
        lock.release()


def install(cache):
    from modules import shared, script_callbacks

    def bind(*_args):
        option = getattr(shared.opts, 'data_labels', {}).get('sd_model_checkpoint')
        if option is None or not hasattr(option, 'onchange'):
            print('[Project Invisible Ideogram 4] checkpoint cleanup hook skipped: '
                  'this Forge build does not expose the expected option contract')
            return
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
