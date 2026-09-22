"""Release a known idle external worker before taking the GPU for Ideogram."""
import sys


def release_idle_qwen_worker():
    runtime = sys.modules.get('pi_qwen21.lib.runtime')
    if runtime is None or getattr(runtime, '_pipe', None) is None:
        return False
    lock = runtime.LOCK
    if not lock.acquire(blocking=False):
        raise RuntimeError('Qwen 2.1 is still generating. Wait for it to finish before starting Ideogram.')
    try:
        runtime.release()
        print('[Invisible-I4] Released idle Qwen 2.1 worker before taking GPU memory')
        return True
    finally:
        lock.release()
