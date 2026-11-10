"""Bind the known API alias after Forge finishes importing its entry points."""
from functools import wraps
import sys


def bind_api_entrypoint(route):
    module = sys.modules.get("modules.api.api")
    original = getattr(module, "process_images", None)
    if not callable(original) or getattr(original, "_pi_i4_api_guard", False):
        return False

    @wraps(original)
    def guarded(p, *args, **kwargs):
        return route(original, p, *args, **kwargs)

    guarded._pi_i4_api_guard = True
    module.process_images = guarded
    return True
