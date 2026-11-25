from types import ModuleType
import sys
from pi_ideogram_forge.entrypoints import bind_api_entrypoint


def test_late_api_binding_preserves_chain_and_is_idempotent(monkeypatch):
    module = ModuleType("modules.api.api")
    calls = []
    def original(p, *args, **kwargs):
        calls.append((p, args, kwargs))
        return "other-model"
    module.process_images = original
    monkeypatch.setitem(sys.modules, module.__name__, module)
    def route(fallback, p, *args, **kwargs):
        return "ideogram" if p == "i4" else fallback(p, *args, **kwargs)
    assert bind_api_entrypoint(route)
    guard = module.process_images
    assert not bind_api_entrypoint(route)
    assert module.process_images is guard
    assert guard("i4") == "ideogram"
    assert guard("other", 3, named=4) == "other-model"
    assert calls == [("other", (3,), {"named":4})]
    assert guard.__wrapped__ is original


def test_missing_api_is_untouched(monkeypatch):
    monkeypatch.delitem(sys.modules, "modules.api.api", raising=False)
    assert not bind_api_entrypoint(lambda *_: None)
