"""Native Settings/no-private-panel contract; requires CPU only."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

EXT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXT))

from pi_ideogram_forge import native_settings


def _engine_ui_node():
    tree = ast.parse((EXT / "scripts/engine.py").read_text(encoding="utf-8"))
    script_class = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.ClassDef) and n.name == "Ideogram4EngineScript")
    return tree, next(n for n in script_class.body
                      if isinstance(n, ast.FunctionDef) and n.name == "ui")


def main():
    tree, ui = _engine_ui_node()
    assert len(ui.body) == 1 and isinstance(ui.body[0], ast.Return)
    assert isinstance(ui.body[0].value, ast.List) and not ui.body[0].value.elts
    forbidden = {"Accordion", "Tab", "Tabs", "Button"}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden]

    saved = {
        "pi_i4_preset": "Auto",
        "pi_i4_adapter_strength": 0.7,
        "pi_i4_spectrum_enabled": True,  # stale/unsupported must never reactivate it
    }
    fake_shared = SimpleNamespace(opts=SimpleNamespace(data=saved))
    fake_modules = SimpleNamespace(shared=fake_shared)
    p = SimpleNamespace(override_settings={
        "pi_i4_preset": "V4_TURBO_12",
        "pi_i4_adapter_strength": "not-a-number",
        "spectrum_enabled": True,
        "first_block_cache_enabled": True,
    })
    with patch.dict(sys.modules, {"modules": fake_modules, "modules.shared": fake_shared}):
        snapshot = native_settings.named_options(p)
        resolved = native_settings.resolve_run_options(p)
    assert snapshot["preset"] == "V4_TURBO_12"
    assert snapshot["spectrum_enabled"] is False
    assert snapshot["first_block_cache_enabled"] is False
    assert resolved["adapter_strength"] == 1.0  # invalid API value safely defaults
    assert resolved["spectrum_enabled"] is False
    assert resolved["first_block_cache_enabled"] is False

    source = (EXT / "pi_ideogram_forge/native_settings.py").read_text(encoding="utf-8")
    assert "OptionHTML(_manual_setup_html())" in source
    assert "Generate never downloads files" in source
    assert "installed Spectrum controls are bridged" in source
    assert "SpectrumNode.patch" in source
    assert "not a claim that Ideogram has a full native Forge engine" in source
    assert "No private Spectrum controls are added" in source
    assert "Download selected files" not in source
    print("UI layout/native Settings tests passed")


if __name__ == "__main__":
    main()
