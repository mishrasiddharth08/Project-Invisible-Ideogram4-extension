import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi_ideogram_forge.lora import apply_lora_set, find_lora_file


def write_lora(path: Path, alias: str | None = None) -> None:
    metadata = {} if alias is None else {"ss_output_name": alias}
    header = json.dumps({"__metadata__": metadata}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(header).to_bytes(8, "little") + header)


class LoraResolutionTests(unittest.TestCase):
    def resolve(self, root: Path, name: str):
        with patch("pi_ideogram_forge.lora.lora_search_dirs", return_value=[root]):
            return find_lora_file(name, [root])

    def test_exact_filename_resolution(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root = Path(td)
            wanted = root / "IDEOGRAM LORA" / "Gray_000002000.safetensors"
            write_lora(wanted, "Gray")
            self.assertEqual(self.resolve(root, "Gray_000002000"), wanted)

    def test_unique_native_alias_resolves_gray_regression(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root = Path(td)
            wanted = root / "IDEOGRAM LORA" / "Gray_000002000.safetensors"
            write_lora(wanted, "Gray")
            self.assertEqual(self.resolve(root, "Gray"), wanted)

    def test_exact_filename_wins_over_same_named_alias(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root = Path(td)
            exact = root / "Gray.safetensors"
            write_lora(exact, "Other")
            write_lora(root / "Gray_000002000.safetensors", "Gray")
            self.assertEqual(self.resolve(root, "Gray"), exact)

    def test_ambiguous_alias_is_refused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root = Path(td)
            write_lora(root / "one.safetensors", "Gray")
            write_lora(root / "two.safetensors", "gray")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                self.resolve(root, "Gray")

    def test_duplicate_exact_names_are_refused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root = Path(td)
            write_lora(root / "one" / "Gray.safetensors")
            write_lora(root / "two" / "Gray.safetensors")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                self.resolve(root, "Gray")

    def test_similar_filename_is_not_fuzzy_matched(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root = Path(td)
            write_lora(root / "Gray_000002000.safetensors", "NotGray")
            self.assertIsNone(self.resolve(root, "Gray"))

    def test_apply_never_silently_skips_missing_adapter(self):
        fake = types.ModuleType("pi_ideogram_lora.ideogram4_lora")
        for name in ("apply_lora_to_model", "classify_adapter", "inspect_lora", "load_lora_state_dict"):
            setattr(fake, name, lambda *args, **kwargs: None)
        fake.LoraRefused = RuntimeError
        with patch.dict(sys.modules, {"pi_ideogram_lora.ideogram4_lora": fake}), \
                patch("pi_ideogram_forge.lora.find_lora_file", return_value=None):
            with self.assertRaisesRegex(ValueError, "generation stopped"):
                apply_lora_set(SimpleNamespace(), [("missing", 1.0, "dual")])


if __name__ == "__main__":
    unittest.main()
