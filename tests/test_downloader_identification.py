import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi_ideogram_assets import downloader as dl
from pi_ideogram_lib import quant_registry

ADAPTER = {"diffusion_model.layers.0.attention.qkv.lora_A.weight": ("U8", [1, 2]),
           "diffusion_model.layers.0.attention.qkv.lora_B.weight": ("U8", [2, 1])}
ENCODER = {"token_embd.weight": ("U8", [1, 4096])}
for layer in range(36):
    for module in ("attn_q", "ffn_down"):
        ENCODER[f"blk.{layer}.{module}.weight"] = ("U8", [1, 1])


def write_file(path, tensors=None, metadata=None, truncate=0):
    tensors = tensors or {"other.weight": ("U8", [4])}
    header = {"__metadata__": metadata or ({"ss_base_model": "ideogram4"} if tensors is ADAPTER else {})}
    payload = bytearray()
    widths = {"U8": 1, "BF16": 2, "F8_E4M3": 1}
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        start = len(payload)
        payload.extend(bytes(count * widths[dtype]))
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(payload)])
    raw = json.dumps(header).encode()
    data = struct.pack("<Q", len(raw)) + raw + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data[:-truncate] if truncate else data)
    return path


class InstalledAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Avoid configuration writes and depend only on the shipped registry.
        cache = patch.object(quant_registry, "_REGISTRY_CACHE", quant_registry.DEFAULT_REGISTRY)
        cache.start()
        self.addCleanup(cache.stop)
        for target, value in (("pi_ideogram_lib.config.get", ""),
                              ("pi_ideogram_lib.detect.forge_selected_modules", [])):
            mock = patch(target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def present(self, asset="checkpoint:cond", quant="fp8_scaled"):
        return dl._asset_present(asset, quant, [self.root])

    def checkpoint(self, name, **kwargs):
        return write_file(self.root / "Stable-diffusion" / name,
                          {"embed_image_indicator.weight": ("U8", [4]),
                           "layer.weight_scale": ("U8", [1])}, **kwargs)

    def test_foreign_quant_name_and_exact_official_name_are_rejected(self):
        for name in ("flux_fp8_scaled.safetensors", "ideogram4_fp8_scaled.safetensors"):
            with self.subTest(name=name):
                write_file(self.root / "Stable-diffusion" / name)
                self.assertIsNone(self.present())

    def test_renamed_checkpoint_identified_by_header(self):
        expected = self.checkpoint("renamed.safetensors")
        self.assertEqual(self.present(), expected)

    def test_manual_checkpoint_status_and_fetch_honor_installed_file(self):
        expected = write_file(self.root / "Stable-diffusion" / "custom_bf16.safetensors",
                              {"embed_image_indicator.weight": ("BF16", [4])})
        row = next(r for r in dl.assets_status([self.root], "bf16") if r["id"] == "checkpoint:cond")
        self.assertTrue(row["present"])
        self.assertEqual(row["path"], str(expected))
        self.assertFalse(row["actionable"])
        with patch.object(dl, "hf_download", side_effect=AssertionError("unexpected download")):
            self.assertIsNone(dl.fetch_asset("checkpoint:cond", "bf16", [self.root]))

    def test_header_quant_overrides_misleading_filename(self):
        self.checkpoint("ideogram4_int8_convrot.safetensors")
        self.assertIsNone(self.present(quant="int8_convrot"))
        self.assertIsNotNone(self.present())

    def test_role_comes_from_detector_metadata(self):
        expected = self.checkpoint("renamed.safetensors", metadata={"model_type": "ideogram4_uncond"})
        self.assertIsNone(self.present())
        self.assertEqual(self.present("checkpoint:uncond"), expected)

    def test_truncated_checkpoint_skipped_for_valid_alternative(self):
        self.checkpoint("a.safetensors", truncate=1)
        expected = self.checkpoint("b.safetensors")
        self.assertEqual(self.present(), expected)

    def test_wrong_quant_uncond_is_not_accepted(self):
        self.checkpoint("unconditional.safetensors")
        self.assertIsNone(self.present("checkpoint:uncond", "int8_convrot"))

    def test_adapter_truncation_and_small_complete_file(self):
        for asset, name in (("adapter:turbotime", dl.HF_TURBOTIME[1]),
                            ("adapter:uncond", dl.HF_UNCOND[1]),
                            ("adapter:gray", dl.CIVITAI_GRAY[1])):
            with self.subTest(asset=asset):
                path = self.root / "Lora" / name
                write_file(path, ADAPTER, truncate=1)
                self.assertIsNone(self.present(asset))
                write_file(path, ADAPTER)
                self.assertEqual(self.present(asset), path)

    def test_truncated_replacement_cannot_satisfy_uncond(self):
        path = self.root / "Lora" / dl.HF_UNCOND[1]
        write_file(path, ADAPTER, truncate=1)
        self.assertIsNone(self.present("checkpoint:uncond"))
        write_file(path, ADAPTER)
        self.assertEqual(self.present("checkpoint:uncond"), path)

    def test_adapter_name_cannot_disguise_foreign_or_dense_weights(self):
        path = self.root / "Lora" / dl.HF_TURBOTIME[1]
        for tensors in (None,
                        {"double_blocks.0.qkv.lora_A.weight": ("U8", [1, 2]),
                         "double_blocks.0.qkv.lora_B.weight": ("U8", [2, 1])},
                        {"layers.0.attention.qkv.lora_A.weight": ("U8", [1, 2])}):
            with self.subTest(tensors=tensors):
                write_file(path, tensors)
                self.assertIsNone(self.present("adapter:turbotime"))

    def test_adapter_explicit_wrong_role_is_rejected(self):
        write_file(self.root / "Lora" / dl.HF_TURBOTIME[1], ADAPTER,
                   metadata={"i4_adapter_kind": "style"})
        self.assertIsNone(self.present("adapter:turbotime"))

    def test_named_encoder_and_vae_require_correct_architecture(self):
        write_file(self.root / "text_encoder" / "qwen3vl_8b_fp8_scaled.safetensors")
        write_file(self.root / "VAE" / "flux2-vae.safetensors")
        self.assertIsNone(self.present("te"))
        self.assertIsNone(self.present("vae"))

    def test_valid_modules_found_after_truncated_named_candidates(self):
        for asset, folder, filename, tensors in (
            ("te", "text_encoder", "qwen3vl_8b_fp8_scaled.safetensors",
             ENCODER),
            ("vae", "VAE", "flux2-vae.safetensors",
             {"encoder.weight": ("U8", [1]), "decoder.conv_in.weight": ("U8", [1, 32, 1, 1])}),
        ):
            with self.subTest(asset=asset):
                write_file(self.root / folder / filename, tensors, truncate=1)
                expected = write_file(self.root / folder / "renamed.safetensors", tensors)
                self.assertEqual(self.present(asset), expected)

    def encoder_directory(self):
        folder = self.root / "encoder"
        write_file(folder / "model-00001-of-00002.safetensors")
        write_file(folder / "model-00002-of-00002.safetensors")
        (folder / "config.json").write_text(json.dumps({"text_config": {
            "hidden_size": 4096, "num_hidden_layers": 36}}), encoding="utf-8")
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors"}}), encoding="utf-8")
        return folder

    def test_supported_sharded_encoder_override_is_preserved(self):
        folder = self.encoder_directory()
        with patch("pi_ideogram_lib.config.get", return_value=str(folder)):
            self.assertEqual(self.present("te"), folder)

    def test_missing_or_truncated_encoder_shard_is_rejected(self):
        folder = self.encoder_directory()
        shard = folder / "model-00002-of-00002.safetensors"
        shard.unlink()
        with patch("pi_ideogram_lib.config.get", return_value=str(folder)):
            self.assertIsNone(self.present("te"))
            write_file(shard, truncate=1)
            self.assertIsNone(self.present("te"))

    def test_wrong_width_encoder_directory_is_rejected(self):
        folder = self.encoder_directory()
        (folder / "config.json").write_text(json.dumps({"hidden_size": 5120}), encoding="utf-8")
        with patch("pi_ideogram_lib.config.get", return_value=str(folder)):
            self.assertIsNone(self.present("te"))

    def test_contradictory_encoder_directory_metadata_is_rejected(self):
        folder = self.encoder_directory()
        (folder / "config.json").write_text(json.dumps({"model_type": "qwen3",
            "hidden_size": 4096, "num_hidden_layers": 36}), encoding="utf-8")
        with patch("pi_ideogram_lib.config.get", return_value=str(folder)):
            self.assertIsNone(self.present("te"))

    def test_embedding_only_named_encoder_cannot_bypass_detector(self):
        from pi_ideogram_lib.detect import find_text_encoder
        folder = self.root / "text_encoder"
        write_file(folder / "qwen3vl_8b_fp8_scaled.safetensors",
                   {"token_embd.weight": ("U8", [1, 4096])})
        self.assertIsNone(find_text_encoder([folder]))
        self.assertIsNone(self.present("te"))

    def test_metadata_conflict_rejected_by_named_detector_and_downloader(self):
        from pi_ideogram_lib.detect import find_text_encoder
        folder = self.root / "text_encoder"
        write_file(folder / "qwen3vl_8b_fp8_scaled.safetensors", ENCODER,
                   metadata={"model_type": "qwen3"})
        self.assertIsNone(find_text_encoder([folder]))
        self.assertIsNone(self.present("te"))

    def test_explicit_encoder_file_selection_is_preserved(self):
        selected = write_file(self.root / "selected.safetensors",
                              ENCODER)
        with patch("pi_ideogram_lib.detect.forge_selected_modules", return_value=[selected]):
            self.assertEqual(self.present("te"), selected)

    def test_invalid_structures_do_not_pass_completeness(self):
        path = self.root / "bad.safetensors"
        base = dict(dtype="U8", shape=[4], data_offsets=[0, 4])
        for header, payload in (
            ({}, b""), ({"__metadata__": {}}, b""),
            ({"x": {}}, b""), ({"x": dict(base, data_offsets=[-1, 3])}, bytes(4)),
            ({"x": dict(base, data_offsets=[0, 1])}, bytes(1)),
            ({"x": base, "y": base}, bytes(4)),
            ({"x": dict(base, data_offsets=[1, 5])}, bytes(5)),
            ({"x": base}, bytes(5)),
        ):
            with self.subTest(header=header):
                raw = json.dumps(header).encode()
                path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
                self.assertFalse(dl._is_complete_safetensors(path))


if __name__ == "__main__":
    unittest.main()
