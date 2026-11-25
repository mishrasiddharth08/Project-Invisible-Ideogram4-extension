import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi_ideogram_lib.detect import _looks_like_qwen3vl_te


def encoder(converted=False, depth=36, prefix="model", width=4096):
    def tensor(shape):
        return {"dtype": "BF16", "shape": shape}
    if converted:
        header = {"token_embd.weight": tensor([1, width])}
        for layer in range(depth):
            for module in ("attn_q", "ffn_down"):
                header[f"blk.{layer}.{module}.weight"] = tensor([1, 1])
    else:
        header = {f"{prefix}.embed_tokens.weight": tensor([1, width]),
                  "model.visual.deepstack_merger_list.0.linear_fc1.weight": tensor([1, 1])}
        for layer in range(depth):
            for module in ("self_attn.q_proj", "mlp.down_proj"):
                header[f"{prefix}.layers.{layer}.{module}.weight"] = tensor([1, 1])
    return header


class EncoderIdentityTests(unittest.TestCase):
    def test_verified_layouts_with_full_depth(self):
        for kwargs in ({"converted": True}, {}, {"prefix": "language_model"},
                       {"prefix": "model.language_model"}):
            with self.subTest(kwargs=kwargs):
                self.assertTrue(_looks_like_qwen3vl_te(encoder(**kwargs)))

    def test_embedding_only_and_shallow_encoders_rejected(self):
        for converted in (False, True):
            for depth in (0, 1, 32, 35):
                with self.subTest(converted=converted, depth=depth):
                    self.assertFalse(_looks_like_qwen3vl_te(encoder(converted, depth)))

    def test_sparse_layer_indices_do_not_prove_depth(self):
        for converted in (False, True):
            header = encoder(converted)
            prefix = "blk.17." if converted else "model.layers.17."
            header = {k: v for k, v in header.items() if not k.startswith(prefix)}
            self.assertFalse(_looks_like_qwen3vl_te(header))

    def test_norms_and_descriptors_are_not_language_weights(self):
        header = encoder(True, 0)
        for layer in range(36):
            header[f"blk.{layer}.attn_norm.weight"] = {"shape": [1]}
            header[f"blk.{layer}.ffn_down.comfy_quant"] = {"shape": [1]}
        self.assertFalse(_looks_like_qwen3vl_te(header))

    def test_contradictory_metadata_rejected_even_at_correct_width_and_depth(self):
        for metadata in ({"model_type": "qwen3"}, {"general.architecture": "llama"},
                         {"modelspec.architecture": "qwen2vl"},
                         {"architectures": '["Qwen3ForCausalLM"]'},
                         {"config": '{"model_type": "qwen3"}'},
                         {"model_config": {"text_config": {"model_type": "qwen2"}}}):
            for converted in (False, True):
                with self.subTest(metadata=metadata, converted=converted):
                    header = encoder(converted)
                    header["__metadata__"] = metadata
                    self.assertFalse(_looks_like_qwen3vl_te(header))

    def test_valid_or_uninformative_metadata_preserves_converted_layout(self):
        for metadata in ({}, {"model_type": "qwen3_vl"},
                         {"architectures": '["Qwen3VLForConditionalGeneration"]'},
                         {"config": '{"text_config": {"model_type": "qwen3_vl_text"}}'},
                         {"modelspec.architecture": "ReaderField(offset=846, name='general.file_type')"}):
            with self.subTest(metadata=metadata):
                header = encoder(True)
                header["__metadata__"] = metadata
                self.assertTrue(_looks_like_qwen3vl_te(header))

    def test_wrong_width_or_missing_vision_rejected(self):
        self.assertFalse(_looks_like_qwen3vl_te(encoder(width=5120)))
        header = encoder()
        header = {k: v for k, v in header.items() if not k.startswith("model.visual.")}
        self.assertFalse(_looks_like_qwen3vl_te(header))


if __name__ == "__main__":
    unittest.main()
