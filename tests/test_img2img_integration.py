"""Fast checks for the Forge-side experimental img2img bridge."""

from pathlib import Path
from types import SimpleNamespace
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_ideogram_forge.run import _img2img_generation_kwargs


def _p(**changes):
    values = dict(init_images=[object()], is_img2img=True, image_mask=None,
                  mask=None, mask_for_overlay=None, latent_mask=None,
                  resize_mode=0, denoising_strength=0.75)
    values.update(changes)
    return SimpleNamespace(**values)


class Img2ImgIntegrationTests(unittest.TestCase):
    def test_strength_zero_is_preserved(self):
        self.assertEqual(
            _img2img_generation_kwargs(_p(denoising_strength=0.0))["denoising_strength"],
            0.0,
        )

    def test_invalid_strength_is_rejected(self):
        for strength in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(strength=strength), self.assertRaisesRegex(ValueError, "between 0 and 1"):
                _img2img_generation_kwargs(_p(denoising_strength=strength))

    def test_masks_and_resize_mode_three_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "masks or inpainting"):
            _img2img_generation_kwargs(_p(image_mask=object()))
        with self.assertRaisesRegex(ValueError, "resize mode 3"):
            _img2img_generation_kwargs(_p(resize_mode=3))

    def test_generate_uses_cyclic_sources_and_sampled_step_count(self):
        source = (Path(__file__).parents[1] / "pi_ideogram_forge" / "generate.py").read_text()
        self.assertIn("init_images[output_index % len(init_images)]", source)
        self.assertIn("forge_images.resize_image(", source)
        self.assertIn("used_steps = image_actual_steps[0]", source)
        self.assertIn('f"Steps: {used_steps}', source)


if __name__ == "__main__":
    unittest.main()
