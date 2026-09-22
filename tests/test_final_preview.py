# requires: cpu
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_ideogram_forge.preview import publish_final, publish_start


class FinalPreviewTests(unittest.TestCase):
    def test_gradient_canvas_then_exact_final(self):
        state = SimpleNamespace(current_image=None, sampling_step=9, id_live_preview=2)
        publish_start(state, 1024, 512)
        self.assertEqual(state.current_image.size, (256, 128))
        self.assertNotEqual(state.current_image.getpixel((0, 0)), state.current_image.getpixel((0, 127)))
        self.assertEqual((state.sampling_step, state.id_live_preview), (0, 3))
        final = object()
        publish_final(state, final, 12)
        self.assertIs(state.current_image, final)

    def test_final_replaces_thumbnail_without_copy_or_decode(self):
        state = SimpleNamespace(current_image=object(), sampling_step=11, id_live_preview=7)
        final = object()
        publish_final(state, final, 12)
        self.assertIs(state.current_image, final)
        self.assertEqual((state.sampling_step, state.id_live_preview), (12, 8))

    def test_zero_step_and_following_batch_image(self):
        state = SimpleNamespace(current_image=None, sampling_step=0, id_live_preview=0)
        for steps in (0, 6, 12):
            final = object()
            publish_final(state, final, steps)
            self.assertIs(state.current_image, final)
            self.assertEqual(state.sampling_step, steps)
        self.assertEqual(state.id_live_preview, 3)


if __name__ == '__main__':
    unittest.main()
