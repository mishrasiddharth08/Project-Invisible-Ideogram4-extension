"""CPU behavior checks for scoped cleanup and Forge-native preview publishing."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi_ideogram_forge.preview import publish_final, publish_frame, publish_start


class HostState:
    def __init__(self):
        self.current_image = None
        self.id_live_preview = 0
        self.sampling_step = 99
        self.steps_seen_by_host = []

    def assign_current_image(self, image):
        self.steps_seen_by_host.append(self.sampling_step)
        self.current_image = image
        self.id_live_preview += 1


class BlueprintContractTests(unittest.TestCase):
    def test_preview_uses_host_assignment_and_publishes_coherent_step(self):
        state = HostState()
        publish_start(state, 64, 32)
        final = object()
        publish_final(state, final, 7)
        self.assertEqual(state.steps_seen_by_host, [0, 7])
        self.assertEqual(state.id_live_preview, 2)
        self.assertIs(state.current_image, final)

    def test_preview_fallback_supports_minimal_compatible_state(self):
        class MinimalState:
            sampling_step = 3

        state = MinimalState()
        final = object()
        publish_final(state, final, 0)
        self.assertIs(state.current_image, final)
        self.assertEqual((state.sampling_step, state.id_live_preview), (0, 1))

    def test_intermediate_frame_uses_host_without_changing_progress(self):
        state = HostState()
        state.sampling_step = 4
        frame = object()
        publish_frame(state, frame)
        self.assertEqual(state.steps_seen_by_host, [4])
        self.assertEqual(state.sampling_step, 4)
        self.assertIs(state.current_image, frame)


if __name__ == '__main__':
    unittest.main()
