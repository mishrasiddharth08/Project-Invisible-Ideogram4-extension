# requires: cpu
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_ideogram_forge import lifecycle


class SwitchReleaseTests(unittest.TestCase):
    def setUp(self):
        lifecycle.selection = 'Ideogram'

    def test_same_selection_keeps_cache(self):
        pipe = Mock()
        cache = {'a': pipe}
        lifecycle.release_changed(cache, 'Ideogram')
        pipe.unload.assert_not_called()
        self.assertTrue(cache)

    def test_switch_unloads_duplicate_aliases_once(self):
        pipe = Mock()
        cache = {'a': pipe, 'b': pipe}
        lifecycle.release_changed(cache, 'Other model')
        pipe.unload.assert_called_once()
        self.assertFalse(cache)

    def test_active_generation_defers_until_safe(self):
        pipe = Mock()
        cache = {'a': pipe}
        with lifecycle.lock:
            worker = threading.Thread(target=lifecycle.release_changed, args=(cache, 'Other'))
            worker.start()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            pipe.unload.assert_not_called()
        lifecycle.release_changed(cache, 'Other')
        pipe.unload.assert_called_once()


if __name__ == '__main__':
    unittest.main()
