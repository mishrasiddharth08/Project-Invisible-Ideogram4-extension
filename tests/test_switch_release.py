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
        lifecycle._wait_for_pending_release()
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
        self.assertTrue(lifecycle._wait_for_pending_release())
        pipe.unload.assert_called_once()

    def test_latest_busy_switch_wins_without_leaking_worker(self):
        pipe = Mock()
        cache = {'a': pipe}

        def switch_twice():
            lifecycle.release_changed(cache, 'Intermediate')
            lifecycle.release_changed(cache, 'Final')

        with lifecycle.lock:
            worker = threading.Thread(target=switch_twice)
            worker.start()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            pipe.unload.assert_not_called()
        self.assertTrue(lifecycle._wait_for_pending_release())
        pipe.unload.assert_called_once()
        self.assertEqual(lifecycle.selection, 'Final')

    def test_one_failed_unload_does_not_block_other_cleanup(self):
        bad = Mock()
        bad.unload.side_effect = RuntimeError('broken cleanup')
        good = Mock()
        cache = {'bad': bad, 'good': good}
        lifecycle.release_changed(cache, 'Other')
        bad.unload.assert_called_once()
        good.unload.assert_called_once()
        self.assertFalse(cache)
        self.assertEqual(lifecycle.selection, 'Other')

    def test_switch_back_while_busy_cancels_stale_release(self):
        pipe = Mock()
        cache = {'a': pipe}

        def switch_away_and_back():
            lifecycle.release_changed(cache, 'Other')
            lifecycle.release_changed(cache, 'Ideogram')

        with lifecycle.lock:
            worker = threading.Thread(target=switch_away_and_back)
            worker.start()
            worker.join(timeout=2)
        self.assertTrue(lifecycle._wait_for_pending_release())
        pipe.unload.assert_not_called()
        self.assertTrue(cache)
        self.assertEqual(lifecycle.selection, 'Ideogram')

    def test_repeated_busy_callbacks_share_one_cleanup_worker(self):
        pipe = Mock()
        cache = {'a': pipe}
        deferred_workers = []

        with lifecycle.lock:
            for selected in ('One', 'Two', 'Final'):
                caller = threading.Thread(
                    target=lifecycle.release_changed, args=(cache, selected))
                caller.start()
                caller.join(timeout=2)
                self.assertFalse(caller.is_alive())
                deferred_workers.append(lifecycle._release_worker)
            self.assertIsNotNone(deferred_workers[0])
            self.assertTrue(all(worker is deferred_workers[0]
                                for worker in deferred_workers))
            pipe.unload.assert_not_called()

        self.assertTrue(lifecycle._wait_for_pending_release())
        pipe.unload.assert_called_once()
        self.assertEqual(lifecycle.selection, 'Final')


if __name__ == '__main__':
    unittest.main()
