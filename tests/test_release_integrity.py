# requires: cpu
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_integrity import verify


class IntegrityTests(unittest.TestCase):
    def test_changes_missing_and_added(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / 'install.py'
            code.write_text('pass\n')
            (root / 'release-manifest.json').write_text(json.dumps({'files': {
                'install.py': hashlib.sha256(code.read_bytes()).hexdigest()}}))
            self.assertEqual(verify(root), [])
            (root / 'config.json').write_text('{}')
            self.assertEqual(verify(root), [])
            code.write_text('changed\n')
            self.assertIn('Changed: install.py', verify(root))
            code.unlink()
            self.assertIn('Missing: install.py', verify(root))
            (root / 'extra.py').write_text('pass\n')
            self.assertIn('Unexpected source: extra.py', verify(root))


if __name__ == '__main__':
    unittest.main()
