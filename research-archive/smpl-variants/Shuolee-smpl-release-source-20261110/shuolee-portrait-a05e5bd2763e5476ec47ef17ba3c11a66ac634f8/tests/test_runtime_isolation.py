"""A Git deployment cannot borrow first-party modules from old experiments."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait import runtime


class RuntimeIsolationTests(unittest.TestCase):
    def test_external_research_path_rejected_before_assets_or_native_modules(self):
        before = sys.path.copy()
        with patch.dict(os.environ, {'SHUOLEE_EXTRA_PYTHONPATH': '/old/research/site-packages'}):
            with self.assertRaisesRegex(RuntimeError, 'Git deployment'):
                runtime.configure('/nonexistent-models')
        self.assertEqual(before, sys.path)


if __name__ == '__main__':
    unittest.main()
