"""Validate the actual unit syntax using the host systemd parser on Linux."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('systemd-analyze'), 'Linux systemd parser required')
class ServiceTests(unittest.TestCase):
    def test_generated_unit_passes_the_systemd_parser(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/install_service119.py'
        spec = importlib.util.spec_from_file_location('install_service119', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            unit = Path(directory) / 'shuolee-unit-test.service'
            unit.write_text(module.service_text(Path(directory) / 'deployment with spaces.json'))
            result = subprocess.run(['systemd-analyze', 'verify', str(unit)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
