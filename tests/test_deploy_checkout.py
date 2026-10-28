"""Exercise production shell entrypoints against real sparse Git checkouts."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('git'), 'Linux Git deployment required')
class DeployCheckoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='smpl-x checkout ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.app = self.root / 'app'
        self.home = self.root / 'home'
        self.home.mkdir()
        self.env = dict(os.environ, HOME=str(self.home), GIT_CONFIG_NOSYSTEM='1',
                        GIT_CONFIG_GLOBAL='/dev/null', SMPLX_ROOT=str(self.root / 'runtime'),
                        SMPLX_ENV_DIR=str(self.root / 'environment'),
                        SMPLX_ASSETS=str(self.root / 'assets'))
        for name in ('deploy.sh', 'scripts/setup_server.sh', 'constraints-linux-cu128.txt',
                     'requirements.txt', 'run_smplx.py', 'watch.py'):
            destination = self.source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, destination)
        for name in ('configs/fixture', 'src/fixture', 'third_party/fixture',
                     'apps/viewer/fixture'):
            destination = self.source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text('checkout fixture\n')
        self.command('git', 'init', '--quiet', self.source)
        self.command('git', '-C', self.source, 'add', '.')
        self.command('git', '-C', self.source, '-c', 'user.name=Fixture',
                     '-c', 'user.email=fixture@example.invalid', 'commit', '--quiet', '-m', 'fixture')
        self.command('git', 'clone', '--quiet', '--no-checkout', self.source, self.app)
        # Stop at the real setup script's first Python call; no packages/models/GPU
        # are mocked as valid, installed, or loaded by these checkout controls.
        executable = self.root / 'environment/bin/python'
        executable.parent.mkdir(parents=True)
        executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$HOME/python-call"\nexit 42\n')
        executable.chmod(0o755)
        (executable.parent.parent / 'pyvenv.cfg').write_text('checkout boundary fixture\n')

    def command(self, *args, expected=0):
        result = subprocess.run(list(map(str, args)), cwd=self.root, env=self.env,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def assert_deploy_reaches_environment_check(self, sparse=True):
        result = self.command('bash', self.app / 'deploy.sh', expected=42)
        self.assertNotIn('Missing pinned dependency constraints', result.stderr)
        self.assertEqual((self.home / 'python-call').read_text().splitlines(),
                         ['-s', '-m', 'pip', 'check'])
        for name in ('deploy.sh', 'constraints-linux-cu128.txt', 'requirements.txt',
                     'run_smplx.py', 'watch.py', 'scripts/setup_server.sh'):
            self.assertEqual((self.app / name).read_bytes(), (self.source / name).read_bytes())
        self.assertTrue((self.app / 'src/fixture').is_file())
        self.assertTrue((self.app / 'third_party/fixture').is_file())
        if sparse:
            cone = self.command('git', '-C', self.app, 'config', '--get', 'core.sparseCheckoutCone')
            self.assertEqual(cone.stdout.strip(), 'true')
            self.assertFalse((self.app / 'apps/viewer/fixture').exists())

    def test_legacy_non_cone_checkout_keeps_root_entrypoints_and_locks(self):
        self.command('git', '-C', self.app, 'sparse-checkout', 'init', '--no-cone')
        self.assert_deploy_reaches_environment_check()

    def test_fresh_cone_checkout_keeps_root_entrypoints_and_locks(self):
        self.command('git', '-C', self.app, 'sparse-checkout', 'init', '--cone')
        self.assert_deploy_reaches_environment_check()

    def test_documented_recovery_restores_previously_excluded_entrypoint(self):
        self.command('git', '-C', self.app, 'sparse-checkout', 'init', '--no-cone')
        self.command('git', '-C', self.app, 'sparse-checkout', 'set', 'scripts', 'src')
        self.assertFalse((self.app / 'deploy.sh').exists())
        self.assertFalse((self.app / 'constraints-linux-cu128.txt').exists())
        self.command('git', '-C', self.app, 'sparse-checkout', 'init', '--cone')
        self.command('git', '-C', self.app, 'pull', '--ff-only')
        self.assert_deploy_reaches_environment_check()

    def test_full_checkout_remains_full(self):
        self.command('git', '-C', self.app, 'checkout', '--quiet')
        self.assert_deploy_reaches_environment_check(sparse=False)
        self.assertTrue((self.app / 'apps/viewer/fixture').is_file())


if __name__ == '__main__':
    unittest.main()
