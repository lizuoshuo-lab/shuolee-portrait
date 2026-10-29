"""Shell routing checks with isolated helper stubs; no deployment or GPU work."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('bash'), 'Linux shell routing test')
class UnifiedEntryScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='unified shell routing ')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.app = self.base / 'app with spaces'
        (self.app / 'scripts').mkdir(parents=True)
        for name in ('deploy.sh', 'scripts/train.sh'):
            shutil.copyfile(PROJECT / name, self.app / name)
        self.home = self.base / 'isolated home'
        self.home.mkdir()
        self.runtime = self.base / 'runtime with spaces'
        self.environment = self.base / 'environment with spaces'
        self.capture = self.base / 'python-call.json'
        self.setup_capture = self.base / 'setup-call.json'
        self.assets = self.base / 'external assets'
        self.assets.mkdir()
        self.make_environment(self.environment)
        self.bin = self.base / 'stub-bin'
        self.bin.mkdir()
        # Only the sparse-checkout query is needed. This fake cannot clone or
        # mutate a checkout, and the test never contacts any remote repository.
        self.executable(self.bin / 'git', '#!/bin/sh\nprintf "false\\n"\n')
        self.env = dict(os.environ, HOME=str(self.home), SMPLX_ROOT=str(self.runtime),
            SMPLX_ENV_DIR=str(self.environment), SMPLX_ASSETS=str(self.assets),
            SHELL_PYTHON_CAPTURE=str(self.capture), SHELL_SETUP_CAPTURE=str(self.setup_capture),
            PATH=str(self.bin) + os.pathsep + os.environ['PATH'])
        for key in ('SHUOLEE_ROOT', 'SHUOLEE_ENV_DIR', 'SMPLX_SERVICE_NAME', 'SHUOLEE_SERVICE_NAME'):
            self.env.pop(key, None)
        (self.app / 'scripts/prepare_assets.py').write_text(
            'import argparse\n'
            'p=argparse.ArgumentParser();p.add_argument("--root");p.add_argument("--environment");p.add_argument("--assets")\n'
            'a=p.parse_args();print(a.assets)\n', encoding='utf-8')
        self.setup_stub = '''#!/usr/bin/env bash
set -euo pipefail
python3 - "$@" <<'PY'
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
Path(os.environ['SHELL_SETUP_CAPTURE']).write_text(json.dumps(args))
profile=Path(args[args.index('--deployment')+1]);profile.parent.mkdir(parents=True,exist_ok=True)
profile.write_text(json.dumps(dict(environment=os.environ['SMPLX_ENV_DIR'],assets=args[args.index('--assets')+1],runs=args[args.index('--runs')+1])))
PY
'''
        (self.app / 'scripts/setup_server.sh').write_text(self.setup_stub, encoding='utf-8')

    @staticmethod
    def executable(path, source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
        path.chmod(0o755)

    def make_environment(self, directory):
        self.executable(directory / 'bin/python', '''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
Path(os.environ['SHELL_PYTHON_CAPTURE']).write_text(json.dumps(dict(executable=sys.argv[0],args=sys.argv[1:],pythonpath=os.environ.get('PYTHONPATH'),extra_pythonpath=os.environ.get('SHUOLEE_EXTRA_PYTHONPATH'),no_user_site=os.environ.get('PYTHONNOUSERSITE'))))
''')
        (directory / 'pyvenv.cfg').write_text('isolated routing fixture\n', encoding='utf-8')

    def command(self, *args, env=None, expected=0):
        result = subprocess.run(list(map(str, args)), cwd=self.base, env=env or self.env,
            capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def profile(self, path=None, environment=None):
        path = path or self.runtime / 'queue/deployment.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(environment=str(environment or self.environment),
            assets=str(self.assets), runs=str(self.runtime / 'runs'))), encoding='utf-8')
        return path

    def test_deploy_only_configures_then_prints_unified_train_command(self):
        result = self.command('bash', self.app / 'deploy.sh')
        self.assertIn('Deployment configuration ready', result.stdout)
        self.assertIn('run_smplx.py train', result.stdout)
        self.assertIn('--camera-config', result.stdout)
        self.assertIn('--reference-frame', result.stdout)
        self.assertFalse(self.capture.exists(), 'Deploy must not invoke the old service/watcher entry.')
        setup = json.loads(self.setup_capture.read_text())
        self.assertEqual(setup[0], '--verify-existing')
        deployment = self.runtime / 'queue/deployment.json'
        profile = json.loads(deployment.read_text())
        self.assertEqual(profile['environment'], str(self.environment))
        self.assertEqual(profile['assets'], str(self.assets))
        self.assertEqual(profile['runs'], str(self.runtime / 'runs'))

    def test_setup_failure_is_not_reported_as_ready(self):
        (self.app / 'scripts/setup_server.sh').write_text('#!/bin/sh\nexit 37\n', encoding='utf-8')
        result = self.command('bash', self.app / 'deploy.sh', expected=37)
        self.assertNotIn('Deployment configuration ready', result.stdout)
        self.assertFalse(self.capture.exists())

    def test_train_forwards_all_arguments_and_uses_profile_environment(self):
        deployment = self.profile()
        env = dict(self.env, PYTHONPATH='unrelated research', SHUOLEE_EXTRA_PYTHONPATH='unrelated code')
        env.pop('SMPLX_ENV_DIR')
        supplied = ['video with spaces.mp4', '--camera-config', 'camera config.json',
            '--reference-frame', 'frame_000031', '--output', 'new run with spaces']
        self.command('bash', self.app / 'scripts/train.sh', *supplied, env=env)
        call = json.loads(self.capture.read_text())
        self.assertEqual(call['executable'], str(self.environment / 'bin/python'))
        self.assertEqual(call['args'][:5], ['-s', str(self.app / 'run_smplx.py'), 'train', '--deployment', str(deployment)])
        self.assertEqual(call['args'][5:], supplied)
        self.assertIsNone(call['pythonpath'])
        self.assertIsNone(call['extra_pythonpath'])
        self.assertEqual(call['no_user_site'], '1')

    def test_explicit_deployment_selects_its_matching_environment(self):
        alternate = self.base / 'another environment'
        self.make_environment(alternate)
        deployment = self.profile(self.base / 'another profile.json', alternate)
        env = dict(self.env)
        env.pop('SMPLX_ENV_DIR')
        supplied = ['clip.mp4', '--camera-config', 'camera.json', '--deployment', str(deployment)]
        self.command('bash', self.app / 'scripts/train.sh', *supplied, env=env)
        call = json.loads(self.capture.read_text())
        self.assertEqual(call['executable'], str(alternate / 'bin/python'))
        self.assertEqual(call['args'][5:], supplied)

    def test_help_and_missing_arguments_do_not_touch_models_or_environment(self):
        self.command('bash', self.app / 'scripts/train.sh', '--help')
        self.command('bash', self.app / 'scripts/train.sh', expected=2)
        self.command('bash', self.app / 'deploy.sh', '--help')
        self.assertFalse(self.capture.exists())
        self.assertFalse(self.setup_capture.exists())

    def test_shell_syntax(self):
        self.command('bash', '-n', self.app / 'deploy.sh')
        self.command('bash', '-n', self.app / 'scripts/train.sh')


if __name__ == '__main__':
    unittest.main()
