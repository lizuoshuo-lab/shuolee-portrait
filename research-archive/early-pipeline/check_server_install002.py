"""Verify the finished disposable install; never installs or resumes a run."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

ROOT = '/opt/gs-slam-lab/shuolee-portrait-clean-002'
SOURCE = '/mnt/d/GS-SLAM-Data/shuolee-server-install-check-002/source'
WIN = Path('//wsl.localhost/Ubuntu-20.04') / ROOT.lstrip('/')
INSTALL = WIN / 'env/install'
assert json.loads((WIN / 'installer-exit.json').read_text(encoding='utf-8-sig'))['exit_code'] == 0
assert json.loads((INSTALL / 'environment-check.json').read_text())['passed']

def run(name, args):
    prefix = ['wsl', '-d', 'Ubuntu-20.04', '--', 'env', '-u', 'PYTHONPATH',
              '-u', 'SHUOLEE_EXTRA_PYTHONPATH', 'PYTHONNOUSERSITE=1',
              'CUDA_HOME=/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8',
              'TORCH_EXTENSIONS_DIR=' + ROOT + '/env/install/torch-extensions',
              ROOT + '/env/bin/python', '-s']
    result = subprocess.run(prefix + args, capture_output=True)
    (INSTALL / (name + '.stderr.log')).write_bytes(result.stderr)
    (INSTALL / (name + '.stdout.log')).write_bytes(result.stdout)
    assert result.returncode == 0, name + ' failed; see retained logs'
    return result.stdout.decode('utf-8')

doctor = json.loads(run('doctor', [SOURCE + '/run.py', 'doctor', '--assets',
                                   '/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets']))
assert doctor['assets']['sha256_verified'] and doctor['prior_configuration_composed']
(INSTALL / 'doctor.json').write_text(json.dumps(doctor, indent=2)+'\n')
print('Doctor: all assets and full prior configuration passed', flush=True)
run('projection-export', [SOURCE + '/scripts/check_projection_export.py',
                         '--output', ROOT + '/env/install/projection-export.json'])
assert json.loads((INSTALL / 'projection-export.json').read_text())['passed']
print('Projection and PLY numeric contract passed', flush=True)
code = ('import importlib.metadata as m,json,re; '
        'print(json.dumps({re.sub(r"[-_.]+","-",d.metadata["Name"]).lower():d.version '
        'for d in m.distributions()},sort_keys=True,indent=2))')
versions = json.loads(run('package-versions', ['-c', code]))
(INSTALL / 'package-versions.json').write_text(json.dumps(versions, indent=2)+'\n')
print(json.dumps({'completed':datetime.now(timezone.utc).isoformat(),
                  'package_count':len(versions), 'no_training_started':True}))
