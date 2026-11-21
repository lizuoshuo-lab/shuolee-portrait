"""Run on WSL; compile one real CUDA unit and check failure handling in isolation."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import fcntl
import hashlib
import re
import shutil

WORK = Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update')
HELPER = WORK / 'server_cuda128.sh'
REAL_CUDA = Path('/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8')
SOURCE = Path('/mnt/d/00/工作/人体重建')
PYTHON = '/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python'
results = []


def environment(home):
    env = dict(os.environ, HOME=str(home), PATH='/usr/bin:/bin', CUDA_HOME='/missing/cuda-12.8')
    for name in ('CUDA_PATH', 'CONDA_PREFIX', 'CONDA_EXE', 'CPATH', 'CPLUS_INCLUDE_PATH',
                 'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'SHUOLEE_EXTRA_PYTHONPATH'):
        env.pop(name, None)
    return env


def run_case(name, home, args, expected, marker, env=None):
    proc = subprocess.run(['bash', str(HELPER), *args], env=env or environment(home),
                          text=True, capture_output=True, timeout=90)
    output = proc.stdout + proc.stderr
    passed = proc.returncode == expected and marker in output
    results.append(dict(name=name, exit_code=proc.returncode, passed=passed, output=output))
    assert passed, results[-1]
    return proc


def mock_toolkit(path, version='12.8', compile_exit=0):
    (path / 'bin').mkdir(parents=True)
    (path / 'include').mkdir()
    for name in ('cuda_runtime.h', 'cuda.h'):
        (path / 'include' / name).write_text('// fixture only\n')
    compiler = path / 'bin/nvcc'
    compiler.write_text(f'''#!/bin/bash
if [[ "$1" == --version ]]; then echo 'Cuda compilation tools, release {version}, V{version}.93'; exit 0; fi
if [[ {compile_exit} != 0 ]]; then echo 'mock compile failure'; exit {compile_exit}; fi
while [[ $# -gt 0 ]]; do
  if [[ "$1" == -o ]]; then printf 'fixture' > "$2"; exit 0; fi
  shift
done
exit 99
''')
    compiler.chmod(0o755)


with tempfile.TemporaryDirectory(prefix='shuolee-cuda128-verification-') as tmp:
    root = Path(tmp)
    subprocess.run(['bash', '-n', str(HELPER)], check=True)
    real = root / 'real'
    real.mkdir()
    run_case('actual_cuda_12_8_compile', real, ['--toolkit', str(REAL_CUDA)], 0, 'CUDA_12_8_READY=')
    env_file = real / 'shuolee/cuda128.env'
    env = environment(real)
    env['PYTHON_BIN'] = PYTHON
    env['SHUOLEE_ENV_DIR'] = str(real / 'python-env-not-created')
    proc = subprocess.run(['bash', '-c',
        'set -e; source "$1"; cd "$2"; bash scripts/setup_server.sh --preflight-only',
        'check', str(env_file), str(SOURCE)], env=env, capture_output=True, text=True, timeout=60)
    passed = proc.returncode == 0 and not Path(env['SHUOLEE_ENV_DIR']).exists()
    results.append(dict(name='frozen_113_actual_preflight', exit_code=proc.returncode,
                        passed=passed, output=proc.stdout + proc.stderr))
    assert passed, results[-1]

    missing = root / 'missing'
    missing.mkdir()
    run_case('missing_compiler_blocks_env_write', missing, ['--toolkit', str(missing / 'no-toolkit')],
             3, 'CUDA_12_8_NOT_FOUND')
    assert not (missing / 'shuolee/cuda128.env').exists()

    wrong = root / 'wrong'
    wrong.mkdir()
    wrong_toolkit = wrong / 'toolkit'
    mock_toolkit(wrong_toolkit, version='12.4')
    run_case('wrong_version_rejected', wrong, ['--toolkit', str(wrong_toolkit)], 3, 'not 12.8')
    assert not (wrong / 'shuolee/cuda128.env').exists()

    fallback = root / 'fallback'
    fallback.mkdir()
    good_toolkit = fallback / 'miniconda3/envs/toolkit with spaces'
    mock_toolkit(good_toolkit)
    env = environment(fallback)
    env['CUDA_HOME'] = str(wrong_toolkit)
    run_case('stale_cuda_home_falls_back_and_quotes_spaces', fallback, [], 0, 'CUDA_12_8_READY=', env)
    proc = subprocess.run(['bash', '-c', 'source "$1"; printf "%s" "$CUDA_HOME"',
                           'check', str(fallback / 'shuolee/cuda128.env')],
                           capture_output=True, text=True, env=env, check=True)
    assert proc.stdout == str(good_toolkit)

    failed = root / 'failed'
    failed.mkdir()
    failed_toolkit = failed / 'toolkit'
    mock_toolkit(failed_toolkit, compile_exit=11)
    run_case('compile_failure_not_marked_ready', failed, ['--toolkit', str(failed_toolkit)],
             7, 'CUDA_12_8_COMPILE_FAILED')
    assert not (failed / 'shuolee/cuda128.env').exists()

    no_conda = root / 'no-conda'
    no_conda.mkdir()
    run_case('no_conda_install_stops_without_writes', no_conda, ['--install'], 4, 'CONDA_NOT_FOUND')
    assert not (no_conda / '.local/share/shuolee-portrait').exists()

    locked = root / 'locked'
    (locked / 'shuolee/logs').mkdir(parents=True)
    old_log = locked / 'shuolee/logs/cuda128-check.log'
    old_log.write_text('existing installation log\n')
    with (locked / 'shuolee/cuda128.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_case('duplicate_helper_blocked_without_touching_log', locked, [], 8, 'CUDA_INSTALL_IN_PROGRESS')
    assert old_log.read_text() == 'existing installation log\n'

    legacy = root / 'legacy-active'
    (legacy / 'shuolee/logs').mkdir(parents=True)
    legacy_log = legacy / 'shuolee/logs/cuda128-check.log'
    legacy_log.write_text('legacy install is collecting metadata\n')
    prefix = legacy / '.local/share/shuolee-portrait/cuda-toolkit-12.8'
    process = subprocess.Popen([PYTHON, '-c', 'import time; print("ready", flush=True); time.sleep(30)',
                                '/fixture/bin/conda', 'create', '--prefix', str(prefix), 'cuda-toolkit=12.8.1'],
                                stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        run_case('legacy_conda_install_detected_without_touching_log', legacy, [], 8, 'CUDA_INSTALL_IN_PROGRESS')
        assert legacy_log.read_text() == 'legacy install is collecting metadata\n'
        assert not (legacy / 'shuolee/cuda128.env').exists()
    finally:
        process.terminate()
        process.wait(timeout=5)

    recovery = root / 'env-recovery'
    recovery.mkdir()
    recovered_toolkit = recovery / 'miniconda3/envs/cuda128'
    mock_toolkit(recovered_toolkit)
    (recovery / 'shuolee/logs').mkdir(parents=True)
    recovery_log = recovery / 'shuolee/logs/cuda128-check.log'
    recovery_log.write_text('retain earlier installer evidence\n')
    run_case('installed_toolkit_regenerates_missing_env_preserving_log', recovery, [], 0, 'CUDA_12_8_READY')
    assert (recovery / 'shuolee/cuda128.env').is_file()
    assert recovery_log.read_text().startswith('retain earlier installer evidence\n')

    guide = Path('/mnt/d/00/工作/SMPL-X/00_部署与训练指南.md').read_text()
    step_c = guide.split('**C．重新核验 CUDA', 1)[1].split('## 5.', 1)[0]
    command = re.search(r'```bash\n(.*?)\n```', step_c, re.S).group(1)
    for label, test_home, expected_exit in [('missing_env_recovers_then_setup', recovery, 0),
                                            ('missing_cuda_stops_before_setup', missing, 3)]:
        project = test_home / 'shuolee/portrait113/scripts'
        project.mkdir(parents=True)
        for script_name in ('setup_server.sh', 'setup_113.sh'):
            (project / script_name).write_text('#!/bin/bash\nprintf "%s\\n" ' + script_name + ' >> "$HOME/setup-trace"\n')
        copied = test_home / 'shuolee/server_cuda128.sh'
        shutil.copyfile(HELPER, copied)
        copied.with_suffix('.sh.sha256').write_text(hashlib.sha256(copied.read_bytes()).hexdigest() + '  server_cuda128.sh\n')
        generated_env = test_home / 'shuolee/cuda128.env'
        if generated_env.exists():
            generated_env.unlink()  # temporary test fixture only
        proc = subprocess.run(['bash', '-c', command], env=environment(test_home),
                              capture_output=True, text=True, timeout=90)
        trace = test_home / 'setup-trace'
        if expected_exit == 0:
            passed = proc.returncode == 0 and generated_env.is_file() and trace.read_text().splitlines() == ['setup_server.sh', 'setup_113.sh']
        else:
            passed = proc.returncode == expected_exit and not trace.exists() and not generated_env.exists()
        results.append(dict(name='guide_step_c_' + label, exit_code=proc.returncode, passed=passed,
                            output=proc.stdout + proc.stderr))
        assert passed, results[-1]

report = dict(date='2026-10-26', passed=all(r['passed'] for r in results),
              local_wsl_only=True, company_server_tested=False,
              toolkit_download_or_install_executed=False,
              description='Actual CUDA compile and frozen setup preflight; remaining cases use mock compilers.',
              cases=results)
(WORK / 'cuda128-helper-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, ensure_ascii=False))
print(json.dumps([dict(name=r['name'], passed=r['passed']) for r in results]))
