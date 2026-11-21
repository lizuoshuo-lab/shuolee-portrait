"""Actual host-header regression plus bounded installer-patch safety checks in /tmp."""
from pathlib import Path
import hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile, zipfile

ROOT=Path(__file__).resolve().parent
PACKAGE=Path('/mnt/d/00/工作/SMPL-X/server113_pip_repair.pyz')
CACHE=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-transfer-cache/pip-bootstrap')
work=Path(tempfile.mkdtemp(prefix='shuolee-header-layout-test-'))
spec=importlib.util.spec_from_file_location('headers',ROOT/'server113_cuda_build_env.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
real=Path('/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8')
split=work/'split-toolkit';(split/'bin').mkdir(parents=True);(split/'targets/x86_64-linux').mkdir(parents=True)
(split/'bin/nvcc').symlink_to(real/'bin/nvcc')
(split/'targets/x86_64-linux/include').symlink_to(real/'include',target_is_directory=True)
(split/'targets/x86_64-linux/lib').symlink_to(real/'lib64',target_is_directory=True)
runtime=Path(sys.prefix)/'lib/python3.10/site-packages/nvidia/cuda_runtime/include'
assert (runtime/'cuda_runtime.h').exists() and not (runtime/'crt/host_config.h').exists()
source=work/'repro.cpp';source.write_text('#include <cuda_runtime.h>\nint x = CUDART_VERSION;\n')
env=os.environ.copy()
for key in ['C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','OBJC_INCLUDE_PATH']:env.pop(key,None)
env['CPATH']=str(runtime)
before=subprocess.run(['g++','-std=c++17','-c',str(source),'-o',str(work/'before.o')],env=env,text=True,capture_output=True)
assert before.returncode!=0 and 'crt/host_config.h' in before.stderr,before.stderr
exports,headers=h.resolve(split,runtime.parent.parent)
after=h.host_probe(exports,work)
assert after['returncode']==0
(work/'before-error.log').write_text(before.stderr)
# Unknown layouts must fail before dependency compilation rather than falsely pass.
empty=work/'missing';(empty/'bin').mkdir(parents=True);(empty/'bin/nvcc').touch()
try:h.resolve(empty,work/'no-wheels');raise AssertionError('missing headers accepted')
except RuntimeError as exc:assert 'header missing' in str(exc)
with zipfile.ZipFile(PACKAGE) as z:
    installer=work/'installer.py';installer.write_bytes(z.read('__main__.py'))
    manifest=json.loads(z.read('install-patch.json'))
    for item in manifest['files']:
        if item['path'].endswith('.sh'):
            shell=subprocess.run(['bash','-n'],input=z.read('install-patch/'+item['path']),capture_output=True)
            assert shell.returncode==0,shell.stderr
spec=importlib.util.spec_from_file_location('installer',installer);i=importlib.util.module_from_spec(spec);spec.loader.exec_module(i)
project=work/'project';(project/'scripts').mkdir(parents=True)
frozen=Path('/mnt/d/00/工作/人体重建')
for name in ['setup_server.sh','setup_113.sh']:shutil.copyfile(frozen/'scripts'/name,project/'scripts'/name)
logs=work/'logs';logs.mkdir();sys.argv[0]=str(PACKAGE)
i.apply_install_patch(project,logs)
first={x['path']:hashlib.sha256((project/x['path']).read_bytes()).hexdigest() for x in manifest['files']}
assert all(first[x['path']]==x['after_sha256'] for x in manifest['files'])
i.apply_install_patch(project,logs) # safe idempotent call
assert json.loads((logs/'cuda-header-installer-patch.json').read_text())['changed']==[]
unknown=project/'scripts/setup_server.sh';unknown.write_text('# user change\n')
try:i.apply_install_patch(project,logs);raise AssertionError('unknown edit overwritten')
except RuntimeError as exc:assert 'Unknown local edit' in str(exc)
assert unknown.read_text()=='# user change\n'
report=dict(passed=True,actual_cpp_before_failed_missing_crt=True,actual_cpp_after_passed=True,
    tested_layout='split toolkit with no top-level include; existing pip cuda_runtime headers',
    header_sources=headers,host_probe=after,fixture=str(work),
    checks=['exact_missing_header_reproduced','host_CPP_development_headers_compiled','missing_headers_fail_closed',
            'bash_syntax','frozen_patch_hashes','idempotence','unknown_edits_preserved'],
    full_nvdiffrast_wheel_built=False,company_server_verified=False,training_started=False)
(CACHE/'header-patch-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
