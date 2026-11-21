"""Test diagnostic classification without changing HOME or installing packages."""
from pathlib import Path
import fcntl,json,os,subprocess,tempfile

BASE=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update')
helper=BASE/'server_cuda128.sh'
real='/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8'
results=[]
def execute(name,args,task,code,markers):
 env=dict(os.environ,SHUOLEE_CUDA_TASK_DIR=str(task))
 p=subprocess.run(['bash',str(helper),*args],env=env,capture_output=True,text=True,timeout=60)
 output=p.stdout+p.stderr
 passed=p.returncode==code and all(m in output for m in markers)
 results.append(dict(name=name,passed=passed,exit_code=p.returncode,output=output))
 assert passed,results[-1]
 return task/'cuda128.env'
def fixture(path,version='12.8',headers='include'):
 (path/'bin').mkdir(parents=True)
 p=path/'bin/nvcc'
 p.write_text('#!/bin/bash\nif [[ "$1" == --version ]]; then echo "Cuda compilation tools, release '+version+', V'+version+'.93"; exit 0; fi\nwhile [[ $# -gt 0 ]]; do if [[ "$1" == -o ]]; then touch "$2"; exit 0; fi; shift; done\nexit 91\n')
 p.chmod(0o755)
 if headers:
  (path/headers).mkdir(parents=True)
  for n in ['cuda.h','cuda_runtime.h']:(path/headers/n).write_text('// mock only\n')

subprocess.run(['bash','-n',str(helper)],check=True)
with tempfile.TemporaryDirectory(prefix='shuolee-cuda-diagnostic-') as d:
 root=Path(d)
 task=root/'actual-readonly';task.mkdir();(task/'cuda128.env').write_text('keep existing environment exactly\n')
 execute('actual_toolkit_readonly_diagnosis',['--diagnose',real],task,0,['release 12.8','DIAGNOSTIC_ONLY','compilers=1'])
 assert (task/'cuda128.env').read_text()=='keep existing environment exactly\n'
 assert not list(task.glob('logs/cuda128-probe.*'))
 execute('actual_compile_after_diagnostic_changes',['--toolkit',real],root/'actual-compile',0,['CUDA_12_8_READY'])
 assert (root/'actual-compile/cuda128.env').is_file()
 old=root/'old';fixture(old,'11.4')
 p=execute('wrong_version_separately_reported',['--toolkit',str(old)],root/'wrong-check',3,['wrong_version=1','CUDA_12_8_NOT_FOUND'])
 assert not p.exists()
 incomplete=root/'incomplete';fixture(incomplete,headers=None)
 p=execute('missing_headers_separately_reported',['--toolkit',str(incomplete)],root/'incomplete-check',3,['CUDA_12_8_HEADERS_MISSING','missing_headers=1'])
 assert not p.exists()
 split=root/'split';fixture(split,headers='targets/x86_64-linux/include')
 execute('split_conda_headers_detected_mock',['--toolkit',str(split)],root/'split-check',0,['Development headers:','targets/x86_64-linux/include','CUDA_12_8_READY'])
 task=root/'busy';task.mkdir()
 with (task/'cuda128.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  execute('read_only_diagnostic_allowed_during_install',['--diagnose',real],task,0,['CUDA_INSTALL_IN_PROGRESS','DIAGNOSTIC_ONLY'])
  assert not (task/'cuda128.env').exists()
  execute('write_check_still_blocked_during_install',['--toolkit',real],task,8,['CUDA_INSTALL_IN_PROGRESS'])
 p=execute('missing_explicit_toolkit_readonly_report',['--diagnose',str(root/'does-not-exist')],root/'absent',0,['compilers=0','DIAGNOSTIC_ONLY'])
 assert not p.exists()

report=dict(passed=all(r['passed'] for r in results),cases=results,company_server_tested=False,
    no_toolkit_download_or_install=True,no_gpu_training=True,
    actual_compiler_checked=real,split_layout_case_is_mock=True)
(BASE/'cuda128-diagnostic-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(passed=report['passed'],cases=len(results),company_server_tested=False)))
