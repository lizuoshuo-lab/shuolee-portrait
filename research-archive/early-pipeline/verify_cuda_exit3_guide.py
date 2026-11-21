"""Check copyable guide syntax and failure handling without installing anything."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib,json,os,re,subprocess,tempfile

BASE=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update')
DOC=Path('/mnt/d/00/工作/SMPL-X')
md=(DOC/'00_部署与训练指南.md').read_text(encoding='utf-8')
page=(DOC/'00_部署与训练指南.html').read_text(encoding='utf-8')
blocks=re.findall(r'```bash\n(.*?)\n```',md,re.S)
for code in blocks:subprocess.run(['bash','-n'],input=code,text=True,check=True)

class CodeBlocks(HTMLParser):
    def __init__(self):super().__init__();self.pre=False;self.code=False;self.blocks=[]
    def handle_starttag(self,t,a):
        if t=='pre':self.pre=True
        if t=='code' and self.pre:self.code=True;self.blocks.append('')
    def handle_endtag(self,t):
        if t=='code':self.code=False
        if t=='pre':self.pre=False
    def handle_data(self,s):
        if self.pre and self.code:self.blocks[-1]+=s

html=CodeBlocks();html.feed(page)
for code in blocks:assert code.strip() in [b.strip() for b in html.blocks]
step_a=md.split('### 4A.',1)[1].split('### 4B.',1)[0]
install=re.search(r'```bash\n(.*?)\n```',step_a,re.S).group(1)
assert 'bash server_cuda128.sh --install' in install
assert install.index('--install') < install.index('test -s') < install.index('echo CUDA_ENV_READY')
assert 'source ' not in install
assert len(re.findall(r'^## \d+\.',md,re.M))==8

results=[]
resume_results=[]
with tempfile.TemporaryDirectory(prefix='shuolee-guide-exit3-') as tmp:
    root=Path(tmp)
    actual_helper=BASE/'server_cuda128.sh'
    subprocess.run(['bash','-n',str(actual_helper)],check=True)
    negative=subprocess.run(['bash',str(actual_helper),'--toolkit',str(root/'absent-toolkit')],
        env=dict(os.environ,SHUOLEE_CUDA_TASK_DIR=str(root/'negative-check')),
        capture_output=True,text=True,timeout=20)
    assert negative.returncode==3
    assert 'CHECK_ONLY_NO_INSTALL' in negative.stdout and 'guide step 4A' in negative.stdout
    assert 'NEXT_INSTALL:' in negative.stdout and '--install' in negative.stdout
    assert not (root/'negative-check/cuda128.env').exists()
    assert 'Continue with guide step 4C' not in actual_helper.read_text()
    for name,status,make_env,expected in [('not_found',3,False,3),('false_success_no_env',0,False,1),('ready',0,True,0)]:
        task=root/name;task.mkdir()
        helper=task/'server_cuda128.sh'
        helper.write_text('#!/bin/bash\n[[ "$1" == --install ]] || exit 92\n'+
            ('echo test > "$SHUOLEE_GUIDE_TEST_DIR/cuda128.env"\n' if make_env else '')+
            f'exit {status}\n',encoding='utf-8')
        (task/'server_cuda128.sh.sha256').write_text(hashlib.sha256(helper.read_bytes()).hexdigest()+'  server_cuda128.sh\n')
        test_code=install.replace('$HOME/shuolee','$SHUOLEE_GUIDE_TEST_DIR')
        p=subprocess.run(['bash'],input=test_code,text=True,capture_output=True,
            env=dict(os.environ,SHUOLEE_GUIDE_TEST_DIR=str(task)),timeout=10)
        assert p.returncode==expected,(name,p.returncode,p.stdout,p.stderr)
        assert ('CUDA_ENV_READY' in p.stdout)==(expected==0)
        results.append(dict(name=name,exit_code=p.returncode,passed=True))

    step_b=md.split('### 4B.',1)[1].split('### 4C.',1)[0]
    resume=[code for code in re.findall(r'```bash\n(.*?)\n```',step_b,re.S)
            if 'bash scripts/setup_server.sh --resume-install' in code]
    assert len(resume)==1
    resume=resume[0]
    assert resume.index('source ') < resume.index('--resume-install') < resume.index('bash scripts/setup_113.sh')
    assert 'route113.py' not in resume and 'rm ' not in resume
    for name,has_env_file,has_venv,base_exit,raster_exit,expected in [
        ('ready_existing_environment',True,True,0,0,0),
        ('missing_cuda_env',False,True,0,0,1),
        ('not_a_venv',True,False,0,0,1),
        ('base_install_failed',True,True,17,0,17),
        ('rasterizer_install_failed',True,True,0,19,19),
    ]:
        task=root/name
        scripts=task/'shuolee/portrait113/scripts';scripts.mkdir(parents=True)
        env=task/'env-cu128';env.mkdir()
        if has_venv:(env/'pyvenv.cfg').write_text('test fixture only\n')
        toolkit=task/'toolkit';(toolkit/'bin').mkdir(parents=True)
        nvcc=toolkit/'bin/nvcc';nvcc.write_text('#!/bin/bash\necho "release 12.8 test fixture"\n');nvcc.chmod(0o755)
        if has_env_file:
            (task/'shuolee/cuda128.env').write_text('export CUDA_HOME="$SHUOLEE_GUIDE_TEST_TOOLKIT"\n')
        (scripts/'setup_server.sh').write_text('#!/bin/bash\n[[ "$1" == --resume-install ]] || exit 98\nprintf "base:%s\\n" "$1" >> "$SHUOLEE_GUIDE_TEST_CALLS"\n'+f'exit {base_exit}\n')
        (scripts/'setup_113.sh').write_text('#!/bin/bash\necho rasterizer >> "$SHUOLEE_GUIDE_TEST_CALLS"\n'+f'exit {raster_exit}\n')
        calls=task/'calls.txt'
        test_code=resume.replace('$HOME/shuolee','$SHUOLEE_GUIDE_TEST_TASK')
        test_code=test_code.replace('$HOME/.local/share/shuolee-portrait/env-cu128','$SHUOLEE_GUIDE_TEST_ENV')
        p=subprocess.run(['bash'],input=test_code,text=True,capture_output=True,timeout=10,
            env=dict(os.environ,SHUOLEE_GUIDE_TEST_TASK=str(task/'shuolee'),
                     SHUOLEE_GUIDE_TEST_ENV=str(env),SHUOLEE_GUIDE_TEST_TOOLKIT=str(toolkit),
                     SHUOLEE_GUIDE_TEST_CALLS=str(calls)))
        assert p.returncode==expected,(name,p.returncode,p.stdout,p.stderr)
        actual_calls=calls.read_text().splitlines() if calls.exists() else []
        wanted=[] if not (has_env_file and has_venv) else ['base:--resume-install']+(['rasterizer'] if base_exit==0 else [])
        assert actual_calls==wanted,(name,actual_calls,wanted)
        resume_results.append(dict(name=name,exit_code=p.returncode,passed=True))

unchanged=json.loads((BASE/'cuda-resume-unchanged-sections.json').read_text())
sections=dict(re.findall(r'(?ms)^## (\d+)\. (.*?)(?=^## \d+\. |\Z)',md))
assert all(hashlib.sha256(sections[k].encode()).hexdigest()==v for k,v in unchanged.items())

canonical=(BASE/'server_cuda128.sh').read_bytes()
assert (DOC/'server_cuda128.sh').read_bytes()==canonical
assert hashlib.sha256(canonical).hexdigest()==(DOC/'server_cuda128.sh.sha256').read_text().split()[0]
report=dict(passed=True,bash_blocks_syntax_checked=len(blocks),html_copy_blocks_identical=True,
    actual_helper_exit3_and_install_step_message_checked=True,
    install_flow_mock_cases=results,dependency_resume_mock_cases=resume_results,
    other_seven_steps_unchanged=True,helper_hash_verified=True,company_server_tested=False,
    no_installation_or_network=True,md_sha256=hashlib.sha256(md.encode()).hexdigest(),
    html_sha256=hashlib.sha256(page.encode()).hexdigest())
(BASE/'cuda128-guide-verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
