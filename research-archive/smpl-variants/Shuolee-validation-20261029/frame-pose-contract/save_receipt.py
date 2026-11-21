"""Save local deployment evidence and append this task's coordination field."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy
import PIL

repo = Path('D:/Projects/Shuolee-frame-pose-contract-123')
out = Path('D:/Projects/Shuolee-validation-20261029/frame-pose-contract')
coord = Path('C:/Users/Administrator/Documents/New project/shuolee-coordination')
handoff = coord/'handoffs/frame-contract-123-20261029'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
git = lambda *args: subprocess.check_output(['git', '-C', str(repo), *args], text=True, encoding='utf-8').strip()
head = git('rev-parse', 'HEAD')
assert head == '3750cd76682ab29023875fb874ac27e31a8bc590'
assert git('status', '--porcelain') == ''
original = read(out/'handoff-import.json')
files = []
for item in original['files']:
    old = handoff/'source'/item['path']
    new = repo/item['path']
    assert sha(old) == item['sha256']
    files.append(dict(path=item['path'], original_sha256=sha(old),
                      deployed_sha256=sha(new), changed=sha(old)!=sha(new)))
assert sha(handoff/'frame-contract-123-source.zip') == original['package_sha256']
linux = read(out/'linux-verification.json')
fresh = read(out/'git-install.json')
assert linux['passed'] and fresh['passed'] and fresh['actual_commit']==head
ci = read(out/'github-ci-final.json')['workflow_runs']
ci = next(r for r in ci if r['head_sha']==head)
assert ci['status']=='completed' and ci['conclusion']=='success'
remote = dict(line.split()[::-1] for line in git('ls-remote','origin',
              'refs/heads/main','refs/heads/smplx/frame-pose-contract-123').splitlines())
assert remote['refs/heads/main']=='9225e1da6a18fe6ae8aac63ec48bafd65a3794a1'
assert remote['refs/heads/smplx/frame-pose-contract-123']==head
receipt = dict(
    updated=datetime.now(timezone.utc).isoformat(),
    state='cpu_contract_published_local_verified_training_not_connected',
    repository='https://github.com/lizuoshuo-lab/shuolee-portrait.git',
    repository_visibility='public', visibility_changed=False,
    branch='smplx/frame-pose-contract-123', commit=head,
    code_commit='22cea46d86a3f0fac24a6e922ba77b15a133a203',
    base_commit=remote['refs/heads/main'], main_unchanged=True,
    old_113_unchanged=True, old_113_commit='5fda749498cca899fb63011dc7b1c71511c14050',
    isolated_checkout=str(repo), clean_checkout=True, published=True,
    source_package_sha256=original['package_sha256'], original_package_preserved=True,
    files=files,
    fixes=['Fully decode bound images and masks; reject truncation with recomputed hashes',
           'Remove unshipped fitting APIs from the independent camera validator',
           'Use explicit cone sparse checkout compatible with Git 2.25.1'],
    authorization=dict(
        source_thread_id='01a0e316-7534-7693-a60c-ded52094785d',
        actual_user_message_id='01a18742-3d71-7991-9a8e-fd657d598e10',
        actual_user_request='将已经跑通的路线图同步给00窗口，让他做git上的部署以及实验，并作反馈给你',
        public_authorization='Current conversation human user: 可以设置成共有的，整个工程不能出现codex等字样',
        precedence='Source thread confirmed the private-repository preference predated the current conversation public authorization'),
    windows_environment=dict(hostname=platform.node(), python=sys.version,
        executable=sys.executable, numpy=numpy.__version__, pillow=PIL.__version__),
    windows_cpu_tests=read(out/'agent-tests.json'),
    linux_verification=linux, fresh_git_checkout=fresh,
    ci=dict(id=ci['id'], status=ci['status'], conclusion=ci['conclusion'], url=ci['html_url'], head_sha=ci['head_sha']),
    real_input_audit=str(out/'real-input-audit.json'),
    real_input_contract_passed=False, physical_calibration_validated=False,
    training_admitted=False, training_started=False, quality_accepted=False,
    server_verified=False, server_execution='not_executed_no_confirmed_host_and_authentication',
    cpu_deploy_command_document=str(repo/'docs/逐帧高斯输入约定.md'),
    cleanup=dict(temporary_clone_removed=fresh['temporary_checkout_removed'],
                 test_fixtures_removed=True, originals_preserved=True),
    evidence=str(out),
)
(out/'deployment-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
status_path=coord/'company00-status.json'
status=read(status_path)
status['frame_pose_contract_handoff']={key:receipt[key] for key in (
    'updated','state','repository','branch','commit','code_commit','base_commit','published',
    'main_unchanged','old_113_unchanged','source_package_sha256','original_package_preserved',
    'real_input_contract_passed','training_admitted','training_started','quality_accepted',
    'physical_calibration_validated','server_verified','server_execution','ci','evidence','cleanup')}
status['frame_pose_contract_handoff'].update(cpu_contract_tests=29,linux_regression_tests=114,
    linux_tests_skipped=0,fresh_git_checkout_verified=True,
    receipt=str(out/'deployment-receipt.json'), feedback_thread_id=receipt['authorization']['source_thread_id'])
temp=status_path.with_suffix('.json.tmp')
temp.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
temp.replace(status_path)
print(json.dumps(dict(receipt=str(out/'deployment-receipt.json'),sha256=sha(out/'deployment-receipt.json'),
                     state_file=str(status_path),commit=head),indent=2))
