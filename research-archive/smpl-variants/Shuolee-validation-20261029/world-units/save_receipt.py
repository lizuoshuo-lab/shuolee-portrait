"""Preserve the checked source handoff and append the local deployment receipt."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

repo=Path('D:/Projects/Shuolee-world-units-contract-123')
out=Path('D:/Projects/Shuolee-validation-20261029/world-units')
coord=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination')
package=coord/'handoffs/world-units-123-20261029'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
git=lambda *args:subprocess.check_output(['git','-C',str(repo),*args],text=True,encoding='utf-8').strip()
manifest=read(package/'source-manifest.json')
head=git('rev-parse','HEAD')
assert head=='8d54497c0395a0a8a8f378a33d4895f5ffcd56be'
assert not git('status','--porcelain')
assert sha(package/'world-units-123-source.zip')==manifest['archive_sha256']
files=[]
patch=[]
for item in manifest['files']:
    old=package/'source'/item['path'];new=repo/item['path']
    assert sha(old)==item['sha256']
    files.append(dict(path=item['path'],original_sha256=sha(old),deployed_sha256=sha(new),
                      changed=sha(old)!=sha(new)))
    if item['path'] in ('src/shuolee_portrait/sequence_fitting.py','tests/test_camera_world_units.py'):
        patch.extend(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),
            new.read_text(encoding='utf-8').splitlines(True),fromfile='a/'+item['path'],tofile='b/'+item['path']))
patch_path=out/'world-units-followup.patch'
patch_path.write_text(''.join(patch),encoding='utf-8',newline='\n')
linux=read(out/'linux-verification.json');fresh=read(out/'git-install.json')
assert linux['passed'] and linux['regression']['tests_run']==139 and linux['regression']['skipped']==0
assert fresh['passed'] and fresh['actual_commit']==head and fresh['tests']['tests_run']==25
ci=next(r for r in read(out/'github-ci.json')['workflow_runs'] if r['head_sha']==head)
assert ci['status']=='completed' and ci['conclusion']=='success'
refs=dict(line.split()[::-1] for line in git('ls-remote','origin','refs/heads/main',
    'refs/heads/smplx/frame-pose-contract-123','refs/heads/smplx/world-units-contract-123').splitlines())
assert refs['refs/heads/main']=='9225e1da6a18fe6ae8aac63ec48bafd65a3794a1'
assert refs['refs/heads/smplx/frame-pose-contract-123']==manifest['base_commit']
assert refs['refs/heads/smplx/world-units-contract-123']==head
assert not git('diff',manifest['base_commit'],'--','src/shuolee_portrait/frame_pose_contract.py',
               'src/shuolee_portrait/alignment_inputs.py','src/shuolee_portrait/camera_pixel_contract.py')
receipt=dict(
    updated=datetime.now(timezone.utc).isoformat(),
    state='cpu_unit_and_provenance_guards_published_local_and_ci_verified',
    repository='https://github.com/lizuoshuo-lab/shuolee-portrait.git',
    branch='smplx/world-units-contract-123',commit=head,base_commit=manifest['base_commit'],
    published=True,clean_checkout=True,main_unchanged=True,old_113_unchanged=True,
    previous_frame_contract_unchanged=True,original_package_preserved=True,
    source_package_sha256=manifest['archive_sha256'],files=files,
    fixes=['Detach fixed K from caller gradient graph while preserving point gradients',
           'Validate frozen cameras and frame order before computation, snapshot and checkpoint save',
           'Save full snapshot frame IDs and explicit training_admitted=false',
           'Reject post-construction camera dtype changes before modifying submodules; preserve float64 source E',
           'Pin CPU PyTorch2.7.1 dependency and verify CPU wheel in CI'],
    tests=dict(original_unit=read(out/'original-unit-tests.json'),fixed_unit=read(out/'fixed-unit-tests.json'),
               linux=linux,fresh_git=fresh),
    pre_fix_evidence=str(out/'pre-fix-reproduction.json'),
    legacy_torch_failure=str(out/'legacy-torch-failure.json'),
    followup_patch=str(patch_path),followup_patch_sha256=sha(patch_path),
    ci=dict(id=ci['id'],url=ci['html_url'],head_sha=ci['head_sha'],status=ci['status'],conclusion=ci['conclusion'],
            python='3.10.22',numpy='1.26.4',torch='2.7.1+cpu',cuda_build=None,tests_run=139,tests_skipped=0),
    command_document=str(repo/'docs/共同相机单位与部署边界.md'),
    model_forwards=0,gaussian_steps=0,real_fit=0,
    real_person_scale_estimated=False,physical_calibration_validated=False,
    training_admitted=False,quality_accepted=False,server_verified=False,
    server_execution='not_executed_no_confirmed_host_and_authentication',
    interpretation=dict(
        physical_meter_unknown_alone_rejects_all_relative_reconstruction=False,
        scope='This interface explicitly converts declared source camera lengths to model meters.',
        pending='Real input admission, verified pose transport and rendering integration remain outside this CPU deployment.',
        old_static_fit_has_scale=True,units_are_not_established_as_the_sole_cause_of_old_ghosting=True,
        peer_relative_warp_research='Not independently executed here; no real-person quality claim is inferred.'),
    cleanup=dict(temporary_git_checkout_removed=fresh['temporary_checkout_removed'],
                 test_fixtures_removed=True,original_sources_and_evidence_preserved=True),
    evidence=str(out))
receipt_path=out/'deployment-receipt.json'
receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
status_path=coord/'company00-status.json';status=read(status_path)
status['world_units_handoff']={k:receipt[k] for k in (
    'updated','state','repository','branch','commit','base_commit','published','main_unchanged',
    'old_113_unchanged','previous_frame_contract_unchanged','source_package_sha256','original_package_preserved',
    'real_person_scale_estimated','physical_calibration_validated','training_admitted','quality_accepted',
    'server_verified','server_execution','ci','interpretation','cleanup','evidence')}
status['world_units_handoff'].update(original_unit_tests=19,fixed_unit_tests=25,linux_regression_tests=139,
    linux_tests_skipped=0,fresh_git_checkout_verified=True,receipt=str(receipt_path),
    followup_patch=str(patch_path),feedback_thread_id='01a0e316-7534-7693-a60c-ded52094785d')
temp=status_path.with_suffix('.json.tmp')
temp.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(status_path)
print(json.dumps(dict(receipt=str(receipt_path),sha256=sha(receipt_path),patch=str(patch_path),
                     patch_sha256=sha(patch_path),commit=head),indent=2))
