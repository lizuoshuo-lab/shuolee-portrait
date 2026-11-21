"""Collect verified deployment evidence without modifying historical results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

out=Path('D:/Projects/Shuolee-validation-20261029/free-clothing-transport')
repo=Path('D:/Projects/Shuolee-free-clothing-transport-123')
status_path=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/company00-status.json')
read=lambda name:json.loads((out/name).read_text(encoding='utf-8-sig'))
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
git=lambda *a:subprocess.check_output(['git','-C',str(repo),*a],text=True,encoding='utf-8').strip()
commit=git('rev-parse','HEAD')
assert not git('status','--porcelain')
original=read('original-tests.json');fixed=read('fixed-transport-tests.json')
linux=read('linux-verification.json');install=read('git-install.json')
assert original['passed'] and original['tests_run']==18
assert fixed['passed'] and fixed['tests_run']==25
assert linux['passed'] and linux['regression']['tests_run']==164 and linux['regression']['skipped']==0
assert install['passed'] and install['actual_commit']==commit and install['temporary_checkout_removed']
ci=read('ci-result.json');assert ci['head_sha']==commit and ci['conclusion']=='success' and ci['tests_run']==164
bindings=read('source-and-patch-verification.json')
for name, hashes in bindings['files'].items():assert sha(repo/name)==hashes['final_sha256']
package=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/handoffs/free-clothing-transport-123-20261029/free-clothing-transport-123-source.zip')
assert sha(package)=='cc4c4b2340313a46bdddbdee0f6dc2b34d00d612a0e283f8189996a8dc6a0799'
expected={'main':'9225e1da6a18fe6ae8aac63ec48bafd65a3794a1',
          'shuolee/113':'5fda749498cca899fb63011dc7b1c71511c14050',
          'smplx/frame-pose-contract-123':'3750cd76682ab29023875fb874ac27e31a8bc590',
          'smplx/world-units-contract-123':'8d54497c0395a0a8a8f378a33d4895f5ffcd56be',
          'smplx/free-clothing-transport-123':commit}
actual={line.split()[1].removeprefix('refs/heads/'):line.split()[0] for line in (out/'remote-refs.txt').read_text(encoding='utf-8').splitlines() if line.strip()}
assert all(actual.get(k)==v for k,v in expected.items())
receipt=dict(updated=datetime.now(timezone.utc).isoformat(),
    state='cpu_free_clothing_transport_published_local_and_ci_verified',
    repository='https://github.com/lizuoshuo-lab/shuolee-portrait.git',
    branch='smplx/free-clothing-transport-123',commit=commit,base_commit=expected['smplx/world-units-contract-123'],
    published=True,main_unchanged=True,old_113_unchanged=True,previous_contract_branches_unchanged=True,
    source_package_sha256=sha(package),original_package_preserved=True,
    file_hashes=bindings['files'],original_tests=18,fixed_tests=25,linux_regression_tests=164,linux_tests_skipped=0,
    local_runtime={k:linux[k] for k in ('platform','python','numpy','torch','torch_cuda_build','cuda_available','cuda_visible_devices')},
    fresh_git_checkout_verified=True,ci=ci,
    fixes=['relative covariance symmetry validation without an absolute tolerance floor',
           'invalid NPZ CRC reported as ValueError with original cause',
           'duplicate NPZ fields rejected before dictionary construction',
           'Torch internal factories explicitly use CPU despite ambient default device',
           'CPU autocast disabled locally to preserve declared dtype and gradients'],
    numerical_controls=fixed['metrics'],training_admitted=False,training_started=False,quality_accepted=False,
    physical_calibration_validated=False,real_fit=0,model_forwards=0,GPU_jobs=0,gaussian_steps=0,
    server_verified=False,server_execution='not_executed_no_confirmed_host_and_authentication',
    interpretation={'physical_meter_unknown_alone_rejects_all_relative_reconstruction':False,
        'common_scale_control':'All world lengths and camera translations scale together; covariance scales quadratically.',
        'pose_compensation_control':'Camera translation changes with per-frame world geometry reconstructed from camera-space geometry.',
        'scope':'CPU geometry/support interface; no renderer or training integration and no real-person quality acceptance.',
        'peer_gpu_probes':'Research thread now reports 14 CUDA synthetic checks passed with 6 Gaussians, 11 forwards and 1 backward. Not independently reproduced here; excluded from this CPU deployment. Earlier failed attempts remain historical evidence.'},
    cleanup={'temporary_git_checkout_removed':install['temporary_checkout_removed'],
             'temporary_patch_checkout_removed':bindings['patch']['temporary_checkout_removed'],
             'original_sources_and_failure_evidence_preserved':True},
    evidence=str(out),receipt=str(out/'deployment-receipt.json'),followup_patch=bindings['patch'],
    document=str(repo/'docs/自由衣物高斯逐帧运输.md'),feedback_thread_id='01a0e316-7534-7693-a60c-ded52094785d')
(out/'deployment-receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
status=json.loads(status_path.read_text(encoding='utf-8-sig'))
status['free_clothing_transport_handoff']=receipt
status_path.write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'commit':commit,'tests':164,'CI':ci['conclusion'],'receipt':str(out/'deployment-receipt.json'),
                  'receipt_sha256':sha(out/'deployment-receipt.json'),'status_updated':True},indent=2))
