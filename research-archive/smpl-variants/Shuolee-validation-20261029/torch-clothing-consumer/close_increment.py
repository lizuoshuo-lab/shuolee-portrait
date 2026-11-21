"""Close the bounded component validation and hand it to the single project."""
from datetime import datetime,timezone
import json
from pathlib import Path

out=Path('D:/Projects/Shuolee-validation-20261029/torch-clothing-consumer')
read=lambda name:json.loads((out/name).read_text(encoding='utf-8-sig'))
bindings=read('source-and-patch-verification.json');fresh=read('git-install.json')
receipt=dict(updated=datetime.now(timezone.utc).isoformat(),
    state='component_cpu_verified_standalone_delivery_superseded_by_single_reconstruction_project',
    branch='smplx/torch-clothing-consumer-123',commit='b74430fd8f9b7d2e97202a69eba7007925ef6f26',
    code_commit='4422cba242fa712c8003bad6d24472f29cc5bd7f',base_commit='d4a3d73fdf8dbec8aed4f661eccd3bbf95f9210b',
    published=True,original_package_preserved=True,
    source_package_sha256='d6868ca069aa681a8b653018d3c350eb3f7ceb785c4d0375fdbd2e65ec816dba',
    file_hashes=bindings['files'],followup_patch=bindings['patch'],
    original_tests=14,fixed_tests=19,local_regression=read('linux-verification.json')['regression'],
    ci=dict(id=40882139928,job_id=123476160229,status='completed',conclusion='success',
            commit='b74430fd8f9b7d2e97202a69eba7007925ef6f26',tests_run=183,
            url='https://github.com/lizuoshuo-lab/shuolee-portrait/actions/runs/40882139928'),
    fresh_git_targeted_passed=fresh['tests']['outcome']=='passed',fresh_git_targeted_tests=19,
    fresh_git_full_regression_passed=False,fresh_git_regression=fresh['full_regression']['regression'],
    fresh_git_regression_failure='The explicitly inherited external HTTP proxy intercepted localhost fixture servers (HTTP502); 8 model-download fixture tests failed. Worktree and CI183 suites passed. Not rerun because component-only validation was ended by user steering.',
    network_failures_recorded=True,temporary_git_checkout_removed=fresh['temporary_checkout_removed'],
    training_admitted=False,quality_accepted=False,server_verified=False,
    GPU_executions=0,body_model_forwards=0,training_steps=0,
    integrated_project='D:/Projects/Shuolee',integrated_branch='smplx/reconstruction',
    current_integration_gap='video/prepared observations to production shared-identity per-frame SMPL-X fitting adapter; legacy fitting and legacy Gaussian trainer are static and are not a substitute',
    research_owner_retains_native_training=True,
    evidence=str(out),receipt=str(out/'deployment-receipt.json'))
(out/'deployment-receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
path=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/company00-status.json')
status=json.loads(path.read_text(encoding='utf-8-sig'))
status['torch_clothing_consumer_handoff']=receipt
status['one_shot_reconstruction']={'updated':receipt['updated'],'state':'integrating_single_project',
    'project':'D:/Projects/Shuolee','branch':'smplx/reconstruction',
    'scope_source':'C:/Users/Administrator/Documents/New project/shuolee-coordination/one-shot-reconstruction-handoff.json',
    'server_verified':False,'end_to_end_real_training_verified':False,'quality_accepted':False,
    'first_missing_interface':receipt['current_integration_gap']}
path.write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status_updated':True,'component_closed':True,'integrated_project':receipt['integrated_project']}))
