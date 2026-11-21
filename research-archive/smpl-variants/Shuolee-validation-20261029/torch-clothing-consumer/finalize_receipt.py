"""Record the independently verified CPU increment and its exact boundaries."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

out=Path('D:/Projects/Shuolee-validation-20261029/torch-clothing-consumer')
repo=Path('D:/Projects/Shuolee-torch-clothing-consumer-123')
handoff=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/handoffs/torch-clothing-consumer-123-20261029')
status_path=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/company00-status.json')
read=lambda name:json.loads((out/name).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
git=lambda *args:subprocess.check_output(['git','-C',str(repo),*args],text=True,encoding='utf-8').strip()
commit=git('rev-parse','HEAD');assert not git('status','--porcelain')
original=read('original-tests.json');fixed=read('fixed-tests.json');linux=read('linux-verification.json')
install=read('git-install.json');bindings=read('source-and-patch-verification.json');ci=read('ci-result.json')
assert original['outcome']=='passed' and original['tests_run']==14
assert fixed['outcome']=='passed' and fixed['failures']==0 and fixed['errors']==0
count=164+fixed['tests_run']
assert linux['passed'] and linux['regression']['tests_run']==count and linux['regression']['skipped']==0
assert install['passed'] and install['actual_commit']==commit and install['temporary_checkout_removed']
assert install['tests']['tests_run']==fixed['tests_run']
assert install['full_regression']['regression']['tests_run']==count
assert install['full_regression']['regression']['skipped']==0
assert ci['head_sha']==commit and ci['conclusion']=='success' and ci['tests_run']==count
for name,hashes in bindings['files'].items():assert sha(repo/name)==hashes['final_sha256']
archive=handoff/'torch-clothing-consumer-123-source.zip'
assert sha(archive)=='d6868ca069aa681a8b653018d3c350eb3f7ceb785c4d0375fdbd2e65ec816dba'
expected={'main':'9225e1da6a18fe6ae8aac63ec48bafd65a3794a1',
    'shuolee/113':'5fda749498cca899fb63011dc7b1c71511c14050',
    'smplx/frame-pose-contract-123':'3750cd76682ab29023875fb874ac27e31a8bc590',
    'smplx/world-units-contract-123':'8d54497c0395a0a8a8f378a33d4895f5ffcd56be',
    'smplx/free-clothing-transport-123':'d4a3d73fdf8dbec8aed4f661eccd3bbf95f9210b',
    'smplx/torch-clothing-consumer-123':commit}
actual={line.split()[1].removeprefix('refs/heads/'):line.split()[0] for line in (out/'remote-refs.txt').read_text(encoding='utf-8').splitlines() if line.strip()}
assert all(actual.get(k)==v for k,v in expected.items())
receipt=dict(updated=datetime.now(timezone.utc).isoformat(),
    state='torch_clothing_consumer_cpu_published_local_fresh_git_and_ci_verified',
    repository='https://github.com/lizuoshuo-lab/shuolee-portrait.git',branch='smplx/torch-clothing-consumer-123',
    commit=commit,base_commit=expected['smplx/free-clothing-transport-123'],published=True,
    main_unchanged=True,old_113_unchanged=True,previous_contract_branches_unchanged=True,
    source_package_sha256=sha(archive),original_package_preserved=True,file_hashes=bindings['files'],
    original_tests=14,fixed_tests=fixed['tests_run'],linux_regression_tests=count,linux_tests_skipped=0,
    fresh_git_targeted_tests=install['tests']['tests_run'],fresh_git_full_regression_tests=count,
    fresh_git_checkout_verified=True,
    fresh_git_network={'existing_proxy_used':install['existing_network_proxy_used'],
        'process_local_http_version':install['process_local_git_http_version'],
        'prior_failure_reports':['git-install-tls-failure.json','git-install-tls-retry-failure.json','git-install-http1-failure.json','git-install-http1-retry-failure.json'],
        'prior_failure_reason':'GnuTLS connection terminated before cloning; no tests executed in these attempts.',
        'independent_network_control':'7 isolated cold clones: 5 passed, 2 failed; identical invocation both passed and failed. Evidence does not distinguish proxy, egress or remote termination.',
        'persistent_git_or_proxy_configuration_changed':False},
    local_runtime={k:linux[k] for k in ('platform','python','numpy','torch','torch_cuda_build','cuda_available','cuda_visible_devices')},
    ci=ci,fixes=['Reject gradients on frozen motion/camera buffers even when Tensor version is unchanged.',
                'Require exactly one 4x4 extrinsic for each named frame during preparation.',
                'Build version-tracked constant tables even when the caller uses inference mode.',
                'Normalize finite nonzero quaternions stably across representable magnitudes and preserve parameter gradients.'],
    numerical_controls=fixed['metrics'],independent_final_audit='final-independent-audit.json',
    reference_quaternion_test_intent='Mathematical normalization compared within 4 eps; XYZ and log scale remain bit-exact at reference.',
    training_admitted=False,training_started=False,quality_accepted=False,
    physical_calibration_validated=False,GPU_jobs=0,body_model_forwards=0,training_steps=0,
    server_verified=False,server_execution='not_executed_no_confirmed_host_and_authentication',
    scope={'generic_source_only':True,'person_data_or_weights_uploaded':False,'production_trainer_connected':False,
           'performance_200k_verified':False,'CPU_tensor_only_verified_here':True,
           'research_GPU_report':'Peer reports 11 CUDA forwards, 1 backward, 83 field assertions on the original consumer; this fixed source has not been run on GPU here.',
           'research_GPU_report_is_native_trainer_or_person_quality_acceptance':False},
    cleanup={'temporary_git_checkout_removed':install['temporary_checkout_removed'],
             'temporary_patch_checkout_removed':bindings['patch']['temporary_checkout_removed'],
             'original_sources_and_failure_evidence_preserved':True},
    evidence=str(out),receipt=str(out/'deployment-receipt.json'),followup_patch=bindings['patch'],
    document=str(repo/'docs/逐帧Torch消费层.md'),feedback_thread_id='01a0e316-7534-7693-a60c-ded52094785d')
(out/'deployment-receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
status=json.loads(status_path.read_text(encoding='utf-8-sig'));status['torch_clothing_consumer_handoff']=receipt
status_path.write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'commit':commit,'targeted_tests':fixed['tests_run'],'full_tests':count,'CI':ci['conclusion'],
                  'receipt':str(out/'deployment-receipt.json'),'receipt_sha256':sha(out/'deployment-receipt.json'),
                  'status_updated':True},indent=2))
