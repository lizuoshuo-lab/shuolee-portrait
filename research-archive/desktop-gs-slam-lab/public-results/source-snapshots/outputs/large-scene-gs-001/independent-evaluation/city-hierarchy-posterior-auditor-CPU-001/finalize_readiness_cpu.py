"""Write CPU-only readiness and the separate readonly supervisor review."""
from pathlib import Path
import datetime,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/large-scene-gs-001/independent-evaluation';HERE=Path(__file__).parent
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
old=ROOT/'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_001.py'
new=ROOT/'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_002.py'
launcher=ROOT/'scripts/large_scene_gs/launch_small_city_hierarchy_stage_002.ps1'
expected=old.read_text().replace("OR Name='GaussianHierarchyMerger.exe'", "OR Name='GaussianHierarchyMerger.exe' OR Name='GaussianHierarchyMergerUnoptimized.exe'")
assert new.read_text()==expected
native=json.loads((ROOT/'outputs/large-scene-gs-001/hierarchy-build/merger-unoptimized/manifest.json').read_text())
assert Path(native['executable']['path']).name=='GaussianHierarchyMergerUnoptimized.exe'
directory=OUT/'city-hierarchy-stage-supervision-002-independent-review-CPU-001'
directory.mkdir(exist_ok=False)
report=dict(status='passed_readonly_static_supervisor002_exact_diff_and_terminal_lifecycle_review',reviewed_utc=stamp,
    old_source_sha256=h(old),new_source_sha256=h(new),new_launcher_sha256=h(launcher),
    actual_native_name_from_manifest='GaussianHierarchyMergerUnoptimized.exe',
    exact_only_Python_diff='Includeactual MergerUnoptimized name in shared Win32_Process filter',
    fresh_dispatch_loop_and_terminal_all_use_same_process_snapshot=True,
    terminal_child_poll_handle_exit_not_CIM_absence=True,PID_birth_handle_CIM_and_exact_argv_tail_checked=True,
    existing_output_receipt_no_overwrite_guard=True,post64_own_namespace_sourcekind_not_formalmerge=True,
    launch_hidden_PS7_exact_path=True,automatic_retry_nextstage=False,runtime_actual_hierarchy_stage_not_executed=True,
    no_app_job_closure_survival_promise=True,CUDA_called=False,process_started_or_stopped=False)
write(directory/'review.json',report)
prior=HERE/'auditor-readiness.json'
shutil.copyfile(prior,HERE/'history/auditor-readiness-before-peer.json')
value=json.loads(prior.read_text(encoding='utf-8'))
for record in value['source_records']:record['sha256']=h(Path(record['path']))
value.update(frozen_utc=stamp,peer_static_review=dict(status='passed_readonly_static_final_bef0_schema_review',
    agent='/root/city_hierarchy_plan_review',reviewed_source_sha256=h(HERE/'audit_city_1404.py'),
    actual_full_audit_execution=False,actual_SSIM_AlexLPIPS_recalculation=False,
    message='Real schema/source/count/input/renderer mask/timing/proof/terminal contract reviewed. No full1404 result or GPU was produced.'),
    source_revision=dict(tested_CPU_control_source_sha256='3d7f27e06ad001a82c612e7dfa779d5ebc75ed1b5a0918c08391128984b011d3',
        frozen_final_source_sha256=h(HERE/'audit_city_1404.py'),
        final_diff='Only durable verifiedframes/canonicalinput/protocol SHA links; completionwrittenlast. Formulas/missinggates/coarsemode unchanged.'),
    supervisor_independent_review_path=str(directory/'review.json'),supervisor_independent_review_sha256=h(directory/'review.json'))
write(prior,value)
shutil.copyfile(HERE/'audit_city_1404.py',HERE/'audit_city_1404-frozen.py')
shutil.copyfile(HERE/'recompute_perceptual_cpu.py',HERE/'recompute_perceptual_cpu-frozen.py')
write(HERE/'history/failed-inline-review-command.json',dict(status='preserved_CPU_inline_shell_quoting_failure',
    error='Python -c SyntaxError: parenthesis was never closed due to double-quote shell escaping',
    any_python_statement_executed=False,files_created_by_failed_command=False,CUDA_called=False,
    resolved_by='Readable saved finalize_readiness_cpu.py; no fallback process or GPU action'))
print(json.dumps(dict(readiness_sha256=h(prior),supervisor_review_sha256=h(directory/'review.json'),frozen_posterior_SHA=h(HERE/'audit_city_1404.py'),frozen_perceptual_SHA=h(HERE/'recompute_perceptual_cpu.py'))))
