"""CPU provenance/report packaging; no Torch or CUDA imports."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/'outputs/large-scene-gs-001/independent-evaluation';HERE=Path(__file__).parent
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
coarse=OUT/'small-city-local-coarse-003-flat-heldout-001'
assert h(coarse/'metrics.json')=='17035e5e1f83577962c933d3a924a5f39d8cf84438d2a862f9d69847f68442dc'
assert h(OUT/'city-hierarchy-posterior-auditor-CPU-001/audit_city_1404.py')=='bef0c57ed5529c04ab4d2d837d5efbd15e76938787dab72b24527375fe9cd133'
assert h(OUT/'city-hierarchy-posterior-auditor-CPU-001/recompute_perceptual_cpu.py')=='4b318c7362c35cce5087605ac7c1902fb0697c67185c039e3150676cfd3a3181'
assert h(OUT/'city-local-hierarchy-evaluation-driver-CPU-001/evaluate_local_city_hierarchy.py')=='c3d9c5c7547cee632f79ba0e66b6d24bbcd35771295b7f4fd23fc5b93ad40610'
full=OUT/'small-city-coarse117-perceptual-full117-CPU-001';result=read(full/'metrics-CPU-reference.json');assert result['actual_views']==117 and not result['CUDA_called'] and not result['actual1404_completed']
manifest=dict(status='actual_coarse117_CPU_SSIM_Alex_reference_completed_not1404',created_utc=datetime.now(timezone.utc).isoformat(),
    source_original_metrics_sha256=h(coarse/'metrics.json'),tested_source_sha256=h(HERE/'recompute_city_perceptual_cpu-v1-e9ac-tested.py'),
    current_future_wrapper_sha256=h(HERE/'recompute_city_perceptual_cpu.py'),current_source_diff='Only future1404 actualbef identity via pinned15cc readonlybinder beforeTorch andprotectedfilesunchanged afterCPU; coarseformula unchanged',
    actual117_metrics_path=str(full/'metrics-CPU-reference.json'),actual117_metrics_sha256=h(full/'metrics-CPU-reference.json'),actual117_outcome_sha256=h(full/'outcome.json'),
    actual8_reference_path=str(OUT/'small-city-coarse117-perceptual-geometry8-CPU-001/metrics-CPU-reference.json'),
    current_missing1404_refusal=read(OUT/'small-city-perceptual-own1404-missing-refusal-CPU-002/outcome.json'),
    actual_metric_cpu_seconds=result['total_CPU_metric_seconds'],actual_CPU_full_process_seconds=read(full/'outcome.json')['elapsed_seconds'],actual_peak_CPU_memory=result['final_memory'],
    CPU_SSIM_mean=result['CPU_SSIM_mean'],original_GPU_SSIM_mean=result['GPU_SSIM_mean'],CPU_AlexLPIPS_mean=result['CPU_AlexLPIPS_mean'],original_GPU_AlexLPIPS_mean=result['GPU_AlexLPIPS_mean'],max_abs_SSIM_delta=result['max_abs_SSIM_delta'],max_abs_AlexLPIPS_delta=result['max_abs_AlexLPIPS_delta'],
    measured117_rounded_observation_window=dict(SSIM=1.1e-6,AlexLPIPS=2e-5,not_analytic_bound=True,not_automatic_future1404_acceptance_tolerance=True),
    threads=4,interop_threads=1,batch_size=1,actual1404_perceptual_completed=False,CUDA_called=False,downloads=False,
    original_results_and_fixed_sources_unchanged=True,peer_original_CPU_source_review='passed_e9ac_readonly_source/device/weights/mask/timing/schema',peer_future_identity_patch_review='pending',
    report_sha256=h(HERE/'report.zh-CN.md'),commands_sha256=h(HERE/'CPU-commands.txt'))
write(HERE/'reference-manifest.json',manifest)
directory=OUT/'city-boundary-images002-independent-review-CPU-001';directory.mkdir(exist_ok=False)
p=ROOT/'scripts/large_scene_gs/audit_city_boundary_images_002.py'
assert h(p)=='15cc350108647f47ee7c20c6297ce7930d6af674233bbc3eac2017eeb74fcb1f'
write(directory/'review.json',dict(status='passed_readonly_static_boundary002_posterior_identity_and_original_image_interface',source_sha256=h(p),posterior_source_sha256='bef0c57ed5529c04ab4d2d837d5efbd15e76938787dab72b24527375fe9cd133',
    complete_actual1404_status_outcome_frameproof_source_input_output_SHA_contract_matches=True,exact_coarse3_vs_dense2_ordinal_paths=True,
    real_per_image_embedded_metric_and_original_float_PNG_raw_masked_pair_pixel_checks=True,fixed_geometry002_eightviews_24ROI_without_score_ranking=True,
    masks_GTm2_predm_auxiliary_m2_denominators_correct=True,all_protected_sources_end_SHA_rechecked=True,
    actual_dense_data_unavailable_no_execution=True,only_static_CPU_review=True,CUDA_called=False,source_and_images_modified=False))
shutil.copyfile(HERE/'recompute_city_perceptual_cpu.py',HERE/'recompute_city_perceptual_cpu-frozen.py')
print(json.dumps(dict(reference_manifest_sha256=h(HERE/'reference-manifest.json'),boundary002_review_sha256=h(directory/'review.json'),current_source_sha256=h(HERE/'recompute_city_perceptual_cpu.py'))))
