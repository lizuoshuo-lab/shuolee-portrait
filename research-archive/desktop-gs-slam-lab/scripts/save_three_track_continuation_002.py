"""Save the checked continuation state without discarding the previous manifest."""
from pathlib import Path
import json,shutil
LAB=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
dest=LAB/'manifests/three-track-continuation-20261013.json'
previous=LAB/'manifests/three-track-continuation-20261013-001.json'
if not previous.exists():shutil.copy2(dest,previous)
checks=read(LAB/'outputs/city-parameter-paging-001/website-http-audit.json')
assert checks['status']=='passed'
record=dict(updated='2026-10-13',status='second_bounded_continuation_completed_research_tracks_not_complete',
    previous_manifest=str(previous),priority=['SplaTAM','FAST-LIVO2','large-scene GS SfM partition LOD'],active_gpu_jobs_from_this_followup=[],active_ROS_jobs_from_this_followup=[],
    tracks=[
        dict(name='SplaTAM',completed='79 actual new continuation frames after saved checkpoints 800 and 1100; 15800 tracking updates and 2370 mapping iterations. Fixed initial support loss drops in 74 frames, with position error rising in 36 of those.',
             metrics='outputs/splatam-room-window-continuation-001/metrics.json',validation='outputs/splatam-room-window-continuation-001/validation.json',script='scripts/splatam_local/continue_room_windows_001.py',
             next='Use saved immediate maps from the NEW continuation (838/1133) for controlled same-map, same-initial-pose color/depth and local-observability comparisons. Capture complete RNG/auxiliary state in a future full run if exact historical reconstruction is required.',
             guard='Sparse old checkpoints cannot reconstruct historical RNG/map-update state. New continuations diverged from historical trajectory. No drift-repair claim and no GT in optimization.'),
        dict(name='FAST-LIVO2',completed='New isolated observation-only full Hilti exp14 replay; 737 visual frames, 736 constrained. 9833 residual tests; 8301 accept branches and 1532 exact position rollbacks; all constrained frame positions crosschecked against state logs.',
             metrics='outputs/fast-livo2-visual-observer-001/metrics.json',validation='outputs/fast-livo2-visual-observer-001/validation.json',script='scripts/fast_livo2_linux/build_visual_observer_001.py',
             run='outputs/fast-livo2-hilti-gt-003/runs/exp14_basement_2_livo_20261013T085415Z_641c515c',isolated_linux_root='/opt/gs-slam-lab/fast-livo2-visual-observer-001',
             next='Add read-only final-return-state residual evaluation and whitened, scale-declared pose-information/gradient diagnostics. Use this same new run for independent reference comparisons before physically justified time/extrinsic controls.',
             guard='357 finest-level exits take a final step without reevaluating its residual. Pre-step acceptance is not final-pose improvement. Do not splice new residuals onto old run GT; printing can perturb scheduling. Baseline environment retained.'),
        dict(name='Large-scene GS SfM partition LOD',completed='Actual bounded LRU GPU parameter cache across all 117 fixed tau3 views; all RGB bitwise exact. 1507328 tree rows plus 100000 background. Resident raw+decoded parameters reduced 77.3628%; CPU independently reproduced every cache transition.',
             metrics='outputs/city-parameter-paging-001/metrics.json',validation='outputs/city-parameter-paging-001/validation.json',raw_run='outputs/large-scene-gs-001/research/city-parameter-paging-023',script='scripts/large_scene_gs/page_city_parameters_001.py',
             next='Connect actual online hierarchy selection and weights to the bounded cache; assess index/tree GPU storage vs CPU traversal. Then use a continuous camera path and test prefetch/transfer overlap; keep exact same-setting RGB controls.',
             guard='Frozen selection/weights are replayed; full source model stays in CPU RAM. Not disk streaming or complete online interactive LOD. Sparse heldout order and single cold-to-warm timing do not prove speedup. Mixed training lineage retained.')],
    website=dict(knowledge_updated=['knowledge/24_splatam_official_rgbd_system.md','knowledge/20_fast_livo2_fusion.md','knowledge/30_large_scene_gs_sfm_partition_lod.md'],
        evidence='experiments/34_three_track_mechanism_followup.md',url='http://localhost:8799/gs-lab/read/three-track-mechanism-followup',export_check='passed',sync_warnings=[],build='passed',
        http_audit='outputs/city-parameter-paging-001/website-http-audit.json',http_checks=checks['checks'],resources_hash_verified=checks['original_resource_hashes_verified'],changed_documents_katex_errors=0,
        main_chapters_single_collapsed_appendix=True,visual_review='New SplaTAM fixed-support and LOD memory formulas inspected in actual browser; FAST theory section present; evidence image displayed; all three new plots visually inspected. No self-drawn theoretical illustrations added.'),
    hardware='Existing RTX 5080 16GB and WSL sufficient for these actual experiments; no new hardware purchase or cloud instance required in this bounded scope.')
dest.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'saved':str(dest),'previous_preserved':str(previous),'http_checks':checks['checks']}))
