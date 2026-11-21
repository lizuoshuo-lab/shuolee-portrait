"""Register one new matcher variable on existing native pixels and queries."""
from pathlib import Path
import itertools,numpy as np
import match_video10_temporal_bridge123 as bridge
b=bridge.base;R=b.R;B=bridge.B;O=bridge.O;N=R/'video10-semantic-role';Q=R/'video10-correspondence-contrast'
assert not Q.exists(), 'Read registered state instead of replacing protocol'
g=b.read(N/'graph-audit-result.json');assert not g['whole_graph_count_gate']
triplets=[[484,510,537],[644,671,698]]
pairs=sorted({p for tri in triplets for p in itertools.combinations(tri,2)})+[(806,859)]
frames=sorted(set(sum([list(p) for p in pairs],[])))
inputs={}; counts={}
for f in frames:
    fp=(B if f in b.FRAMES else O)/'run'/f'features_{f:06d}.npz'
    for p in [fp,bridge.imgpath(f)]:inputs[str(p)]=b.sha(p)
    with np.load(fp) as d:counts[str(f)]=len(d['uv'])
for name in ['secondary-observations.json','secondary-families.json','material-visual-review.json','material-support.json']:
    inputs[str(N/name)]=b.sha(N/name)
weights=b.read(R.parent/'portrait-alignment-open-source-116/roma/weight-acquisition.json')['files']
Q.mkdir()
plan=dict(registered=b.now(),state='registered_runtime_preflight_pending',
    motivation='Pixel semantics restored candidate eligibility but113family review still leaves a disconnected support graph. Test whether a learned image-pair matcher differs from the frozen RootSIFT correspondences at the exact same query locations.',
    new_causal_variable='Correspondence operator only: local RootSIFT matching versus existing official RoMa Outdoor+DINOv2. No feature detection, image processing, K, poses, 3D or body variables change.',
    subject='video10.mp4',native_size_wh=[2704,1520],frames=frames,triplets=triplets,pairs=[list(p) for p in pairs],
    pair_selection='Two existing poorly supported temporal triplets484/510/537 and644/671/698;806/859 positive coarse-material control. Chosen before RoMa outputs. Not a random generalization benchmark.',
    input_sha256=inputs,queries=dict(policy='Every existing selected SIFT uv in both images, without relocalizing sources or choosing by RoMa score',per_frame_counts=counts),
    weights=weights,weight_root='C:/Users/Administrator/Downloads/shuolee-roma116',
    runtime='/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python',
    source_paths=['/opt/gs-slam-lab/shuolee-roma116/python','/opt/gs-slam-lab/shuolee-roma116/RoMa'],
    model=dict(entry='roma_outdoor',coarse_resolution_hw=[448,798],upsample_resolution_hw=[756,1344],
        reason='Landscape transpose of the previously validated portrait aspect-preserving budget. Network rescaling is internal only; predictions map back to original2704x1520 integer pixel centers.',
        symmetric=True,use_custom_corr=False,do_compile=False,amp_dtype='float16',matmul_precision='highest'),
    coordinate_adapter='C:/Users/Administrator/Documents/New project/shuolee-slide-update/roma116_coordinates.py',
    coordinate_adapter_sha256=b.sha(Path(__file__).parent/'roma116_coordinates.py'),
    frozen_gate=dict(min_bidirectional_confidence=.9,max_fb_native_px=1.0,max_threeview_cycle_native_px=1.0),
    lineage='All existing FIT/HELD/quarantine identities remain frozen. New endpoints do not become independent GT. Any endpoint connecting old FIT andHELD families is quarantined for future use; do not move HEld to FIT. Raw predictions retained, including failed/unknown.',
    visual_review=dict(max_representatives=72,max_context_rows=16,
        selection='Before inference, deterministic per-pair native3x3 cells select up to8existing background queries by SHA with1HELD+1FIT per cell when available, capped8; add up to2existing headROI queries by frozenfamilySHA. Include all pre-reviewed control correspondences on806/859 within cap. Review both accepted and rejected predictions. Sharedqueries deduplicated peroldfamily. Unknown material remains unknown.',
        no_residual_selection=True,threshold_search=False),
    budget=dict(attempts=1,pairs=7,wall_seconds=900,GPU_tasks=1,new_weights=0,new_features=0,camera_operations=0,training_steps=0),
    stop='Hash/coordinate mismatch, runningGPU task, deadline, memory error or absent local model: preserve evidence and stop. No download/threshold/repeated inference. After bounded review, inadequate physical/graph support ends this matcher candidate, not a new automatic parameter sweep.',
    interpretation='Confidence, round-trip and cycle residuals test self-consistency only. Even high retained counts cannot establish fullcamera registration, <1pxfacephysicalaccuracy or training admission.',
    material_truth='Prior review provides coarse local recognizability, not precise physical point GT. No negative truth is invented from ambiguous dartboard/tiles. Dynamic display is invalid for fixedmaterial even if location matches.')
b.save(Q/'plan.json',plan)
b.save(Q/'progress.json',dict(updated=b.now(),state='registered_no_inference',pid=None,pairs=7,training_started=False))
print(dict(registered=True,frames=frames,query_counts=counts,weights_not_downloaded=True,inference_started=False))
