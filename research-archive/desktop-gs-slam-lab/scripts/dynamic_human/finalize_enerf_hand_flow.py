from pathlib import Path
import json,hashlib,shutil
from datetime import datetime,timezone
import numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-open-hand-flow-083'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=read(O/'numerical-audit.json');c=read(O/'comparison/summary.json');v=read(O/'render-visual-review.json');vd=read(O/'visibility-diagnostic/summary.json')
assert len(v['files'])==36 and not v['adopt_as_development_baseline'];assert all(sha(O/r['path'])==r['sha256'] for r in v['files'])
visible_checks={}
for b,r in vd['branches'].items():
    arrays=[dict(np.load(O/'visibility-diagnostic'/f'{b}-camera{x["camera"]}-frame{x["source_frame"]:06d}-to{x["target_frame"]:06d}.npz')) for x in r['records']]
    for k in arrays[0]:
        value=np.concatenate([x[k] for x in arrays]);assert len(value)==1527
        expected=r['statistics'][k]
        for metric,recomputed in [('mean',value.mean()),('median',np.median(value)),('p90',np.quantile(value,.9)),('maximum',value.max())]:assert abs(float(recomputed)-expected[metric])<1e-6
    visible_checks[b]=dict(count=1527,arrays=len(arrays),current_mean_normal_error_px=r['statistics']['current_normal_error']['mean'])
save(O/'visibility-diagnostic/numerical-audit.json',dict(status='all129arrays_and_saved_visibility_statistics_recomputed',branches=visible_checks))
next_action='Preserve079baseline.083normal-flow-only graph intervention rejected by108heldoutRGB and12hand/objectwindows; current-visibility rerender still reduces motionresidual, so do not blame fixedweights alone orrepeat scalarweight sweeps. Next test bounded hand-local joint geometry/appearance/motion adaptation fromsame079parent withmatched photo-onlycontrol, source-defined localGaussian support and unchanged120nativeeval/splits. Recompute currentvisibility forflow andkeep nonhand/body/cameras/clocks frozen; verify activehandregion first. Need actualRGB/finger improvement, notonly1527training residuals. Do not promote082unreliable finger labels. Full scene/background andnewindependentscene remain outstanding.'
out=dict(status='matched_hand_normal_flow_training_and_native_evaluation_complete_candidate_rejected',scene='ENeRF-Outdoor actor5_6',frames=[720,732],training=dict(branches=2,additional_steps_each=1500,parent_steps=15000,Gaussian_points=252984,active_graph_nodes=27,inactive_graph_nodes=357,allGaussian_parameters_frozen=True,training_motion_constraints=1527,training_motion_pairs=43),
    normal_error_px={b:{k:r[k] for k in ['mean_absolute_error_px','median_px','p90_px']} for b,r in a['branches'].items()},normal_error_reduction_fraction=a['normal_error_reduction_fraction'],
    heldout_comparison=c['splits'],detail_windows=a['detail_windows'],temporal_residuals=c['temporal_residuals'],current_visibility_diagnostic=visible_checks,
    actual_render_panels_reviewed=36,independent_PNG_checks=480,native_resolution=[1920,1080],
    decision='Motion training residual improves but heldoutRGB andhand/objectwindows regress; handghosts/toyfragmentation persist. Neither083branch replaces079baseline. Same-scene development, not independent new-scene acceptance. Feet numerically unchanged.',
    limitations=['Qualified normal flow is estimated from photographs, not ground-truth flow. Mostly handback/wrist; frontal open-palm04 only2points.','Only localgraphmotion was optimized; canonical geometry,colors,opacity,covariance remainlocked.','Sourcevisibility wasfixed inloss; read-onlycurrentvisibility stillshowsmotionresidualreduction butdoesnotvalidatecorrectsurfacegeometry.','Fullbackground+human jointscene notcompleted.'],
    adopt_as_development_baseline=False,next_action=next_action,goal_complete=False)
save(O/'final-summary.json',out)
for r in ['infer_enerf_open_hand_raft.py','prepare_enerf_open_hand_flow.py','segment_enerf_open_hand.py','prepare_enerf_hand_normal_flow.py','approve_enerf_hand_normal_flow.py','enerf_hand_flow_model.py','diagnose_enerf_hand_flow_visibility.py','finalize_enerf_hand_flow.py']:
    shutil.copy2(L/'scripts/dynamic_human'/r,O/r)
p=L/'manifests/dynamic-human-work.json';m=read(p);m.update(status='hand_normal_flow_training_completed_candidate_rejected_clarity_unresolved',current_runs=[],outdoor_hand_normal_flow='outputs/'+O.name+'/final-summary.json',next_action=next_action,goal_turn_classification='Progress: official RAFT,45handmaskcrops,1527qualifiednormalconstraints;matched2x1500graphupdates;240nativeRGBand480PNGsverified;36renderpanelsviewed;currentvisibilitydiagnostic129arrays. Negativeclarityresult retained.',updated_at=datetime.now(timezone.utc).isoformat());m['completed'].append('083:2x1500matchedgraphupdates/27nodes,1527normalconstraints;motionMAE1.245to0.807px butheldoutPSNR-0.107/-0.052dB andhandwindows-0.243dB;36renderpanels reviewed,480PNGs audited;candidate rejected,079baseline retained.');save(p,m)
print('083 finalized; candidate rejected; original baseline preserved; goal active')
