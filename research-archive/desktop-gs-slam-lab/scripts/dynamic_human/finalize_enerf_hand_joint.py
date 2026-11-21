"""Finalize matched local joint training after numeric and actual image review."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
import numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-joint-084'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=read(O/'comparison/summary.json');t=read(O/'temporal/summary.json');s=read(O/'spatial-diagnostic/summary.json')
files=read(O/'reviewed-files.json');assert len(files)==40 and len(set(files))==40
observations=[
    'Camera04 original palm opens away from the toy during728..732. All three rendered models retain fragmented skin and toy-colored wisps; neither candidate restores separated fingers.',
    'Camera13 hand back/wrist remains recognizable but hand/toy contact contains persistent black-white fragmentation, particularly728..732; no stable sharpening across the13frames.',
    'All12 hand/object panels at04/06/07/12 times721/727/731 inspected; full-person scale can hide hand defects. Large clothing/body shapes remain recognizable.',
    '12full-person panels include all unique top-three adverse heldout contexts for both candidates; no widespread change outside the active region is apparent. Exact frozen parameters and unchanged feet checked numerically.',
    '6target-right-hand native160px windows and8sequence review sheets viewed. The latter cover all13consecutive frames at each of04/13;2three-frame contact sheets also viewed. Videos are encoded at5fps for review, not a source capture-rate claim.',
]
review=dict(status='40actual_render_panels_viewed',files=[dict(path=p,sha256=sha(O/p)) for p in files],observations=observations,adopt_as_development_baseline=False,goal_complete=False)
save(O/'render-visual-review.json',review)
assert c['PNG_checks']==720 and c['max_recomputed_metric_difference']<1e-9
assert t['PNG_checks']==78 and len(t['videos'])==2
assert c['detail_summary']['both_person_feet']['changes_db']=={'joint_photo':0.,'joint_flow':0.}
training={}
for b in ['joint_photo','joint_flow']:
    f=read(O/b/'frozen-state-audit.json');m=read(O/b/'completion.json')
    assert f['frozen_Gaussians']==246654 and f['frozen_nodes']==362 and f['all_optimizer_steps']==3000
    assert sha(O/b/'resume.pt')==f['checkpoint_sha256']==c['checkpoints'][b]
    assert len(read(O/b/'native-evaluation/metrics.json')['records'])==120
    assert t['normal_flow_checks'][b]['count']==1527 and t['normal_flow_checks'][b]['max_recomputation_difference_px']<2e-6
    training[b]=m
a,b=[dict(np.load(O/br/'schedule.npz')) for br in ['joint_photo','joint_flow']]
assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
assert sha(L/'outputs/dynamic-human-enerf-second-interval-079/local_bspline/resume.pt')==c['checkpoints']['parent']
audit=dict(status='360native_predictions_720PNGs_plus78temporalPNGs_previously_recomputed_and_provenance_rechecked',native_PNG_checks=720,temporal_PNG_checks=78,max_native_metric_difference=c['max_recomputed_metric_difference'],normal_constraints_each=1527,identical_training_schedule=True,nonhand_Gaussians_exactly_frozen=246654,inactive_nodes_exactly_frozen=362,checkpoints=c['checkpoints'],runtime_smoke=read(O/'runtime-smoke.json')['status'],reviewed_files=40,goal_complete=False)
save(O/'numerical-audit.json',audit)
next_action='Preserve079baseline;084both localjoint branches rejected. Next inspect projected32local node proposal against source hand masks, then implement a zero-initialized hand-local residual graph applied only to selected6330Gaussians on top of original frozen motion. Verify exact initial RGB and inactive geometry before matched photo-only/normal-motion training. Keep120native eval,13frame two-camera videos, original/source-defined ROIs and masks. Do not repeat scalar weight sweeps or claim spatial density proves cause. Existing hand labels unreliable; proposal not anatomical ground truth. Fullstaticbackground and independentnewscene remain.'
summary=dict(status='matched_hand_joint_training_native_and_temporal_evaluation_complete_both_candidates_rejected',scene='ENeRF-Outdoor actor5_6',frames=[720,732],training=training,active_Gaussians=6330,active_existing_nodes=22,total_Gaussians=252984,heldout_comparison=c['splits'],detail_summary=c['detail_summary'],normal_flow_checks=t['normal_flow_checks'],consecutive_frame_temporal_residuals=t['temporal_residuals'],spatial_support=s,actual_render_panels_reviewed=40,independent_PNG_checks=798,
    decision='No stable fingertip clarity improvement. Joint photo is near parent; adding estimated normal motion reduces training residual while heldout RGB and hand/object regions regress. Both084branches rejected;079retained.',
    limitations=['All validation is same-scene development, not untouched final test.','Selected support is training-derived hand/wrist Gaussian material with possible fringe contamination, not anatomical truth.','Normal flow is estimated; source visibility recomputed each step but detached for gradient, projected Gaussian center motion is approximate.','32node spatial proposal is untrained and unreviewed in images; no causality or clarity claim.','Full scene background reconstruction and independent new scene validation remain incomplete.'],
    adopt_as_development_baseline=False,next_action=next_action,goal_complete=False)
save(O/'final-summary.json',summary)
for script in ['enerf_hand_joint_model.py','prepare_enerf_hand_joint_support.py','train_enerf_hand_joint.py','evaluate_enerf_hand_joint.py','render_enerf_hand_joint_temporal.py','audit_enerf_hand_joint_temporal.py','finalize_enerf_hand_joint.py']:
    shutil.copy2(L/'scripts/dynamic_human'/script,O/script)
p=L/'manifests/dynamic-human-work.json';m=read(p)
m.update(status='hand_joint_training_complete_both_candidates_rejected_clarity_unresolved',current_runs=[],outdoor_hand_joint='outputs/'+O.name+'/final-summary.json',next_action=next_action,goal_turn_classification='Progress:2x3000matchedjointupdates,6330Gaussians/22nodes;120nativecontexts/3models720PNG plus78temporalPNG/26frames/2videos;40panelsreviewed;spatialnodecoveragequantified. Negativeclarityresult retained.',updated_at=datetime.now(timezone.utc).isoformat())
m['completed'].append('084:2x3000handjointupdates;6330Gaussian parameters and22nodes updated,246654others frozen. NormalMAE1.252to0.731px but no heldoutclaritygain.798PNG and40panels verified;bothcandidates rejected. Existing handbbox has0graphnodes;32localnodes proposed, untrained.')
save(p,m);print('084 finalized: both candidates rejected;079 retained; goal active')

