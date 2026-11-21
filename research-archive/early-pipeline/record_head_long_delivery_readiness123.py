import shutil
import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now
root=a.REC/'demo-vhap-initialization';long=root/'head-gaussian-long300k';snap=long/'gaussian-run/step-060000'
r=a.read(snap/'result.json');metrics=a.read(snap/'review/metrics.json')
a.save(snap/'visual-review.json',dict(updated=now(),actual_all59_review=True,reviewed_sheets=5,native_scale=1,reviewed_frames=59,
    world_ply_sha256=r['world_ply_sha256'],opacity_reset_checkpoint=True,quality_accepted=False,continuous_novel_views_reviewed=0,
    summary_zh='已审阅60k全部59帧：快照紧接透明度重置，帽脸整体变暗；不作为稳定收敛质量对比。继续当前授权长训练。',
    comparable_to_stable_18k=False,metrics_source=str(snap/'review/metrics.json')))
record=a.lp('D:/00/记录/demo')
for f in a.lp(snap/'review/frames').glob('*.png'):shutil.copy2(f,record/('高斯_长训练300k_60000步_透明度重置后_'+f.name))
p=a.read(long/'progress.json')
files=[a.H/x for x in ['export_demo_head_gaussians123.py','verify_demo_head_export123.py','package_demo_head_gaussians123.mjs','head-serialized-camera-check123.json','head-final-package-camera-gate-check123.json','head-long-delivery-preparation123.json']]
prep=dict(updated=now(),stage='long300k_delivery_prepared_and60k_reviewed',
    helper_hashes=[dict(path=str(f),sha256=a.sha(f)) for f in files],
    current_live_training=dict(pid=p['pid'],step=p['step'],points=p['points'],progress=str(long/'progress.json')),
    modified_running_source=False,GPU_export_or_replay_executed=False,new_PLY_or_ZIP_generated=False,
    CPU_serialized_camera_controls=9,CPU_final_package_gate_controls=5,
    original_interface_proposal='Offline only; not a drop-in trainer replacement, no second training authorized by this record.',
    camera_check_change='Final native replay now consumes actual cameras.json and binds its SHA; old replay evidence is not upgraded by CPU parser checks.',
    final_sequence=['export_demo_head_gaussians123.py --run head-gaussian-long300k --step 300000',
                    'verify_demo_head_export123.py --run head-gaussian-long300k --step 300000',
                    'package_demo_head_gaussians123.mjs 300000 --run head-gaussian-long300k'],
    next_action='Wait verified live300k job; next scheduled100k checkpoint is not an opacity-reset step. Review stable native59 frames then continue authorized job. After completion/native replay/UI inspect, publish paired final result.',
    full_pipeline_15min_accepted=False,quality_accepted=False)
a.save(root/'long-delivery-readiness.json',prep)
for path in [a.R/'queue-state.json',a.REC/'queue-state.json',a.C/'local-research-status.json']:
    d=a.read(path);d['latest_long_delivery_preparation']=prep
    if 'latest_demo_head_gaussian' in d:d['latest_demo_head_gaussian'].update(updated=now(),step=p['step'],points=p['points'],training_running=True)
    a.save(path,d)
ledger=a.R/'evidence-ledger.json';d=a.read(ledger)
entry=dict(updated=now(),stage=prep['stage'],source=str(root/'long-delivery-readiness.json'),
    conclusion='60k all59 native frames reviewed as post-opacity-reset; no stable quality claim. Delivery helpers support distinct long run and actual serialized camera replay; CPU guards passed. Training unchanged and live.',goal_complete=False)
if isinstance(d,list):d.append(entry)
else:d.setdefault('entries',[]).append(entry)
a.save(ledger,d)
print(dict(step=p['step'],points=p['points'],pid=p['pid'],quality_accepted=False,next_quality_checkpoint=100000))
