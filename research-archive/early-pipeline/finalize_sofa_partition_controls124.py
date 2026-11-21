"""Record both rejected parameter controls and the all-person anchor audit."""
from datetime import datetime,timezone
import audit_gaussianavatars_params123 as a

root=a.R.parent/'local-clothed-surface-123/cross-subject-sh0-124'
run=root/'sofa-temporal-warmstart-v2'
stamp=datetime.now(timezone.utc).isoformat()
controls={}
for label,folder in [('frozen','body-protected-final-stage-control'),('local','local-head-wrist-final-stage-control')]:
    directory=run/folder
    r=a.read(directory/'result.json');v=a.read(directory/'visual-review.json')
    assert not r['face_hand_comparative_gate_passed']
    assert all(x==0 for x in r['prefix_parity'].values())
    assert all(r['final_body_parameters_unchanged'].values())
    if not v['actual_visual_review']:
        v.update(updated=stamp,actual_visual_review=True,state='reviewed_rejected_face_and_hand_regression',
            conclusion='All four native unscaled body/face/hands/feet comparisons actually viewed. Knee improved to the original stage3 result, but face and hands regress against the saved all-parameter endpoint. Both registered regional protection checks failed. Preserve this control; no full sequence or Gaussian.',
            quality_accepted=False,gaussian_admitted=False,full_sequence_admitted=False)
        a.save(directory/'visual-review.json',v)
    controls[label]=dict(result=str(directory/'result.json'),visual_review=str(directory/'visual-review.json'),
        metrics=r['metrics'],prefix_exact=True,body_parameters_unchanged=True,actual_visual_review=True,
        final_stage_parameters=r['actual_final_parameters'],images=v['images'],accepted=False)
duplicate_path=root/'duplicate-wrist-anchor-audit/result.json'
duplicate=a.read(duplicate_path)
stats={s:{k:v for k,v in row.items() if k!='records'} for s,row in duplicate['subjects'].items()}
value=dict(updated=stamp,state='both_partition_controls_rejected_input_contract_conflict',controls=controls,
    duplicate_wrist_audit=str(duplicate_path),duplicate_wrist_summary=stats,
    summary='两项参数隔离对照均未通过。固定身体保住左膝（101.23→30.66px），但脸/手误差4.19/6.52→8.37/21.78px；开放头颈与手腕后为4.55/11.62px，仍未通过保护。随后只读核查四人物373帧，744组有效的身体手腕／手部根点都指向同一个3D关节、却相差超过2px。第16帧左右差24.56/12.34px，且两套置信度均高。不能继续靠扩大姿态自由度消化这个输入冲突。',
    limits=['The1px contradiction refers only to fitting both duplicate wrist targets simultaneously, not a claim about the facial1px requirement.',
        'Multiple noisy observations of one3D point are not inherently invalid. Current relative confidences do not establish independence, calibration or which2D location is physically correct.',
        'Neither the duplicate audit nor these parameter controls uniquely explain all pose errors or ghosting.',
        'Local joint parameters have not been integrated into full-sequence temporal seeding or the Gaussian pipeline; do not deploy them.'],
    quality_accepted=False,gaussian_admitted=False,current_pid=None,
    next_action='Both matched final-stage parameter controls are complete and rejected. Do not rerun them, loosen their gates, expand joint subsets automatically or start a full sequence/GS. Read duplicate-wrist-anchor-audit/result.json and the original RTMW/COCO-WholeBody-to-SMPL-X mapping semantics. Prepare one deterministic unique-anchor observation contract (canonical source/fallback explicitly defined before fitting), retaining all original2D coordinates and a separate audit of the unused duplicate; no residual-based point selection or warping. Compare any proposed contract with the earlier winterhat12 hand-confidence control rather than repeating it. Register a new causal input-only test and all-region protections before execution. Keep existing temporal weights, original optimizer and current demo SH0 model; the four-person goal remains active.')
dest=run/'parameter-partition-conclusion.json'
assert not a.lp(dest).exists();a.save(dest,value)
print(value['summary'])
