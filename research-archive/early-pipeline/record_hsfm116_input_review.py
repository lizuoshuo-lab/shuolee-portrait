from pathlib import Path
import datetime as dt,json
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116')
H=O/'hsfm';now=dt.datetime.now(dt.timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
review=dict(updated=now,state='visually_reviewed_input_identity_only',
    inspected=['body23-observations-1.jpg','body23-observations-2.jpg','body23-observations-3.jpg'],
    all13_target_id_correct=True,
    findings=['All panels place the target detections on the red-jacket person, not the other people in the scene.',
        'Front views roughly locate visible shoulders, wrists, knees and footwear; no precise pixel accuracy is established by these overview sheets.',
        'In side views the far arm, hip and leg are partly hidden; their inferred joints are not visible material-point measurements.',
        'Background includes different moving people. Entire background cannot be treated as a guaranteed rigid match set.',
        'These sequential frames do not establish a perfectly constant body pose.'],
    RGB_preprocess_review='rtmlib0.0.16 RTMPose.preprocess and BaseTool.inference do not reorder channels; the PIL RGB input matches the ONNX package RGB normalization.',
    mapping_verified='COCO body23 names only; no face133 material identity or surface-boundary accuracy claim',
    next='Use visible, consistently named observations with an explicit pre-frozen visibility policy; retain uncertainty rather than forcing all23 as exact constraints.',
    physical_camera_accuracy_verified=False,quality_accepted=False)
save(H/'input-pilot/visual-review.json',review)
r=json.loads((H/'input-pilot/result.json').read_text(encoding='utf-8-sig'))
r.update(state='fresh_input_observations_identity_reviewed',visual_review='visual-review.json',updated=now)
save(H/'input-pilot/result.json',r)
q=json.loads((O/'queue-state.json').read_text(encoding='utf-8-sig'))
q.update(updated=now,state='P3_native_forward_launch_prepared',stage='S1_P3_native_interface_smoke',
    active_pid=None,active_process_kind='bounded_native_DUSt3R_forward_pending_launch',
    active_log=str(H/'native-forward.log'),active_stderr=None,
    runtime_evidence='hsfm/native-model-check.json',
    next_action='Run exactly one fixed4-directed-pair native forward under300s timeout. No camera/body optimization. Then visually review native outputs and register the single paired joint/scene diagnostic. No new large downloads.')
q['no_repeat']=list(dict.fromkeys(q['no_repeat']+['P3_DUSt3R_download','P3_CPU_adapter_fixtures','P3_HMR2_strict_load','P3_RTMw_schema_check','P3_native_checkpoint_strict_load','P3_RTMw_input_inference']))
save(O/'queue-state.json',q)
print('Input review saved; queue prepared for one bounded native forward.')
