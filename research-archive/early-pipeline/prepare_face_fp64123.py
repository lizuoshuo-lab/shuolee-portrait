"""Create one precision-only variant, retaining the original completed source."""
from pathlib import Path
import ast
import difflib

H=Path(__file__).parent
source=H/'refine_smplifyx_face_fixed_body123.py'
target=H/'refine_smplifyx_face_fp64123.py'
text=source.read_text(encoding='utf-8')
assert not target.exists()
prepare_node=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name=='prepare')
new_prepare='''def prepare():
    assert not PLAN.exists() and not OUT.exists()
    old_plan=BODY/'face-fixed-body-refinement-plan.json'
    old_result=BODY/'face-fixed-body-refinement/result.json'
    stationarity=BODY/'face-fixed-body-refinement/stationarity-audit/result.json'
    p=read(old_plan);s=read(stationarity)
    assert s['frames']==59 and s['summary']['optimizer_gradient_pass_count']==0
    assert read(old_result)['frames']==59
    for f,h in {**p['source_hashes'],**p['input_hashes']}.items():assert sha(f)==h,f
    p.update(updated=now(),state='prepared',hypothesis='Float32 objective rounding may stop the original local-face LBFGS before its gradient condition. Change only numerical dtype tofloat64, from the same pre-face-stage saved states, to test this.',
        intervention='Numerical dtype only: original SMPL-X, original final loss and LBFGS, original tolerances/maxiters, same59 initial states and fresh optimizers. This remains a local face-only stage adaptation, not a full official pipeline.',
        computation_dtype='float64',comparison_dtype='float32',comparison_result=str(old_result),
        camera_dtype_contract='Full K and camera parameters unchanged. Computation usesfloat64; old rounded K_compute_float32 is retained as input_K_compute_float32, not claimed to be the new computation matrix.',
        runtime_budget_seconds=600,expected_seconds=240,
        primary_success='All59 fixed face/body/feet protections, elbow preservation and native visual review. Report numerical limb invariant separately; no gate relaxation or automatic downstream admission.',
        failure_stop='One completed uniform59 call only. Do not increase iterations, vary tolerances/weights, select frames or rerun to chase success.',
        source_of_next_hypothesis=str(stationarity))
    p['input_hashes'].update({str(f):sha(f) for f in (old_plan,old_result,stationarity)})
    p['source_hashes'].pop(str(Path(__file__).with_name('refine_smplifyx_face_fixed_body123.py').resolve()),None)
    p['source_hashes'][str(Path(__file__).resolve())]=sha(__file__)
    p['source_hashes'][str(Path(__file__).with_name('refine_smplifyx_face_fixed_body123.py').resolve())]=sha(Path(__file__).with_name('refine_smplifyx_face_fixed_body123.py'))
    p['source_hashes'][str(b.OFFICIAL/'smplifyx/optimizers/lbfgs_ls.py')]=sha(b.OFFICIAL/'smplifyx/optimizers/lbfgs_ls.py')
    save(PLAN,p);print('PREPARED_FP64_FACE_ONLY_ALL59',flush=True)
'''
lines=text.splitlines(keepends=True)
changed=''.join(lines[:prepare_node.lineno-1])+new_prepare+''.join(lines[prepare_node.end_lineno:])
changed=changed.replace("OUT=BODY/'face-fixed-body-refinement'","OUT=BODY/'face-fixed-body-fp64'")
changed=changed.replace("PLAN=BODY/'face-fixed-body-refinement-plan.json'","PLAN=BODY/'face-fixed-body-fp64-plan.json'")
changed=changed.replace("HANDOFF=BODY/'face-fixed-body-refinement-gpu-handoff.json'","HANDOFF=BODY/'face-fixed-body-fp64-gpu-handoff.json'")
for old,new in (
    ('torch.float32','torch.float64'),
    ("torch.tensor(saved['body_pose_axis_angle'],device='cuda')","torch.tensor(saved['body_pose_axis_angle'],dtype=torch.float64,device='cuda')"),
    ("torch.tensor(saved['pose_embedding'],device='cuda')","torch.tensor(saved['pose_embedding'],dtype=torch.float64,device='cuda')"),
    ("torch.tensor(saved['camera_translation'],device='cuda')","torch.tensor(saved['camera_translation'],dtype=torch.float64,device='cuda')"),
    ("torch.tensor(aggregate['observed_native_keypoints'][ids.index(frame):ids.index(frame)+1],device='cuda')","torch.tensor(aggregate['observed_native_keypoints'][ids.index(frame):ids.index(frame)+1],dtype=torch.float64,device='cuda')"),
    ("torch.full((1,118),1. if frame=='frame_000033' else 2.,device='cuda')","torch.full((1,118),1. if frame=='frame_000033' else 2.,dtype=torch.float64,device='cuda')"),
    ("actual={k:v.copy() for k,v in saved.items()}","actual={k:v.copy() for k,v in saved.items()}\n                actual['input_K_compute_float32']=actual.pop('K_compute_float32')\n                actual['K_compute_float64']=K.detach().cpu().numpy().copy()\n                actual['computation_dtype']=np.array('float64')"),
    ("info=dict(frame=frame,source_sha256=", "info=dict(frame=frame,computation_dtype='float64',source_sha256="),
):
    assert old in changed,old
    changed=changed.replace(old,new)
ast.parse(changed)
target.write_text(changed,encoding='utf-8',newline='\n')
(H/'face-fp64-only123.patch').write_text(''.join(difflib.unified_diff(text.splitlines(keepends=True),changed.splitlines(keepends=True),fromfile=source.name,tofile=target.name)),encoding='utf-8')
print('CREATED_PRECISION_ONLY_VARIANT',target)
