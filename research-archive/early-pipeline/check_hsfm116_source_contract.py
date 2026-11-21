"""CPU-only source contract fixtures. No candidate fitting or model weights."""
from pathlib import Path
import ast
import datetime as dt
import hashlib
import json
import time
import numpy as np
import PIL.Image
import torch

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/hsfm')
S=R/'source'
now=dt.datetime.now(dt.timezone.utc).isoformat()
def save(name,obj): (R/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in S.iterdir() if p.is_file()}
assert len(sources)==10
save('source-check-protocol.json',dict(created=now,scope='CPU source contract fixtures only, not an HSfM fit',
    source_sha256=sources,gpu=False,model_weights=False,source_images_modified=False,
    hypotheses=[
        'Hard-coded initialization principal point assumes landscape; test portrait dimensions without changing source.',
        'COCO first body keypoint is nose while root_transl is consumed as pelvis; test exact initializer on constant-depth known skeleton.',
        'Bounding boxes are stored as xyxy but loss normalization uses x2*y2; test translation-invariant equal-size boxes.',
        'Shared human parameters are projected to all cameras; no per-view dynamic human is introduced.'
    ],
    stop_on='Any unexpected source SHA or fixture error: stop and record, no full pipeline launch',
    budget_seconds=120,planned_next='If compatibility fails, register minimal isolated adapter fixes before runtime; do not call these failures the historical ghosting cause.'))
t0=time.monotonic()
def extract(path,names,ns):
    tree=ast.parse(path.read_text(encoding='utf8'))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert {x.name for x in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

ns=extract(S/'align_world_env_and_smpl_hsfm_optim.py',['estimate_initial_trans'],dict(torch=torch))
# All selected joints have the same depth, so the scale estimate is exact.
relative=torch.tensor([[0.,.60,0.],[.20,.40,0.],[-.20,.40,0.],[.15,-.45,0.],[-.15,-.45,0.]],dtype=torch.float64)
true_root=torch.tensor([0.,0.,3.],dtype=torch.float64)
f=torch.tensor(500.,dtype=torch.float64);c=torch.tensor([144.,256.],dtype=torch.float64)
cam=relative+true_root
obs=torch.cat([f*cam[:,:2]/cam[:,2:]+c,torch.ones((len(cam),1),dtype=torch.float64)],dim=1)
pred=ns['estimate_initial_trans'](relative,obs,f,c,[(0,1),(0,2),(1,3),(2,4)],device='cpu')
root_error=float(torch.linalg.norm(pred-true_root))
assert root_error>.59 and root_error<.61
joint_tree=ast.parse((S/'joint_names.py').read_text(encoding='utf8'))
names={}
for node in joint_tree.body:
    if isinstance(node,ast.Assign) and isinstance(node.value,ast.List):
        for target in node.targets:
            if isinstance(target,ast.Name):names[target.id]=ast.literal_eval(node.value)
assert names['COCO_WHOLEBODY_KEYPOINTS'][0]=='nose'
assert names['ORIGINAL_SMPLX_JOINT_NAMES'][0]=='pelvis'

ns2=extract(S/'get_world_env_dust3r_for_hsfm.py',['preprocess_and_get_transform'],dict(PIL=PIL,np=np))
sample=R/'portrait-size-fixture.png'
PIL.Image.new('RGB',(1520,2704),(127,127,127)).save(sample)
crop,A=ns2['preprocess_and_get_transform'](str(sample))
center=np.array(crop.size,dtype=float)/2
hardcoded=np.array([256.,144.])
# Matrix returned by source maps processed coords to original-image coords.
resized=np.rint(np.array([1520,2704])*512/2704).astype(int)
pixel_center_offset=(np.array([1520,2704])/resized-1)/2
boxes=np.array([[10.,20.,110.,220.],[110.,120.,210.,320.]])
code_area=boxes[:,2]*boxes[:,3]
actual_area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
assert actual_area[0]==actual_area[1] and code_area[0]!=code_area[1]
text=(S/'align_world_env_and_smpl_hsfm_optim.py').read_text(encoding='utf8')
assert 'smplx_j3d.unsqueeze(1).expand(-1, N, -1, -1)' in text
assert 'bbox_area = bbox[2] * bbox[3]' in text
assert 'init_princpt = [256., 144.]' in text
assert all(hashlib.sha256((S/n).read_bytes()).hexdigest()==h for n,h in sources.items())
result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),state='source_contract_issues_confirmed_CPU_only',elapsed_seconds=time.monotonic()-t0,
    official_runtime_verified=False,official_helper_executed_on_CPU=True,full_HSfM_fit=False,gpu_used=False,
    principal_point=dict(original_wh=[1520,2704],processed_wh=list(crop.size),hardcoded_xy=hardcoded.tolist(),image_midpoint_xy=center.tolist(),delta_processed_px=(hardcoded-center).tolist(),
        limit='Image midpoint is a dimension sanity check, not a measured physical principal point. Must use each camera K in the processed image convention.'),
    root_initialization=dict(coco_first_joint='nose',template_root_joint='pelvis',true_root=true_root.tolist(),returned_init=pred.tolist(),error_fixture_m=root_error,
        explanation='Exact equal-depth synthetic skeleton exposes nose anchor used as pelvis translation; later optimization may compensate. Not a measured subject error.'),
    bbox_weight=dict(boxes_xyxy=boxes.tolist(),source_area=code_area.tolist(),correct_area=actual_area.tolist(),
        explanation='Equal-size boxes receive different source loss normalization when translated; xyxy has not become xywh in this call path.'),
    pixel_center_review=dict(source_inverse_affine=A.tolist(),PIL_center_aware_inverse_translation_offset_original_px=pixel_center_offset.tolist(),
        state='Requires explicit source-vs-target pixel convention fixture before integration; do not infer historical half-pixel error.'),
    human_parameter_sharing='One body/shape/root per person, same3D joints projected to all views. Static shared-person assumption compatible with task only if observational micro-motion is bounded.',
    current_data_assumptions_unresolved=['Physical source K/R/t','Staticity across sequential capture','Independent face/glasses material identity'],
    historical_ghosting_cause_established=False,
    next='Prepare explicit local adapter requirements and known dependency inputs before runtime. Preserve official files unchanged; do not run unmodified run_hsfm.sh or download additional large weights automatically.')
assert result['elapsed_seconds']<120
save('source-contract-check.json',result)
print(json.dumps(result,ensure_ascii=False))
