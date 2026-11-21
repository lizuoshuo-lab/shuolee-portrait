"""CPU-only completion of already saved CUDA trajectory; no second solve."""
import os
from pathlib import Path
import numpy as np
from PIL import Image
import audit_sofa_temporal_objective124 as a
from run_official_gaussians123 import read,save,sha,now
from review_full_sequence123 import mesh_renderer,crop_box

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
out=a.RUN/'stage33-diagnostic-replay'
assert not (out/'result-completed.json').exists()
failure=read(out/'failure.json')
assert failure['error']=="TypeError(\"VPoser.decode() got an unexpected keyword argument 'output_type'\")"
parity=read(out/'endpoint-parity.json')
assert all(v==0 for v in parity['differences'].values())
inputs=[out/'trajectory.npz',out/'endpoint-parity.json',out/'plan.json',out/'failure.json',a.RUN/'official-fit/frames/frame_000032/winner-camera-space.npz']
save(out/'cpu-completion-plan.json',dict(updated=now(),reason='One CUDA replay finished with exact endpoint parity. Only a later diagnostic decode used the wrong API. Recover stage boundaries and rotations from saved arrays, without rerunning the fit.',
    new_fitter_calls=0,gpu_tasks=0,limitation='Per-step loss/stop-reason metadata was not serialized before the diagnostic exception. Do not infer those values.',
    source_sha256=sha(__file__),frozen={str(p):sha(p) for p in inputs}))
with np.load(out/'trajectory.npz',allow_pickle=False) as z:data=dict(z)
with np.load(inputs[-1],allow_pickle=False) as z:previous=dict(z)
import torch
from smplx.lbs import batch_rodrigues
torch.set_num_threads(1)
loader=a.entry.module(a.PROJECT/'src/avatar_core/shuolee_vposer.py','_trace_completion124_vposer')
loader.MODEL,loader.CODE=a.ASSETS/'vposer/V02_05',a.PROJECT/'third_party/human_body_prior'
vposer,_,_=loader.load_vposer('cpu');vposer.eval()
indices=sorted(int(k.split('_')[1]) for k in data if k.endswith('_pose_embedding'))
boundaries=sorted(int(k.split('_')[1]) for k in data if k.endswith('_vertices_camera'))
assert indices==list(range(len(indices))) and len(boundaries)==12
pairs=list(zip(boundaries[::2],boundaries[1::2]))
assert pairs[0][0]==0 and pairs[-1][1]==indices[-1]
assert all(b+1==pairs[i+1][0] for i,(_,b) in enumerate(pairs[:-1]))
assert np.array_equal(data[f'state_{pairs[1][0]:03d}_pose_embedding'],previous['pose_embedding'])
with torch.no_grad():
    latent=torch.from_numpy(np.concatenate([data[f'state_{i:03d}_pose_embedding'] for i in indices]))
    poses=vposer.decode(latent)['pose_body'].reshape(-1,21,3)
    rotations=batch_rodrigues(poses.reshape(-1,3)).reshape(-1,21,3,3)
    ref=batch_rodrigues(torch.from_numpy(previous['body_pose_axis_angle'].reshape(-1,3)))
    tr=(ref[None].transpose(-1,-2)@rotations).diagonal(dim1=-2,dim2=-1).sum(-1)
    angles=torch.rad2deg(torch.acos(((tr-1)/2).clamp(-1,1))).numpy()
records=[]
row=read(a.SRC/'data/inputs.json')['records'][32];raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
box=np.array(crop_box(mask.getbbox(),raw.size,30));K=np.array(row['K_integer_centers']);render,_=mesh_renderer()
for index,(before,after) in enumerate(pairs):
    record=dict(kind='camera' if index==0 else 'body',stage=index-1,before_state=before,after_state=after,
        recorded_optimizer_steps=after-before-1,
        right_hip_from32_before_deg=float(angles[before,1]),right_hip_from32_after_deg=float(angles[after,1]),
        right_knee_from32_before_deg=float(angles[before,4]),right_knee_from32_after_deg=float(angles[after,4]),
        latent_squared_before=float(latent[before].square().sum()),latent_squared_after=float(latent[after].square().sum()))
    if index:
        parts=[raw.crop(box)]
        for i in (before,after):parts.append(render(raw.crop(box),box,dict(vertices_camera=data[f'state_{i:03d}_vertices_camera'],faces=data['faces'],K_integer_centers=K)))
        for i in (before,after):
            v=data[f'state_{i:03d}_vertices_camera'][0];center=(v.min(0)+v.max(0))/2;extent=float(np.max(np.ptp(v,axis=0)))
            rot=np.array([[0,0,1],[0,1,0],[-1,0,0]])
            vv=(v-center)@rot.T;vv[:,2]+=3*extent
            parts.append(render(Image.new('RGB',(340,560),'black'),np.array([0,0,340,560]),dict(vertices_camera=vv[None],faces=data['faces'],K_integer_centers=np.array([[1250,0,170],[0,1250,280],[0,0,1]]))))
        panel=Image.new('RGB',(1700,560),'black')
        for j,im in enumerate(parts):im.thumbnail((340,560));panel.paste(im,(j*340+(340-im.width)//2,0))
        path=Path('/mnt/d/00/记录/006')/f'smpl-x_连续帧124_frame000033_stage{index-1}_原图_阶段前后mesh_阶段前后侧面.png'
        assert not path.exists();panel.save(path);record['image']=str(path)
    records.append(record)
save(out/'result-completed.json',dict(updated=now(),state='diagnostic_complete_pending_visual_review',original_failure_preserved=True,
    endpoint_parity_passed=True,endpoint_parity=parity['differences'],additional_fitter_calls=0,cpu_only_completion=True,
    stage_mapping='Six original monitor blocks: camera then body0..4. Boundaries reconstructed from saved before/after geometry keys and contiguous state indexes.',
    records=records,rotation_trajectory=[dict(state=i,right_hip_from32_deg=float(angles[i,1]),right_knee_from32_deg=float(angles[i,4])) for i in indices],
    actual_visual_review=False,quality_accepted=False,no_new_training_candidate=True,
    limitation='No saved per-step loss/stop-reason metadata. A fixed-state audit is separate; do not fabricate a stop cause.'))
for record in records:print(record,flush=True)
