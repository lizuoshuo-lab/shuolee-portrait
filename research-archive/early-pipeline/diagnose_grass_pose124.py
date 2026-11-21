"""Read-only provenance and pose-depth audit, no fitter/training invocation."""
import os,json
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read,save,sha,now
from review_full_sequence123 import mesh_renderer,joined,crop_box
root=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/grass')
out=root/'pose-cause-audit';out.mkdir(exist_ok=True)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
save(out/'plan.json',dict(updated=now(),request='草地侧视姿态错误，定位首次出现环节',mode='read-only CPU',
    comparisons=['cached official SMPLify-X vs Gaussian input','initial vs final mesh, same reference and orbit camera','joint depth and native 2D error, all94 frames','world-to-preview rigid inverse validation'],
    no_new_optimization=True,no_parameter_change=True,no_training_relaunch=True))
import cv2
cv2.setNumThreads(1)
rows=read(root/'data/training-inputs.json')['records'];sel=read(root/'selection.json')['selected_frames']
with np.load(root/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
with np.load(root/'gaussian-run/step-018000/current-meshes.npz',allow_pickle=False) as z:final=dict(z)
source_path=read(root/'plan.json')['source_fit']
with np.load(source_path,allow_pickle=False) as z:source=dict(z)
ids=fit['frame_ids'].astype(str).tolist();assert ids==[r['frame'] for r in rows]==final['frame_ids'].astype(str).tolist()
source_ids=source['frame_ids'].astype(str).tolist()
source_error=max(float(np.max(abs(fit['vertices_camera'][i]-source['vertices_camera'][source_ids.index(name)]))) for i,name in enumerate(ids))
assert source_error==0
mesh_delta=np.linalg.norm(final['vertices_camera']-fit['vertices_camera'],axis=-1)
metrics=[]
for i,name in enumerate(ids):
    j=fit['joints_camera'][i];pred=fit['joints_projected_native'][i];obs=fit['observed_native_keypoints'][i]
    metrics.append(dict(frame=name,training=name in sel,time=rows[i]['time_seconds'],
        right_shoulder_camera_xyz=j[2].tolist(),right_elbow_camera_xyz=j[3].tolist(),right_wrist_camera_xyz=j[4].tolist(),
        wrist_minus_shoulder_depth_model_m=float(j[4,2]-j[2,2]),
        right_shoulder_error_px=float(np.linalg.norm(pred[2]-obs[2,:2])),right_elbow_error_px=float(np.linalg.norm(pred[3]-obs[3,:2])),right_wrist_error_px=float(np.linalg.norm(pred[4]-obs[4,:2])),
        confidence=obs[[2,3,4],2].tolist(),max_mesh_motion_model_m=float(mesh_delta[i].max())))
ordered=sorted(range(len(ids)),key=lambda i:ids[i]);transitions=[]
for b,c in zip(ordered[:-1],ordered[1:]):
    transitions.append(dict(previous=ids[b],current=ids[c],
        relative_arm_depth_change_model_m=metrics[c]['wrist_minus_shoulder_depth_model_m']-metrics[b]['wrist_minus_shoulder_depth_model_m'],
        native_wrist_2d_change_px=float(np.linalg.norm(fit['observed_native_keypoints'][c,4,:2]-fit['observed_native_keypoints'][b,4,:2]))))
render,_=mesh_renderer();delivery=root/'delivery/step-018000';R=np.array(read(delivery/'validation.json')['rotation'])
novel=read(delivery/'novel/cameras.json')['records'];ref=ids.index('frame_000032');faces=fit['faces']
E=np.array(rows[ref]['w2c']);initial_world=(fit['vertices_camera'][ref]-E[:3,3])@E[:3,:3];final_world=(final['vertices_camera'][ref]-E[:3,3])@E[:3,:3]
initial_preview=initial_world@R.T;final_preview=final_world@R.T
panels=[]
for ix in (0,4,8,12,18,24,28,32):
    row=novel[ix];Ne=np.array(row['w2c']);K=np.array(row['K']);K[0,2]-=.5;K[1,2]-=.5
    w,h=row['wh'];blank=Image.new('RGB',(w,h),'black');parts=[]
    for vs in (initial_preview,final_preview):
        vc=vs@Ne[:3,:3].T+Ne[:3,3]
        parts.append(render(blank,np.array([0,0,w,h]),dict(vertices_camera=vc[None],faces=faces,K_integer_centers=K)))
    parts.append(Image.open(delivery/f'novel/novel_{ix:03d}.png').convert('RGB'))
    panel=joined(parts);panel.save(out/f'orbit_{ix:03d}_initial_final_gaussian.png');panels.append(panel)
for k in (0,4):
    p=Image.new('RGB',(panels[0].width*2,panels[0].height*2),'black')
    for n,pic in enumerate(panels[k:k+4]):p.paste(pic,((n%2)*pic.width,(n//2)*pic.height))
    p.save(out/f'orbit_mesh_compare_{k//4+1}.png')
native=[]
for frame in ('frame_000030','frame_000032','frame_000044','frame_000045','frame_000046'):
    i=ids.index(frame);r=rows[i];raw=Image.open(r['image']).convert('RGB');mask=Image.open(r['mask']).convert('L');box=crop_box(mask.getbbox(),raw.size,30);pic=raw.crop(box)
    overlay=render(pic,np.array(box),dict(vertices_camera=fit['vertices_camera'][i:i+1],faces=faces,K_integer_centers=np.array(r['K_integer_centers'])))
    p=joined([pic,overlay]);f=out/f'{frame}_source_initial.png';p.save(f);native.append(str(f))
summary=dict(updated=now(),source_fit=source_path,source_fit_sha256=sha(source_path),input_vertices_exactly_equal=True,
    source_to_training_initial_max_error_model_m=source_error,final_mesh_displacement_max_model_m=float(mesh_delta.max()),
    reference=metrics[ref],selected_training=[x for x in metrics if x['training']],all_frames=sorted(metrics,key=lambda x:x['frame']),
    transitions=transitions,export_validation=read(delivery/'validation.json'),
    pose_frozen_in_gaussian_training=True,reference_in_training='frame_000032' in sel,
    source_images=native,actual_visual_review=False,quality_accepted=False,
    unit_warning='Depth is the fitted SMPL-X model meter convention; no independent scene scale claim.',
    causal_limit='Locates error before Gaussian training; monocular depth ambiguity/optimizer basin/observation/prior balance need separate controlled fitting evidence for unique attribution.')
save(out/'result.json',summary);print('POSE_CAUSE_AUDIT_READY',summary['reference'],flush=True)
