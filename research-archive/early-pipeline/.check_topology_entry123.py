from pathlib import Path
import sys,json,time,numpy as np,torch,smplx
H=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update');sys.path.insert(0,str(H))
import run_official_sequence123_topology as adapter
import official_smplifyx_entry123 as entry
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control')
F=R/'official-first-cycle/full94-shared-shape'
utils,_,_,_=entry.load_official(R/'official')
body=smplx.SMPLX('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz',batch_size=1,num_betas=10,num_expression_coeffs=10,use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,create_transl=False,joint_mapper=utils.JointMapper(utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')),dtype=torch.float32).cuda()
z=dict(np.load(F/'frames/frame_000032/winner-camera-space.npz',allow_pickle=False));body.reset_params(**{k:z[k] for k,_ in body.named_parameters()});pose=torch.as_tensor(z['body_pose_axis_angle'],device='cuda')
with torch.no_grad():
 a=body(body_pose=pose,return_verts=True,return_full_pose=True)
 info=adapter.cache_static_topology(body)
 b=body(body_pose=pose,return_verts=True,return_full_pose=True)
 pointer=body.parents.data_ptr();adapter.cache_static_topology(body)
 assert body.parents.data_ptr()==pointer and torch.equal(a.vertices,b.vertices) and torch.equal(a.joints,b.joints)
 report=dict(state='actual_CUDA_model_hook_verified',vertices_identical=True,joints_identical=True,idempotent=True,parameters_on_CUDA=all(p.device.type=='cuda' for p in body.parameters()),parent_device=str(body.parents.device),adapter_sha256=entry.sha(adapter.__file__),metadata=info,optimizer_updates=0,complete94_fit_run=False)
(F/'reviews/official-topology-full-fit/runtime-entry-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))