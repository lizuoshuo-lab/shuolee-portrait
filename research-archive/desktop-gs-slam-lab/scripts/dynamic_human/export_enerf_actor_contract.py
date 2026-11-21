"""Lossless scene-node interface for actual persistent foreground states.

No claim of SMPL articulation, metric scale, learned background or OmniRe
checkpoint compatibility. Unassigned Gaussians remain as an explicit node.
"""
from pathlib import Path
import json,argparse,hashlib
import cv2,numpy as np,torch
from enerf_native_loader import EnerfNativeDataset

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--root',required=True);a=p.parse_args();out=Path(a.out);dest=out/'actor-interface';dest.mkdir(exist_ok=True);assert not (dest/'scene-contract.json').exists();group=out/'offset-index';motion=group/'motion';completion=json.loads((motion/'completion.json').read_text());frames=completion['frames'];assert frames==list(range(34,59))
    first=torch.load(motion/'frame-034.pt',weights_only=False,map_location='cpu');xyz=first['means'].numpy().astype('float64');N=len(xyz);votes=np.zeros((N,2),np.uint8);ds=EnerfNativeDataset(group/'appearance-manifest.json','train',a.root);camera_checks=[]
    for i,r in enumerate(ds.rows):
        if r['common_frame']!=34:continue
        o=ds[i];_,labels,stats,_=cv2.connectedComponentsWithStats((o['foreground_matte']>.5).astype('uint8'));components=sorted([(j+1,s) for j,s in enumerate(stats[1:]) if s[4]>10000],key=lambda x:x[1][0]);assert len(components)==2
        camera=xyz@o['w2c'][:3,:3].T+o['w2c'][:3,3];projected=camera@o['K'].T;uv=np.rint(projected[:,:2]/projected[:,2:]).astype(int);h,w=labels.shape;valid=(camera[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);take=np.flatnonzero(valid);lab=labels[uv[take,1],uv[take,0]]
        for j,(index,_) in enumerate(components):votes[take,j]+=(lab==index)
        camera_checks.append(dict(camera=o['camera'],source_frame=o['frame'],source_sha256=o['source_sha256'],component_boxes=[s[:4].tolist() for _,s in components]))
    win=votes.argmax(1);support=votes.max(1);margin=np.abs(votes[:,0].astype(int)-votes[:,1].astype(int));actor=np.where((support>=8)&(margin>=4),win+1,0).astype('uint8');assert len(camera_checks)==15
    np.savez_compressed(dest/'gaussian-instance-index.npz',gaussian_id=np.arange(N,dtype=np.int32),foreground_instance=actor,votes=votes)
    counts={str(k):int((actor==k).sum()) for k in [0,1,2]};assert counts['1']>10000 and counts['2']>10000;trajectories={str(k):[] for k in [0,1,2]};states=[];maximum_error=0.
    for t in frames:
        path=motion/f'frame-{t:03d}.pt';state=torch.load(path,weights_only=False,map_location='cpu');points=state['means'].numpy().astype('float64');assert len(points)==N
        for k,v in state.items():assert torch.isfinite(v).all(),(t,k)
        for k in ['scales','opacities']:assert torch.equal(state[k],first[k]),(t,k)
        restored=np.empty_like(points)
        for k in [0,1,2]:
            mask=actor==k
            if mask.any():center=np.median(points[mask],axis=0);local=points[mask]-center;restored[mask]=local+center
            else:center=np.zeros(3)
            trajectories[str(k)].append(dict(common_frame=t,translation_world=center.tolist(),root_rotation_wxyz=[1,0,0,0]))
        err=float(np.abs(restored-points).max());maximum_error=max(maximum_error,err);assert np.array_equal(restored.astype('float32'),state['means'].numpy())
        states.append(dict(common_frame=t,file=path.relative_to(out).as_posix(),sha256=sha(path)))
    contract=dict(schema='persistent_foreground_scene_nodes_v0.1',status='candidate_motion_export_losslessly_verified_not_clarity_accepted',source_model='offset-index',source_manifest_sha256=sha(group/'appearance-manifest.json'),frame_coordinates='Common image index anchored to camera08; NOT physical seconds',frames=frames,world_coordinates='Released OpenCV calibration world; metric scale not independently established',camera_coordinates='Xcamera=Rcw Xworld+tcw, native1920x1080 K with source distortion rectified',nodes=[dict(id=k,type='unassigned_foreground' if k==0 else 'person_associated_deformable_foreground',points=counts[str(k)],semantic_scope='Includes held toy and possible matting artifacts; no claim of human-only/SMPL mesh') for k in [0,1,2]],index_file='gaussian-instance-index.npz',assignment='At first common frame34, left-to-right connected foreground components in15training views; >=8votes and>=4vote margin. Ambiguous points retained in node0; no Gaussians pruned.',state_files=states,node_trajectories=trajectories,local_geometry='For each node/time: local_center=source_global_center-node_translation. World transform has identity rotation and this translation; remaining per-point motion lives in local deformation. Global quaternions remain unchanged.',quaternion='Gaussian quaternion order wxyz as consumed by gsplat; root rotations identity; body orientation and skeleton NOT estimated',render_composition='All nodes must be recombined in original gaussian_id order and rendered in a shared depth/alpha ordering, including unassigned node0. Do not render each node independently then overlay.',joint_scene_boundary='Future static background node must share verified coordinate frame/scale and camera times. This manifest is an explicit interface, not an OmniRe checkpoint or a learned scene-background result.',no_test_photos_for_assignment=True,lossless_check=dict(states=len(states),all_source_gaussians_retained=N,maximum_float64_position_error=maximum_error,float32_reconstructed_positions_exact=True,frozen_scale_and_opacity_all_states=True),source_camera_assignment_checks=camera_checks)
    (dest/'scene-contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=contract['status'],counts=counts,states=len(states),lossless=contract['lossless_check']),indent=2))
if __name__=='__main__':main()
