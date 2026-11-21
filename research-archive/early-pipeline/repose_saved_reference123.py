"""Recover an immutable pre-densification binding by seeded replay, verify it
against six saved own-pose renders, then export a reviewed real source pose.
No nearest-face reassignment, optimizer state or optimization is used.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import fcntl
import traceback
import numpy as np
from PIL import Image
from review_full_sequence123 import read,sha,joined
from inspect_limb_reference123 import loaded,OUT,N,F,save

sys.path.insert(0,'/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')
from shuolee_portrait import runtime as rt
from shuolee_portrait.training import camera,image
from shuolee_portrait.export import rotate_ply
import torch
from torch import nn
from types import SimpleNamespace
from scene.gaussian_model import GaussianModel
from scene.smplx_gaussian_model import SMPLXGaussianModel
from roma import quat_product,quat_wxyz_to_xyzw,quat_xyzw_to_wxyz
from arguments import OptimizationParams
import argparse

TARGET='frame_000051'
DEST=OUT/'selected-reference'
P=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')

def main():
    assert not DEST.exists(),'Do not repeat or overwrite this bounded export.'
    DEST.mkdir()
    cfg=read(N/'gaussian-run/config.json');run=read(N/'gaussian-run/result.json')
    snapshot=N/'gaussian-run'/run['final_snapshot'];fit,rows=loaded()
    source=snapshot/'world.ply';old=read(N/'export/validation.json')
    args=argparse.ArgumentParser();opts=OptimizationParams(args).extract(args.parse_args([]))
    assert run['steps']==1000 and opts.densify_from_iter>1000 and cfg['initial_points']==200000
    logs=[json.loads(v) for v in (N/'gaussian-run/loss.jsonl').read_text().splitlines()]
    assert all(v['points']==200000 for v in logs) and logs[-1]['step']==1000
    assert cfg['seed']==6060 and cfg['fit_sha256']==sha(F/'fit/fitted.npz')
    ids=[v['frame'] for v in rows];reference=ids.index(old['reference_frame_id']);target=ids.index(TARGET)
    ranking=read(OUT/'reference-ranking.json');assert ranking['ordered_frames'][0]==TARGET
    save(DEST/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),
        hypothesis='The1000-step run had no densification/pruning; frozen faces, seed6060 and the same initializer can replay its original point-to-triangle order. Verify this against six saved renders before any new pose export.',
        change='Output pose only, from frame32 to the visually reviewed real source frame51. No optimizer updates, no new observations, no nearest-face reassignment.',
        source_ply_sha256=sha(source),fit_sha256=sha(F/'fit/fitted.npz'),
        source_code_sha256={str(p):sha(p) for p in [P/'avatar_core/scene/smplx_gaussian_model.py',P/'avatar_core/scene/gaussian_model.py',P/'avatar_core/utils/graphics_utils.py']},
        seed=cfg['seed'],densify_from_iter=opts.densify_from_iter,recorded_steps=run['steps'],target_frame=TARGET,
        selection_basis='Top combined rank of body/feet/face fit and body-pose representativeness. Native face image and own-pose body/hand/foot overlays actually reviewed; visible eyes open. Existing fit/geometry still not quality accepted.',
        acceptance=dict(reference_float_render_mean_max=2e-6,reference_float_render_max_max=.005,
            saved_own_pose_byte_mean_max=.00051,saved_own_pose_byte_max_max=2,views=6),
        budget=dict(binding_replays=1,optimization_steps=0,pose_candidates=1,novel_views=36),
        stop='Any identity/count, geometry, native-reference or six historical-render mismatch stops export. No threshold tuning.'))
    checks=[]
    with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        model=SMPLXGaussianModel(3);model.load_fresh_meshes(F/'fit/fitted.npz')
        model.initialize_surface(cfg['initial_points'],cfg['seed'])
        world=GaussianModel(3);world.load_ply(str(source));world.active_sh_degree=1
        model.select_mesh_by_timestep(reference)
        with torch.no_grad():
            bindings=model.binding
            delta=(world.get_xyz-model.face_center[bindings])/model.face_scaling[bindings]
            model._xyz=nn.Parameter(torch.bmm(model.face_orien_mat[bindings].transpose(1,2),delta[:,:,None]).squeeze(-1))
            q=model.rotation_activation(model.face_orien_quat[bindings])*torch.tensor([1.,-1.,-1.,-1.],device='cuda')
            local_q=quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(q),quat_wxyz_to_xyzw(world.get_rotation)))
            model._rotation=nn.Parameter(local_q)
            model._scaling=nn.Parameter((world.get_scaling/model.face_scaling[bindings]).log())
            for field in ['_features_dc','_features_rest','_opacity']:
                setattr(model,field,nn.Parameter(getattr(world,field).detach().clone()))
            model.active_sh_degree=1
            xyz_delta=float(abs(model.get_xyz-world.get_xyz).max())
            cov_delta=float(abs(model.get_covariance()-world.get_covariance()).max())
            assert xyz_delta<2e-7 and cov_delta<1e-7,(xyz_delta,cov_delta)
            pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
            ca,_=camera(rows[reference],reference,1520,pose_index=reference)
            a=render_image(ca,model,pipe,bg);b=render_image(ca,world,pipe,bg);difference=abs(a-b)
            reference_check=dict(mean=float(difference.mean()),maximum=float(difference.max()))
            assert reference_check['mean']<2e-6 and reference_check['maximum']<.005,reference_check
            for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
                model.select_mesh_by_timestep(int(i));cam,_=camera(rows[i],int(i),1520,pose_index=int(i))
                replay=image(render_image(cam,model,pipe,bg));historical=snapshot/f'{ids[i]}_render.png'
                delta=np.abs(np.array(replay,dtype=np.int16)-np.array(Image.open(historical),dtype=np.int16))
                item=dict(frame=ids[i],source_png_sha256=sha(historical),mean_byte=float(delta.mean()),max_byte=int(delta.max()),pixels_differing=int((delta>0).any(-1).sum()))
                checks.append(item);save(DEST/'recovery-checks.json',dict(state='checking',checks=checks,reference_check=reference_check))
                assert item['mean_byte']<.00051 and item['max_byte']<=2,item
            # Successful historical own-pose render replay is the recovery evidence.
            model.save_ply(str(DEST/'bound-local.ply'))
            model.export_world_ply(DEST/'world.ply',target)
            portable_check=rotate_ply(DEST/'world.ply',DEST/'point_cloud.ply',np.array(old['rotation']),np.array(old['origin_world']))
            portable_check.update(reference_frame_id=TARGET,reference_pose_index=target,active_sh_degree=1,
                original_training_steps=1000,new_training_steps=0,reposed_not_retrained=True,quality_accepted=False)
            portable=GaussianModel(3);portable.load_ply(str(DEST/'point_cloud.ply'));portable.active_sh_degree=1
            mapped=read(N/'export/cameras.json');mapped['world_to_viewer']=portable_check
            mapped['default_reference_frame']=TARGET;save(DEST/'cameras.json',mapped)
            allcams={r['frame']:r for r in mapped['records']}
            render_checks=[]
            for i in sorted(set([reference,target,0,62,93])):
                ca,_=camera(rows[i],i,1520,pose_index=target);cb,_=camera(allcams[ids[i]],i,1520,pose_index=target)
                a=render_image(ca,model,pipe,bg);b=render_image(cb,portable,pipe,bg);d=abs(a-b)
                item=dict(frame=ids[i],mean_abs=float(d.mean()),max_abs=float(d.max()));render_checks.append(item)
                assert item['mean_abs']<2e-6 and item['max_abs']<.005,item
                image(b).save(DEST/f'{ids[i]}-pose51-render.png')
            portable_check['native_render_checks']=render_checks;portable_check['format_passed']=True
            save(DEST/'validation.json',portable_check)
            raw=Image.open(rows[target]['image']).convert('RGB');im=Image.open(DEST/f'{TARGET}-pose51-render.png').convert('RGB')
            candidate=read(OUT/'candidate-meshes/result.json')['records']
            for item in [x for x in candidate if x['frame']==TARGET]:
                crop=item['crop_xyxy'];joined([raw.crop(crop),im.crop(crop)]).save(DEST/f'{TARGET}-{item["region"]}-raw-gaussian.png')
            novel=DEST/'novel';novel.mkdir()
            xyz=portable.get_xyz.cpu().numpy();center=np.median(xyz,axis=0)
            height=float(np.quantile(xyz[:,1],.99)-np.quantile(xyz[:,1],.01));radius=height*1.4
            for i,angle in enumerate(np.linspace(0,2*np.pi,36,endpoint=False)):
                eye=center+radius*np.array([np.sin(angle),.12,np.cos(angle)]);forward=center-eye;forward/=np.linalg.norm(forward)
                right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=np.cross(forward,right)
                E=np.eye(4);E[:3,:3]=np.stack([right,down,forward]);E[:3,3]=-E[:3,:3]@eye
                row=dict(frame=f'novel_{i:03d}',wh=[512,768],K=[[768,0,256],[0,768,384],[0,0,1]],w2c=E.tolist())
                cam,_=camera(row,i,pose_index=target);image(render_image(cam,portable,pipe,bg)).save(novel/f'{i:03d}.png')
    result=dict(state='exported_pending_actual_visual_and_UI_review',reference_frame=TARGET,reference_source_index=rows[target]['source_frame_index'],
        source_time_seconds=rows[target]['time_seconds'],reference_check=reference_check,historical_checks=checks,
        source_ply_sha256=sha(source),fit_sha256=sha(F/'fit/fitted.npz'),bound_local_sha256=sha(DEST/'bound-local.ply'),
        seeded_binding_replay_verified=True,point_order_changed=False,nearest_face_rebinding=False,
        recovered_not_originally_saved=True,optimizer_created=False,optimizer_steps=0,training_steps=0,
        active_sh_degree=1,torch_version=torch.__version__,position_reconstruction_max=xyz_delta,covariance_reconstruction_max=cov_delta,
        portable=portable_check,quality_accepted=False,model_published=False)
    save(DEST/'result.json',result)
    print(json.dumps(dict(state=result['state'],reference=TARGET,checks=checks,new_training_steps=0,quality_accepted=False)),flush=True)

def render_image(camera,model,pipe,bg):
    from gaussian_renderer import render
    return render(camera,model,pipe,bg)['render']

if __name__=='__main__':
    try:main()
    except Exception:
        if DEST.is_dir():
            save(DEST/'failure.json',dict(traceback=traceback.format_exc(),training_started=False,published=False))
        raise
