"""Bounded reference-pose ranking and saved geometry inspection, without fitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from review_full_sequence123 import read, sha, relocate, joined, mesh_renderer

O = relocate('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle')
F = O/'full94-shared-shape'
N = O/'native-resolution-1000'
OUT = O/'limb-and-reference-review'
BODY = np.array([0,16,15,18,17,5,2,6,3,7,4,13,10,14,11])

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def stats(a):
    a=np.asarray(a)
    if not a.size:
        return dict(count=0,median=None,p90=None,maximum=None)
    return dict(count=int(a.size),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max()))

def loaded():
    with np.load(F/'fit/fitted.npz',allow_pickle=False) as z:
        fit=dict(z)
    rows=read(F/'data/training-inputs.json')['records']
    assert [r['frame'] for r in rows]==fit['frame_ids'].astype(str).tolist()
    assert sha(F/'fit/fitted.npz')==sha(N/'fit/fitted.npz')
    return fit,rows

def rank():
    assert not OUT.exists(),'Review the existing result instead of repeating it.'
    OUT.mkdir()
    fit,rows=loaded()
    save(OUT/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn='progress',user_direction='Inspect twisted Gaussian limbs and select a real source pose with good fit, representative body action and visible open eyes.',
        hypothesis='Separate original SMPL-X geometry, learned Gaussian support and exported coordinates before changing the fitter.',
        sources={str(p):sha(p) for p in [F/'fit/fitted.npz',F/'data/training-inputs.json',N/'gaussian-run/step-001000/world.ply',N/'export/point_cloud.ply']},
        rank='Equal percentile ranks of body median, feet median, face median and median pairwise body-joint rotation distance; all94 frames. Open-eye and occlusion acceptance require native image review.',
        geometry='Three existing camera positions, one identical frozen pose32. Compare original, saved mesh and trained Gaussians; no new pose or coordinate warp. Nearest-surface distances are descriptive, not an anatomical error.',
        budget=dict(rank_frames=94,rank_preview_candidates=12,geometry_views=3,new_features=0,fit_updates=0,Gaussian_updates=0),
        stop='Any hash/identity/nonfinite discrepancy; no parameter or threshold search. Existing PLY has no per-point binding and will not be silently reposed.'))
    y=fit['observed_native_keypoints'];p=fit['joints_projected_native'];err=np.linalg.norm(p-y[:,:,:2],axis=-1)
    rotations=Rotation.from_rotvec(fit['body_pose_axis_angle'].reshape(-1,3)).as_quat().reshape(94,21,4)
    cos=np.abs(np.einsum('ijd,kjd->ikj',rotations,rotations)).clip(0,1)
    angle=np.degrees(2*np.arccos(cos)).mean(-1)
    pose=np.array([np.median(np.delete(angle[i],i)) for i in range(94)])
    b=np.median(err[:,BODY],axis=1);feet=np.median(err[:,19:25],axis=1);face=np.median(err[:,67:118],axis=1)
    score=sum((rankdata(v,method='average')-1)/93 for v in [b,feet,face,pose])/4
    results=[]
    for i,r in enumerate(rows):
        eyes=[]
        for start in [86,92]:
            eye=y[i,start:start+6,:2]
            width=np.linalg.norm(eye[0]-eye[3])
            eyes.append(float((np.linalg.norm(eye[1]-eye[5])+np.linalg.norm(eye[2]-eye[4]))/(2*max(width,1e-6))))
        results.append(dict(frame_id=r['frame'],source_frame_index=r['source_frame_index'],time_seconds=r['time_seconds'],
            body_median_px=float(b[i]),feet_median_px=float(feet[i]),face_median_px=float(face[i]),face_maximum_px=float(err[i,67:118].max()),
            median_body_rotation_difference_deg=float(pose[i]),combined_rank=float(score[i]),
            eye_aspect_ratios=eyes,eye_state_verified=False,independent_accuracy=False,
            image_sha256=r['source_sha256'],mask_sha256=r['mask_sha256']))
    order=np.argsort(score,kind='stable');selected=order[:12].tolist()
    # Include the currently exported pose as a separate comparison, without changing ranking.
    if 31 not in selected:selected.append(31)
    faces=[]
    for i in selected:
        r=rows[i];path=relocate(r['image']);assert sha(path)==r['source_sha256']
        raw=Image.open(path).convert('RGB')
        xy=y[i,67:118,:2];cx,cy=np.mean([xy.min(0),xy.max(0)],axis=0)
        box=[int(round(cx))-80,int(round(cy))-100,int(round(cx))+80,int(round(cy))+100]
        im=raw.crop(box);im.save(OUT/f'{r["frame"]}-head.png');faces.append(im)
        results[i]['preview_head_crop_xyxy']=box
    for g in range((len(faces)+5)//6):
        subset=faces[g*6:(g+1)*6]
        while len(subset)<6:subset.append(Image.new('RGB',(160,200),(240,240,240)))
        top=joined(subset[:3]);bottom=joined(subset[3:])
        sheet=Image.new('RGB',(480,400));sheet.paste(top,(0,0));sheet.paste(bottom,(0,200))
        sheet.save(OUT/f'candidate-heads-{g:02d}.png')
    save(OUT/'reference-ranking.json',dict(state='ranked_requires_eye_visibility_and_mesh_review',rows=results,
        ordered_frames=[rows[i]['frame'] for i in order],preview_frames=[rows[i]['frame'] for i in selected],
        previous_reference='frame_000032',selection=None,changed_export=False,source_pose_interpolated=False,
        caution='Median residuals are to fitted detector targets. Ranking is not a quality pass; EAR alone cannot establish open eyes or visibility.'))
    print(json.dumps(dict(previews=[results[i] for i in selected]),ensure_ascii=False),flush=True)

def geometry():
    import fcntl
    plan=read(OUT/'plan.json')
    assert not (OUT/'geometry-result.json').exists()
    for p,h in plan['sources'].items():assert sha(relocate(p))==h
    fit,rows=loaded();slot=31
    source=N/'gaussian-run/step-001000/world.ply'
    sys.path.insert(0,'/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.training import camera,image
    import torch
    from types import SimpleNamespace
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    mesh,mesh_source=mesh_renderer();records=[]
    with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        model=GaussianModel(3);model.load_ply(str(source))
        # This snapshot was saved at1000, when training had activated degree1.
        model.active_sh_degree=1
        bg=torch.zeros(3,device='cuda');pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        xyz=model.get_xyz.detach().cpu().numpy();scale=model.get_scaling.detach().cpu().numpy();opacity=model.get_opacity.detach().cpu().numpy()[:,0]
        distances,nearest=cKDTree(fit['vertices_world'][slot]).query(xyz,workers=4)
        inv=read(N/'export/validation.json');preview=GaussianModel(3);preview.load_ply(str(N/'export/point_cloud.ply'))
        inverse=preview.get_xyz.detach().cpu().numpy()@np.array(inv['rotation'])+np.array(inv['origin_world'])
        export_delta=float(abs(inverse-xyz).max());assert export_delta<1e-5
        with torch.no_grad():
            for view in [31,62,93]:
                r=rows[view];cam,cam_info=camera(r,view,1520,pose_index=slot)
                result=render(cam,model,pipe,bg);rgb=image(result['render'])
                alpha=image(render(cam,model,pipe,bg,override_color=torch.ones_like(model.get_xyz))['render'])
                E=np.array(r['w2c']);vc=fit['vertices_world'][slot]@E[:3,:3].T+E[:3,3]
                K=np.array(r['K_integer_centers']);uv=vc@K.T;uv=uv[:,:2]/uv[:,2:]
                box=np.r_[np.floor(uv.min(0)-60),np.ceil(uv.max(0)+60)].astype(int)
                box[:2]=np.maximum(box[:2],0);box[2:]=np.minimum(box[2:],r['wh'])
                canvas=Image.new('RGB',tuple(r['wh']),(12,15,16))
                mesh_only=mesh(canvas,np.array([0,0,*r['wh']]),dict(vertices_camera=vc[None],faces=fit['faces'],K_integer_centers=K))
                mesh_over_gaussian=mesh(rgb,np.array([0,0,*r['wh']]),dict(vertices_camera=vc[None],faces=fit['faces'],K_integer_centers=K))
                panels=[mesh_only.crop(box),rgb.crop(box),mesh_over_gaussian.crop(box)]
                if view==slot:
                    assert sha(relocate(r['image']))==r['source_sha256']
                    panels.insert(0,Image.open(relocate(r['image'])).convert('RGB').crop(box))
                output=OUT/f'pose32-camera{view+1:02d}-mesh-gaussians.png';joined(panels).save(output)
                rgb.save(OUT/f'camera{view+1:02d}-gaussian-full.png');alpha.save(OUT/f'camera{view+1:02d}-alpha-full.png')
                record=dict(camera_frame=r['frame'],pose_frame='frame_000032',crop_xyxy=box.tolist(),image=str(output),sha256=sha(output),camera=cam_info,
                    source_photo_comparable=view==slot,rendered_SH_degree=1)
                if view==slot:
                    target=np.array(Image.open(relocate(r['mask'])).convert('L'))/255
                    a=np.array(alpha)[:,:,0]/255
                    fg=target>.99
                    record['silhouette']=dict(foreground_area_fraction=float(fg.mean()),global_MAE=float(abs(a-target).mean()),foreground_MAE=float(abs(a-target)[fg].mean()),
                        outside_alpha_sum=float(a[~fg].sum()),foreground_target_sum=float(fg.sum()),
                        existing_weighted_global_silhouette_loss=float(.1*abs(a-target).mean()))
                records.append(record)
    result=dict(state='complete_visual_review_pending',reference_frame='frame_000032',records=records,
        nearest_vertex_distance_m=stats(distances),maximum_Gaussian_axis_m=stats(scale.max(1)),
        alpha_over_half_nearest_vertex_m=stats(distances[opacity>.5]),
        export_position_inverse_max=export_delta,
        point_binding_available_in_snapshot=False,new_bindings_inferred=False,
        geometry_updates=0,optimizer_updates=0,quality_accepted=False,mesh_renderer_sha256=sha(mesh_source),
        interpretation='Original fitted mesh and saved Gaussian snapshot share the same native coordinates and reference pose. Distances to naked mesh are descriptive; clothing/hair should depart from it. An exported-only PLY cannot prove the original per-point mesh binding.')
    save(OUT/'geometry-result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)

def candidates():
    ranking=read(OUT/'reference-ranking.json');fit,rows=loaded()
    target=OUT/'candidate-meshes';target.mkdir()
    ids=['frame_000051','frame_000053','frame_000044','frame_000032']
    save(target/'plan.json',dict(frames=ids,new_fitting=False,rendering='Saved actual fitted mesh on its own unwarped original pixels; native integer crops.',
        purpose='Verify body/hands/feet and eye visibility before accepting a ranked output pose.'))
    mesh,_=mesh_renderer();records=[]
    for name in ids:
        i=next(i for i,r in enumerate(rows) if r['frame']==name);r=rows[i]
        path=relocate(r['image']);assert sha(path)==r['source_sha256']
        raw=Image.open(path).convert('RGB');xy=fit['observed_native_keypoints'][i,:,:2]
        foreground=np.array(Image.open(relocate(r['mask'])).convert('L'))>250
        yy,xx=np.where(foreground);body=np.array([xx.min()-25,yy.min()-25,xx.max()+26,yy.max()+26])
        boxes={'body':body,'head':next(v['preview_head_crop_xyxy'] for v in ranking['rows'] if v['frame_id']==name)}
        for region,indices in [('feet',np.arange(19,25)),('left-hand',np.arange(25,46)),('right-hand',np.arange(46,67))]:
            sub=xy[indices];boxes[region]=np.r_[np.floor(sub.min(0)-25),np.ceil(sub.max(0)+26)].astype(int)
        for region,b in boxes.items():
            b=np.array(b,int);b[:2]=np.maximum(b[:2],0);b[2:]=np.minimum(b[2:],r['wh'])
            im=raw.crop(b);overlay=mesh(im,b,dict(vertices_camera=fit['vertices_camera'][i:i+1],faces=fit['faces'],K_integer_centers=fit['K_integer_centers'][i]))
            file=target/f'{name}-{region}.png';joined([im,overlay]).save(file)
            records.append(dict(frame=name,region=region,crop_xyxy=b.tolist(),file=str(file),sha256=sha(file)))
    save(target/'result.json',dict(state='complete_visual_review_pending',records=records,new_fitting=False))
    print(json.dumps(dict(records=len(records),directory=str(target))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['rank','geometry','candidates'])
    args=parser.parse_args();globals()[args.mode]()
