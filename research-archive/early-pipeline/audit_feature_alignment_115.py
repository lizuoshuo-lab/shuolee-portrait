"""Read-only replay of real image correspondences in the CURRENT camera pipeline.

No optimization, new training, image warping, point rejection or camera update.
The older 101 point-only fit is a frozen diagnostic reference, not ground truth.
"""
from pathlib import Path
from types import SimpleNamespace
import sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

G = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee')
sys.path.insert(0, str(G))
import shuolee_common as c

O = c.BASE / 'current/camera-feature-root-cause-115'
P = c.BASE / 'current/avatar-repository-pair-114'
F = c.BASE / 'current/shared-point-bundle-convergence-101'
LABELS = ['background', 'face_lower', 'face_upper', 'torso', 'legs_shoes']

def stats(x):
    x = np.asarray(x).reshape(-1)
    assert np.isfinite(x).all()
    return dict(n=int(x.size), median=float(np.median(x)), p90=float(np.quantile(x,.9)),
                maximum=float(x.max())) if x.size else dict(n=0)

def project(x, e, k):
    q = np.einsum('nij,nj->ni', e[:,:3,:3], x) + e[:,:3,3]
    p = q @ k.T
    return p[:,:2]/p[:,2:]-.5, q[:,2]

def main():
    assert not (O/'feature-result.json').exists(), 'Completed evidence is immutable.'
    st = time.monotonic()
    sources = [F/'input.npz', F/'fixed_cameras.npz', P/'smplx/data/training-inputs.json',
               P/'smplx/data/observations.npz', P/'smplx/gaussian-run/point_cloud.ply']
    hashes = {str(p):c.sha(p) for p in sources}
    c.save(O/'feature-protocol.json', dict(created=c.stamp(),source_sha256=hashes,
        question='Do the same real image correspondences meet at one XYZ in the current fixed cameras?',
        scope='Replay frozen101 point-only fits with current113/114 input records; no new fitting or training.',
        selection='All existing observations and fixed holdout flags; no residual-based rejection.',
        units='Source rectified1514x2694 pixel centers; scaled863x1536 values reported separately.',
        evaluation='Image-only texture middle-third holdout; named landmarks held out in35 five-frame blocks.',
        limits=['Neither correspondences nor fitted XYZ are physical ground truth.',
                'Held out from point fitting does not mean withheld from Gaussian RGB training.',
                'Different regional errors do not uniquely distinguish calibration error, feature bias and micromotion.'],
        no_optimizer=True,new_training=False,source_updates=0))
    a=np.load(sources[0]); fit=np.load(sources[1])
    rows=c.read(sources[2])['records']; current=np.load(sources[3])
    E=np.array([r['w2c'] for r in rows]); K=np.array(rows[0]['K'])
    assert len(rows)==175 and np.max(abs(E-a['w2c']))<1e-12
    assert np.max(abs(E-fit['w2c']))<1e-10 and np.max(abs(K-a['K']))<1e-12
    wh=np.array(rows[0]['wh']); ratio=np.array([863,1536])/wh
    ob=a['observations']; off=a['offsets']; fi=ob[:,0].astype(int)
    ids=np.repeat(np.arange(len(off)-1),np.diff(off)); held=a['held']; groups=a['groups']
    pred,z=project(fit['xyz'][ids],E[fi],K)
    assert (z>0).all()
    old_pred_difference=float(np.max(abs(pred-fit['scene_prediction'])))
    errors=pred-ob[:,1:]; regions={}
    for j,label in enumerate(LABELS):
        sel=held&(groups==j)
        regions[label]=dict(source_pixels=stats(np.linalg.norm(errors[sel],axis=-1)),
            training_pixels=stats(np.linalg.norm(errors[sel]*ratio,axis=-1)),
            tracks=int(len(np.unique(ids[sel]))),frames=int(len(np.unique(fi[sel]))))
    fo=a['face_observations']; ff=fo[:,0].astype(int); fpids=a['face_point_indices']
    named_ids=a['face_ids'][fpids]; fh=a['face_held']
    fp,fz=project(fit['face_xyz'][fpids],E[ff],K)
    assert (fz>0).all()
    fe=fp-fo[:,1:]
    # The wholebody133 face segment is the68 landmarks starting at23.
    current_uv=current['keypoints'][ff,23+named_ids]
    current_detector_delta=np.linalg.norm(current_uv-fo[:,1:],axis=-1)
    named={}
    for label,sel in [('all',np.ones(len(fo),bool)),('nose',(named_ids>=27)&(named_ids<=35)),
                      ('mouth',named_ids>=48)]:
        named[label]=dict(held_source_pixels=stats(np.linalg.norm(fe[sel&fh],axis=-1)),
            held_training_pixels=stats(np.linalg.norm(fe[sel&fh]*ratio,axis=-1)))
    frame_records=[]
    for i in range(175):
        rs={}
        for j,label in enumerate(LABELS):
            sel=held&(fi==i)&(groups==j)
            rs[label]=dict(error=stats(np.linalg.norm(errors[sel],axis=-1)),
                median_vector=np.median(errors[sel],axis=0).tolist() if sel.any() else None)
        sel=fh&(ff==i)
        frame_records.append(dict(raw_frame=i+1,texture=rs,
            named_held=stats(np.linalg.norm(fe[sel],axis=-1))))
    # Visual evidence: same crop, same camera, no independent image alignment.
    torch=c.runtime()
    from scene.cameras import CameraProj2
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    model=GaussianModel(3); model.load_ply(str(sources[-1])); model.active_sh_degree=3
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',21)
    small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',17)
    visuals=[]
    # These four source frames are in the pre-existing point-fit holdout.
    for raw in [38,63,88,138]:
        i=raw-1;r=rows[i];e=E[i];kt=np.diag([*ratio,1.])@K
        cam=CameraProj2(i,e[:3,:3].T,e[:3,3],kt[0,0],kt[1,1],'white',863,None,1536,r['image'],r['frame'],i,cx=kt[0,2],cy=kt[1,2])
        with torch.no_grad(): rgb=render(cam,model,pipe,torch.ones(3,device='cuda'))['render']
        rendered=Image.fromarray((rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype('uint8'))
        photo=Image.open(r['image']).convert('RGB').resize((863,1536),Image.Resampling.LANCZOS)
        available=np.flatnonzero((ff==i)&fh)
        assert len(available)>0
        obs=(fo[available,1:]+.5)*ratio-.5; pr=(fp[available]+.5)*ratio-.5
        # Include whole head, not a different crop centered on the prediction.
        landmarks=(current['keypoints'][i,23:91]+.5)*ratio-.5
        low=landmarks.min(0)-[25,35];high=landmarks.max(0)+[25,20]
        box=tuple([int(max(low[0],0)),int(max(low[1],0)),int(min(np.ceil(high[0]),863)),int(min(np.ceil(high[1]),1536))])
        aspect=(box[2]-box[0])/(box[3]-box[1]);height=420;width=round(height*aspect)
        if width>350:width=350;height=round(width/aspect)
        canvas=Image.new('RGB',(1140,570),'#f8fafc');draw=ImageDraw.Draw(canvas)
        titles=['原视频同相机区域','实测特征与单一三维点投影','114B 已训练高斯：同相机']
        for col,im in enumerate([photo,photo,rendered]):
            patch=im.crop(box).resize((width,height),Image.Resampling.LANCZOS)
            x=col*380+(380-width)//2;y=72
            canvas.paste(patch,(x,y));draw.text((col*380+14,14),titles[col],fill='#172c40',font=font)
            if col==1:
                chosen=np.isin(named_ids[available],[30,33,48,51,54,57])
                for uv,qv,lm in zip(obs[chosen],pr[chosen],named_ids[available][chosen]):
                    aa=(uv-np.array(box[:2]))*[width/(box[2]-box[0]),height/(box[3]-box[1])]+[x,y]
                    bb=(qv-np.array(box[:2]))*[width/(box[2]-box[0]),height/(box[3]-box[1])]+[x,y]
                    draw.line([tuple(aa),tuple(bb)],fill='#ffee55',width=2)
                    draw.ellipse((aa[0]-4,aa[1]-4,aa[0]+4,aa[1]+4),outline='#00e9bd',width=2)
                    draw.line((bb[0]-4,bb[1]-4,bb[0]+4,bb[1]+4),fill='#ff712c',width=2)
                    draw.line((bb[0]-4,bb[1]+4,bb[0]+4,bb[1]-4),fill='#ff712c',width=2)
        draw.text((16,510),f'帧{raw:06d}｜绿色圈：原图检测；橙色叉：同一三维点投影；黄色线：误差（未放大）。',font=small,fill='#20354b')
        draw.text((16,540),'此帧未参与三维点拟合；高斯训练使用过此图。检测点和三角化点均不是真值。',font=small,fill='#20354b')
        path=O/f'feature-source-{raw:06d}.jpg';canvas.save(path,quality=95)
        visuals.append(dict(raw_frame=raw,path=path.name,crop_training_xyxy=list(box),
                            same_crop=True,point_fit_holdout=True,Gaussian_train_holdout=False))
    assert all(c.sha(p)==h for p,h in hashes.items())
    result=dict(created=c.stamp(),state='current_real_correspondence_replayed',frames=175,
        current_vs_frozen_K_E_passed=True,reprojection_vs_cached_max_abs_px=old_pred_difference,
        current_named_detector_vs_previous_observation_source_px=stats(current_detector_delta),
        texture_held=regions,named_landmarks=named,named_shared_xyz=len(fit['face_xyz']),
        named_held_frames=int(len(np.unique(ff[fh]))),visuals=visuals,
        evidence='Current K/E reproduce the prior independent texture and named-feature measurements. No global coordinate transfer defect was found. Real semantic observations still have materially larger residuals than background tracks.',
        unresolved=['Estimated physical calibration accuracy','View-dependent anatomical keypoint bias/occlusion','Subject micromotion','Depth/geometry ambiguity of trained Gaussians'],
        no_unique_root_cause=True,new_training_started=False,camera_updates=0,source_updates=0,
        optimizer_updates=0,quality_accepted=False,source_hashes_unchanged=True,seconds=time.monotonic()-st)
    np.savez_compressed(O/'feature-residuals.npz',scene_residual=errors,named_residual=fe,
                        scene_frame=fi,scene_held=held,scene_group=groups,named_frame=ff,
                        named_held=fh,named_id=named_ids)
    c.save(O/'feature-frame-records.json',frame_records);c.save(O/'feature-result.json',result)
    print(c.json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
