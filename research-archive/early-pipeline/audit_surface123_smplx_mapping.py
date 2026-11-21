"""Read-only audit of frozen round123 fits. No optimizer or renderer mutation."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, time
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/opt/gs-slam-lab/shuolee-local-surface-123')
REC = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT = REC/'smplx-mapping-audit'
MODEL = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def stats(x):
    x=np.asarray(x).ravel()
    return dict(n=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(np.max(x)),at_least_1px=int((x>=1).sum())) if len(x) else dict(n=0)
def project(x,E,K):
    p=np.einsum('nij,nkj->nki',E[:,:3,:3],x)+E[:,:3,3,None].transpose(0,2,1)
    q=np.einsum('nij,nkj->nki',K,p)
    return q[...,:2]/q[...,2:]-.5,p[...,2]
def occluded(C,P,tri,world_scale):
    # Independent CPU segment/triangle test. Model-derived visibility, not truth.
    e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];v=C-tri[:,0]
    q=np.cross(v,e1);out=[]
    for pt in P:
        d=pt-C;h=np.cross(d,e2);a=np.einsum('ij,ij->i',e1,h)
        valid=np.abs(a)>1e-12;inv=np.zeros_like(a);inv[valid]=1/a[valid]
        u=np.einsum('ij,ij->i',v,h)*inv;b=(q@d)*inv;t=np.einsum('ij,ij->i',e2,q)*inv
        eps=1e-4*world_scale/max(np.linalg.norm(d),1e-12)
        out.append(bool(np.any(valid&(u>=-1e-7)&(b>=-1e-7)&(u+b<=1+1e-7)&(t>1e-6)&(t<1-eps))))
    return np.asarray(out)

def run(subject,model):
    start=time.monotonic();root=ROOT/subject
    f=dict(np.load(root/'fit/fitted.npz'));obs=dict(np.load(root/'data/observations.npz'));w=dict(np.load(root/'fit/observation-weights.npz'))
    rows=read(root/'data/inputs.json')['records'];E=np.asarray([r['w2c'] for r in rows]);K=np.asarray([r['K'] for r in rows]);n=len(rows)
    ids=[r['frame'] for r in rows]
    assert list(f['frame_ids'])==list(obs['frame_ids'])==ids
    assert [r['official_prediction_slot'] for r in rows]==list(range(n))
    assert [r['source_frame_index'] for r in rows]==read(root/'frame-plan.json')['selected_source_indices']
    assert obs['joint_names'][40:91].tolist()==[f'face-{i}' for i in range(17,68)]
    assert obs['joint_names'][17:23].tolist()==['left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']
    tri=f['vertices_world'][0].astype(float)[f['faces']]
    faceids=model['lmk_faces_idx'];bary=model['lmk_bary_coords'];fp=(tri[faceids]*bary[:,:,None]).sum(1)
    fn=np.cross(tri[faceids,1]-tri[faceids,0],tri[faceids,2]-tri[faceids,0])
    pf,zf=project(np.broadcast_to(fp,(n,*fp.shape)),E,K)
    pc,zc=project(f['coco17_world'],E,K);pj,zj=project(f['joints_world'],E,K)
    diff={k:float(np.max(np.abs(a-b))) for k,a,b in [('coco17',pc,f['projected_coco17']),('joints',pj,f['projected_joints']),('face51',pf,f['projected_face51'])]}
    corner=np.asarray([r['K_integer_centers'] for r in rows]);corner[:,:2,2]+=.5
    assert np.max(np.abs(K-corner))<1e-10
    centers=-np.einsum('nji,nj->ni',E[:,:3,:3],E[:,:3,3])
    back=np.einsum('nli,li->nl',centers[:,None]-fp[None],fn)<=0
    occ=[]
    for i in range(n):
        if time.monotonic()-start>120:raise RuntimeError('Read-only 120-second per-subject budget exhausted.')
        occ.append(occluded(centers[i],fp,tri,float(f['world_scale'])))
    occ=np.asarray(occ);unsupported=back|occ|(zf<=0)
    faceerr=np.linalg.norm(pf-obs['keypoints'][:,40:91],axis=-1)
    bodyerr=np.linalg.norm(pc-w['body_target'],axis=-1);bw=w['body']*w['visibility'];fw=w['face']*w['face_surface_visibility']
    # Semantic SMPL-X foot/hand selection is explicit, not an assumed index range.
    from smplx.joint_names import JOINT_NAMES
    feetids=[JOINT_NAMES.index(str(k)) for k in obs['joint_names'][17:23]]
    handnames=['wrist','thumb1','thumb2','thumb3','thumb','index1','index2','index3','index','middle1','middle2','middle3','middle','ring1','ring2','ring3','ring','pinky1','pinky2','pinky3','pinky']
    handids=[JOINT_NAMES.index(side+'_'+k) for side in ['left','right'] for k in handnames]
    footerr=np.linalg.norm(pj[:,feetids]-obs['keypoints'][:,17:23],axis=-1)
    handerr=np.linalg.norm(pj[:,handids]-obs['keypoints'][:,91:133],axis=-1)
    hs=(obs['boxes'][:,3]-obs['boxes'][:,1]);active=fw>0
    records=[]
    for i,r in enumerate(rows):
        records.append(dict(frame=r['frame'],source_frame_index=r['source_frame_index'],face_positive_weight_count=int(active[i].sum()),
            face_model_unsupported_positive_weight_count=int((unsupported[i]&active[i]).sum()),face_model_backfacing_positive_weight_count=int((back[i]&active[i]).sum()),
            face_weight_sum=float(fw[i].sum()),face_model_unsupported_weight_sum=float(fw[i][unsupported[i]].sum()),
            face_errors_active=stats(faceerr[i][active[i]]),body_reliable_errors=stats(bodyerr[i][bw[i]>.25])))
    arrays=dict(face_errors_px=faceerr,body_errors_px=bodyerr,feet_errors_px=footerr,hands_errors_px=handerr,
        face_weight=fw,face_model_backfacing=back,face_model_occluded=occ,body_weight=bw)
    np.savez_compressed(OUT/f'{subject}-observations.npz',**arrays)
    result=dict(subject=subject,frames=n,seconds=time.monotonic()-start,read_only=True,no_parameters_optimized=True,
        frame_and_slot_identity_passed=True,face_17_67_and_feet_semantic_order_passed=True,
        independent_projection_max_abs_px=diff,renderer_K_minus_integer_K_exactly_half_pixel=True,
        shared_pose_max_difference=float(np.max(np.abs(f['body_pose']-f['body_pose'][0]))),
        static_mesh_max_difference=float(np.max(np.abs(f['vertices_world']-f['vertices_world'][0]))),
        anatomical_visibility=read(root/'fit/protocol.json')['anatomical_visibility'],
        saved_face_surface_visibility_all_one=bool(np.all(w['face_surface_visibility']==1)),
        saved_feet_surface_visibility_all_one=bool(np.all(w['feet_surface_visibility']==1)),
        body_reliable_native_error=stats(bodyerr[bw>.25]),face_nonzero_weight_native_error=stats(faceerr[active]),
        feet_nonzero_weight_native_error=stats(footerr[w['feet']>0]),hands_nonzero_weight_native_error=stats(handerr[w['hands']>0]),
        body_height_px=stats(hs),gate_allowed_median_px_range=(.03*np.array([hs.min(),hs.max()])).tolist(),
        gate_allowed_p90_px_range=(.08*np.array([hs.min(),hs.max()])).tolist(),
        active_face_observations=int(active.sum()),model_unsupported_positive_weight_observations=int((unsupported&active).sum()),
        model_backfacing_positive_weight_observations=int((back&active).sum()),
        model_unsupported_fraction_of_face_weight=float(fw[unsupported].sum()/fw.sum()),
        model_backfacing_fraction_of_face_weight=float(fw[back].sum()/fw.sum()),
        face_weight_slots_always_zero=np.flatnonzero((fw==0).all(0)).tolist(),
        measurement_is_detector_residual_not_independent_material_truth=True,
        visibility_is_frozen_fitted_model_diagnostic_not_true_human_visibility=True,
        no_unique_cause_or_fix_claim=True,frames_detail=records,
        hashes={str(p.relative_to(root)):sha(p) for p in [root/'fit/fitted.npz',root/'fit/observation-weights.npz',root/'data/observations.npz',root/'data/inputs.json']})
    save(OUT/f'{subject}-result.json',result)
    if subject=='grayT':
        gallery=REC/'grayT-smplx-audit-review';gallery.mkdir(exist_ok=False);images=[]
        for i in np.unique(np.linspace(0,n-1,6).astype(int)):
            r=rows[i];rgb=Image.open(r['image']).convert('RGB');box=np.asarray(r['face_box_xyxy'],float)
            margin=.5*max(box[2:]-box[:2]);xy=np.rint(np.clip(box+[-margin,-margin,margin,margin],[0,0,0,0],[*rgb.size,*rgb.size])).astype(int)
            panels=[rgb.copy() for _ in range(3)]
            for im,pts,col in [(panels[1],obs['keypoints'][i,40:91],(0,220,255)),(panels[2],pf[i],(255,70,60))]:
                draw=ImageDraw.Draw(im)
                for point in pts[active[i]]:
                    x,y=point;draw.ellipse((x-2,y-2,x+2,y+2),fill=col)
            crops=[im.crop(tuple(xy)) for im in panels];out=Image.new('RGB',(crops[0].width*3,crops[0].height))
            for j,im in enumerate(crops):out.paste(im,(j*im.width,0))
            name=r['frame']+'_native_face_raw_detector_model.png';out.save(gallery/name)
            images.append(dict(frame=r['frame'],region='native_face',file=name,columns=['原图','参与拟合的检测点','对应SMPL-X投影点'],native_crop_xyxy=xy.tolist(),no_resampling=True,sha256=sha(gallery/name)))
        save(gallery/'index.json',dict(subject=subject,phase='smplx_audit',images=images,image_labels_burned_in=False,actual_review_pending=True))
    return {k:v for k,v in result.items() if k not in ('frames_detail','hashes')}

if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    save(OUT/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),reason='User: smpl-x对不齐',
        subjects=['winterhat','whiteT','grayT'],cpu_seconds_per_subject=120,read_only=True,
        hypotheses=['Frame/slot/index or pixel convention mismatch','Invisible facial observations remain active','Body-fraction gate hides large absolute residuals','One static pose cannot fit inconsistent observations'],
        checks=['Frozen semantic order','Independent float64 projection','Fitted-model segment occlusion and back-facing weights','Per-region absolute residuals without error filtering'],
        no_gpu=True,no_refit=True,no_camera_changes=True,no_training=True,no_error_based_observation_deletion=True,
        limits='No isolated causal intervention: this audit cannot rank physical camera, detector and motion causes or establish ground truth.'))
    with np.load(MODEL,allow_pickle=False) as z:model={k:z[k] for k in ['lmk_faces_idx','lmk_bary_coords']}
    res=[run(s,model) for s in ['winterhat','whiteT','grayT']]
    save(OUT/'summary.json',res)
    print(json.dumps(res,ensure_ascii=False,indent=2))
