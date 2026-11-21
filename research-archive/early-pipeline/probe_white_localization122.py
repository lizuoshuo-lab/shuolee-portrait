"""N12e local photometric localization diagnosis. Never rewrites observations/cameras."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import argparse,hashlib,json,os,time,traceback
import numpy as np
import cv2
BASE=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I=BASE/'video-update-20261027'; O=I/'clip03-grid-coverage/array-cache-repair'; M=I/'clip03-scene-support'; D=I/'clip03-localization'
PLAN=I/'clip03-localization-plan.json'; SIZE=41; CENTER=np.array([20.,20.,1.]); LIMIT=12.
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(str(v).encode()).hexdigest()
def patch(image,uv):
    h,w=image.shape; x,y=map(float,uv)
    if min(x,y,w-1-x,h-1-y)<22:return None
    return cv2.getRectSubPix(image,(SIZE,SIZE),(x,y)).astype(np.float32)/255.
def estimate(a,b):
    if a is None or b is None:return dict(state='source_or_target_border_support_missing')
    W=np.eye(2,3,dtype=np.float32)
    try:
        cc,W=cv2.findTransformECC(a,b,W,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,50,1e-6),None,5)
    except cv2.error as e:return dict(state='ECC_failed',error=str(e).split('\n')[-1])
    H=np.eye(3);H[:2]=W
    if not np.isfinite(H).all() or not np.isfinite(cc):return dict(state='nonfinite_ECC')
    shift=H@CENTER-CENTER; singular=np.linalg.svd(H[:2,:2],compute_uv=False)
    state='measured'
    if np.linalg.norm(shift[:2])>LIMIT:state='outside_preregistered_local_support'
    elif np.linalg.det(H[:2,:2])<=0 or singular.min()<.5 or singular.max()>2:state='nonlocal_affine_deformation'
    out=dict(state=state,H=H.tolist(),ECC=float(cc),center_delta_px=shift[:2].tolist(),center_delta_norm_px=float(np.linalg.norm(shift[:2])),affine_singular_values=singular.tolist())
    if state!='measured':return out
    ys,xs=np.mgrid[4:SIZE-4,4:SIZE-4]; xy=np.stack((xs.ravel(),ys.ravel(),np.ones(xs.size)),1)
    tgt=xy@H[:2].T; valid=np.all((tgt>=1)&(tgt<=SIZE-2),axis=1)
    if valid.sum()<100:out['sensitivity_state']='insufficient_warp_support';return out
    xy,tgt=xy[valid],tgt[valid]
    mx=tgt[:,0].astype('float32').reshape(-1,1);my=tgt[:,1].astype('float32').reshape(-1,1)
    sample=lambda im:cv2.remap(im,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel().astype(float)
    target=sample(b); template=a[xy[:,1].astype(int),xy[:,0].astype(int)].astype(float)
    gb=np.linalg.lstsq(np.c_[template,np.ones(len(template))],target,rcond=None)[0]
    residual=target-(gb[0]*template+gb[1]);noise=float(np.sqrt(np.sum(residual**2)/max(1,len(residual)-8)))
    gx=sample(cv2.Sobel(b,cv2.CV_32F,1,0,ksize=3,scale=.125));gy=sample(cv2.Sobel(b,cv2.CV_32F,0,1,ksize=3,scale=.125))
    x=(xy[:,0]-20.)/20.;y=(xy[:,1]-20.)/20.
    nuisance=np.stack((gx*x,gx*y,gy*x,gy*y,template,np.ones(len(template))),1)
    Jt=np.stack((gx,gy),1);projected=Jt-nuisance@np.linalg.lstsq(nuisance,Jt,rcond=1e-10)[0]
    info=projected.T@projected;eig=np.linalg.eigvalsh(info)
    out.update(sensitivity_state='conditional_white_noise_only',sample_pixels=int(len(residual)),photometric_RMS_0_1=noise,
      gain_bias=gb.tolist(),translation_information_eigenvalues=eig.tolist(),translation_information_condition=float(eig[-1]/max(eig[0],1e-20)),
      conditional_one_sigma_major_px=float(noise/np.sqrt(eig[0])) if eig[0]>1e-12 else None,
      calibrated_measurement_uncertainty=False)
    return out
def point(H):return (np.asarray(H)@CENTER)[:2]
def controls():
    cv2.setNumThreads(1)
    rng=np.random.default_rng(12213);a=cv2.GaussianBlur(rng.random((SIZE,SIZE)).astype('float32'),(5,5),.9)
    ident=estimate(a,a);assert ident['state']=='measured' and ident['center_delta_norm_px']<1e-4
    W=cv2.getRotationMatrix2D((20.,20.),1.5,1.01);W[:,2]+=np.array([.35,-.4])
    b=cv2.warpAffine(a,W,(SIZE,SIZE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    est=estimate(a,b);truth=W@CENTER;err=float(np.linalg.norm(point(est['H'])-truth))
    assert est['state']=='measured' and err<.15,(est,err)
    reverse=estimate(b,a);cycle=float(np.linalg.norm((np.asarray(reverse['H'])@np.asarray(est['H'])@CENTER-CENTER)[:2]))
    assert cycle<.2,cycle
    flat=estimate(np.ones_like(a),np.ones_like(a));assert flat['state']!='measured'
    # The direction is template/source pixel -> input/target pixel; translation signs must be explicit.
    assert np.dot(np.asarray(est['center_delta_px']),[.35,-.4])>0
    return dict(checks=5,passed=True,identity_center_error_px=ident['center_delta_norm_px'],known_affine_center_error_px=err,
      known_affine_forward_backward_px=cycle,flat_patch_state=flat['state'],scope='Implementation and resampling control only; not real material truth or camera admission.')
def register():
    assert not PLAN.exists() and not D.exists()
    c=controls();v=read(O/'visual-index.json');r=read(O/'visual-review.json')
    assert len(v['rows'])==83 and len(r['rows'])==83 and r['inherited_geometry_gate']['passed'] is False
    paths=[O/n for n in ['visual-index.json','visual-review.json','observations.json','families.json','result.json']]+[M/f'frame_{f:06d}.png' for f in v['columns']]
    pairs=[]
    for t in v['rows']:
        obs=sorted(t['observations'],key=lambda a:a['frame'])[:3]
        assert len(obs)==3
        pairs.append(dict(row=t['row'],track=t['track'],family=t['family'],split=t['split'],observations=obs))
    save(PLAN,dict(stage='N12e',registered=now(),code_sha256=sha(Path(__file__)),
      hypothesis='N12d improved coarse material support but SIFT keypoint identity does not establish precise localization. Fixed local affine photometric measurements can diagnose detector anchor bias and repeatability independently of camera fitting. Self-consistent reflections/occlusions remain possible.',
      new_causal_variable='One local affine ECC measurement around already frozen SIFT anchors. No detection, descriptor, pairing, track membership, K or camera change. This is a localization diagnostic, not another global matching or geometry candidate.',
      fixed='Same native7images,83preregistered reviewed families and old splits. Every row first3chronological observations; all three pairs both directions. All failures retained; no residual-based family rejection or relabeling.',
      solver=dict(patch=41,iterations=50,epsilon=1e-6,motion='affine',gaussian_filter=5,initial_warp='identity relative to fixed source/target SIFT centers',local_center_limit_px=12,affine_scale_bounds=[.5,2.],opencv_threads=1),
      measures=['Correction relative to SIFT anchor, not ground-truth error','Independently solved forward/backward and3viewcycle closure','Photometric affine nuisance marginalized translation sensitivity, uncalibrated white-noise assumption','All source material categories and FIT/heldout separately'],
      controls=c,inputs={str(p):sha(p) for p in paths},rows=pairs,
      budget=dict(runs=1,CPU_wall_seconds=120,rows=83,directed_real_pair_solves=498,GPU=0,camera_solves=0,training=0,raw_review_rows=[14,18,19,24,9,27,52,55,56,57,59,60]),
      stop='Any input/control failure stops. Keep all nonconvergence/nonlocal measurements in denominator. No threshold, window, scale or iteration scan. Even perfect closure cannot relax N12d material/camera gates.',
      observation_export='Diagnostic-only measurements with explicit source hashes; never overwrite raw observations or feed updated coordinates into camera/training.',
      next_condition='Decide whether detector anchoring contributes measurable instability. A negative or self-inconsistent result cannot justify training. Physical ambiguity and camera/motion remain separate.'))
    print(json.dumps(dict(plan=str(PLAN),controls=c),ensure_ascii=False))
def summarize(values):
    if not values:return dict(n=0)
    a=np.asarray(values,float);return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()),strictly_under1=int(np.sum(a<1.)))
def run():
    p=read(PLAN);assert sha(Path(__file__))==p['code_sha256']
    assert not D.exists();D.mkdir()
    start=time.monotonic();pid=os.getpid()
    def progress(stage,**kw):
        assert time.monotonic()-start<120,'Registered CPU wall budget exceeded'
        save(D/'progress.json',dict(state=stage,pid=pid,updated=now(),elapsed_seconds=time.monotonic()-start,**kw))
    try:
        progress('verify_inputs')
        for path,h in p['inputs'].items():assert sha(Path(path))==h,path
        c=controls();cv2.setNumThreads(1)
        frames=read(O/'visual-index.json')['columns'];gray={f:cv2.imread(str(M/f'frame_{f:06d}.png'),cv2.IMREAD_GRAYSCALE) for f in frames}
        assert all(a.shape==(2704,1520) for a in gray.values())
        labels={r['row']:r for r in read(O/'visual-review.json')['rows']};out=[]
        for item in p['rows']:
            obs=item['observations']; patches=[patch(gray[o['frame']],o['uv']) for o in obs];pairs={}
            for a,b in [(0,1),(1,0),(1,2),(2,1),(0,2),(2,0)]:pairs[f'{a}{b}']=estimate(patches[a],patches[b])
            fbs=[]
            for a,b in [(0,1),(1,2),(0,2)]:
                f,g=pairs[f'{a}{b}'],pairs[f'{b}{a}']
                if f['state']==g['state']=='measured':fbs.append(dict(pair=[a,b],error_px=float(np.linalg.norm((np.asarray(g['H'])@np.asarray(f['H'])@CENTER-CENTER)[:2]))))
            cycles=[]
            for a,b,cidx in [(0,1,2),(2,1,0)]:
                paths=[pairs[f'{a}{b}'],pairs[f'{b}{cidx}'],pairs[f'{a}{cidx}']]
                if all(q['state']=='measured' for q in paths):
                    h1,h2,h3=[np.asarray(q['H']) for q in paths]
                    cycles.append(dict(path=[a,b,cidx],error_px=float(np.linalg.norm((h2@h1@CENTER-h3@CENTER)[:2]))))
            out.append(dict(**item,material_label=labels[item['row']]['reviewed_material'],coarse_static_candidate=labels[item['row']]['coarse_static_candidate'],
               pairs=pairs,forward_backward=fbs,cycle=cycles,all6_measured=all(q['state']=='measured' for q in pairs.values()),physical_accuracy_certified=False))
            if len(out)%10==0:progress('local_measurements',completed_rows=len(out),total=83)
        save(D/'measurements.json',dict(rows=out,source_observations_unchanged=True,coordinates_for_training=False))
        groups={}
        for name,subset in [('all',out),('coarse_static',[t for t in out if t['coarse_static_candidate']]),('roof',[t for t in out if t['material_label']=='roof_structure_coarse']),('face',[t for t in out if t['material_label']=='face_surface_coarse']),('glass_ambiguous',[t for t in out if t['material_label']=='window_reflection_or_repeated_texture_uncertain']),('foliage',[t for t in out if t['material_label']=='foliage']),('person_other',[t for t in out if t['row']>=51 and t['material_label']!='face_surface_coarse'])]:
            groups[name]=dict(rows=len(subset),all6_measured=sum(t['all6_measured'] for t in subset),states=Counter(q['state'] for t in subset for q in t['pairs'].values()),
                center_correction_px=summarize([q['center_delta_norm_px'] for t in subset for q in t['pairs'].values() if q['state']=='measured']),
                forward_backward_px=summarize([q['error_px'] for t in subset for q in t['forward_backward']]),cycle_px=summarize([q['error_px'] for t in subset for q in t['cycle']]))
        result=dict(stage='N12e',state='complete_pending_actual_review',updated=now(),elapsed_seconds=time.monotonic()-start,controls=c,groups=groups,
            measured_pairs=sum(q['state']=='measured' for t in out for q in t['pairs'].values()),attempted_pairs=498,
            source_hashes=p['inputs'],measurements_sha256=sha(D/'measurements.json'),
            caveats=['All summaries condition on successful bounded local measurements; every failure remains in states/denominator.',
              'ECC changes the measurement model; low closure and high correlation do not prove fixed 3D material or calibrated uncertainty.',
              'Local diagnostic sampling is not a warp of training images. Raw SIFT coordinates, family splits and K are unchanged.',
              'No physical pixel-error ground truth exists here. Never interpret delta relative to SIFT as reprojection error.'],
            K_changed=False,new_camera=False,training_started=False,physical_subpixel_accepted=False)
        for path,h in p['inputs'].items():assert sha(Path(path))==h
        save(D/'result.json',result);progress('complete_pending_actual_review',completed_rows=83)
        print(json.dumps(dict(result=str(D/'result.json'),elapsed=result['elapsed_seconds'],groups=groups),ensure_ascii=False))
    except BaseException as e:
        save(D/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start,pid=pid))
        save(D/'progress.json',dict(state='failed',pid=pid,updated=now(),error=repr(e)))
        raise
if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('action',choices=['register','run']);a=args.parse_args()
    register() if a.action=='register' else run()
