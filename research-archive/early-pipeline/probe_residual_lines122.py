"""One bounded native plumb-line probe; observed-to-ideal output mapping is explicit."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,time,sys
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import cv2
from scipy.optimize import least_squares

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=ROOT/'residual-distortion-direction'; DATA=Path('/mnt/d/GS-SLAM-Data/robust122-whiteT-native-supervision/rgb')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
P=read(O/'plan.json');K=np.array(P['model']['K']);F=K.diagonal()[:2];PP=K[:2,2]
cv2.setNumThreads(2)

def mapping(uv,c):
    xy=(np.asarray(uv)-PP)/F;x,y=xy.T;r2=x*x+y*y;k1,k2,p1,p2=c
    d=np.stack([x*(k1*r2+k2*r2*r2)+2*p1*x*y+p2*(r2+2*x*x),
                y*(k1*r2+k2*r2*r2)+p1*(r2+2*y*y)+2*p2*x*y],-1)
    return np.asarray(uv)+d*F

def basis(uv):
    x,y=((np.asarray(uv)-PP)/F).T;r=x*x+y*y
    return np.stack([np.stack([x*r,x*r*r,2*x*y,r+2*x*x],-1)*F[0],
                     np.stack([y*r,y*r*r,r+2*y*y,2*x*y],-1)*F[1]],1)

def spatial_jacobian(uv,c):
    x,y=((np.asarray(uv)-PP)/F).T;r=x*x+y*y;k1,k2,p1,p2=c
    h=k1+2*k2*r;s=1+k1*r+k2*r*r
    J=np.empty((len(x),2,2))
    J[:,0,0]=s+2*x*x*h+2*p1*y+6*p2*x
    J[:,0,1]=(2*x*y*h+2*p1*x+2*p2*y)*F[0]/F[1]
    J[:,1,0]=(2*x*y*h+2*p1*x+2*p2*y)*F[1]/F[0]
    J[:,1,1]=s+2*y*y*h+6*p1*y+2*p2*x
    return J

def forward_inverse(ideal,c):
    obs=np.array(ideal,float).copy()
    for _ in range(12):obs-=np.linalg.solve(spatial_jacobian(obs,c),(mapping(obs,c)-ideal)[...,None])[...,0]
    return obs

def tls(uv,normal):
    mean=uv.mean(0);_,_,V=np.linalg.svd(uv-mean,full_matrices=False);n=V[-1]
    if np.dot(n,normal)<0:n=-n
    return (uv-mean)@n

def controls():
    points=np.array([[30.,20.],[1450,80],[700,800],[230,2400],[1510,2690]])
    c=np.array([.003,-.001,.0007,-.0005]);b=basis(points)
    eps=1e-6
    fd=np.stack([(mapping(points,c+np.eye(4)[i]*eps)-mapping(points,c-np.eye(4)[i]*eps))/(2*eps) for i in range(4)],-1)
    sj=np.stack([(mapping(points+np.eye(2)[i]*1e-3,c)-mapping(points-np.eye(2)[i]*1e-3,c))/(2e-3) for i in range(2)],-1)
    ce=float(np.abs(fd-b).max());se=float(np.abs(sj-spatial_jacobian(points,c)).max())
    re=float(np.linalg.norm(forward_inverse(mapping(points,c),c)-points,axis=1).max())
    assert ce<1e-6 and se<1e-7 and re<1e-8
    assert np.array_equal(mapping(points,np.zeros(4)),points)
    return dict(coefficient_jacobian_max=ce,spatial_jacobian_max=se,forward_inverse_roundtrip_px=re,zero_identity=True)

def extract():
    assert not (O/'measurements.json').exists()
    save(O/'progress.json',dict(state='native_line_extraction',pid=os.getpid(),updated=stamp()))
    cfg=P['extraction'];records=[];cache={};out=O/'line-review'
    old_images={p.name:sha(p) for p in out.glob('*.png')} if out.exists() else {}
    out.mkdir(exist_ok=True)
    for line in P['line_seeds']:
        f=line['frame'];path=DATA/f'frame_{f:06d}.png'
        if f not in cache:
            assert sha(path)==P['input_hashes'][path.name]
            cache[f]=cv2.imread(str(path));assert cache[f].shape==(2704,1520,3)
        rgb=cache[f];gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY).astype(np.float32)
        a,b=np.array(line['a'],float),np.array(line['b'],float);tangent=(b-a)/np.linalg.norm(b-a);normal=np.array([-tangent[1],tangent[0]])
        fraction=np.linspace(0,1,cfg['samples_per_line']);center=a[None]+fraction[:,None]*(b-a)
        offs=np.arange(-cfg['normal_search_half_width_px'],cfg['normal_search_half_width_px']+.001,cfg['normal_step_px'])
        profiles=[]
        for toff in cfg['tangent_average_offsets_px']:
            pos=center[:,None]+offs[None,:,None]*normal+tangent*toff
            profiles.append(cv2.remap(gray,pos[:,:,0].astype(np.float32),pos[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT))
        profile=np.mean(profiles,0);grad=np.gradient(profile,cfg['normal_step_px'],axis=1)
        rowpeak=np.argmax(np.abs(grad[:,1:-1]),axis=1)+1
        sign=1. if np.median(grad[np.arange(len(grad)),rowpeak])>=0 else -1.
        g=grad*sign;points=[];reasons=[];strength=[];offsets=[]
        for j in range(len(center)):
            peak=int(np.argmax(g[j,1:-1])+1);val=float(g[j,peak]);reason=[]
            if any(lo<=fraction[j]<=hi for lo,hi in line['exclude_fraction']):reason.append('preregistered_occlusion')
            if peak<=2 or peak>=len(offs)-3:reason.append('peak_near_search_limit')
            if val<cfg['weak_gradient_limit_gray_per_px']:reason.append('weak_gradient')
            # Competing separated local maxima only; same peak's shoulder is not a different edge.
            alternatives=[k for k in range(1,len(offs)-1) if abs(offs[k]-offs[peak])>=2.5 and g[j,k]>=g[j,k-1] and g[j,k]>g[j,k+1]]
            if alternatives and max(g[j,k] for k in alternatives)>=.9*val:reason.append('ambiguous_peak')
            den=float(g[j,peak-1]-2*g[j,peak]+g[j,peak+1]);delta=0 if abs(den)<1e-10 else .5*float(g[j,peak-1]-g[j,peak+1])/den
            delta=float(np.clip(delta,-.5,.5));offset=float(offs[peak]+delta*cfg['normal_step_px']);uv=center[j]+offset*normal
            if not (8<uv[0]<1512 and 8<uv[1]<2696):reason.append('border')
            points.append(uv.tolist());reasons.append(reason);strength.append(val);offsets.append(offset)
        eligible=np.array([not x for x in reasons]);unoccluded=np.array([not any(lo<=u<=hi for lo,hi in line['exclude_fraction']) for u in fraction])
        frac=float(eligible.sum()/max(1,unoccluded.sum()))
        qualified=eligible.sum()>=cfg['min_samples'] and frac>=cfg['min_valid_fraction']
        box=[int(max(0,min(a[0],b[0])-20)),int(max(0,min(a[1],b[1])-20)),int(min(1520,max(a[0],b[0])+21)),int(min(2704,max(a[1],b[1])+21))]
        cv2.imwrite(str(out/(line['id']+'.png')),rgb[box[1]:box[3],box[0]:box[2]])
        records.append(dict(**line,normal=normal.tolist(),points=points,valid=eligible.tolist(),reasons=reasons,
            strength=strength,offsets=offsets,valid_count=int(eligible.sum()),valid_fraction=frac,
            measurement_qualified=bool(qualified),native_review_box=box,median_offset=float(np.median(np.array(offsets)[eligible])) if eligible.any() else None))
    for name,digest in old_images.items():assert sha(out/name)==digest
    save(O/'controls.json',controls());save(O/'measurements.json',dict(updated=stamp(),records=records,source_plan_sha256=sha(O/'plan.json'),recovered_images_byte_unchanged=len(old_images)))
    save(O/'progress.json',dict(state='extraction_complete_pending_actual_line_review',updated=stamp(),pid=None))
    print(json.dumps([dict(id=x['id'],split=x['split'],count=x['valid_count'],fraction=x['valid_fraction'],qualified=x['measurement_qualified'],offset=x['median_offset']) for x in records]))

def fit():
    assert not (O/'result.json').exists()
    rr=read(O/'line-review.json');assert rr['actual_review_complete']
    M=read(O/'measurements.json');assert M['source_plan_sha256']==sha(O/'plan.json')
    lines=[x for x in M['records'] if x['measurement_qualified'] and x['id'] in rr['accepted_for_measurement']]
    fitlines=[x for x in lines if x['split']=='fit'];held=[x for x in lines if x['split']=='heldout']
    gate=P['acceptance'];support=(len(fitlines)>=gate['min_fit_lines'] and len(held)>=gate['min_heldout_lines']
      and len(set(x['physical_group'] for x in fitlines))>=gate['min_fit_groups'] and len(set(x['physical_group'] for x in held))>=gate['min_heldout_groups'])
    if not support:
        save(O/'result.json',dict(state='measurement_support_insufficient',fit_lines=len(fitlines),heldout_lines=len(held),candidate_promoted=False,training=False));return
    start=time.monotonic();save(O/'progress.json',dict(state='single_residual_fit',pid=os.getpid(),updated=stamp()))
    def pts(l):return np.array(l['points'])[l['valid']]
    def fun(c):
        if time.monotonic()-start>P['fit']['max_seconds']:raise TimeoutError('registered CPU budget')
        return np.concatenate([tls(mapping(pts(l),c),np.array(l['normal']))*np.sqrt(100/len(pts(l))) for l in fitlines])
    bounds=np.array(P['fit']['bounds']);opt=least_squares(fun,np.zeros(4),bounds=(-bounds,bounds),
      loss=P['fit']['loss'],f_scale=P['fit']['f_scale_px'],max_nfev=P['fit']['max_nfev'],xtol=1e-11,ftol=1e-11,gtol=1e-11)
    c=opt.x;perline=[]
    for l in lines:
        before=tls(pts(l),np.array(l['normal']));after=tls(mapping(pts(l),c),np.array(l['normal']))
        perline.append(dict(id=l['id'],split=l['split'],physical_group=l['physical_group'],count=len(before),
           before_rms_px=float(np.sqrt(np.mean(before**2))),after_rms_px=float(np.sqrt(np.mean(after**2))),
           before_max_px=float(np.abs(before).max()),after_max_px=float(np.abs(after).max()),
           before_residual_px=before.tolist(),after_residual_px=after.tolist()))
    # Independent finite difference of the profiled line objective, with dimensionless parameter scaling.
    eps=1e-6;J=np.stack([(fun(c+np.eye(4)[i]*eps)-fun(c-np.eye(4)[i]*eps))/(2*eps) for i in range(4)],-1)
    singular=np.linalg.svd(J*bounds,compute_uv=False);rank=int(np.sum(singular>singular[0]*1e-7));condition=float(singular[0]/singular[-1])
    grid=np.stack(np.meshgrid(np.linspace(0,1519,41),np.linspace(0,2703,61)),-1).reshape(-1,2)
    shift=np.linalg.norm(mapping(grid,c)-grid,axis=1);det=np.linalg.det(spatial_jacobian(grid,c))
    hg=[x for x in perline if x['split']=='heldout'];bg=float(np.sqrt(np.mean([x['before_rms_px']**2 for x in hg])));ag=float(np.sqrt(np.mean([x['after_rms_px']**2 for x in hg])))
    guards=dict(solver_success=bool(opt.success),full_rank=rank==4,condition=condition<gate['max_scaled_condition'],
      heldout_improves=ag<=bg*(1-gate['heldout_rms_improvement_fraction']),
      all_heldout_protected=all(x['after_rms_px']<=x['before_rms_px']+gate['heldout_max_rms_worsening_px'] for x in hg),
      full_field_displacement=float(shift.max())<=gate['max_full_frame_displacement_px'],
      invertible_grid=float(det.min())>=gate['min_jacobian_determinant'],not_on_bound=bool(np.all(np.abs(c)<.99*bounds)))
    result=dict(stage='D1_residual_lines',state='complete_pending_review',updated=stamp(),seconds=time.monotonic()-start,
      coefficients=c.tolist(),coefficient_order=['k1','k2','p1','p2'],model=P['model']['name'],
      fit_lines=len(fitlines),heldout_lines=len(held),nfev=opt.nfev,solver_message=opt.message,
      scaled_jacobian_singular_values=singular.tolist(),scaled_jacobian_rank=rank,scaled_jacobian_condition=condition,
      heldout_equal_line_rms_before_px=bg,heldout_equal_line_rms_after_px=ag,
      max_full_field_displacement_px=float(shift.max()),min_grid_jacobian_determinant=float(det.min()),
      guards=guards,line_gate_pass=all(guards.values()),candidate_promoted=False,training=False,K_changed=False,
      image_resampling=False,old_camera_changed=False,physical_face_acceptance=False,
      coverage_px=dict(x_min=float(min(pts(l)[:,0].min() for l in lines)),x_max=float(max(pts(l)[:,0].max() for l in lines)),
                       y_min=float(min(pts(l)[:,1].min() for l in lines)),y_max=float(max(pts(l)[:,1].max() for l in lines))),
      per_line=perline,plan_sha256=sha(O/'plan.json'),measurements_sha256=sha(O/'measurements.json'))
    save(O/'result.json',result);save(O/'progress.json',dict(state='complete_pending_review',pid=None,updated=stamp()))
    print(json.dumps({k:v for k,v in result.items() if k!='per_line'}))

if __name__=='__main__':
    if sys.argv[1]=='extract':extract()
    elif sys.argv[1]=='fit':fit()
    else:raise SystemExit('extract or fit')
