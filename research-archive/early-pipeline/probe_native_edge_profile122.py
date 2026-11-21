"""D4 native-pixel edge-spread localization. A bounded measurement experiment, no calibration.

The effective decoded-intensity edge model is deliberately not a physical camera
calibration model. Save every estimate, failure, and original occlusion label.
"""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,time,traceback
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import cv2
from scipy.special import ndtr
from scipy.optimize import least_squares
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
D=R/'residual-distortion-direction';A=R/'residual-edge-association';O=R/'native-edge-profile'
DATA=Path('/mnt/d/GS-SLAM-Data/robust122-whiteT-native-supervision/rgb')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
nodes,weights=np.polynomial.legendre.leggauss(5)
sub=np.stack(np.meshgrid(nodes*.5,nodes*.5),-1).reshape(-1,2)
sw=(weights[:,None]*weights[None,:]).reshape(-1)*.25

def patch_coordinates(center,tangent):
    center=np.asarray(center,float);tangent=np.asarray(tangent,float);normal=np.array([-tangent[1],tangent[0]])
    x,y=np.meshgrid(np.arange(int(np.floor(center[0]))-12,int(np.floor(center[0]))+13),
                    np.arange(int(np.floor(center[1]))-12,int(np.floor(center[1]))+13))
    uv=np.stack([x,y],-1).reshape(-1,2);xy=uv-center;nt=xy@normal;tt=xy@tangent
    ok=(np.abs(nt)<=8)&(np.abs(tt)<=4)
    return uv[ok],xy[ok],normal,tangent

def spread(xy,normal,tangent,p):
    d,angle,ls=p;sigma=np.exp(ls);n=normal*np.cos(angle)+tangent*np.sin(angle)
    return ndtr(((xy[:,None,:]+sub[None])@n-d)/sigma)@sw

def fit_patch(xy,rgb,normal,tangent,initial):
    # Raw integer-center observations, no bilinear image remap. Linear photometric
    # coefficients are eliminated for each three-parameter geometry evaluation.
    plane=np.stack([np.ones(len(xy)),xy@normal/8,xy@tangent/4],-1)
    low=np.array([-7.99,-.15,np.log(.35)]);high=np.array([7.99,.15,np.log(3.)])
    def evaluate(p):
        E=spread(xy,normal,tangent,p);X=np.column_stack([plane,E]);coef=np.linalg.lstsq(X,rgb,rcond=None)[0]
        return X@coef-rgb,coef
    opt=least_squares(lambda p:evaluate(p)[0].reshape(-1),np.array([np.clip(initial,-7.9,7.9),0.,np.log(.8)]),
        bounds=(low,high),max_nfev=40,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    residual,coef=evaluate(opt.x);sing=np.linalg.svd(opt.jac,compute_uv=False)
    rank=int((sing>max(sing[0]*1e-7,1e-9)).sum());dof=max(1,residual.size-15)
    formal_cov=np.linalg.pinv(opt.jac.T@opt.jac,rcond=1e-10)*(residual**2).sum()/dof
    n=normal*np.cos(opt.x[1])+tangent*np.sin(opt.x[1])
    # The reported point lies on the original normal cross-section t=0.
    offset=float(opt.x[0]/np.dot(n,normal))
    return dict(offset=offset,normal_offset=float(opt.x[0]),angle=float(opt.x[1]),sigma=float(np.exp(opt.x[2])),
      rms_gray=float(np.sqrt(np.mean(residual**2))),contrast_rgb=coef[-1].tolist(),plane_coefficients=coef[:-1].tolist(),
      formal_offset_std_px=float(np.sqrt(max(0,formal_cov[0,0]))),formal_uncertainty_not_physical=True,
      numerical_success=bool(opt.success),nfev=int(opt.nfev),rank=rank,
      bound_hit=bool(np.min(np.minimum(opt.x-low,high-opt.x))<1e-3),
      all_native_pixels_used=True)

def independent_synthetic(xy,n,offset,sigma,gamma=1.,two_edges=False):
    # Midpoint area integration, independent from the estimator Gauss-Legendre rule.
    ss=(np.arange(32)+.5)/32-.5;u,v=np.meshgrid(ss,ss);s=np.stack([u,v],-1).reshape(-1,2)
    q=(xy[:,None,:]+s[None])@n
    E=ndtr((q-offset)/sigma).mean(1)
    signal=.08+.78*E
    if two_edges:signal=.08+.38*E+.4*ndtr((q-offset-3.0)/sigma).mean(1)
    signal=np.power(signal,1/gamma)
    return np.column_stack([signal,signal*.9,signal*.7])*255

def old_peaks(image,center,tangent):
    normal=np.array([-tangent[1],tangent[0]]);offs=np.arange(-8,8.001,.25);profiles=[]
    gray=cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)
    for t in [-2,-1,0,1,2]:
      xy=np.asarray(center)+offs[:,None]*normal+t*tangent
      profiles.append(cv2.remap(gray,xy[:,0].astype(np.float32)[None],xy[:,1].astype(np.float32)[None],cv2.INTER_LINEAR)[0])
    g=np.gradient(np.mean(profiles,0),.25);w=int(np.argmax(abs(g[1:-1]))+1);g*=1 if g[w]>=0 else -1
    out=[float(offs[i]) for i in range(3,len(offs)-3) if g[i]>=3 and g[i]>=g[i-1] and g[i]>g[i+1]]
    return out

def controls():
    positive=[];counter=[]
    for angle in [0.,.025,.35,1.2]:
      tangent=np.array([np.cos(angle),np.sin(angle)]);uv,xy,n,t=patch_coordinates([20.3,20.6],tangent)
      for offset in [-.4,0.,.4]:
       for sigma in [.6,1.2,2.]:
        rgb=independent_synthetic(xy,n,offset,sigma);f=fit_patch(xy,rgb,n,t,0.)
        positive.append(dict(angle=angle,true_offset=offset,true_sigma=sigma,estimate=f['offset'],error=abs(f['offset']-offset)))
    assert max(x['error'] for x in positive)<.05
    tangent=np.array([1.,0.]);uv,xy,n,t=patch_coordinates([20.3,20.6],tangent)
    for kind,gamma,two in [('gamma_after_blur',2.2,False),('two_separate_edges',1.,True)]:
      f=fit_patch(xy,independent_synthetic(xy,n,0.,1.2,gamma,two),n,t,0.)
      counter.append(dict(kind=kind,known_first_edge_offset=0.,result=f,absolute_bias=abs(f['offset'])))
    # Same physical straight edge, but standard interpolated profiles can have more than one peak.
    x,y=np.meshgrid(np.arange(41),np.arange(41));grid=np.stack([x,y],-1).reshape(-1,2);peak_examples=[]
    for angle in [0.,.025,.35,1.2]:
      t=np.array([np.cos(angle),np.sin(angle)]);n=np.array([-t[1],t[0]])
      values=independent_synthetic(grid-np.array([20.3,20.6]),n,.2,.6).reshape(41,41,3)
      image=np.clip(np.rint(values),0,255).astype(np.uint8)
      peaks=old_peaks(image,[20.3,20.6],t);peak_examples.append(dict(angle=angle,local_peaks=peaks,within2px=sum(abs(z-.2)<2 for z in peaks)))
    return dict(known_edge_cases=len(positive),max_known_edge_error_px=max(x['error'] for x in positive),
       positive=positive,model_mismatch_controls=counter,interpolated_single_edge_peak_examples=peak_examples,
       source_images_smoothed=False,physical_accuracy_claim=False)

def main():
    assert not O.exists();O.mkdir();P=read(D/'plan.json');M=read(D/'measurements.json');AP=read(A/'plan.json')
    start=time.monotonic();budget=240.
    plan=dict(stage='D4_native_edge_spread',updated=now(),
      hypothesis='Quarter-pixel bilinear derivative extrema can split a single blurred boundary; local intensity edge-spread fitting on actual native integer pixels can avoid that peak artifact. Multiple physical edges and unknown processing response may still bias it.',
      controlled_change='Replace gradient-peak localization with native-pixel local edge-spread localization. Use identical registered11sourceedges/120locations/3PNGs. No new straightness, distortion, pose, or material-track optimization.',
      sampling='Actual decoded RGB integer pixels with normal distance<=8px and tangent distance<=4px from each original line sample. No source remap or image smoothing.',
      model='Pixel-area integrated Gaussian-blurred step plus separate per-channel affine intensity planes and contrast. Three local geometry parameters: offset, orientation, log-sigma; linear intensity parameters eliminated by least squares.',
      bounds=dict(offset_px=[-7.99,7.99],angle_rad=[-.15,.15],sigma_px=[.35,3.]),
      init='One unchanged approximate D3 normal-offset seed per physical line at every location; no strongest-peak init, restarts, propagation or parameter sweep.',
      controls='36 independent area-integrated known single-edge cases; gamma-after-blur and two-physical-edge counterexamples; compare old interpolated extrema on known single edges.',
      acceptance='Synthetic implementation must localize all36knowncases within0.05px before real processing. Real estimates remain proposals: report all fits, bounds, raw residuals and original occlusions, without residual-based deletion or calibration admission. Formal iid uncertainty is not physical uncertainty.',
      budget=dict(cpu_seconds=budget,max_nfev_per_patch=40,images=3,lines=11,locations_per_line=120,coefficient_or_camera_fits=0),
      exposure='These architectural groups are development-exposed; no new independent holdout is claimed.',
      source_sha256={str(p):sha(p) for p in [D/'plan.json',D/'measurements.json',A/'plan.json']},
      stop='No second run or threshold/model sweep. Stop on implementation failure or elapsed budget. No D1 refit or human training. Model mismatch is retained and reported.',no_GPU=True)
    save(O/'plan.json',plan);save(O/'progress.json',dict(state='controls',pid=os.getpid(),updated=now()))
    c=controls();save(O/'controls.json',c)
    cache={};records=[]
    for line in M['records']:
      if line['id'] not in AP['anchors']:continue
      f=line['frame'];path=DATA/f'frame_{f:06d}.png'
      if f not in cache:
        assert sha(path)==P['input_hashes'][path.name];cache[f]=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
      a,b=np.array(line['a'],float),np.array(line['b'],float);t=(b-a)/np.linalg.norm(b-a);samples=[]
      for i in range(120):
        if time.monotonic()-start>budget:raise TimeoutError('D4 registered CPU budget')
        center=a+(b-a)*i/119;uv,xy,n,t=patch_coordinates(center,t)
        inside=(uv[:,0]>=0)&(uv[:,0]<1520)&(uv[:,1]>=0)&(uv[:,1]<2704);uv=uv[inside];xy=xy[inside]
        rgb=cache[f][uv[:,1],uv[:,0]].astype(float)
        fit=fit_patch(xy,rgb,n,t,AP['anchors'][line['id']]['normal_offset_px'])
        fit.update(index=i,uv=(center+n*fit['offset']).tolist(),native_pixels=len(uv),
          original_eligible=line['valid'][i],original_reasons=line['reasons'][i],original_offset_px=line['offsets'][i],
          preregistered_occlusion=any(lo<=i/119<=hi for lo,hi in line['exclude_fraction']),physical_identity_confirmed=False)
        samples.append(fit)
      records.append(dict(id=line['id'],frame=f,split=line['split'],physical_group=line['physical_group'],samples=samples,
        seed=AP['anchors'][line['id']],a=line['a'],b=line['b'],normal=line['normal']))
      save(O/'partial.json',dict(updated=now(),records=records,terminal=False))
      save(O/'progress.json',dict(state='native_localization',pid=os.getpid(),updated=now(),completed_lines=[x['id'] for x in records],seconds=time.monotonic()-start))
      print(line['id'], 'done',round(time.monotonic()-start,2),flush=True)
    for path,digest in plan['source_sha256'].items():assert sha(Path(path))==digest
    result=dict(stage='D4',state='complete_pending_review',updated=now(),seconds=time.monotonic()-start,records=records,
      source_hashes_unchanged=True,source_pixels_changed=False,K_changed=False,coefficient_fit=False,camera_fit=False,training=False,
      physical_accuracy_improved=None,physical_calibration_accepted=False)
    save(O/'result.json',result);save(O/'progress.json',dict(state='complete_pending_review',pid=None,updated=now(),seconds=time.monotonic()-start))
    print(json.dumps(dict(state=result['state'],lines=len(records),seconds=result['seconds'])))

if __name__=='__main__':
  cv2.setNumThreads(2)
  try:main()
  except Exception as e:
    if O.exists():save(O/'failure.json',dict(updated=now(),type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));save(O/'progress.json',dict(state='failed',pid=None,updated=now()))
    raise
