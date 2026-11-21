"""Evaluate one saved candidate on immutable new and historical image observations."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import signal
import time
import traceback
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont

B=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S=B/'current/full-scene-localization-074'
O=B/'current/portrait-alignment-open-source-116/pixsfm';V=O/'independent-validation';R=O/'ba-vgg-candidate'
LABELS=['background','face_lower','face_upper','torso','legs_shoes']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def save(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
    return h.hexdigest()
def quant(a):
    a=np.asarray(a);finite=np.isfinite(a);valid=a[finite]
    return dict(n=len(a),invalid=int((~finite).sum()),median=float(np.median(valid)),p90=float(np.percentile(valid,90)),rms=float(np.sqrt(np.mean(valid**2)))) if len(valid) else dict(n=len(a),invalid=len(a))
def project(x,E,k):
    c=E[:,:3,:3]@x+E[:,:3,3];n=c[:,:2]/c[:,2:]
    return n*(1+k[3]*(n*n).sum(1,keepdims=True))*k[0]+k[1:3]-.5,c[:,2]
def point_fit(ob,held,E,k):
    frames=ob[:,0].astype(int);fit=~held;w=E[frames[fit]]
    nd=(ob[fit,1:]+.5-k[1:3])/k[0];u=nd.copy()
    for _ in range(12):u=nd/(1+k[3]*(u*u).sum(1,keepdims=True))
    A=np.concatenate([u[:,0,None]*w[:,2]-w[:,0],u[:,1,None]*w[:,2]-w[:,1]])
    _,_,vt=np.linalg.svd(A,full_matrices=False);x=vt[-1,:3]/vt[-1,3]
    sol=least_squares(lambda x:(project(x,w,k)[0]-ob[fit,1:]).ravel(),x,loss='soft_l1',f_scale=2,
       max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    p,z=project(sol.x,E[frames],k)
    return sol.x,p,z,sol.success
def original_raw(xy,K,k):
    n=(xy+.5-K[:2,2])/[K[0,0],K[1,1]]
    return n*(1+k[3]*(n*n).sum(-1,keepdims=True))*k[0]+k[1:3]-.5
def timeout(*_):raise TimeoutError('Independent evaluation exceeded 30 minutes')

def main():
    assert (R/'result.json').exists() and read(R/'progress.json')['state']=='completed'
    assert not (R/'evaluation.json').exists() and not (R/'evaluation-progress.json').exists(),'Never silently repeat a completed/running evaluation.'
    registration=read(O/'ba-preregistration.json');vr=read(V/'result.json');assert vr['tracks_sha256']==sha(V/'tracks.npz')
    epath=O/'evaluation-preregistration.json';ep=read(epath);assert ep['runner_sha256']==sha(Path(__file__))
    seconds=min(1800,int((dt.datetime.fromisoformat(ep['overall_deadline_utc'])-dt.datetime.now(dt.timezone.utc)).total_seconds()))
    assert seconds>0,'P1 full diagnostic budget exhausted; do not start evaluation.'
    start=time.monotonic();signal.signal(signal.SIGALRM,timeout);signal.alarm(seconds)
    def progress(stage,**kw):save(R/'evaluation-progress.json',dict(updated=now(),state='running',pid=os.getpid(),stage=stage,**kw))
    try:
        with np.load(R/'candidate-cameras.npz') as z:E0=z['source_w2c'];E1=z['w2c'];k0=z['source_intrinsics'];k1=z['intrinsics']
        source=read(S/'cameras.json');assert np.max(abs(E0-np.array([r['w2c'] for r in source['records']])))<1e-9
        assert np.max(abs(k0-source['intrinsics']))<1e-9
        with np.load(V/'tracks.npz') as z:new={k:z[k] for k in z.files}
        new['groups']=np.repeat(new['groups'],np.diff(new['offsets']));new['labels']=LABELS
        oldfile=B/'current/shared-point-bundle-convergence-101/input.npz'
        with np.load(oldfile) as z:olddata={k:z[k] for k in z.files}
        assert sha(oldfile)==ep['old_bank_sha256']
        old={k:olddata[k].copy() for k in ['observations','held','offsets','groups']};old['labels']=LABELS
        old['observations'][:,1:]=original_raw(old['observations'][:,1:],olddata['K'],k0)
        # Anatomical IDs are an additional descriptive check, not physical texture GT.
        order=np.argsort(olddata['face_point_indices'],kind='stable');idx=olddata['face_point_indices'][order]
        fo=olddata['face_observations'][order].copy();fo[:,1:]=original_raw(fo[:,1:],olddata['K'],k0)
        ids=olddata['face_ids'][idx];ng=np.where((ids>=27)&(ids<=35),0,np.where(ids>=48,1,2))
        named=dict(observations=fo,held=olddata['face_held'][order],offsets=np.r_[0,np.flatnonzero(np.diff(idx))+1,len(idx)],groups=ng,labels=['nose','mouth','upper'])
        banks={'new_image_tracks':new,'historical_texture_validation':old,'historical_named_features':named}
        output={};newpred={};base_depth=[];failures=[]
        for name,bank in banks.items():
            ob=bank['observations'];off=bank['offsets'];h=bank['held'];g=bank['groups'];output[name]={}
            for arm,E,k in [('source074',E0,k0),('pixsfm_vgg',E1,k1)]:
                pred=np.empty((len(ob),2));xyz=[];positive=True;failed=[];depths=[]
                progress('fresh_point_only_validation',bank=name,arm=arm,tracks_total=len(off)-1)
                for tid,(lo,hi) in enumerate(zip(off[:-1],off[1:])):
                    x,p,z,ok=point_fit(ob[lo:hi],h[lo:hi],E,k);pred[lo:hi]=p;xyz.append(x);depths.extend(z.tolist())
                    if not ok:failed.append(int(tid))
                    positive=positive and bool(np.isfinite(p).all() and np.isfinite(x).all() and (z>0).all())
                    if (tid+1)%2000==0:progress('fresh_point_only_validation',bank=name,arm=arm,tracks_complete=tid+1,tracks_total=len(off)-1)
                err=np.linalg.norm(pred-ob[:,1:],axis=1)
                result=dict(all_held=quant(err[h]),regions={label:quant(err[h&(g==j)]) for j,label in enumerate(bank['labels'])},
                    point_fit_solver_failures=failed,positive_all=positive,tracks=len(off)-1,observation_count=len(ob),held_observation_count=int(h.sum()))
                output[name][arm]=result
                np.savez_compressed(R/f'eval-{name}-{arm}.npz',prediction=pred,xyz=np.array(xyz),errors=err)
                if name=='new_image_tracks':
                    newpred[arm]=pred
                    if arm=='source074':base_depth=np.array(depths)
                print(json.dumps(dict(bank=name,arm=arm,held=result['regions'],positive_all=positive,failed_point_fits=len(failed))),flush=True)
        gates={}
        for name in ['new_image_tracks','historical_texture_validation']:
            control=output[name]['source074'];candidate=output[name]['pixsfm_vgg'];a=control['regions'];b=candidate['regions']
            gates[name]=dict(lowerface_median=b['face_lower']['median']<=.8*a['face_lower']['median'],lowerface_p90=b['face_lower']['p90']<=a['face_lower']['p90'],
                background=b['background']['p90']<=a['background']['p90']+.2,
                body=all(b[l]['p90']<=a[l]['p90']+.5 for l in ['torso','legs_shoes']),
                positive=control['positive_all'] and candidate['positive_all'],
                point_fits_converged=not(control['point_fit_solver_failures'] or candidate['point_fit_solver_failures']))
        C0=-np.einsum('fji,fj->fi',E0[:,:3,:3],E0[:,:3,3]);C1=-np.einsum('fji,fj->fi',E1[:,:3,:3],E1[:,:3,3])
        angles=Rotation.from_matrix(E1[:,:3,:3]@E0[:,:3,:3].transpose(0,2,1)).magnitude()*180/np.pi
        shifts=np.linalg.norm(C1-C0,axis=1)/np.median(base_depth)
        camera=dict(max_rotation_degrees=float(angles.max()),max_center_shift_over_baseline_median_depth=float(shifts.max()),
                    shared_focal_fractional_change=float(k1[0]/k0[0]-1),radial_change=float(k1[3]-k0[3]),gauge_fixed_pose_max_difference=float(abs(E1[0]-E0[0]).max()))
        gates['camera_sanity']=bool(angles.max()<=2 and shifts.max()<=.02 and abs(k1[0]/k0[0]-1)<=.02)
        solver=read(R/'result.json')['solver'];gates['native_solver_converged']=all('CONVERGENCE' in str(s.get('termination_type','')).upper() and 'NO_CONVERGENCE' not in str(s.get('termination_type','')).upper() for s in solver)
        # Fixed preregistered visual sample; observations and both projected points share one source crop.
        font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',15);panels=[];off=new['offsets'];ob=new['observations']
        for tid in vr['visual_sample_track_ids']:
            lo,hi=off[tid:tid+2];heldidx=np.flatnonzero(new['held'][lo:hi])+lo
            ids=np.unique(np.round(np.linspace(0,len(heldidx)-1,3)).astype(int));panel=Image.new('RGB',(1200,310),'white');d=ImageDraw.Draw(panel)
            label=LABELS[int(new['groups'][lo])];d.text((8,5),f'Track {tid} | {label} | held source-image observations; no crop deformation',font=font,fill='black')
            for col,j in enumerate(heldidx[ids]):
                fi,x,y=ob[j];origin=np.round([x-80,y-80]).astype(int)
                with Image.open(S/'images'/f'raw_{int(fi)+1:06d}.png') as im:crop=im.convert('RGB').crop((*origin,*(origin+160))).resize((256,256),Image.Resampling.LANCZOS)
                panel.paste(crop,(col*400+12,30))
                for xy,color in [(ob[j,1:],'#009f70'),(newpred['source074'][j],'#0065d0'),(newpred['pixsfm_vgg'][j],'#ed6a00')]:
                    px,py=(xy-origin)*1.6+[col*400+12,30];d.ellipse((px-5,py-5,px+5,py+5),outline=color,width=2)
                d.text((col*400+278,50),f'raw{int(fi)+1:06d}',font=font,fill='black')
                for n,(arm,color) in enumerate([('source074','#0065d0'),('pixsfm_vgg','#ed6a00')]):
                    err=np.linalg.norm(newpred[arm][j]-ob[j,1:]);d.text((col*400+278,82+n*22),f'{err:.2f} raw px',font=font,fill=color)
            d.text((8,290),'Green: independent image track | Blue:074 fresh point fit | Orange:PixSfM fresh point fit. Same observations, all tracks retained.',font=font,fill='#444444');panels.append(panel)
        figs=[]
        for startidx in range(0,len(panels),5):
            sub=panels[startidx:startidx+5];board=Image.new('RGB',(1200,310*len(sub)),'white')
            for j,p in enumerate(sub):board.paste(p,(0,310*j))
            name=f'held-projection-review-{startidx//5+1}.jpg';board.save(R/name,quality=95);figs.append(name)
        allgates=all(all(v.values()) if isinstance(v,dict) else v for v in gates.values())
        assert sha(V/'tracks.npz')==vr['tracks_sha256'] and sha(oldfile)==ep['old_bank_sha256']
        result=dict(completed=now(),elapsed_seconds=time.monotonic()-start,banks=output,camera=camera,gates=gates,
            numeric_screen_passed=allgates,visual_review='pending',figures=figs,
            observations_modified=False,source_model_modified=False,new_training_started=False,quality_accepted=False,root_cause_resolved=False,
            interpretation='Held observations fit neither camera nor test point; each camera arm freshly triangulates its own point from the same outer observations. Error reduction still cannot establish physical camera truth, static material identity, or Gaussian ghosting repair.')
        save(R/'evaluation.json',result);save(R/'evaluation-progress.json',dict(updated=now(),state='completed',pid=os.getpid(),numeric_screen_passed=allgates));print(json.dumps(dict(gates=gates,camera=camera)),flush=True)
    except BaseException as e:
        save(R/'evaluation-failure.json',dict(updated=now(),error=str(e),traceback=traceback.format_exc()))
        save(R/'evaluation-progress.json',dict(updated=now(),state='failed',pid=os.getpid(),error=str(e)));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
