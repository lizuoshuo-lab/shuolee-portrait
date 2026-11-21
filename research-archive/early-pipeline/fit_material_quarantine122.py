"""N10: one observed material-identity correction, fixed K, fresh geometry."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import sys, time, traceback
import numpy as np
import cv2
from scipy.sparse import coo_matrix
import shared_background122 as common
from audit_shared_solver122 import Problem, huber_cost
from repair_shared_solver122 import schur_step, linear_control

B=common.B; I=common.I; O=B/'material-quarantine'; P=B/'material-quarantine-plan.json'
read,save,sha,now=common.read,common.save,common.sha,common.now

def register():
    assert not P.exists() and not O.exists()
    review=read(B/'offplane-support/visual-review.json')
    assert review['material_contradictions']==[399]
    raw=read(B/'observations.json')['tracks']
    quarantine=[t for t in raw if t['family']==399]
    assert len(quarantine)==1 and quarantine[0]['split']=='fit'
    files=[B/'observations.json',I/'background-geometry-plan.json',B/'offplane-support/visual-review.json',
        B/'offplane-support/track399-context-index.json',B/'offplane-support/track399_native_context.png',
        I/'native-temporal-material/observations.npz',I/'native-temporal-material/seeds.json',
        Path(__file__).parent/'audit_shared_solver122.py',Path(__file__).parent/'repair_shared_solver122.py',Path(__file__).parent/'shared_background122.py']
    save(P,dict(stage='N10',registered=now(),source_sha256=sha(Path(__file__)),
        hypothesis='An actually inspected FIT material switch falsely appears as useful off-plane/parallax support and can bias the shared camera path. Test its causal effect separately from K, mask support, motion and matching algorithms.',
        evidence='N9 native320px crops: family399 frames246/262 near sandbar; frame278 different far-bank material. Selection was via the preregistered topdecile conditional plane-distance audit, not reprojection-error pruning.',
        single_change='Quarantine whole FIT family399 (one track,3 observations); keep it in raw data and an explicitly excluded diagnostic group. Never reassign it to independent heldout.',
        unchanged=['Same7 nativeframes','Same supplied firstK4000 resize/crop hypothesis','Same other934FIT tracks and190heldout tracks','Same seed1225/PoseLib/PnP/positive-depth/parallax initialization policies','Same Huber1px and checked Schur optimizer','Same firstcamera identity and unit firstbaseline gauge'],
        initialization='Recompute first-pair PoseLib E from original FIT pixels; then fresh DLT and PnP. Do not load any old optimized camera, point, parameter vector, Gaussian, or checkpoint as initialization.',
        validation='Every original heldout observation retained. Target-excluded DLT for the same9face,38clothing,20mixedfootROI and240ground tracks; ground may physically overlap FIT, so it is descriptive only. Do not turn raw face candidates into certified ground truth.',
        budget=dict(attempts=1,CPU_seconds=180,max_iterations=80,damping_trials=8,GPU=0),
        stop='Preserve failure on registration, identity/hash or budget failure; no relaxed thresholds or rescanning. A small/no improvement refutes this single mismatch as the main cause; do not clean additional points based on residuals.',
        admission='Diagnostic7frame path only. No fullframe camera promotion or training; physical point certification and fullvideo coverage still required.',
        quarantine_families=[399],input_hashes={str(f):sha(f) for f in files}))
    print('N10_REGISTERED',flush=True)

class FreshProblem(Problem):
    def __init__(self):
        self.plan=read(I/'background-geometry-plan.json');self.frames=self.plan['frames']
        self.K=np.array(self.plan['K_integer_centers'],dtype=float)
        raw=read(B/'observations.json')['tracks']
        fit=[t for t in raw if t['split']=='fit' and t['family']!=399]
        assert len(fit)==934
        self.fit=fit
        sys.path.insert(0,'/opt/gs-slam-lab/camera-contract120-tools/poselib-2.0.5')
        import poselib
        first=[]
        for t in fit:
            obs={o['frame']:o['uv'] for o in t['observations']}
            if all(f in obs for f in self.frames[:2]):first.append((obs[self.frames[0]],obs[self.frames[1]]))
        a=np.array([x for x,y in first]);b=np.array([y for x,y in first]);K=self.K
        cam=dict(model='PINHOLE',width=1520,height=2704,params=[K[0,0],K[1,1],K[0,2],K[1,2]])
        pose,info=poselib.estimate_relative_pose(a,b,cam,cam,dict(max_epipolar_error=1.,min_iterations=100,max_iterations=10000,seed=1225),{})
        assert pose is not None
        cameras={self.frames[0]:np.eye(4),self.frames[1]:np.eye(4)}
        cameras[self.frames[1]][:3,:3]=pose.R;cameras[self.frames[1]][:3,3]=pose.t/np.linalg.norm(pose.t)
        points={};init=[]
        for f in self.frames[2:]:
            for t in fit:
                if t['track'] not in points:
                    X=common.track_initial_point(K,cameras,t)
                    if X is not None:points[t['track']]=X
            available=[(t['track'],ob['uv']) for t in fit if t['track'] in points for ob in t['observations'] if ob['frame']==f]
            assert len(available)>=30
            xyz=np.array([points[t] for t,uv in available]);uv=np.array([uv for t,uv in available])
            ok,rv,tv,ins=cv2.solvePnPRansac(xyz,uv,K,None,iterationsCount=1000,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
            assert ok and ins is not None and len(ins)>=30
            M=np.eye(4);M[:3,:3]=cv2.Rodrigues(rv)[0];M[:3,3]=tv.ravel();cameras[f]=M
            init.append(dict(frame=f,available=len(available),PnP_inliers=len(ins)))
        points={t['track']:X for t in fit if (X:=common.track_initial_point(K,cameras,t)) is not None}
        self.tids=np.array(sorted(points));self.slices={};params=[]
        for i,f in enumerate(self.frames[1:],1):
            rv=cv2.Rodrigues(cameras[f][:3,:3])[0].ravel();t=cameras[f][:3,3]
            v=np.r_[rv,[np.arctan2(t[1],t[0]),np.arcsin(np.clip(t[2],-1,1))]] if i==1 else np.r_[rv,t]
            self.slices[f]=slice(len(params),len(params)+len(v));params.extend(v)
        self.ncam=len(params);self.x0=np.r_[params,np.array([points[t] for t in self.tids]).ravel()]
        lookup={int(t):i for i,t in enumerate(self.tids)}
        obs=[(o['frame'],lookup[t['track']],o['uv']) for t in fit if t['track'] in lookup for o in t['observations']]
        self.target=np.array([v for f,i,v in obs]);self.indices=np.array([i for f,i,v in obs])
        self.groups={f:np.where([ff==f for ff,i,v in obs])[0] for f in self.frames}
        rows=[];cols=[]
        for r,(f,i,uv) in enumerate(obs):
            cc=list(range(self.ncam+3*i,self.ncam+3*i+3))
            if f in self.slices:cc+=list(range(self.slices[f].start,self.slices[f].stop))
            for rr in [2*r,2*r+1]:rows.extend([rr]*len(cc));cols.extend(cc)
        self.sparsity=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(2*len(obs),len(self.x0))).tocsr()
        self.initialization=dict(source='fresh raw-pixel E+PnP+DLT, no old optimized vector loaded',first_pair_fit=len(first),first_pair_inliers=int(info['num_inliers']),
            first_pair_R=pose.R.tolist(),first_pair_t=cameras[self.frames[1]][:3,3].tolist(),PnP=init,initialized_tracks=len(points),
            raw_fit_tracks=len(fit),initialization_excluded_track_ids=[t['track'] for t in fit if t['track'] not in points])

def run():
    assert not O.exists();plan=read(P);assert plan['source_sha256']==sha(Path(__file__))
    for f,h in plan['input_hashes'].items():assert sha(Path(f))==h
    O.mkdir();start=time.monotonic();cv2.setNumThreads(2);cv2.setRNGSeed(1225)
    save(O/'progress.json',dict(state='fresh_initialization',pid=os.getpid(),updated=now()))
    p=FreshProblem();save(O/'initialization.json',p.initialization)
    np.savez_compressed(O/'fresh-initialization.npz',parameters=p.x0,track_ids=p.tids)
    x=p.x0.copy();checks=dict(schur_dense_error=linear_control())
    rng=np.random.default_rng(122);v=rng.normal(size=len(x));v/=np.max(abs(v));h=1e-7
    analytic=p.jacobian(x)@v;numeric=(p.residual(x+h*v)-p.residual(x-h*v))/(2*h)
    checks['fresh_directional_relative_error']=float(np.linalg.norm(analytic-numeric)/np.linalg.norm(analytic))
    assert checks['fresh_directional_relative_error']<1e-6
    cost=huber_cost(p.residual(x));initial_cost=cost;damping=.001;history=[];reason='max_iterations'
    for it in range(80):
        if time.monotonic()-start>180:reason='wall_budget';break
        r=p.residual(x);j=p.jacobian(x);accepted=False
        for trial in range(8):
            dx,g,d=schur_step(j,r,p.ncam,damping);xr=x+dx;tr=p.residual(xr);cams,xyz=p.decode(xr)
            positive=all(np.all(common.project(p.K,cams[f],xyz[p.indices[ix]])[1]>0) for f,ix in p.groups.items())
            tc=huber_cost(tr) if np.isfinite(tr).all() and positive else float('inf')
            if tc<cost:
                prev=cost;x=xr;cost=tc;accepted=True;used=damping;damping=max(damping/3,1e-12);break
            damping*=10
        item=dict(iteration=it,accepted=accepted,cost=cost,gradient_inf=float(np.max(abs(g))),damping=float(used if accepted else damping),trials=trial+1,elapsed_seconds=time.monotonic()-start)
        history.append(item);save(O/'progress.json',dict(state='optimizing_diagnostic_only',pid=os.getpid(),updated=now(),last=item));print(item,flush=True)
        if not accepted:reason='no_descent_with_registered_budget';break
        if max(abs(dx))<1e-9 or np.max(abs(g))<1e-5:reason='numerical_stationarity';break
        if prev-cost<1e-9*max(1,prev):reason='relative_cost_change';break
    cams,xyz=p.decode(x);raw=read(B/'observations.json')['tracks'];held=[t for t in raw if t['split']=='heldout']
    validation=dict(background=common.evaluate(p.K,cams,held),quarantined399=common.evaluate(p.K,cams,[t for t in raw if t['family']==399]))
    seeds=read(I/'native-temporal-material/seeds.json')['seeds'];d=np.load(I/'native-temporal-material/observations.npz')
    for group in ['face_candidate','clothing_candidate','footwear_candidate','ground_candidate']:
        ts=[dict(track=s['id'],family=s['family'],observations=[dict(frame=f,uv=d['uv'][f,s['id']].tolist()) for f in p.frames if d['valid'][f,s['id']]]) for s in seeds if s['group']==group]
        ev=common.evaluate(p.K,cams,ts)
        ev['raw_slots']=len(ts)*len(p.frames);ev['numeric_invalid_slots']=ev['raw_slots']-ev['summary']['n_total']
        ev['count_ge1']=sum(o['valid_projection'] and o['error_px']>=1 for o in ev['observations'])
        validation[group]=ev
    base=read(B/'shared-path/solver-repair/result.json');n8=read(B/'material-region-comparison/result.json')
    comparisons={g:dict(baseline=n8['groups'][g],candidate=validation[g]['summary']) for g in ['face_candidate','clothing_candidate','footwear_candidate','ground_candidate']}
    comparisons['background']=dict(baseline=base['validation']['background']['summary'],candidate=validation['background']['summary'])
    # Baseline results first enter here, after fresh initialization and optimization.
    pose_difference=[]
    for c in base['cameras']:
        f=c['frame'];before=np.array(c['w2c']);delta=cams[f][:3,:3]@before[:3,:3].T
        pose_difference.append(dict(frame=f,rotation_degrees=float(np.degrees(np.linalg.norm(cv2.Rodrigues(delta)[0]))),translation_l2_baseline_units=float(np.linalg.norm(cams[f][:3,3]-before[:3,3]))))
    result=dict(stage='N10',updated=now(),state='complete_pending_review',elapsed_seconds=time.monotonic()-start,
        causal_variable='Whole physically contradicted FIT family399 quarantined; all other observations and K fixed',K_integer_centers=p.K.tolist(),K_optimized=False,
        actual_registered_frames=p.frames,fit_tracks=len(p.tids),quarantine_family_ids=[399],heldout_tracks=len(held),
        initialization=p.initialization,controls=checks,optimizer=dict(initial_cost=initial_cost,final_cost=cost,reason=reason,accepted_steps=sum(h['accepted'] for h in history),history=history,
        gradient_inf=float(np.max(abs(p.jacobian(x).T@np.clip(p.residual(x),-1,1))))),
        cameras=[dict(frame=f,w2c=cams[f].tolist()) for f in p.frames],validation=validation,comparisons=comparisons,pose_difference=pose_difference,
        original_observations_modified=False,all_training_frames_registered=False,physical_face_accepted=False,training_started=False,
        limitations='The actual quarantined error is proven, not its dominance. All other physical identities, fixedK export mapping, lowparallax support and personmotion remain uncertain. Existing face candidates are not independent subpixel ground truth.')
    save(O/'result.json',result);np.savez_compressed(O/'fit-diagnostic.npz',parameters=x,points=xyz,track_ids=p.tids)
    for f,h in plan['input_hashes'].items():assert sha(Path(f))==h
    save(O/'progress.json',dict(state='complete_pending_review',pid=None,updated=now()))
    print({g:v['summary'] for g,v in validation.items()},flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException as e:
        if O.exists():save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        raise
