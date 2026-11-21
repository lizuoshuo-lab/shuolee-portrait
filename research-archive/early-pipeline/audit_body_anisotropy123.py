"""Read-only geometry census of two immutable snapshots; no fitting or pruning."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
from plyfile import PlyData

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/body-gaussian-upstream-offset')
OUT=ROOT/'geometry-compensation-audit'
MODEL=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
STEPS=(1000,120000)
PARTS={'head':[15,22,23,24],'neck':[12],'torso':[0,3,6,9],
 'left_upper_arm':[13,16],'right_upper_arm':[14,17],
 'left_forearm':[18],'right_forearm':[19],
 'left_hand':[20]+list(range(25,40)),'right_hand':[21]+list(range(40,55)),
 'left_thigh':[1],'right_thigh':[2],'left_shin':[4],'right_shin':[5],
 'left_foot':[7,10],'right_foot':[8,11]}

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-20)
def ply(p):return PlyData.read(p)['vertex'].data
def fields(a,names):return np.stack([a[n] for n in names],-1).astype(np.float64)
def summary(x,w=None):
    x=np.asarray(x);assert len(x) and np.all(np.isfinite(x))
    r=dict(median=float(np.median(x)),p90=float(np.quantile(x,.9)),p99=float(np.quantile(x,.99)),max=float(np.max(x)))
    if w is not None:
        order=np.argsort(x);c=np.cumsum(w[order]);c/=c[-1]
        r['opacity_weighted_median']=float(np.interp(.5,c,x[order]))
        r['opacity_weighted_p90']=float(np.interp(.9,c,x[order]))
    return r

def triangle_distance(p,t):
    a,b,c=t[:,0],t[:,1],t[:,2];u=b-a;v=c-a
    n=unit(np.cross(u,v));signed=np.sum((p-a)*n,axis=1)
    projected=p-signed[:,None]*n;d=projected-a
    uu=np.sum(u*u,1);uv=np.sum(u*v,1);vv=np.sum(v*v,1)
    du=np.sum(d*u,1);dv=np.sum(d*v,1);den=uu*vv-uv*uv
    assert np.all(den>1e-24),'Degenerate bound triangle: report and stop'
    beta=(du*vv-dv*uv)/den;gamma=(dv*uu-du*uv)/den
    inside=(beta>=0)&(gamma>=0)&(beta+gamma<=1)
    edge=[]
    for x,y in ((a,b),(b,c),(c,a)):
        e=y-x;fraction=np.clip(np.sum((p-x)*e,1)/np.sum(e*e,1),0,1)
        edge.append(np.linalg.norm(p-(x+fraction[:,None]*e),axis=1))
    distance=np.where(inside,np.abs(signed),np.min(edge,axis=0))
    return distance,np.abs(signed),inside

def prepare():
    assert not OUT.exists();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    files=[Path(__file__),MODEL,ROOT/'fit/fitted.npz',ROOT/'data/training-inputs.json']
    for step in STEPS:
        folder=ROOT/'gaussian-run'/f'step-{step:06d}'
        files += [folder/n for n in ('world.ply','bound-local.ply','binding.npz','current-meshes.npz','result.json')]
    OUT.mkdir()
    save(OUT/'plan.json',dict(created=now(),state='prepared',steps=list(STEPS),points='Every Gaussian, including low opacity; additionally report opacity-weighted summaries.',
      question='Are elongated Gaussians and mesh-relative center offsets present and concentrated anatomically? This census does not prove why the optimizer created them.',
      quantities=['sigma_max/sigma_mid elongation','sigma_mid/sigma_min flattening','sigma_max/sigma_min total anisotropy',
       'distance to own bound triangle, not globally nearest surface','absolute distance to own triangle plane','Gaussian-to-mesh-centroid distance',
       'change of fitted mesh vertices since Gaussian initialization'],parts=PARTS,reference_frame='frame_000033',cpu_budget_seconds=120,
      validation=dict(world_xyz_max_abs_m=2e-6,scale_max_abs_m=2e-6),
      scope=['Snapshot distributions have different point counts after adaptive density; they are not matched point trajectories.',
       'Flat surface Gaussians, clothing offsets and low-opacity outliers are not by themselves errors.',
       'Part labels use mean triangle skinning weights, not true clothing segmentation.',
       'Fitted metre units are not independently measured physical scale.',
       'No thresholds select training data or prune Gaussians. No reconstruction or new renderer run.',
       'Current Gaussian snapshots used the old VHAP long fit, not the newer body-weight fitting candidate.'],
      inputs={str(p):sha(p) for p in files},quality_accepted=False))
    print('ANISOTROPY_PLAN_PREPARED',flush=True)

def verify():
    p=read(OUT/'plan.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    return p

def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    p=verify();assert not (OUT/'progress.json').exists()
    save(OUT/'progress.json',dict(state='running_cpu',pid=os.getpid()))
    started=time.monotonic()
    def deadline(a,b):raise TimeoutError('Preregistered CPU budget exceeded')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(p['cpu_budget_seconds'])
    try:
        # Known triangle sanity checks, including outside-plane projection.
        tri=np.tile(np.array([[0.,0,0],[1,0,0],[0,1,0]]),(3,1,1))
        d,_,_=triangle_distance(np.array([[.2,.2,2],[2,0,0],[.2,.2,0]]),tri)
        assert np.allclose(d,[2,1,0],atol=1e-12)
        initial=dict(np.load(ROOT/'fit/fitted.npz',allow_pickle=False))
        weights=np.load(MODEL,allow_pickle=False)['weights']
        rows=read(ROOT/'data/training-inputs.json')['records'];reports=[];arrays={}
        assert rows[0]['frame']=='frame_000033'
        for step in p['steps']:
            folder=ROOT/'gaussian-run'/f'step-{step:06d}'
            snap=read(folder/'result.json');assert snap['reference_slot']==0 and snap['reference_frame']==rows[0]['frame']
            assert sha(folder/'world.ply')==snap['world_ply_sha256']
            world=ply(folder/'world.ply');local=ply(folder/'bound-local.ply')
            xyz=fields(world,['x','y','z']);loc=fields(local,['x','y','z'])
            scales=np.exp(fields(world,['scale_0','scale_1','scale_2']))
            local_scales=np.exp(fields(local,['scale_0','scale_1','scale_2']))
            opacity=1/(1+np.exp(-world['opacity'].astype(np.float64)))
            bind=np.load(folder/'binding.npz',allow_pickle=False)['triangle_ids']
            mesh=dict(np.load(folder/'current-meshes.npz',allow_pickle=False))
            assert list(mesh['frame_ids'])==[r['frame'] for r in rows]==list(initial['frame_ids'])
            assert np.array_equal(mesh['faces'],initial['faces']) and len(bind)==len(xyz)==snap['points']
            E=np.asarray(rows[0]['w2c']);verts=(mesh['vertices_camera'][0]-E[:3,3])@E[:3,:3]
            all_tri=verts[mesh['faces']];t=all_tri[bind]
            e0=unit(t[:,1]-t[:,0]);e1=unit(np.cross(e0,t[:,2]-t[:,0]));e2=-unit(np.cross(e1,e0))
            orient=np.stack([e0,e1,e2],-1)
            factor=.5*(np.linalg.norm(t[:,1]-t[:,0],axis=1)+np.abs(np.sum(e2*(t[:,2]-t[:,0]),1)))
            reconstructed=np.einsum('nij,nj->ni',orient,loc)*factor[:,None]+t.mean(1)
            xe=float(np.max(np.abs(reconstructed-xyz)));se=float(np.max(np.abs(scales-local_scales*factor[:,None])))
            assert xe<p['validation']['world_xyz_max_abs_m'] and se<p['validation']['scale_max_abs_m'],(xe,se)
            ordered=np.sort(scales,axis=1);dist,plane,inside=triangle_distance(xyz,t)
            values={'elongation':ordered[:,2]/ordered[:,1],'flattening':ordered[:,1]/ordered[:,0],
              'total_anisotropy':ordered[:,2]/ordered[:,0], 'sigma_max_mm':ordered[:,2]*1000,
              'bound_triangle_distance_mm':dist*1000,'bound_plane_distance_mm':plane*1000,
              'centroid_distance_mm':np.linalg.norm(xyz-t.mean(1),axis=1)*1000}
            joint=weights[mesh['faces']].mean(1).argmax(1);gaussian_joint=joint[bind]
            vjoint=weights.argmax(1);delta=np.linalg.norm(mesh['vertices_camera']-initial['vertices_camera'],axis=-1)*1000
            grouping={}
            for name,joints in {'all':list(range(55)),**p['parts']}.items():
                selected=np.isin(gaussian_joint,joints);vmask=np.isin(vjoint,joints)
                assert selected.any() and vmask.any(),name
                grouping[name]=dict(points=int(selected.sum()),triangles=int(np.unique(bind[selected]).size),
                  opacity=summary(opacity[selected]),metrics={k:summary(v[selected],opacity[selected]) for k,v in values.items()},
                  mesh_displacement_mm_all59=summary(delta[:,vmask].ravel()),
                  center_orthogonal_projection_inside_bound_triangle_fraction=float(np.mean(inside[selected])))
            arrays[f's{step}_binding']=bind;arrays[f's{step}_part_joint']=gaussian_joint
            arrays.update({f's{step}_{k}':v for k,v in values.items()})
            reports.append(dict(step=step,points=len(xyz),world_reconstruction_max_abs_m=xe,scale_reconstruction_max_abs_m=se,regions=grouping))
        verify();np.savez_compressed(OUT/'point-metrics-only.npz',**arrays)
        result=dict(updated=now(),state='complete',reports=reports,plan_sha256=sha(OUT/'plan.json'),
          elapsed_seconds=time.monotonic()-started,limitations=p['scope'],training_changed=False,renderer_or_coefficients_changed=False,
          quality_accepted=False,compensation_causal_proof=False)
        save(OUT/'result.json',result);save(OUT/'progress.json',dict(state='complete',pid=None))
        for r in reports:
            print('SNAPSHOT',r['step'],r['points'],json.dumps(r['regions']['all']['metrics']),flush=True)
    except BaseException as e:
        save(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        save(OUT/'progress.json',dict(state='failed_preserved',pid=None));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run']);args=a.parse_args()
    prepare() if args.mode=='prepare' else run()
