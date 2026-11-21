"""Read-only audit of two independently completed fits of the same video.

This is not a single-factor experiment: upstream resolution, matches, masks and
software differ. No source/camera/model is modified or promoted. Reuses the
existing M0 anchor/held split without selecting tracks by the result.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
H=Path(__file__).parent
REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT=REC/'grayT-facial-articulation-review'
A=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/grayT')
B=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stats(x):
    a=np.array(x).reshape(-1);return dict(n=int(a.size),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max()))
def main():
    assert not (OUT/'cross-run-started.json').exists()
    p=read(OUT/'cross-run-plan.json');files=[Path(x['path']) for x in p['sources']]
    assert all(sha(f)==x['sha256'] for f,x in zip(files,p['sources']))
    save(OUT/'cross-run-started.json',dict(updated=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(OUT/'cross-run-plan.json'),source_sha256=sha(Path(__file__))))
    ra=read(A/'data/inputs.json')['records'];rb=read(B/'data/inputs.json')['records']
    assert len(ra)==len(rb)==95 and [x['source_frame_index'] for x in ra]==[x['source_frame_index'] for x in rb]
    assert [x['time_seconds'] for x in ra]==[x['time_seconds'] for x in rb]
    fa=np.load(A/'fit/fitted.npz');fb=np.load(B/'fit/fitted.npz');oa=np.load(A/'data/observations.npz');ob=np.load(B/'data/observations.npz')
    wa=np.load(A/'fit/observation-weights.npz');wb=np.load(B/'fit/observation-weights.npz')
    assert list(fa['frame_ids'])==list(fb['frame_ids'])==list(oa['frame_ids'])==list(ob['frame_ids'])
    assert list(oa['joint_names'])==list(ob['joint_names'])
    wha=np.array([r['wh'] for r in ra]);whb=np.array([r['wh'] for r in rb]);scale=whb/wha
    Ka=np.array([r['K_integer_centers'] for r in ra]);Kb=np.array([r['K_integer_centers'] for r in rb]);Kn=Kb.copy()
    Kn[:,0,:]/=scale[:,0,None];Kn[:,1,:]/=scale[:,1,None]
    Kn[:,0,2]+=.5/scale[:,0]-.5;Kn[:,1,2]+=.5/scale[:,1]-.5
    uvb=(ob['keypoints']+.5)/scale[:,None,:]-.5
    pb=(fb['projected_face51']+.5)/scale[:,None,:]-.5
    facea=wa['face']*wa['face_surface_visibility'];faceb=wb['face']
    valid=(facea>0)&(faceb>0)
    ta=oa['keypoints'][:,40:91];tb=uvb[:,40:91]
    Ea=np.array([r['w2c'] for r in ra]);Eb=np.array([r['w2c'] for r in rb])
    assert np.max(np.abs(Ea[0]-np.eye(4)))==np.max(np.abs(Eb[0]-np.eye(4)))==0
    dR=Ea[:,:3,:3]@Eb[:,:3,:3].transpose(0,2,1)
    angle=np.rad2deg(Rotation.from_matrix(dR).magnitude())
    Ca=-np.einsum('nji,nj->ni',Ea[:,:3,:3],Ea[:,:3,3]);Cb=-np.einsum('nji,nj->ni',Eb[:,:3,:3],Eb[:,:3,3])
    s=float(np.sum(Ca*Cb)/np.sum(Cb*Cb));cd=np.linalg.norm(Ca-s*Cb,axis=-1)
    md=read(REC/'grayT-skin-localization-review/result.json')['observations'];ii=np.array([x['frame_number']-1 for x in md]);uv=np.array([x['xy'] for x in md]);anchor=np.array([x['role']=='ANCHOR' for x in md])
    assert [x['frame_number'] for x in md if x['role']=='ANCHOR']==[40,48,56,64,72,80]
    def evaluate(E,K):
        E=E[ii];K=K[ii];P=K@E[:,:3];rows=[]
        for mat,pt in zip(P[anchor],uv[anchor]):rows.extend([pt[0]*mat[2]-mat[0],pt[1]*mat[2]-mat[1]])
        _,_,v=np.linalg.svd(rows);x=v[-1,:3]/v[-1,3]
        def project(x):
            xyz=np.einsum('nij,j->ni',E[:,:3,:3],x)+E[:,:3,3];q=np.einsum('nij,nj->ni',K,xyz)
            return q[:,:2]/q[:,2:],xyz[:,2]
        opt=least_squares(lambda x:(project(x)[0][anchor]-uv[anchor]).ravel(),x,method='lm',max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        pp,z=project(opt.x);err=np.linalg.norm(pp-uv,axis=-1)
        return dict(success=bool(opt.success),point=opt.x.tolist(),anchors=stats(err[anchor]),held=stats(err[~anchor]),held_ge1=int(np.sum(err[~anchor]>=1)),positive_depth=bool((z>0).all()),observations=[dict(frame_number=r['frame_number'],role=r['role'],xy=u.tolist(),projected=p.tolist(),error_px=float(e)) for r,u,p,e in zip(md,uv,pp,err)])
    a=evaluate(Ea,Ka);b=evaluate(Eb,Kn)
    assert all(sha(f)==x['sha256'] for f,x in zip(files,p['sources']))
    result=dict(updated=datetime.now(timezone.utc).isoformat(),same95_frames_and_timestamps=True,baseline_wh=wha[0].tolist(),company00_wh=whb[0].tolist(),
        K_native_max_abs=float(np.max(np.abs(Ka-Kn))),first_camera_gauge_identical=True,
        relative_rotation_difference_deg=stats(angle),center_scale_B_to_A=s,center_difference_A_world=stats(cd),center_relative_to_A_extent=stats(cd/np.max(np.linalg.norm(Ca,axis=-1))),
        common_valid_semantic_nodes=int(valid.sum()),semantic_observation_difference_native_px=stats(np.linalg.norm(ta-tb,axis=-1)[valid]),
        baseline_face_semantic_residual_native_px=stats(np.linalg.norm(fa['projected_face51']-ta,axis=-1)[valid]),company00_face_semantic_residual_native_px=stats(np.linalg.norm(pb-tb,axis=-1)[valid]),
        baseline_M0=a,company00_M0=b,sources_unchanged=True,
        not_single_factor_comparison=True,physical_camera_accuracy_accepted=False,training_started=False,
        interpretation='Same video, source frame identities and effective conditional K; independently estimated E differs. Processing resolution, feature sets, masks and fitter revisions also differ. Existing M0 split reused as secondary validation, not new ground truth. No rotation difference is automatically a motion/accuracy estimate.')
    save(OUT/'cross-run-result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['baseline_M0','company00_M0']},ensure_ascii=False))
    print(json.dumps({k:{s:v[s] for s in ['success','anchors','held','held_ge1','positive_depth']} for k,v in [('baseline_M0',a),('company00_M0',b)]}))
if __name__=='__main__':main()
