"""Read-only triangulation of the already selected eight observations.

This tests whether the existing conditional cameras can supply missing joint
depth. No camera/pose optimization, new detections, or training is performed.
The result is not an independently measured 3D skeleton.
"""
from pathlib import Path
from datetime import datetime, timezone
import time
import hashlib
import numpy as np
from PIL import Image, ImageDraw
import audit_gaussianavatars_params123 as a

ROOT = a.REC/'cross-subject-sh0-124'
OUT = ROOT/'quality-first-eight124/grass-multiview-joint-check'
SRC = ROOT/'grass-input/data'
JOINTS = dict(left_shoulder=5, right_shoulder=6, left_elbow=7,
              right_elbow=8, left_wrist=9, right_wrist=10,
              left_hip=11, right_hip=12)


def clock():
    return datetime.now(timezone.utc).isoformat()


def local(path):
    path = str(path)
    if path.startswith('/mnt/'):
        path = path[5].upper()+':/'+path[7:]
    elif path.startswith('/opt/'):
        return Path('//wsl.localhost/Ubuntu-20.04'+path)
    return a.lp(path)


def project(point, K, E):
    camera = E[:, :3, :3]@point + E[:, :3, 3]
    pixel = np.einsum('nij,nj->ni', K, camera)
    return pixel[:, :2]/pixel[:, 2, None], camera[:, 2]


def triangulate(xy, confidence, K, E):
    # Fixed cameras, confidence-weighted linear pixel equations; no RANSAC,
    # error-based removal, threshold search, or nonlinear fitting.
    P = K@E[:, :3]
    eq = xy[:, :, None]*P[:, None, 2]-P[:, :2]
    weighted = eq*confidence[:, None, None]
    matrix = weighted[:, :, :3].reshape(-1, 3)
    rhs = -weighted[:, :, 3].reshape(-1)
    point, _, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=None)
    return point, int(rank), float(singular[0]/singular[-1])


def controls():
    K = np.repeat(np.array([[900.,0,321.25],[0,880.,238.75],[0,0,1.]])[None],8,0)
    E = np.repeat(np.eye(4)[None],8,0)
    E[:, 0, 3] = np.linspace(-1.,1.,8)
    E[:, 1, 3] = np.sin(np.arange(8))*.2
    known = np.array([.2,-.1,4.])
    xy, _ = project(known,K,E)
    q,rank,cond = triangulate(xy,np.ones(8),K,E)
    assert rank == 3 and np.max(abs(q-known)) < 1e-10
    shift=np.array([.8,.4,-.1]); F=E.copy();F[:,:3,3]-=shift
    moved,_,_=triangulate(xy,np.ones(8),K,F)
    assert np.max(abs(moved-q-shift))<1e-10
    scaled=E.copy();scaled[:,:3,3]*=2
    got,_,_=triangulate(xy,np.ones(8),K,scaled)
    assert np.max(abs(got-q*2))<1e-10
    altered=xy.copy();altered[0]+=900
    w=np.ones(8);w[0]=0
    got,_,_=triangulate(altered,w,K,E)
    assert np.max(abs(got-known))<1e-10
    return dict(count=4,passed=True,synthetic_only=True)


def main():
    assert not a.lp(OUT).exists(), 'Do not repeat this bounded fixed-camera diagnostic'
    rows = a.read(SRC/'inputs.json')['records']; ids=[r['frame'] for r in rows]
    selected = a.read(ROOT/'grass-input/selection.json')['training_frames']
    slots=[ids.index(n) for n in selected]; assert len(slots)==8
    # Four equal time targets from remaining frames. Chosen without errors,
    # fit quality, or detector confidence; never enter the triangulation.
    ts=np.array([r['time_seconds'] for r in rows]); remaining=set(range(len(rows)))-set(slots)
    holdout=[]
    for t in np.linspace(ts[0],ts[-1],6)[1:-1]:
        i=min(remaining,key=lambda i:(abs(ts[i]-t),i));holdout.append(i);remaining.remove(i)
    paths=[SRC/'inputs.json',SRC/'observations.npz',ROOT/'grass-input/selection.json',Path(__file__)]
    sources={str(p):a.sha(p) for p in paths}
    a.save(OUT/'plan.json',dict(updated=clock(),state='registered_read_only',
        hypothesis='Frozen conditional camera rays may constrain wrist depth beyond monocular SMPLify-X and rotation smoothing.',
        fit_frames=selected,verification_frames=[ids[i] for i in holdout],
        joints=JOINTS,assumption='Same anatomical joint approximately static over these sampled times; not established material correspondence.',
        independent_camera_calibration=False,static_body_confirmed=False,
        method='Confidence-weighted fixed-camera linear triangulation; no rejection by residual and no sweep.',
        budget=dict(cpu_seconds=60,executions=1,new_fit_calls=0,new_camera_updates=0,new_gaussian_steps=0),
        sources=sources,old_poses_used=False,training_admitted=False))
    started=time.monotonic()
    with np.load(local(SRC/'observations.npz'),allow_pickle=False) as z:
        assert z['frame_ids'].astype(str).tolist()==ids
        xy=z['keypoints'].astype(float);scores=z['scores'].astype(float)
        names=z['joint_names'].astype(str)
    K=np.array([r['K_integer_centers'] for r in rows]);E=np.array([r['w2c'] for r in rows])
    assert np.isfinite(K).all() and np.isfinite(E).all()
    assert np.allclose(E[:,:3,:3]@np.swapaxes(E[:,:3,:3],1,2),np.eye(3),atol=1e-5)
    assert np.allclose(E[:,3],[0,0,0,1])
    for i,r in enumerate(rows):
        inside=(xy[i,:,0]>=0)&(xy[i,:,0]<r['wh'][0])&(xy[i,:,1]>=0)&(xy[i,:,1]<r['wh'][1])
        scores[i]*=inside
    tests=controls(); results={}; points={}
    for name,j in JOINTS.items():
        assert name in names[j].lower().replace(' ','_'), (name,names[j])
        point,rank,condition=triangulate(xy[slots,j],scores[slots,j],K[slots],E[slots])
        uv,depth=project(point,K,E); errors=np.linalg.norm(uv-xy[:,j],axis=-1)
        loo=[triangulate(xy[sub,j],scores[sub,j],K[sub],E[sub])[0] for omit in slots
             for sub in [[s for s in slots if s!=omit]]]
        points[name]=point
        results[name]=dict(point_conditional_world=point.tolist(),matrix_rank=rank,condition=condition,
            positive_depth_in_all_selected=bool(np.all(depth[slots]>0)),
            contributing_observations=int(np.count_nonzero(scores[slots,j]>0)),
            per_selected=[dict(frame=ids[i],error_px=float(errors[i]),confidence=float(scores[i,j])) for i in slots],
            per_verification=[dict(frame=ids[i],error_px=float(errors[i]),confidence=float(scores[i,j])) for i in holdout],
            leave_one_out_displacement_world=[float(np.linalg.norm(q-point)) for q in loo])
    images=[]
    for i in sorted(slots+holdout):
        r=rows[i];raw=Image.open(local(r['image'])).convert('RGB');assert hashlib.sha256(local(r['image']).read_bytes()).hexdigest()==r['source_sha256']
        box=np.array(r['initial_box_xyxy']);w,h=box[2:]-box[:2];pad=.1*max(w,h)
        crop=(max(0,int(box[0]-pad)),max(0,int(box[1]-pad)),min(raw.width,int(box[2]+pad)),min(raw.height,int(box[3]+pad)))
        overlay=raw.copy();draw=ImageDraw.Draw(overlay)
        for name,j in JOINTS.items():
            uv,_=project(points[name],K[i:i+1],E[i:i+1]);p=uv[0]
            if np.isfinite(p).all():
                x,y=p;draw.ellipse((x-5,y-5,x+5,y+5),outline=(255,60,200),width=3)
            x,y=xy[i,j];draw.ellipse((x-4,y-4,x+4,y+4),outline=(0,255,80),width=3)
        raw=raw.crop(crop);overlay=overlay.crop(crop)
        board=Image.new('RGB',(raw.width*2,raw.height));board.paste(raw,(0,0));board.paste(overlay,(raw.width,0))
        p=Path('D:/00/记录/005')/f'分析_胸前手臂124_{ids[i]}_原图_二维观测绿_条件三维投影紫.png'
        assert not a.lp(p).exists();board.save(a.lp(p))
        images.append(dict(frame=ids[i],role='triangulation_input' if i in slots else 'verification_only',path=str(p),sha256=a.sha(p)))
    shoulder_width=float(np.linalg.norm(points['left_shoulder']-points['right_shoulder']))
    for r in results.values():r['leave_one_out_displacement_over_shoulder_width']=[x/shoulder_width for x in r['leave_one_out_displacement_world']]
    assert all(a.sha(p)==d for p,d in sources.items())
    elapsed=time.monotonic()-started;assert elapsed<60
    a.save(OUT/'result.json',dict(updated=clock(),state='fixed_camera_observation_check_complete',
        joints=results,conditional_shoulder_width=shoulder_width,synthetic_controls=tests,seconds=elapsed,
        images=images,originals_unchanged=True,actual_image_review=False,
        training_admitted=False,physical_3d_truth=False,new_fit_calls=0,new_gaussian_steps=0))
    print({n:dict(selected_mean=np.mean([v['error_px'] for v in r['per_selected']]),
                 verify_mean=np.mean([v['error_px'] for v in r['per_verification']]),
                 verify_max=max(v['error_px'] for v in r['per_verification'])) for n,r in results.items()})


if __name__=='__main__':main()
