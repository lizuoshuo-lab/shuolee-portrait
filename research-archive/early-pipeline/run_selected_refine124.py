"""One finite original-stage refinement of the existing eight-view body pose."""
from pathlib import Path
import os,sys,time,signal,fcntl,traceback
import numpy as np
from PIL import Image
import torch
import run_grass_selected_views124 as shared
import smplify_seeded_refine124 as adapter
import official_smplifyx_entry123 as entry
from run_official_gaussians123 import read,save,sha,now

BASE=shared.OUT;OUT=shared.ROOT/'grass-selected-view-refine124';H=Path(__file__).resolve().parent
def load(p):
    with np.load(p,allow_pickle=False) as z:return dict(z)

def prepare():
    assert not OUT.exists()
    previous=read(BASE/'conclusion.json');assert previous['actual_native_body_face_feet_and_sides_reviewed']
    p=read(BASE/'plan.json');sources=[Path(__file__),Path(adapter.__file__),Path(entry.__file__),
        H/'control_smplifyx_root_seed123.py',H/'smplify_temporal124.py',Path(shared.__file__),
        BASE/'selected-fit.npz',BASE/'plan.json',BASE/'conclusion.json',
        shared.SRC/'data/inputs.json',shared.SRC/'data/observations.npz',shared.SRC/'selection.json']
    sources+=[BASE/'frames'/f/'winner-camera-space.npz' for f in p['frames']]
    sources+=[shared.OFFICIAL/k for k in entry.PINNED]
    code,delta=adapter.source(shared.OFFICIAL)
    OUT.mkdir();(OUT/'interface.diff').write_text(delta,encoding='utf-8')
    save(OUT/'plan.json',dict(updated=now(),state='registered',frames=p['frames'],rows=p['rows'],
        hypothesis='A multi-view common pose corrects arm depth but overconstrains small local motions. Test per-view original stage3/4 refinement from that one common estimate with the existing body-rotation penalty30.',
        variable='Free per-view VPoser local body pose around the learned common pose; every view starts from the same coherent archive.',
        initialization='Same saved8-view body/face/hand/root/placement parameters, no new zero or neighbour initialization.',
        unchanged=['fixed full K','frozen shared betas','native pixels and masks','118 observation mapping and all confidences','upstream last-two-stage weights,30 iterations and LBFGS','per-view root,placement,face,hand optimization'],
        regularizer='Reuse smplify_temporal124.rotation_energy;body_weight30,root_weight0,dt=nominal_dt=1. This is a shared-pose tether, not an adjacent-time or measured3D term.',
        risks=['The per-view solve may undo the corrected arm depth.','Ambiguous occluded observations can still distort a limb.','The shared-pose penalty may resist true motion.'],
        budget=dict(candidates=1,frames=8,max_wall_seconds=600,original_stage_indices=[3,4],maxiters=30),
        admission='Review all8 candidate native body/face/feet and common side geometry. Then select about8 eligible poses; no all94-frame gate, no threshold changes or stitched winners. No automatic GS.',
        source_hashes={str(p):sha(p) for p in sources},quality_accepted=False,gaussian_admitted=False))
    print('SELECTED_REFINEMENT_REGISTERED',flush=True)

def verify():
    p=read(OUT/'plan.json')
    for path,d in p['source_hashes'].items():assert sha(path)==d,path
    return p

def preflight():
    p=verify();assert not (OUT/'preflight.json').exists() and not (OUT/'progress.json').exists()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.manual_seed(124)
    body,vposer=shared.make_models(1);rows,target=shared.data()
    # Use all8 saved seeds: identity, complete K and seed replay must hold for
    # every selected frame before any one of them is optimized.
    checks=[]
    for i,row in enumerate(rows):
        s=load(BASE/'frames'/row['frame']/'winner-camera-space.npz')
        raw=np.asarray(Image.open(row['image']).convert('RGB'),np.float32)/255
        out,receipt=adapter.fit(saved_seed=s,replay_only=True,official_root=shared.OFFICIAL,
            image=raw,keypoints=target[i:i+1],K_integer_centers=row['K_integer_centers'],
            body_model=body,vposer2=vposer,result_path=OUT/'replay-only-unused.pkl',shared_betas=s['betas'],
            freeze_betas=True,device='cuda',diff_path=OUT/'executed-interface.diff')
        diff={k:float(np.max(abs(s[k]-out[k]))) for k in ['vertices_camera','joints_camera','joints_projected_native',
            'camera_translation','body_pose_axis_angle','pose_embedding','global_orient','expression','jaw_pose','left_hand_pose','right_hand_pose','betas']}
        assert diff['vertices_camera']<1e-6 and diff['joints_projected_native']<.001,diff
        assert all(diff[k]==0 for k in ['camera_translation','pose_embedding','global_orient','betas'])
        assert receipt['anchor_calls']==0 and not (OUT/'replay-only-unused.pkl').exists()
        checks.append(dict(frame=row['frame'],max_abs=diff))
    # At the anchor the added energy and gradient must be zero; a bent arm
    # must have a finite restoring gradient. Face and root stay unaffected.
    from smplify_temporal124 import rotation_energy
    pose=torch.zeros((1,165),device='cuda',requires_grad=True)
    expected=pose.detach().clone();energy=rotation_energy(pose,expected,1,1,30,0)
    g=torch.autograd.grad(energy,pose)[0];assert energy==0 and g.abs().max()==0
    expected[0,52]=.3;e=rotation_energy(pose,expected,1,1,30,0);g=torch.autograd.grad(e,pose)[0]
    assert torch.isfinite(g).all() and g.abs().max()>0 and g[:,:3].abs().max()==0 and g[:,66:].abs().max()==0
    save(OUT/'preflight.json',dict(updated=now(),actual_eight_seed_replays=checks,optimizer_steps=0,
        anchor_zero_energy_gradient=True,anchor_finite_restoring_gradient=True,root_face_not_anchored=True,quality_accepted=False))
    print('EIGHT_SEEDS_GPU_REPLAY_PASSED',flush=True)

def run():
    p=verify();assert (OUT/'preflight.json').exists() and not (OUT/'progress.json').exists()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('one600srefinementbudget')));signal.alarm(600)
    torch.set_num_threads(4);torch.manual_seed(124)
    started=time.monotonic();receipts=[];arrays_all=[]
    save(OUT/'launch.json',dict(updated=now(),pid=os.getpid(),proc_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],command=sys.argv))
    save(OUT/'progress.json',dict(updated=now(),state='running',pid=os.getpid(),done=0,total=8))
    try:
        body,vposer=shared.make_models(1);rows,target=shared.data()
        for i,row in enumerate(rows):
            name=row['frame'];folder=OUT/'frames'/name;folder.mkdir(parents=True)
            s=load(BASE/'frames'/name/'winner-camera-space.npz')
            raw=np.asarray(Image.open(row['image']).convert('RGB'),np.float32)/255;start=time.monotonic()
            arr,rec=adapter.fit(saved_seed=s,official_root=shared.OFFICIAL,image=raw,keypoints=target[i:i+1],
                K_integer_centers=row['K_integer_centers'],body_model=body,vposer2=vposer,
                result_path=folder/'official-result.pkl',shared_betas=s['betas'],freeze_betas=True,
                device='cuda',diff_path=folder/'interface.diff')
            assert np.array_equal(s['betas'],arr['betas']) and rec['anchor_calls']>0
            q=arr['joints_camera']@arr['K_integer_centers'].T
            parity=float(np.max(abs(q[...,:2]/q[...,2:]-arr['joints_projected_native'])));assert parity<.001
            for k in ['observed_native_keypoints','image_sha256','mask_sha256','source_frames']:arr[k]=s[k]
            arr['frame_id']=np.array(name);np.savez_compressed(folder/'winner-camera-space.npz',**arr)
            rec.update(frame=name,seconds=time.monotonic()-start,projection_replay_max_px=parity,
                seed_sha256=sha(BASE/'frames'/name/'winner-camera-space.npz'),quality_accepted=False)
            save(folder/'result.json',rec);receipts.append(rec);arrays_all.append(arr)
            save(OUT/'progress.json',dict(updated=now(),state='running',pid=os.getpid(),done=i+1,total=8,frame=name))
            print('FRAME_REFINED',name,round(rec['seconds'],3),flush=True)
        static={'faces','K_integer_centers','K_compute_float32','frame_id'}
        combined={k:arrays_all[0][k] if k in static else np.concatenate([x[k] for x in arrays_all],axis=0) for k in arrays_all[0] if k!='frame_id'}
        combined['frame_ids']=np.array(p['frames']);np.savez_compressed(OUT/'selected-fit.npz',**combined)
        verify();save(OUT/'result.json',dict(updated=now(),state='completed_review_pending',seconds=time.monotonic()-started,
            frames=p['frames'],records=receipts,fit_sha256=sha(OUT/'selected-fit.npz'),quality_accepted=False,gaussian_admitted=False))
        save(OUT/'progress.json',dict(updated=now(),state='completed_review_pending',pid=None,done=8,total=8))
        print('SELECTED_REFINEMENT_COMPLETED',time.monotonic()-started,flush=True)
    except BaseException:
        save(OUT/'failure.json',dict(updated=now(),traceback=traceback.format_exc(),done=len(receipts)))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None,done=len(receipts)));raise
    finally:signal.alarm(0)

if __name__=='__main__':{'prepare':prepare,'preflight':preflight,'run':run}[sys.argv[1]]()
