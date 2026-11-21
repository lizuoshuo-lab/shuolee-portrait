"""One bounded, input-only chronological control; original SMPLify-X solver."""
import argparse
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import official_smplifyx_entry123 as entry
import control_smplifyx_root_seed123 as seed
import run_official_sequence123 as sequence
from run_official_sequence123_topology import cache_static_topology
from run_official_gaussians123 import ASSETS, PROJECT, read, save, sha, now
from smplify_temporal124 import install_loss_factory
from smplify_face_support124 import abstain_inner_face

HERE=Path(__file__).resolve().parent
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
ROOT=REC/'cross-subject-sh0-124'
SRC=ROOT/'indoor-input'
INPUT=ROOT/'face-absence-input124'
OUT=INPUT/'indoor-transition-control'
OFFICIAL=REC/'smplifyx-reference-control/official'
PROTECTED=[2,3,4,5,6,7]


def context():
    rows=read(SRC/'data/inputs.json')['records']
    with np.load(INPUT/'indoor-diagnostic-input.npz',allow_pickle=False) as z:d=dict(z)
    assert d['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]
    assert np.array_equal(d['candidate_mapped'],abstain_inner_face(d['original_mapped'],d['associated_count']))
    start=int(np.flatnonzero(d['associated_count']==0)[0]);assert start>=2
    indices=list(range(start-2,start+2))
    assert np.all(d['associated_count'][start:start+2]==0)
    return rows,d,indices


def verify():
    p=read(OUT/'plan.json')
    for name,digest in p['frozen'].items():assert sha(name)==digest,name
    return p


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
    rows,d,indices=context();pre=read(INPUT/'preflight.json')
    assert sha(INPUT/'indoor-diagnostic-input.npz')==pre['subjects']['indoor']['sha256']
    for name,digest in read(INPUT/'plan.json')['frozen'].items():assert sha(name)==digest,name
    paths=[HERE/x for x in ['control_face_absence124.py','official_smplifyx_entry123.py',
        'control_smplifyx_root_seed123.py','run_official_sequence123.py',
        'run_official_sequence123_topology.py','smplify_temporal124.py','smplify_face_support124.py']]
    paths += [OFFICIAL/x for x in entry.PINNED]
    paths += [SRC/'data/inputs.json',SRC/'data/observations.npz',SRC/'reference-fit/winner-camera-space.npz',
        INPUT/'indoor-diagnostic-input.npz',INPUT/'plan.json',INPUT/'preflight.json',
        ROOT/'sfd-face-presence124/full167/result.json',ROOT/'sfd-face-presence124/full167/visual-review.json']
    for i in indices:
        row=rows[i]
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        assert np.array_equal(row['w2c'],np.eye(4))
        paths.extend([Path(row['image']),Path(row['mask'])])
    cfg=entry.load_official_config(OFFICIAL);assert cfg['maxiters']==30
    assert len(cfg['body_pose_prior_weights'])==5
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),state='registered',subject='indoor',diagnostic_only=True,
        hypothesis='Determine the effect of withholding unsupported inner-face51 targets at the first frozen no-face transition, while keeping the existing continuous-frame constraint and original fitting settings.',
        selection='First zero-associated SFD count in chronological roster; the two immediately preceding frames form one shared prefix, followed by the first two absent frames in each arm. No selection by fitted error.',
        frame_ids=[rows[i]['frame'] for i in indices],indices=indices,
        original_roster=111,first_detection_absence=rows[indices[2]]['frame'],
        original_stages=5,maxiters_per_stage=30,configuration=cfg,random_seed=124,
        nominal_dt_seconds=float(np.median(np.diff([r['time_seconds'] for r in rows]))),
        shared_prefix='Original zero/mean38, then same-run39. This local diagnostic prefix is not a good-pose or complete-sequence initialization claim.',
        branch_initialization='Both40 fits start from exactly the same newly fitted39. Each41 starts from its own40. Camera initialization algorithm unchanged;40 camera must match exactly;41 downstream changes are recorded, not called a direct isolated face effect.',
        intervention='Only original118-layout67:118 confidence zero when associated_count==0. Coordinates, body/head5/hands/feet, K, frozen beta105, priors, five stages, optimizer, temporal weights30 and actual timestamp scaling unchanged.',
        temporal='Detached adjacent same-run root/body rotations; not a bidirectional joint solver. No old pose/vertices, no hand/face temporal energy.',
        budget=dict(fitter_calls=6,distinct_frames=4,max_seconds=900,new_detectors=0,new_features=0),
        protection=dict(original_body_weighted_mean_tolerance_px=1.,per_protected_joint_tolerance_px=2.,protected_joint_ids=PROTECTED,
            actual_native_and_synthetic_side_review=True,all_raw_observations_retained=True),
        input_visual_review=dict(actual=True,frames=[rows[i]['frame'] for i in indices],
            finding='Posterior/near-posterior head; eyes,nose,mouth mostly hidden. Hands rest at waist; lower legs outside raster. SFD positive38/39 is not complete face visibility.'),
        limitations=['Posterior face51 errors are recorded but cannot count as valid facial truth or a quality improvement.',
            'Body nose/eye/ear predictions, occluded wrists/legs and single-view depth remain unresolved.',
            'A cold-start local prefix can be wrong and can propagate; compare both arms, never promote this slice to a sequence initializer.',
            'Reference105 supplies beta only, not independent shape truth or calibrated camera trajectory.'],
        stop='Six fits once. Stop on nonfinite values, source mismatch or900s. No sweep, full111 restart, winner stitching or GS admission. Review every endpoint.',
        frozen={str(p):sha(p) for p in paths},quality_accepted=False,default_applied=False))
    print('FACE_ABSENCE_CONTROL_REGISTERED',[rows[i]['frame'] for i in indices],flush=True)


def metric(projected,targets):
    e=np.linalg.norm(projected[0]-targets[0,:,:2],axis=1)
    c=targets[0,:,2].copy();c[[1,9,12]]=0
    out=dict(distance_px=e.tolist(),original_confidence=targets[0,:,2].tolist())
    for name,ids in [('body',np.r_[2:9,10:12,13:15]),('hands',np.r_[25:67]),('face_unverified',np.r_[67:118]),('feet_unverified',np.r_[19:25])]:
        w=c[ids];out[name]=float(np.dot(w,e[ids])/w.sum()) if w.sum()>0 else None
    return out


def worker():
    p=verify();assert not (OUT/'progress.json').exists()
    save(OUT/'progress.json',dict(updated=now(),state='running',pid=os.getpid(),completed=0))
    originals=entry.adapted_entry_source,entry.load_official_config,entry.load_official
    started=time.monotonic();completed=[]
    try:
        import torch,smplx
        torch.set_num_threads(4)
        rows,d,indices=context();utils,_,_,_=originals[2](OFFICIAL)
        mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        loader=entry.module(PROJECT/'src/avatar_core/shuolee_vposer.py','_face_absence_vposer124')
        loader.MODEL,loader.CODE=ASSETS/'vposer/V02_05',PROJECT/'third_party/human_body_prior'
        vposer,_,_=loader.load_vposer('cuda')
        body=smplx.SMPLX(str(ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
            use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
            create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).to('cuda')
        cache_static_topology(body)
        with np.load(SRC/'reference-fit/winner-camera-space.npz',allow_pickle=False) as z:beta=z['betas'].copy()
        active={};entry._root_seed_original_source=originals[0]
        def config(root):
            c=originals[1](root);assert c==p['configuration']
            c.update(_seed_pose_embedding=active['latent'],_seed_global_orient=active['root'],_seed_root_enabled=active['previous'] is not None)
            return c
        def load(root):
            modules=originals[2](root)
            install_loss_factory(modules[2],active['neighbour'],active['dt'],p['nominal_dt_seconds'],active['receipt'])
            return modules
        entry.adapted_entry_source,entry.load_official_config,entry.load_official=seed.adapted_source,config,load
        bank={};results={}
        schedule=[('prefix',i) for i in indices[:2]]+[(arm,i) for arm in ('original','face_absence') for i in indices[2:]]
        for arm,i in schedule:
            random.seed(p['random_seed']);np.random.seed(p['random_seed']);torch.manual_seed(p['random_seed'])
            row=rows[i];name=row['frame'];previous=None
            if i!=indices[0]:previous=('prefix' if i<=indices[2] else arm,rows[i-1]['frame'])
            prev=bank[previous] if previous else None
            active.update(previous=previous,latent=prev['pose_embedding'].copy() if prev else np.zeros((1,32),np.float32),
                root=prev['global_orient'].copy() if prev else np.zeros((1,3),np.float32),neighbour=None,
                dt=row['time_seconds']-rows[i-1]['time_seconds'] if previous else p['nominal_dt_seconds'],receipt={})
            if prev:
                active['neighbour']=np.zeros((1,165),np.float32)
                active['neighbour'][:,:3]=prev['global_orient'];active['neighbour'][:,3:66]=prev['body_pose_axis_angle']
            target=d['candidate_mapped' if arm=='face_absence' else 'original_mapped'][i:i+1].copy()
            dest=OUT/arm/name;dest.mkdir(parents=True)
            save(OUT/'progress.json',dict(updated=now(),state='running',pid=os.getpid(),completed=len(completed),arm=arm,frame=name))
            tick=time.monotonic()
            arrays,report=entry.fit_one(official_root=OFFICIAL,image=sequence.checked_image(row),keypoints=target,
                K_integer_centers=row['K_integer_centers'],body_model=body,vposer2=vposer,result_path=dest/'official-result.pkl',
                shared_betas=beta,freeze_betas=True,device='cuda',diff_path=dest/'adapter.diff')
            trace={k:v.tolist() for k,v in entry._root_seed_trace.items()}
            assert np.array_equal(arrays['betas'],beta) and np.array_equal(arrays['K_integer_centers'],row['K_integer_centers'])
            assert np.array_equal(arrays['camera_translation'],np.array(trace['camera_translation']))
            if previous:
                assert active['receipt']['calls']>0 and active['receipt']['temporal_gradient_finite']
                assert np.array_equal(trace['body_fit_initial_root'],active['root'])
            arrays.update(frame_id=np.asarray(name),image_sha256=np.asarray(row['source_sha256']),
                original_native_observations=d['original_mapped'][i:i+1],fitting_native_observations=target)
            sequence.save_npz(dest/'winner-camera-space.npz',**arrays);bank[arm,name]=arrays
            temporal={k:(float(v.cpu()) if hasattr(v,'cpu') else v) for k,v in active['receipt'].items()}
            result=dict(updated=now(),frame=name,arm=arm,seconds=time.monotonic()-tick,previous=previous,
                original_report=report,temporal_receipt=temporal,seed_trace=trace,initial_latent=active['latent'].tolist(),
                metrics=metric(arrays['joints_projected_native'],d['original_mapped'][i:i+1]),quality_accepted=False)
            save(dest/'result.json',result);results[arm+'/'+name]=result;completed.append(arm+'/'+name)
            print('FIT_COMPLETE',completed[-1],round(result['seconds'],3),flush=True)
        comparisons={}
        for i in indices[2:]:
            name=rows[i]['frame'];old=bank['original',name];new=bank['face_absence',name]
            a=results['original/'+name];b=results['face_absence/'+name]
            camera_delta=float(np.max(np.abs(old['camera_translation']-new['camera_translation'])))
            if i==indices[2]:
                assert camera_delta==0 and a['initial_latent']==b['initial_latent'] and a['seed_trace']==b['seed_trace']
            gates=dict(body_protected=b['metrics']['body']<=a['metrics']['body']+1.,
                protected_joints=all(b['metrics']['distance_px'][j]<=a['metrics']['distance_px'][j]+2. for j in PROTECTED))
            comparisons[name]=dict(original=a['metrics'],face_absence=b['metrics'],camera_translation_max_delta_m=camera_delta,
                gates=gates,direct_same_seed_control=i==indices[2],downstream_branch_propagation=i!=indices[2])
        verify()
        save(OUT/'result.json',dict(updated=now(),state='complete_pending_visual_review',seconds=time.monotonic()-started,
            fitting_calls=len(completed),completed=completed,comparisons=comparisons,source_hashes_verified=True,
            actual_visual_review=False,quality_accepted=False,full_sequence_admitted=False,gaussian_admitted=False,default_applied=False))
        save(OUT/'progress.json',dict(updated=now(),state='complete',pid=None,completed=len(completed)))
        print('FACE_ABSENCE_CONTROL_FINISHED',flush=True)
    except BaseException as exc:
        save(OUT/'failure.json',dict(updated=now(),error=repr(exc),completed=completed))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None,completed=len(completed)));raise
    finally:
        entry.adapted_entry_source,entry.load_official_config,entry.load_official=originals


def run():
    import fcntl
    p=verify();assert not (OUT/'progress.json').exists()
    lease=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (OUT/'fit.log').open('x') as log:
        try:
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,
                timeout=p['budget']['max_seconds'],check=True)
        except subprocess.TimeoutExpired:
            save(OUT/'progress.json',dict(updated=now(),state='time_budget_exhausted',pid=None));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','worker']);args=ap.parse_args()
    dict(prepare=prepare,run=run,worker=worker)[args.mode]()
