"""Bounded head5 absence contrast using the original temporal SMPLify-X adapter."""
import argparse
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import control_face_absence124 as base

HERE=Path(__file__).resolve().parent
OUT=base.ROOT/'head-absence-input124'
HEAD5=np.array([0,15,16,17,18])
entry=base.entry
read,save,sha,now=base.read,base.save,base.sha,base.now

def abstain_head(mapped,associated_count):
    """Diagnostic abstention, not a visibility label; never change coordinates."""
    out=base.abstain_inner_face(mapped,associated_count)
    absent=np.flatnonzero(np.asarray(associated_count)==0)
    out[absent[:,None],HEAD5[None,:],2]=0
    return out

def verify():
    p=read(OUT/'plan.json')
    for path,digest in p['frozen'].items():assert sha(path)==digest,path
    return p

def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
    previous=base.verify()
    assert read(base.OUT/'conclusion.json')['actual_visual_review']
    assert read(base.ROOT/'shared-shape-head5-audit124/conclusion.json')['actual_visual_review']
    paths=[HERE/'control_head_absence124.py']+[Path(k) for k in previous['frozen']]
    paths += [base.OUT/'plan.json',base.OUT/'result.json',base.OUT/'conclusion.json']
    paths += [base.OUT/'prefix/frame_000039/winner-camera-space.npz']
    paths += [base.OUT/arm/f'frame_{frame:06d}'/file for arm in ('original','face_absence')
              for frame in (40,41) for file in ('winner-camera-space.npz','result.json')]
    paths += [base.ROOT/'shared-shape-head5-audit124/conclusion.json']
    subjects={};frozen={str(p):sha(p) for p in paths};OUT.mkdir()
    for subject in ('sofa','indoor'):
        src=base.INPUT/(subject+'-diagnostic-input.npz')
        with np.load(src,allow_pickle=False) as z:d=dict(z)
        candidate=abstain_head(d['original_mapped'],d['associated_count'])
        old=d['candidate_mapped']; changed=candidate[...,2]!=old[...,2]
        allowed=np.zeros_like(changed);allowed[np.flatnonzero(d['associated_count']==0)[:,None],HEAD5[None,:]]=True
        assert not np.any(changed&~allowed)
        assert np.array_equal(candidate[...,:2],old[...,:2])
        assert np.array_equal(candidate[d['associated_count']>0],old[d['associated_count']>0])
        assert np.array_equal(candidate[:,19:],old[:,19:])
        assert np.array_equal(abstain_head(candidate,d['associated_count']),candidate)
        assert np.all(candidate[...,2]<=old[...,2])
        dest=OUT/(subject+'-diagnostic-input.npz')
        np.savez_compressed(dest,**d,head_absence_mapped=candidate)
        frozen[str(src)]=sha(src);frozen[str(dest)]=sha(dest)
        subjects[subject]=dict(frames=len(candidate),changed_entries=int(changed.sum()),
            absent_frames=d['frame_ids'][d['associated_count']==0].astype(str).tolist(),
            original_coordinates_and_other_data_unchanged=True,sha256=sha(dest))
    save(OUT/'plan.json',dict(updated=now(),state='registered',
        hypothesis='After removing unsupported innerface51, the still-active body nose/eyes/ears5 may pull the posterior head and shoulders toward false2D targets. Test only this added abstention.',
        intervention='Only118-layout0/15/16/17/18 confidences additionally set to zero when frozen associated face count is zero. Original observations and coordinates retained. No positive detection promotion.',
        selection='Preserved first absence40, followed by41 only if all preregistered visible-body protections pass at40. No replacement frames or error-based winner selection.',
        frame_ids=['frame_000040','frame_000041'],configuration=previous['configuration'],
        nominal_dt_seconds=previous['nominal_dt_seconds'],random_seed=previous['random_seed'],
        seed='40 uses the exact saved39 local diagnostic seed from the prior matched control, not a complete-sequence initialization or accepted pose.41 uses this candidate40 only.',
        budget=dict(max_fit_calls=2,max_seconds=300,new_detectors=0,new_features=0),
        protections=dict(body_mean_px=1.,per_joint_px=2.,ids=base.PROTECTED,
            compare_with=['face_absence','original'],stop_before41_if_any40_protection_fails=True,
            actual_native_head_body_hands_and_common_side_review=True),
        unchanged='Shared beta105; fullK; camera initialization; original five stages and30 iterations; LBFGS,VPoser,priors; temporal body/root weights30; original raster/masks.',
        risks=['A missing face detection does not prove ears or all head5 are invisible; real ear evidence may be lost.',
            'SFD positive side-head false detections remain untouched.','Unaccepted shared shape and local39 seed remain confounders for absolute quality.',
            'No independent3D or visibility truth. Lower data error is not quality acceptance.'],
        stop='At most2new fits once; source mismatch, nonfinite or300s stops. No retry, full sequence, threshold/weight sweep, old-run restart, winner stitching or Gaussian.',
        frozen=frozen,subjects=subjects,default_applied=False,quality_accepted=False))
    save(OUT/'preflight.json',dict(updated=now(),state='passed_input_contract',subjects=subjects,
        frames_checked=167,new_fit_calls=0,new_detector_calls=0,default_applied=False,quality_accepted=False))
    print('HEAD_ABSENCE_REGISTERED',subjects,flush=True)

def metrics_and_gates(arrays,raw,name,p):
    candidate=base.metric(arrays['joints_projected_native'],raw)
    comparisons={}
    for arm in p['protections']['compare_with']:
        old=read(base.OUT/arm/name/'result.json')['metrics']
        gates=dict(body=candidate['body']<=old['body']+p['protections']['body_mean_px'],
            protected_joints=all(candidate['distance_px'][j]<=old['distance_px'][j]+p['protections']['per_joint_px'] for j in base.PROTECTED))
        comparisons[arm]=dict(old=old,candidate=candidate,gates=gates)
    return candidate,comparisons

def worker():
    p=verify();assert not (OUT/'progress.json').exists()
    started=time.monotonic();completed=[]
    originals=entry.adapted_entry_source,entry.load_official_config,entry.load_official
    save(OUT/'progress.json',dict(updated=now(),state='running',pid=os.getpid(),completed=0))
    try:
        import torch,smplx
        torch.set_num_threads(4)
        rows={r['frame']:r for r in read(base.SRC/'data/inputs.json')['records']}
        with np.load(OUT/'indoor-diagnostic-input.npz',allow_pickle=False) as z:d=dict(z)
        ids=d['frame_ids'].astype(str).tolist()
        utils,_,_,_=originals[2](base.OFFICIAL)
        mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        loader=entry.module(base.PROJECT/'src/avatar_core/shuolee_vposer.py','_head_absence_vposer124')
        loader.MODEL,loader.CODE=base.ASSETS/'vposer/V02_05',base.PROJECT/'third_party/human_body_prior'
        vposer,_,_=loader.load_vposer('cuda')
        body=smplx.SMPLX(str(base.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
            use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
            create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).to('cuda')
        base.cache_static_topology(body)
        with np.load(base.SRC/'reference-fit/winner-camera-space.npz',allow_pickle=False) as z:beta=z['betas'].copy()
        with np.load(base.OUT/'prefix/frame_000039/winner-camera-space.npz',allow_pickle=False) as z:previous=dict(z)
        active={};entry._root_seed_original_source=originals[0]
        def config(root):
            c=originals[1](root);assert c==p['configuration']
            c.update(_seed_pose_embedding=active['latent'],_seed_global_orient=active['root'],_seed_root_enabled=True)
            return c
        def load(root):
            modules=originals[2](root)
            base.install_loss_factory(modules[2],active['neighbour'],active['dt'],p['nominal_dt_seconds'],active['receipt'])
            return modules
        entry.adapted_entry_source,entry.load_official_config,entry.load_official=base.seed.adapted_source,config,load
        records=[];stop_reason='completed_registered_frames'
        for name in p['frame_ids']:
            random.seed(p['random_seed']);np.random.seed(p['random_seed']);torch.manual_seed(p['random_seed'])
            i=ids.index(name);row=rows[name];prev_name=ids[i-1]
            assert d['associated_count'][i]==0
            neighbour=np.zeros((1,165),np.float32)
            neighbour[:,:3]=previous['global_orient'];neighbour[:,3:66]=previous['body_pose_axis_angle']
            active.update(latent=previous['pose_embedding'].copy(),root=previous['global_orient'].copy(),
                neighbour=neighbour,dt=row['time_seconds']-rows[prev_name]['time_seconds'],receipt={})
            target=d['head_absence_mapped'][i:i+1].copy();raw=d['original_mapped'][i:i+1]
            dest=OUT/name;dest.mkdir()
            save(OUT/'progress.json',dict(updated=now(),state='running',pid=os.getpid(),completed=len(completed),frame=name))
            tick=time.monotonic()
            arrays,report=entry.fit_one(official_root=base.OFFICIAL,image=base.sequence.checked_image(row),keypoints=target,
                K_integer_centers=row['K_integer_centers'],body_model=body,vposer2=vposer,result_path=dest/'official-result.pkl',
                shared_betas=beta,freeze_betas=True,device='cuda',diff_path=dest/'adapter.diff')
            trace={k:v.tolist() for k,v in entry._root_seed_trace.items()}
            old_report=read(base.OUT/'face_absence'/name/'result.json')
            assert np.array_equal(arrays['betas'],beta)
            assert np.array_equal(arrays['K_integer_centers'],row['K_integer_centers'])
            assert active['receipt']['calls']>0 and active['receipt']['temporal_gradient_finite']
            if name==p['frame_ids'][0]:
                assert old_report['initial_latent']==active['latent'].tolist() and old_report['seed_trace']==trace
            assert np.array_equal(arrays['camera_translation'],np.asarray(old_report['seed_trace']['camera_translation']))
            assert all(np.isfinite(x).all() for x in arrays.values() if isinstance(x,np.ndarray) and x.dtype.kind in 'fc')
            arrays.update(frame_id=np.asarray(name),image_sha256=np.asarray(row['source_sha256']),
                original_native_observations=raw,fitting_native_observations=target)
            base.sequence.save_npz(dest/'winner-camera-space.npz',**arrays)
            metrics,comparisons=metrics_and_gates(arrays,raw,name,p)
            gates_passed=all(all(item['gates'].values()) for item in comparisons.values())
            rec=dict(updated=now(),frame=name,seconds=time.monotonic()-tick,
                previous=prev_name,initial_latent=active['latent'].tolist(),seed_trace=trace,
                temporal_receipt={k:float(v.cpu()) if hasattr(v,'cpu') else v for k,v in active['receipt'].items()},
                original_report=report,metrics=metrics,comparisons=comparisons,protected_body_passed=gates_passed,
                direct_same_seed_control=name==p['frame_ids'][0],quality_accepted=False)
            save(dest/'result.json',rec);records.append(rec);completed.append(name);previous=arrays
            print('HEAD_CONTROL_FIT_COMPLETE',name,round(rec['seconds'],2),'body_protection',gates_passed,flush=True)
            if not gates_passed:
                stop_reason='visible_body_protection_failed';break
        verify()
        save(OUT/'result.json',dict(updated=now(),state='complete_pending_visual_review',seconds=time.monotonic()-started,
            fitting_calls=len(completed),completed=completed,records=records,stop_reason=stop_reason,
            skipped_frames=[n for n in p['frame_ids'] if n not in completed],original_sources_unchanged=True,
            actual_visual_review=False,default_applied=False,quality_accepted=False,gaussian_admitted=False))
        save(OUT/'progress.json',dict(updated=now(),state='complete',pid=None,completed=len(completed),stop_reason=stop_reason))
        print('HEAD_ABSENCE_CONTROL_FINISHED',stop_reason,flush=True)
    except BaseException as exc:
        save(OUT/'failure.json',dict(updated=now(),error=repr(exc),completed=completed))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None,completed=len(completed)));raise
    finally:entry.adapted_entry_source,entry.load_official_config,entry.load_official=originals

def run():
    import fcntl
    p=verify();assert not (OUT/'progress.json').exists()
    lease=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (OUT/'fit.log').open('x') as log:
        try:subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,timeout=p['budget']['max_seconds'],check=True)
        except subprocess.TimeoutExpired:
            save(OUT/'progress.json',dict(updated=now(),state='time_budget_exhausted',pid=None));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','worker'])
    args=ap.parse_args();dict(prepare=prepare,run=run,worker=worker)[args.mode]()
