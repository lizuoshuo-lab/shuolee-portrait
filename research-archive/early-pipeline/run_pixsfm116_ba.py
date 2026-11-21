"""One bounded PixSfM feature-metric camera candidate. No Gaussian launcher."""
from pathlib import Path
import collections
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT=Path('/opt/gs-slam-lab/shuolee-pixsfm116')
B=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S=B/'current/full-scene-localization-074'
O=B/'current/portrait-alignment-open-source-116/pixsfm'
V=O/'independent-validation'
R=O/'ba-vgg-candidate'
LABELS=['background','face_lower','face_upper','torso','legs_shoes','other_foreground']
WEIGHT=Path('/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth')
WEIGHT_SHA='397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def alarm(*_):raise TimeoutError('P1 full175 diagnostic budget reached')
def camera_arrays(rec):
    E=np.tile(np.eye(4),(175,1,1));camera_ids=set()
    for i in range(1,176):
        im=rec.images[i];assert im.name==f'raw_{i:06d}.png';E[i-1,:3]=im.projection_matrix();camera_ids.add(im.camera_id)
    assert len(camera_ids)==1
    camera=rec.cameras[next(iter(camera_ids))]
    assert camera.model_name=='SIMPLE_RADIAL' and camera.width==1520 and camera.height==2704
    return E,np.array(camera.params,dtype=float).copy()

def main():
    lock=(ROOT/'alignment116.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reg=read(O/'ba-preregistration.json');assert reg['runner_sha256']==sha(__file__)
    val=read(V/'result.json');assert val['coverage_screen_passed']
    assert isinstance(val['visual_review'],dict) and val['visual_review']['usable_for_bounded_diagnostic']
    assert val['tracks_sha256']==sha(V/'tracks.npz') and val['exclusion_sha256']==sha(V/'excluded-sfm-point-ids.npy')
    recovery=R.exists()
    if recovery:
        assert {p.name for p in R.iterdir()}=={'progress.json','failure.json'}, 'Never overwrite/restart a prepared, running or completed BA candidate.'
        assert read(R/'progress.json')['state']=='failed'
        assert reg.get('preflight_repair',{}).get('no_model_or_optimizer_created') is True
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
    assert not gpu.stdout.strip(),'Another GPU task is active.'
    if recovery:(R/'failure.json').rename(R/'preflight-failure-01.json')
    else:R.mkdir()
    start=time.monotonic();deadline=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=reg['wall_budget_seconds'])).isoformat()
    signal.signal(signal.SIGALRM,alarm);signal.alarm(reg['wall_budget_seconds'])
    def status(stage,**kw):
        save(R/'progress.json',dict(updated=now(),state='running',stage=stage,pid=os.getpid(),deadline=deadline,started=reg['execution_registered_at'],gaussian_training_started=False,**kw))
    try:
        status('verify_and_copy_preregistered_source_model')
        files=[S/'cameras.json']+sorted((S/'refined_sparse').glob('*.bin'))
        source_hash={str(p):sha(p) for p in files};assert source_hash==reg['source_hashes']
        assert sha(WEIGHT)==WEIGHT_SHA
        import pycolmap
        import torch
        from pixsfm.refine_colmap import PixSfM
        from pixsfm import extract
        from pixsfm._pixsfm import _bundle_adjustment as ba
        from omegaconf import OmegaConf
        torch.set_num_threads(4);torch.manual_seed(116)
        def no_download(*a,**k):raise RuntimeError('No model download in this candidate.')
        torch.hub.set_dir('/root/.cache/torch/hub');torch.hub.download_url_to_file=no_download
        copy=R/'input-model';copy.mkdir()
        for f in (S/'refined_sparse').glob('*.bin'):shutil.copyfile(f,copy/f.name)
        rec=pycolmap.Reconstruction(str(copy));E0,K0=camera_arrays(rec)
        assert len(rec.images)==175 and len(rec.reg_image_ids())==175
        source=read(S/'cameras.json');assert np.max(abs(E0-np.array([r['w2c'] for r in source['records']])))<1e-9
        assert np.max(abs(K0-np.array(source['intrinsics'])))<1e-9
        with np.load(O/'database-regions.npz') as z:regions={int(k):z[k] for k in z.files}
        exclude=set(np.load(V/'excluded-sfm-point-ids.npy').tolist())
        pools=collections.defaultdict(list);before=len(rec.points3D)
        for pid,p in rec.points3D.items():
            if pid in exclude:continue
            els=list(p.track.elements)
            if len(set(int(e.image_id) for e in els))<3:continue
            labels=[int(regions[e.image_id][e.point2D_idx]) for e in els]
            group,count=collections.Counter(labels).most_common(1)[0]
            if count/len(labels)<.8:group=5
            pools[group].append(int(pid))
        selected=[];selection_counts={}
        for group,label in enumerate(LABELS):
            # Fixed hash order, never residual-, convergence- or result-selected.
            order=sorted(pools[group],key=lambda pid:hashlib.sha256(f'116-P1-{pid}'.encode()).hexdigest())
            chosen=order[:reg['region_point_caps'][group]];selected.extend(chosen)
            selection_counts[label]=dict(available=len(order),selected=len(chosen))
        assert selection_counts['face_lower']['selected']>=30,'Insufficient disjoint lower-face source support.'
        keep=set(selected)
        for pid in list(rec.points3D):
            if pid not in keep:rec.delete_point3D(pid)
        per_image={str(i):rec.images[i].num_points3D() for i in range(1,176)}
        assert min(per_image.values())>=30,'Insufficient source constraints in at least one camera.'
        observations=sum(per_image.values());estimate=observations*8*8*128*2
        assert estimate<=reg['feature_cache_estimate_limit_bytes']
        assert shutil.disk_usage(R).free>estimate*2+8*1024**3
        rec.write(str(copy));np.save(R/'selected-source-point-ids.npy',np.array(sorted(keep),np.int64))
        C=-np.einsum('fji,fj->fi',E0[:,:3,:3],E0[:,:3,3]);far=int(np.argmax(np.linalg.norm(C-C[0],axis=1)))
        axis=int(np.argmax(abs(E0[far,:3,:3]@C[0]+E0[far,:3,3])))
        setup=ba.BundleAdjustmentSetup();setup.add_images(set(range(1,176)));setup.set_constant_pose(1);setup.set_constant_tvec(far+1,[axis])
        manifest=dict(created=now(),source_hashes=source_hash,source_points=before,selected_points=len(keep),selected_by_region=selection_counts,
          observations=observations,observations_per_image=per_image,all175_frame_ids_preserved=True,validation_source_points_excluded=len(exclude),
          validation_bank_sha256=val['tracks_sha256'],validation_exclusion_sha256=val['exclusion_sha256'],
          feature_patch_estimated_bytes=estimate,gauge=dict(fixed_pose_raw_frame=1,fixed_translation_raw_frame=far+1,fixed_translation_axis=axis),
          raw_pixels_unchanged=True,source_model_modified=False)
        save(R/'input-manifest.json',manifest)
        sfm=PixSfM(reg['runtime_config']);state=torch.load(WEIGHT,map_location='cpu',weights_only=True)
        for k,v in sfm.extractor.model.encoder.state_dict().items():assert torch.equal(v,state['features.'+k])
        del state
        save(R/'resolved-config.json',OmegaConf.to_container(sfm.conf,resolve=True))
        status('extract_featuremaps_all175',points=len(keep),observations=observations,estimated_cache_bytes=estimate)
        features=extract.features_from_reconstruction(sfm.extractor,rec,S/'images',cache_path=R/'featuremaps.h5')
        torch.cuda.synchronize();status('featuremetric_camera_point_bundle_adjustment')
        data=sfm.bundle_adjuster.refine_multilevel(rec,features,problem_setup=setup)
        status('write_separate_candidate_and_check_integrity')
        E1,K1=camera_arrays(rec);assert np.isfinite(E1).all() and np.isfinite(K1).all() and K1[0]>0
        assert np.max(abs(E1[0]-E0[0]))<1e-9
        assert np.max(abs(K1[1:3]-K0[1:3]))<1e-9,'Principal point was registered fixed.'
        candidate=R/'candidate-model';candidate.mkdir();rec.write(str(candidate))
        np.savez_compressed(R/'candidate-cameras.npz',w2c=E1,intrinsics=K1,source_w2c=E0,source_intrinsics=K0)
        reports=[]
        for s in data['summary']:
            row={}
            for key in ['initial_cost','final_cost','num_successful_steps','num_unsuccessful_steps','termination_type','total_time_in_seconds']:
                v=getattr(s,key,None);row[key]=v if isinstance(v,(str,int,float,bool,type(None))) else str(v)
            for method in ['brief_report','BriefReport']:
                if hasattr(s,method):row['brief_report']=getattr(s,method)();break
            reports.append(row)
        assert source_hash=={str(p):sha(p) for p in files}
        assert val['tracks_sha256']==sha(V/'tracks.npz')
        result=dict(completed=now(),elapsed_seconds=time.monotonic()-start,state='candidate_completed_pending_independent_evaluation',
           frames=175,source_hashes_unchanged=True,validation_bank_unchanged=True,source_intrinsics=K0.tolist(),candidate_intrinsics=K1.tolist(),
           solver=reports,feature_variant=reg['feature_variant'],official_default_s2dnet=False,
           source_camera_updates=0,gaussian_training_started=False,alignment_quality_accepted=False,root_cause_resolved=False)
        save(R/'result.json',result);save(R/'progress.json',dict(updated=now(),state='completed',stage='candidate_pending_evaluation',pid=os.getpid(),deadline=deadline))
        print(json.dumps(result),flush=True)
    except BaseException as e:
        save(R/'failure.json',dict(updated=now(),exception=str(e),traceback=traceback.format_exc()))
        save(R/'progress.json',dict(updated=now(),state='failed',pid=os.getpid(),exception=str(e),deadline=deadline));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
