"""Two unchanged official SMPLify-X fits with only face51 native XY replaced."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl,json,os,sys,time,traceback
import numpy as np
from PIL import Image
import official_smplifyx_entry123 as entry
from check_local_face_input123 import F,O,REF,read,save,sha

def main():
    plan=read(O/'plan.json');review=read(O/'observation-review.json')
    assert review['actual_visual_review'] and not review['quality_accepted']
    selected=plan['conditional_fit_budget']['frames'];assert selected==review['admitted_for_bounded_fit_comparison']
    assert sha(O/'observations.npz')==review['observations_sha256']
    for p,h in plan['source_hashes'].items():assert sha(p)==h,p
    out=O/'official-fits';assert not out.exists();out.mkdir()
    source_files=[Path(__file__),Path(entry.__file__),O/'plan.json',O/'observations.npz',O/'observation-review.json']
    hashes={str(p):sha(p) for p in source_files}
    save(out/'execution-plan.json',dict(created=datetime.now(timezone.utc).isoformat(),source_hashes=hashes,
         frames=selected,variable='Only51 face XY. Original face confidence and all other observations/K/beta remain byte-equal.',
         fresh_official_zero_mean_initialization=True,official_objective_and_optimizer_unchanged=True,
         maximum_seconds=180,new_sequence_solver=False,new_gaussian_training=False))
    start=time.monotonic();save(out/'progress.json',dict(state='loading',pid=os.getpid(),frames_completed=0))
    try:
        project=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118');assets=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
        sys.path.insert(0,str(project/'src'))
        from shuolee_portrait import runtime as rt
        rt.configure(assets,out)
        import torch,smplx
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        with np.load(F/'fit/fitted.npz',allow_pickle=False) as z:
            ids=list(z['frame_ids'].astype(str));targets=z['observed_native_keypoints'].copy();betas=z['shared_betas'].copy();Ks=z['K_integer_centers'].copy()
        with np.load(O/'observations.npz',allow_pickle=False) as z:
            fan_ids=list(z['frame_ids'].astype(str));fan=z['fan68_native'].copy()
        rows=read(F/'data/training-inputs.json')['records']
        with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            utils,_,_,_=entry.load_official(REF/'official')
            mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
            loader=entry.module(project/'src/avatar_core/shuolee_vposer.py','_local_face_vposer')
            loader.MODEL=assets/'vposer/V02_05';loader.CODE=project/'third_party/human_body_prior'
            vposer,_,_=loader.load_vposer('cuda')
            body=smplx.SMPLX(str(assets/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
                use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
                create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).cuda()
            from run_official_sequence123_topology import cache_static_topology
            cache_static_topology(body)
            original_load=entry.load_official
            def bounded_load(path):
                modules=original_load(path);fitting=modules[2];original=fitting.FittingMonitor.run_fitting
                def bounded(self,optimizer,closure,*a,**k):
                    def observe(*ca,**ck):
                        if time.monotonic()-start>180:raise RuntimeError('Predeclared180second fit budget exhausted')
                        return closure(*ca,**ck)
                    return original(self,optimizer,observe,*a,**k)
                fitting.FittingMonitor.run_fitting=bounded
                return modules
            entry.load_official=bounded_load
            runs=[]
            for frame in selected:
                slot=ids.index(frame);row=rows[slot];assert row['frame']==frame and sha(row['image'])==row['source_sha256']
                target=targets[slot:slot+1].copy();old=target.copy();target[:,67:118,:2]=fan[fan_ids.index(frame),17:].astype(target.dtype)
                assert np.array_equal(target[:,:67],old[:,:67]) and np.array_equal(target[:,:,2],old[:,:,2])
                directory=out/frame;directory.mkdir();raw=Image.open(row['image']).convert('RGB')
                torch.manual_seed(6032);torch.cuda.manual_seed_all(6032)
                save(out/'progress.json',dict(state='official_fit_running',pid=os.getpid(),frame_id=frame,frames_completed=len(runs),total=2))
                begin=time.monotonic()
                arrays,report=entry.fit_one(official_root=REF/'official',image=np.asarray(raw,dtype=np.float32)/255.,keypoints=target,
                    K_integer_centers=Ks[slot],body_model=body,vposer2=vposer,result_path=directory/'official-result.pkl',
                    shared_betas=betas,freeze_betas=True,device='cuda',diff_path=directory/'input-output-adapter.diff')
                arrays.update(frame_id=np.asarray(frame),observed_native_keypoints=target,baseline_native_keypoints=old,
                    source_frame=np.asarray(row['source_frame_index']),image_sha256=np.asarray(row['source_sha256']))
                assert np.array_equal(arrays['betas'],betas) and np.array_equal(arrays['K_integer_centers'],Ks[slot])
                np.savez_compressed(directory/'winner-camera-space.npz',**arrays)
                report.update(state='complete',frame_id=frame,elapsed_seconds=time.monotonic()-begin,only_face_XY_changed=True,
                    all_confidence_unchanged=True,model_quality_accepted=False)
                save(directory/'result.json',report);runs.append(dict(frame_id=frame,seconds=report['elapsed_seconds'],winning_loss=report['winning_loss']))
                print(json.dumps(runs[-1]),flush=True)
            entry.load_official=original_load
        assert all(sha(p)==h for p,h in hashes.items())
        assert all(sha(p)==h for p,h in plan['source_hashes'].items())
        save(out/'result.json',dict(state='complete_visual_review_pending',runs=runs,original94_preserved=True,
            no_gaussian_training=True,only_face_XY_changed=True,losses_not_comparable_across_changed_targets=True,
            quality_accepted=False,seconds=time.monotonic()-start))
        save(out/'progress.json',dict(state='complete',pid=None,frames_completed=2))
    except BaseException as e:
        save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        save(out/'progress.json',dict(state='failed',pid=None));raise

if __name__=='__main__':main()
