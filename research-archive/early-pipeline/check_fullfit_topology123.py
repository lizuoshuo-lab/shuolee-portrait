"""Paired complete official fit with invariant topology device as sole variable."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import sys
import time
import traceback

import numpy as np
from PIL import Image
import official_smplifyx_entry123 as entry


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);args=parser.parse_args()
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
    save=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plan_path=Path(args.plan);plan=read(plan_path);root=Path(plan['root']);official=Path(plan['official_root'])
    out=root/'reviews/official-topology-full-fit';out.mkdir(exist_ok=False)
    for p,h in plan['source_hashes'].items():assert entry.sha(p)==h,p
    preflight=read(root/'reviews/official-topology-cost-repair/result.json')
    assert all(all(t['parity']['gradients_max']==0 for t in f['timings']) for f in preflight['frames'])
    started=time.monotonic();stage_log=[];active=None
    save(out/'progress.json',dict(state='preparing',pid=__import__('os').getpid()))
    try:
        project=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118');assets=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
        sys.path.insert(0,str(project/'src'))
        from shuolee_portrait import runtime as rt
        rt.configure(assets,out)
        import torch
        import smplx
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            utils,_,_,_=entry.load_official(official)
            loader=entry.module(project/'src/avatar_core/shuolee_vposer.py','_fullfit_topology_vposer')
            loader.MODEL=assets/'vposer/V02_05';loader.CODE=project/'third_party/human_body_prior'
            vposer,_,_=loader.load_vposer('cuda')
            mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
            body=smplx.SMPLX(str(assets/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,
                num_expression_coeffs=10,use_pca=True,num_pca_comps=12,flat_hand_mean=False,
                use_face_contour=False,create_body_pose=False,create_transl=False,
                joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).cuda()
            parents=body.parents.detach().cpu().clone()
            with np.load(root/'fit/fitted.npz',allow_pickle=False) as z:
                slot=list(z['frame_ids'].astype(str)).index(plan['frame_id'])
                target=z['observed_native_keypoints'][slot:slot+1].copy();beta=z['shared_betas'].copy();K=z['K_integer_centers'][slot].copy()
            row=read(root/'data/training-inputs.json')['records'][slot]
            assert entry.sha(row['image'])==row['source_sha256']
            with Image.open(row['image']) as im:rgb=np.asarray(im.convert('RGB'),dtype=np.float32)/255
            original_load=entry.load_official
            def instrumented_load(path):
                modules=original_load(path);fitting=modules[2];original=fitting.FittingMonitor.run_fitting
                def run_monitor(self,optimizer,closure,*a,**k):
                    begin=time.monotonic();closures=0;steps=0;step_function=optimizer.step
                    def observe_closure(*ca,**ck):
                        nonlocal closures
                        if time.monotonic()-started>plan['max_seconds']:raise RuntimeError('Paired full-fit time budget exceeded.')
                        closures+=1
                        return closure(*ca,**ck)
                    def observe_step(*sa,**sk):
                        nonlocal steps
                        steps+=1
                        return step_function(*sa,**sk)
                    optimizer.step=observe_step
                    try:return original(self,optimizer,observe_closure,*a,**k)
                    finally:
                        optimizer.step=step_function
                        entry_log=dict(mode=active,stage_in_mode=sum(x['mode']==active for x in stage_log),
                            monitor_step_calls=steps,closure_evaluations=closures,seconds=time.monotonic()-begin)
                        stage_log.append(entry_log);save(out/'stages.json',stage_log)
                        print(json.dumps(entry_log),flush=True)
                fitting.FittingMonitor.run_fitting=run_monitor
                return modules
            entry.load_official=instrumented_load
            runs=[];reference=None
            for active in ['cuda_parent_indices','cpu_parent_indices']:
                destination=out/active;destination.mkdir()
                body.parents=parents.to('cuda' if active.startswith('cuda') else 'cpu')
                torch.manual_seed(6032);torch.cuda.manual_seed_all(6032)
                save(out/'progress.json',dict(state='official_fit_running',mode=active,pid=__import__('os').getpid(),elapsed_seconds=time.monotonic()-started))
                begin=time.monotonic()
                arrays,report=entry.fit_one(official_root=official,image=rgb,keypoints=target,K_integer_centers=K,
                    body_model=body,vposer2=vposer,result_path=destination/'official-result.pkl',
                    shared_betas=beta,freeze_betas=True,device='cuda',diff_path=destination/'input-output-adapter.diff')
                torch.cuda.synchronize();seconds=time.monotonic()-begin
                np.savez_compressed(destination/'winner-camera-space.npz',**arrays);save(destination/'result.json',report)
                differences={}
                if reference is None:reference=arrays
                for name,value in arrays.items():
                    assert value.shape==reference[name].shape
                    differences[name]=float(np.max(np.abs(value.astype(np.float64)-reference[name].astype(np.float64))))
                runs.append(dict(mode=active,seconds=seconds,winning_loss=report['winning_loss'],orientation_count=report['orientation_count'],
                    per_field_difference_from_cuda=differences,exact_output_equal=all(v==0 for v in differences.values())))
            entry.load_official=original_load
            for p,h in plan['source_hashes'].items():assert entry.sha(p)==h,p
            result=dict(state='completed',frame_id=plan['frame_id'],plan_sha256=entry.sha(plan_path),runs=runs,stages=stage_log,
                paired_outputs_identical=all(x['exact_output_equal'] for x in runs),elapsed_seconds=time.monotonic()-started,
                original_94_results_unchanged=True,production_code_changed=False,Gaussian_training_started=False,
                loss_or_optimizer_algorithm_changed=False,shared_beta_unchanged=True,
                limitation='One complete official frame with fixed shared beta. Not94-frame runtime, quality improvement, or video-to-export15-minute acceptance.')
            save(out/'result.json',result);save(out/'progress.json',dict(state='complete',pid=None));print(json.dumps(result),flush=True)
    except BaseException as error:
        save(out/'failure.json',dict(mode=active,error=repr(error),traceback=traceback.format_exc(),source_results_preserved=True))
        save(out/'progress.json',dict(state='failed_preserved',pid=None));raise


if __name__=='__main__':main()
