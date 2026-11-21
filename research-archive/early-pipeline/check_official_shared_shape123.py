"""One bounded paired official fit: fixed versus updatable shared beta."""
import argparse
import fcntl
import json
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
import official_smplifyx_entry123 as entry


def main(plan_path):
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
    plan=read(plan_path); root=Path(plan['source_root']); out=Path(plan['output'])
    save=lambda p,d:Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    assert not out.exists()
    for path,digest in plan['source_hashes'].items():assert entry.sha(path)==digest,path
    out.mkdir(parents=True)
    save(out/'progress.json',dict(state='preparing',pid=os.getpid()))
    started=time.monotonic(); active=None
    try:
        project=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
        assets=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
        sys.path.insert(0,str(project/'src'))
        from shuolee_portrait import runtime as rt
        rt.configure(assets,out)
        import torch
        import smplx
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        with np.load(root/'fit/fitted.npz',allow_pickle=False) as z:
            slot=list(z['frame_ids'].astype(str)).index(plan['frame_id'])
            target=z['observed_native_keypoints'][slot:slot+1].copy()
            beta=z['shared_betas'].copy(); K=z['K_integer_centers'][slot].copy()
        row=read(root/'data/training-inputs.json')['records'][slot]
        assert entry.sha(row['image'])==row['source_sha256']
        raw=Image.open(row['image']).convert('RGB'); rgb=np.asarray(raw,dtype=np.float32)/255
        with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            utils,_,_,_=entry.load_official(plan['official_root'])
            loader=entry.module(project/'src/avatar_core/shuolee_vposer.py','_shape_control_vposer')
            loader.MODEL=assets/'vposer/V02_05';loader.CODE=project/'third_party/human_body_prior'
            vposer,_,_=loader.load_vposer('cuda')
            mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
            body=smplx.SMPLX(str(assets/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,
                num_expression_coeffs=10,use_pca=True,num_pca_comps=12,flat_hand_mean=False,
                use_face_contour=False,create_body_pose=False,create_transl=False,
                joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).cuda()
            from run_official_sequence123_topology import cache_static_topology
            cache_static_topology(body)
            original_load=entry.load_official
            def bounded_load(path):
                modules=original_load(path); fitting=modules[2]
                original_run=fitting.FittingMonitor.run_fitting
                def run_monitor(self,optimizer,closure,*args,**kwargs):
                    def checked_closure(*a,**k):
                        if time.monotonic()-started > plan['max_seconds']:
                            raise RuntimeError('Shared shape control time budget exceeded.')
                        return closure(*a,**k)
                    return original_run(self,optimizer,checked_closure,*args,**kwargs)
                fitting.FittingMonitor.run_fitting=run_monitor
                return modules
            entry.load_official=bounded_load
            results=[];saved={}
            groups=dict(body17=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]),
                feet6=np.arange(19,25),left_hand21=np.arange(25,46),right_hand21=np.arange(46,67),face51=np.arange(67,118))
            for active,freeze in [('fixed_beta',True),('updatable_beta',False)]:
                dest=out/active;dest.mkdir()
                torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
                save(out/'progress.json',dict(state='official_fit_running',mode=active,pid=os.getpid()))
                begin=time.monotonic()
                arrays,report=entry.fit_one(official_root=plan['official_root'],image=rgb,keypoints=target,
                    K_integer_centers=K,body_model=body,vposer2=vposer,result_path=dest/'official-result.pkl',
                    shared_betas=beta,freeze_betas=freeze,device='cuda',diff_path=dest/'input-output-adapter.diff')
                torch.cuda.synchronize()
                arrays.update(frame_id=np.asarray(plan['frame_id']),observed_native_keypoints=target,
                    source_frame=np.asarray(row['source_frame_index']),image_sha256=np.asarray(row['source_sha256']))
                np.savez_compressed(dest/'winner-camera-space.npz',**arrays)
                report.update(state='complete',frame_id=plan['frame_id'],elapsed_seconds=time.monotonic()-begin)
                save(dest/'result.json',report)
                errors=np.linalg.norm(arrays['joints_projected_native'][0]-target[0,:,:2],axis=1)
                valid=target[0,:,2]>0;valid[[1,9,12]]=False
                stats={}
                for label,ids in groups.items():
                    e=errors[ids[valid[ids]]]
                    stats[label]=dict(count=len(e),mean=float(e.mean()),median=float(np.median(e)),maximum=float(e.max()))
                results.append(dict(mode=active,seconds=report['elapsed_seconds'],winning_loss=report['winning_loss'],
                    groups=stats,beta=arrays['betas'].tolist(),beta_change_l2=float(np.linalg.norm(arrays['betas']-beta))))
                saved[active]=arrays
                print(json.dumps(results[-1]),flush=True)
            entry.load_official=original_load
        from review_full_sequence123 import mesh_renderer,joined
        mesh,_=mesh_renderer()
        overlays={name:mesh(raw,np.array([0,0,*raw.size]),arrays) for name,arrays in saved.items()}
        old_review=read(root/f'frames/{plan["frame_id"]}/saved-mesh-review.json')
        records=Path('/mnt/d/00/记录/005')
        images=[]
        for item in old_review['files']:
            if item['panels'] != '原图_官方SMPLX透明覆盖':continue
            p=Path(item['path']); label=next((n for n in ['全身','面部','双脚','左手','右手'] if '_'+n+'_' in p.name),None)
            if label is None:continue
            box=item['crop_integer_xyxy']
            panel=joined([raw.crop(box),overlays['fixed_beta'].crop(box),overlays['updatable_beta'].crop(box)])
            alias=out/f'{label}.png';panel.save(alias)
            dest=records/('smpl-x' if label=='全身' else '分析')/f'20261030_共享体型约束对照_{plan["frame_id"]}_{label}_左原图_中固定体型_右允许更新.png'
            panel.save(dest)
            images.append(dict(label=label,crop_integer_xyxy=box,path=str(alias),record=str(dest),sha256=entry.sha(alias)))
        for path,digest in plan['source_hashes'].items():assert entry.sha(path)==digest,path
        result=dict(state='completed_visual_review_pending',frame_id=plan['frame_id'],plan_sha256=entry.sha(plan_path),
            runs=results,images=images,source94_unchanged=True,model_or_camera_promoted=False,
            gaussian_training_started=False,loss_optimizer_mapping_unchanged=True,actual_visual_review=False,
            interpretation='Same training observations; lower loss alone cannot establish better physical shape. A new shared shape must also pass the front view before use.')
        save(out/'result.json',result);save(out/'progress.json',dict(state='complete',pid=None))
        print(json.dumps(dict(output=str(out),state=result['state'])),flush=True)
    except BaseException as error:
        save(out/'failure.json',dict(mode=active,error=repr(error),traceback=traceback.format_exc()))
        save(out/'progress.json',dict(state='failed_preserved',pid=None));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    main(p.parse_args().plan)
