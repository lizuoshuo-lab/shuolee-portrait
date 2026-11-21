"""Full AV2 scene joint baseline without SMPL; isolated local research runner."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import logging
import os
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity

ROOT=Path('/opt/gs-slam-lab/dynamic-human')
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/dynamic-human-street-joint-009'
SCENE=LAB/'data/dynamic-human/argoverse2/processed/training/000'
REPO=ROOT/'repos/drivestudio'
sys.path.insert(0,str(REPO))
os.chdir(REPO)
from datasets.driving_dataset import DrivingDataset
from datasets.argoverse.argoverse_sourceloader import ArgoVerseCameraData
from models.trainers import MultiTrainer
from street_capture_time_adapter import install as install_capture
from street_training_only_adapter import install as install_training_only


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def config():
    cfg=OmegaConf.load(REPO/'configs/omnire.yaml')
    cfg.data=OmegaConf.load(REPO/'configs/datasets/argoverse/7cams.yaml').data
    cfg.seed=73
    cfg.dataset='argoverse/7cams'
    cfg.data.data_root=str(SCENE.parent)
    cfg.data.start_timestep=0
    cfg.data.end_timestep=155
    cfg.data.pixel_source.test_image_stride=10
    cfg.data.pixel_source.load_smpl=False
    # Fixed calibrated cameras and identity exposure on both train and test.
    # No untrained per-image test embeddings are used at evaluation.
    for name in ('SMPLNodes','CamPose','Affine'):
        cfg.model.pop(name)
    return cfg


def person_rois(dataset):
    annotations=json.loads((SCENE/'instances/instances_info.json').read_text())
    reference=np.linalg.inv(np.loadtxt(SCENE/'ego_pose/000.txt'))
    signs=np.array(list(itertools.product((-1.,1.),repeat=3)))
    rois=[]
    for identity,item in annotations.items():
        if item['class_name']!='PEDESTRIAN':
            continue
        a=item['frame_annotations']
        for index,frame in enumerate(a['frame_idx']):
            if frame not in dataset.test_timesteps:
                continue
            box=signs*np.asarray(a['box_size'][index])[None]/2
            obj=reference@np.asarray(a['obj_to_world'][index])
            xyz=box@obj[:3,:3].T+obj[:3,3]
            for camera,cam in dataset.pixel_source.camera_data.items():
                w2c=np.linalg.inv(cam.cam_to_worlds[frame].numpy())
                camera_xyz=xyz@w2c[:3,:3].T+w2c[:3,3]
                if camera_xyz[:,2].min()<=0.1:
                    continue
                p=camera_xyz@cam.intrinsics[frame].numpy().T
                uv=p[:,:2]/p[:,2,None]
                lower=np.floor(uv.min(0)).astype(int)
                upper=np.ceil(uv.max(0)).astype(int)
                lo=np.maximum(lower,0)
                hi=np.minimum(upper,[cam.WIDTH,cam.HEIGHT])
                width,height=hi-lo
                area=width*height
                full_area=np.prod(upper-lower)
                if width<12 or height<24 or area<512 or area<0.5*full_area:
                    continue
                rois.append(dict(frame=int(frame),camera=int(camera),instance=int(identity),
                    box=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])],
                    image_size=[int(cam.WIDTH),int(cam.HEIGHT)],source='projected_annotated_3D_box'))
    return rois


def to_cuda(info):
    return {k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in info.items()}


def save_rgb(path, array):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.uint8(np.clip(array,0,1)*255+0.5)).save(path)


def score(gt,pred):
    mse=float(np.mean((gt.astype(np.float64)-pred.astype(np.float64))**2))
    return dict(psnr_db=float(-10*np.log10(max(mse,1e-12))),
        ssim=float(structural_similarity(gt,pred,data_range=1.,channel_axis=-1)))


@torch.no_grad()
def evaluate(trainer,dataset,rois,out,label,complete=False):
    trainer.set_eval()
    trainer.render_each_class=False
    rows=[]
    roi_rows=[]
    directory=out/'evaluation'/label
    directory.mkdir(parents=True,exist_ok=True)
    indices=dataset.test_indices if complete else [130*7+4,140*7+4,150*7+6]
    for img_idx in indices:
        info,cam=dataset.full_image_set.get_image(img_idx,1.)
        info,cam=to_cuda(info),to_cuda(cam)
        output=trainer(info,cam,novel_view=True)
        prediction=output['rgb'].clamp(0,1).cpu().numpy()
        target=info['pixels'].cpu().numpy()
        frame,camera=divmod(img_idx,7)
        stem=f'frame{frame:03d}-camera{camera}'
        save_rgb(directory/f'{stem}-prediction.png',prediction)
        save_rgb(out/'ground-truth'/f'{stem}.png',target)
        rows.append(dict(frame=frame,camera=camera,**score(target,prediction)))
        for roi in rois:
            if roi['frame']!=frame or roi['camera']!=camera:
                continue
            x0,y0,x1,y1=roi['box']
            roi_rows.append(dict(**roi,**score(target[y0:y1,x0:x1],prediction[y0:y1,x0:x1])))
            if roi['instance']==79:
                save_rgb(directory/f'{stem}-person79.png',prediction[y0:y1,x0:x1])
                save_rgb(out/'ground-truth'/f'{stem}-person79.png',target[y0:y1,x0:x1])
    result=dict(label=label,optimizer_updates=trainer.completed_updates,schedule_step=int(trainer.step),views=rows,person_box_regions=roi_rows,
        mean_full_psnr_db=float(np.mean([r['psnr_db'] for r in rows])),
        mean_person_box_psnr_db=float(np.mean([r['psnr_db'] for r in roi_rows])) if roi_rows else None,
        mean_person_box_ssim=float(np.mean([r['ssim'] for r in roi_rows])) if roi_rows else None,
        region_definition='Predeclared projected annotated pedestrian boxes; includes background and occluded people; not segmentation GT',
        image_scale='Author AV2 default loading scale 1/3, no image enhancement',
        heldout_boundary='Every tenth time across all cameras; calibrated trajectories known. No heldout RGB or heldout LiDAR initialization. No future motion claim.')
    write_json(directory/'metrics.json',result)
    trainer.set_train()
    return result


def checkpoint(trainer,out,step):
    # Model-only checkpoints are the upstream interchange format.
    torch.save(trainer.state_dict(only_model=True),out/f'checkpoint-{step:05d}.pt')
    # Preserve optimizer/RNG/densification accumulators for a later explicit resume adapter.
    state=dict(model=trainer.state_dict(only_model=True),optimizer=trainer.optimizer.state_dict(),
        grad_scaler=trainer.grad_scaler.state_dict(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
        accumulators={name:{key:getattr(model,key,None) for key in ('xys_grad_norm','vis_counts','max_2Dsize')}
                      for name,model in trainer.models.items()},updates=step,
        limitation='Full state saved; resume path requires a separate tested adapter before use')
    temporary=out/'latest-training-state.pt.tmp'
    torch.save(state,temporary)
    temporary.replace(out/'latest-training-state.pt')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--run-label',default='baseline-bounded')
    args=parser.parse_args()
    if not args.run_label.replace('-','').isalnum():
        raise ValueError('Run label must contain only letters, digits and hyphens')
    out=BASE/('preflight' if args.preflight else args.run_label)
    if (out/'protocol.json').exists():
        raise RuntimeError('Existing run protocol; use a new explicitly versioned run, do not overwrite')
    receipt=LAB/'outputs/dynamic-human-street-005/training-only-initialization-audit.json'
    audit=json.loads(receipt.read_text())
    assert audit['status'].startswith('passed_')
    assert audit['adapter_sha256']==sha(Path(__file__).with_name('street_training_only_adapter.py'))
    if not args.preflight:
        preflight=json.loads((BASE/'preflight/completion.json').read_text())
        assert preflight['status']=='full_scene_joint_forward_backward_and_heldout_render_passed'
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    cfg=config()
    OmegaConf.save(cfg,out/'config.yaml')
    scripts={name:sha(Path(__file__).with_name(name)) for name in (
        Path(__file__).name,'street_capture_time_adapter.py','street_training_only_adapter.py')}
    for name in scripts:
        shutil.copy2(Path(__file__).with_name(name),out/name)
    write_json(out/'protocol.json',dict(status='sealed_before_model_updates',preflight=args.preflight,
        seed=73,total_updates=30000,frames=[0,155],cameras=list(range(7)),loading_scale=3,
        training_images=987,heldout_images=105,heldout_times=list(range(10,156,10)),
        source='DriveStudio configs/omnire.yaml and MultiTrainer; local no-body-template baseline',
        source_trainer_sha256=sha(REPO/'models/trainers/scene_graph.py'),
        configuration_sha256=sha(out/'config.yaml'),scripts=scripts,initialization_audit_sha256=sha(receipt),
        preflight_receipt_sha256=sha(BASE/'preflight/completion.json') if not args.preflight else None,
        allocator_policy={'maximum_process_bytes':10_000_000_000,'empty_cache_every_updates':50,
            'reason':'Previous uncapped run reserved17.27GB with4.16GB peak allocated and stalled; keep original scientific configuration'},
        modifications=['SMPLNodes omitted; all moving pedestrians use DeformableNodes',
            'camera extrinsics fixed at actual image exposure; CamPose disabled',
            'Affine disabled; identity train/eval exposure',
            'training-only LiDAR geometry/color initialization; deterministic nearest-hit sparse depth',
            'exactly30000updates; no full-image error cache; heldout images never used by loss or sampling'],
        schedule='Original 30000-step optimization,800k background LiDAR seeds plus100k near/far random seeds; deformation after3000;1/3 source images',
        evaluation='All105 heldout images and deterministic projected pedestrian boxes, fixed before optimization',
        limitations=['A single public AV2 log; no-body-template ablation, not complete OmniRe',
            'Image-time ego poses but LiDAR-time object annotations; not motion interpolation or rolling shutter correction',
            'Heldout boxes/trajectories and calibrations remain known; cannot claim future action prediction',
            '1/3 loading scale limits recoverable fine details; assess native crops and do not enhance them']))
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(10_000_000_000/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)
    torch.cuda.manual_seed_all(73)
    install_capture(ArgoVerseCameraData)
    install_training_only(DrivingDataset)
    started=time.perf_counter()
    dataset=DrivingDataset(cfg.data)
    assert len(dataset.train_image_set)==987 and len(dataset.test_image_set)==105
    rois=person_rois(dataset)
    write_json(out/'person-rois.json',dict(regions=rois,definition='all visible-front-facing projected pedestrian boxes,clipped area>=50%,width>=12,height>=24,area>=512pixels; occlusion not filtered'))
    trainer=MultiTrainer(**cfg.trainer,num_timesteps=dataset.num_img_timesteps,
        model_config=OmegaConf.create(OmegaConf.to_container(cfg.model)),
        num_train_images=len(dataset.train_image_set),num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,scene_aabb=dataset.aabb.reshape(2,3),device=torch.device('cuda'))
    # Capture local instance-index to author track mapping before model compacts IDs.
    get_init=dataset.get_init_objects
    instance_map={}
    def record_init(*a,**kw):
        result=get_init(*a,**kw)
        kind=kw.get('cur_node_type',a[0] if a else None)
        instance_map[kind]=[{ 'model_index':i,'dataset_index':int(k),
            'author_instance':int(dataset.pixel_source.instances_true_id[k]),'initial_points':v['num_pts']}
            for i,(k,v) in enumerate(result.items())]
        return result
    dataset.get_init_objects=record_init
    trainer.init_gaussians_from_dataset(dataset)
    trainer.initialize_optimizer()
    assert 'DeformableNodes' in trainer.models
    assert any(r['author_instance']==79 for r in instance_map['DeformableNodes'])
    write_json(out/'initialization.json',dict(gaussians=trainer.get_gaussian_count(),instances=instance_map,
        seconds=time.perf_counter()-started,device=torch.cuda.get_device_name(),
        allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
        lidar_points=dataset.lidar_source.num_points,roi_count=len(rois),
        main_person_roi_count=sum(r['instance']==79 for r in rois)))
    steps=[0,1001,3001] if args.preflight else range(30000)
    log=(out/'training.jsonl').open('a',buffering=1)
    for completed_updates,step in enumerate(steps,1):
        trainer.set_train()
        trainer.preprocess_per_train_step(step)
        trainer.optimizer_zero_grad()
        info,cam=dataset.train_image_set.next(trainer._get_downscale_factor())
        image_id=int(info['img_idx'].flatten()[0])
        assert image_id in dataset.train_indices and image_id not in dataset.test_indices
        info,cam=to_cuda(info),to_cuda(cam)
        outputs=trainer(info,cam)
        trainer.update_visibility_filter()
        losses=trainer.compute_losses(outputs,info,cam)
        assert all(bool(torch.isfinite(v).all()) for v in losses.values())
        trainer.backward(losses)
        trainer.completed_updates=completed_updates
        # Explicitly exercise the non-rigid branch during preflight without calling
        # its skipped-interval densification schedule. These updates are discarded.
        if not args.preflight:
            trainer.postprocess_per_train_step(step)
        if args.preflight or step%50==0 or step==3001 or step==29999:
            deformation_gradients={}
            if step>3000:
                for name,param in trainer.models['DeformableNodes'].deform_network.named_parameters():
                    if param.grad is not None:
                        assert torch.isfinite(param.grad).all()
                        deformation_gradients[name]=float(param.grad.abs().max())
                # Some sampled cameras contain no visible moving person. A zero
                # gradient there is valid; report it rather than abort training.
            row=dict(step=step,updates=completed_updates,image_id=image_id,seconds=time.perf_counter()-started,
                losses={k:float(v.detach()) for k,v in losses.items()},
                gaussians=trainer.get_gaussian_count(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),deformation_gradient_max=deformation_gradients)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            write_json(out/'progress.json',row)
            print(json.dumps(row),flush=True)
            torch.cuda.empty_cache()
        del outputs,losses,info,cam
        if not args.preflight and step+1 in (500,3000,10000,20000,30000):
            checkpoint(trainer,out,step+1)
            if step+1 in (500,3000,10000):
                evaluate(trainer,dataset,rois,out,f'updates-{step+1:05d}')
    if args.preflight:
        evaluate(trainer,dataset,rois,out,'discarded-smoke')
        result=dict(status='full_scene_joint_forward_backward_and_heldout_render_passed',
            discarded_optimizer_steps=[0,1001,3001],quality_claim=False)
    else:
        result=evaluate(trainer,dataset,rois,out,'final',complete=True)
        result['status']='30000_updates_and_all105_heldout_images_rendered_pending_independent_visual_audit'
    result.update(seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(out/'completion.json',result)
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
