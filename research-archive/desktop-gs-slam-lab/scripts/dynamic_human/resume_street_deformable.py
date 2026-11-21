"""Restore the full 10000-update joint model, audit restoration, then finish 30k.

The source runner and scientific configuration are sealed in the parent run.
This changes CUDA allocation policy and logging/checkpoint frequency only.
"""
from pathlib import Path
import copy
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf

LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
ROOT=Path('/opt/gs-slam-lab/dynamic-human')
BASE=LAB/'outputs/dynamic-human-street-joint-009'
PARENT=BASE/'baseline-bounded'
OUT=BASE/'resume-10000-expandable'
SOURCE=PARENT/'latest-training-state.pt'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def equal_tree(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape,path
        assert torch.equal(a.detach().cpu(),b.detach().cpu()),path
    elif isinstance(a,dict):
        assert set(a)==set(b),path
        for key in a:equal_tree(a[key],b[key],f'{path}.{key}')
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal_tree(x,y,f'{path}[{i}]')
    else:
        assert a==b,(path,a,b)


def cuda_tree(value):
    if isinstance(value,torch.Tensor):return value.to('cuda')
    if isinstance(value,dict):return {k:cuda_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cuda_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cuda_tree(v) for v in value)
    return value


def main():
    assert not (OUT/'protocol.json').exists(),'Preserve previous attempts; no overwrite'
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True,garbage_collection_threshold:0.8'
    parent=json.loads((PARENT/'protocol.json').read_text())
    assert sha(PARENT/'config.yaml')==parent['configuration_sha256']
    for name,digest in parent['scripts'].items():
        assert sha(PARENT/name)==digest,name
        if name!='train_street_deformable.py':
            assert sha(Path(__file__).with_name(name))==digest,name
    spec=importlib.util.spec_from_file_location('street_sealed_runner',PARENT/'train_street_deformable.py')
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    OUT.mkdir(parents=True)
    shutil.copy2(Path(__file__),OUT/Path(__file__).name)
    shutil.copy2(PARENT/'config.yaml',OUT/'config.yaml')
    shutil.copy2(PARENT/'person-rois.json',OUT/'person-rois.json')
    runner.write_json(OUT/'protocol.json',dict(status='sealed_before_resume',parent_run=str(PARENT),
        original_target_updates=30000,resume_completed_updates=10000,remaining_updates=20000,
        full_checkpoint_sha256=sha(SOURCE),model_checkpoint_sha256=sha(PARENT/'checkpoint-10000.pt'),
        configuration_sha256=parent['configuration_sha256'],parent_protocol_sha256=sha(PARENT/'protocol.json'),
        resume_script_sha256=sha(Path(__file__)),allocator_limit_bytes=12_000_000_000,
        allocator_configuration=os.environ['PYTORCH_CUDA_ALLOC_CONF'],
        unchanged='data split, camera timing, initialization policy, Gaussian models, network, losses, learning-rate schedules, ROI definitions and total update budget',
        restore='model parameters and buffers, Adam moments and steps, gradient scaler, CPU/CUDA/NumPy/Python RNG, densification accumulators',
        required_checks=['full state model equals standalone10000checkpoint',
            'loaded model/optimizer/accumulators equal saved values',
            'restored10000render matches existing checkpoint render',
            'first resumed sampled training image equals original step10000 image'],
        additional_checkpoints=[11000,15000,20000,25000,30000],
        final_evaluation='all105 heldout images with original435 projected pedestrian boxes; pending independent image and visual review'))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(12_000_000_000/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(73)
    torch.cuda.manual_seed_all(73)
    np.random.seed(73)
    random.seed(73)
    runner.install_capture(runner.ArgoVerseCameraData)
    runner.install_training_only(runner.DrivingDataset)
    cfg=OmegaConf.load(OUT/'config.yaml')
    started=time.perf_counter()
    dataset=runner.DrivingDataset(cfg.data)
    assert len(dataset.train_image_set)==987 and len(dataset.test_image_set)==105
    rois=json.loads((OUT/'person-rois.json').read_text())['regions']
    assert runner.person_rois(dataset)==rois
    trainer=runner.MultiTrainer(**cfg.trainer,num_timesteps=dataset.num_img_timesteps,
        model_config=OmegaConf.create(OmegaConf.to_container(cfg.model)),
        num_train_images=987,num_full_images=1092,test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.aabb.reshape(2,3),device=torch.device('cuda'))
    original_log=(ROOT/'logs/street-joint-baseline-bounded.log').read_text()
    expected_radius=float(re.search(r'scene radius: ([0-9.e+-]+)',original_log).group(1))
    assert trainer.scene_radius==expected_radius,(trainer.scene_radius,expected_radius)
    # These are our own local checkpoints. The full file includes NumPy/Python RNG states.
    state=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True)
    standalone=torch.load(PARENT/'checkpoint-10000.pt',map_location='cpu',weights_only=False,mmap=True)
    assert state['updates']==10000 and state['model']['step']==9999
    equal_tree(state['model'],standalone,'standalone_model_agreement')
    del standalone
    trainer.load_state_dict(cuda_tree(state['model']),load_only_model=True,strict=True)
    equal_tree(state['model'],trainer.state_dict(only_model=True),'loaded_model')
    trainer.initialize_optimizer()
    trainer.optimizer.load_state_dict(state['optimizer'])
    trainer.grad_scaler.load_state_dict(state['grad_scaler'])
    equal_tree(state['optimizer'],trainer.optimizer.state_dict(),'loaded_Adam')
    for name,fields in state['accumulators'].items():
        for key,value in fields.items():
            setattr(trainer.models[name],key,cuda_tree(value))
            equal_tree(value,getattr(trainer.models[name],key),f'accumulator.{name}.{key}')
    trainer.completed_updates=10000
    torch.cuda.empty_cache()
    recovered=runner.evaluate(trainer,dataset,rois,OUT,'recovered-10000')
    previous=json.loads((PARENT/'evaluation/updates-10000/metrics.json').read_text())
    comparisons=[]
    for row_old,row_new in zip(previous['views'],recovered['views']):
        assert (row_old['frame'],row_old['camera'])==(row_new['frame'],row_new['camera'])
        stem=f"frame{row_old['frame']:03d}-camera{row_old['camera']}-prediction.png"
        before=np.asarray(Image.open(PARENT/'evaluation/updates-10000'/stem),dtype=np.int16)
        after=np.asarray(Image.open(OUT/'evaluation/recovered-10000'/stem),dtype=np.int16)
        difference=np.abs(before-after)
        row=dict(image=stem,max_channel_difference_8bit=int(difference.max()),
            mean_channel_difference_8bit=float(difference.mean()),
            psnr_difference_db=row_new['psnr_db']-row_old['psnr_db'])
        assert row['max_channel_difference_8bit']<=1,row
        assert abs(row['psnr_difference_db'])<1e-4,row
        comparisons.append(row)
    assert len(comparisons)==3
    # Evaluation must not alter accumulation used by the next refinement step.
    for name,fields in state['accumulators'].items():
        for key,value in fields.items():
            equal_tree(value,getattr(trainer.models[name],key),f'after_eval_accumulator.{name}.{key}')
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    del state
    gc.collect()
    torch.cuda.empty_cache()
    original_rows=[json.loads(line) for line in (PARENT/'training.jsonl').read_text().splitlines() if line.strip()]
    original_first=next(r for r in original_rows if r['step']==10000)
    audit=dict(status='model_Adam_RNG_accumulators_restored_render_verified_first_sample_pending',
        source_updates=10000,scene_radius=trainer.scene_radius,
        comparisons=comparisons,gaussians=trainer.get_gaussian_count(),
        optimizer_groups=[g['name'] for g in trainer.optimizer.param_groups],
        initialization_seconds=time.perf_counter()-started)
    runner.write_json(OUT/'recovery-audit.json',audit)
    log=(OUT/'training.jsonl').open('a',buffering=1)
    for step in range(10000,30000):
        trainer.set_train()
        trainer.preprocess_per_train_step(step)
        trainer.optimizer_zero_grad()
        info,cam=dataset.train_image_set.next(trainer._get_downscale_factor())
        image_id=int(info['img_idx'].flatten()[0])
        assert image_id in dataset.train_indices and image_id not in dataset.test_indices
        if step==10000:
            assert image_id==original_first['image_id'],(image_id,original_first['image_id'])
            audit.update(status='passed_full_state_restore_render_and_next_training_image',first_resumed_image_id=image_id)
            runner.write_json(OUT/'recovery-audit.json',audit)
        info,cam=runner.to_cuda(info),runner.to_cuda(cam)
        outputs=trainer(info,cam)
        trainer.update_visibility_filter()
        losses=trainer.compute_losses(outputs,info,cam)
        assert all(bool(torch.isfinite(v).all()) for v in losses.values())
        trainer.backward(losses)
        trainer.completed_updates=step+1
        trainer.postprocess_per_train_step(step)
        if step%50==0 or step==29999:
            grad={name:float(p.grad.abs().max()) for name,p in trainer.models['DeformableNodes'].deform_network.named_parameters() if p.grad is not None}
            assert all(math.isfinite(v) for v in grad.values())
            row=dict(step=step,updates=step+1,resumed_updates=step-9999,image_id=image_id,
                seconds=time.perf_counter()-started,losses={k:float(v.detach()) for k,v in losses.items()},
                gaussians=trainer.get_gaussian_count(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),deformation_gradient_max=grad)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            runner.write_json(OUT/'progress.json',row)
            print(json.dumps(row),flush=True)
        del outputs,losses,info,cam
        if step%50==0:torch.cuda.empty_cache()
        if step+1 in (11000,15000,20000,25000,30000):
            runner.checkpoint(trainer,OUT,step+1)
    result=runner.evaluate(trainer,dataset,rois,OUT,'final',complete=True)
    result.update(status='30000_total_updates_and_all105_heldout_images_rendered_pending_independent_visual_audit',
        resumed_from_updates=10000,resumed_optimizer_updates=20000,
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    runner.write_json(OUT/'completion.json',result)
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        failure=traceback.format_exc()
        if OUT.exists():(OUT/'failure.txt').write_text(failure)
        print(failure,flush=True)
        raise
