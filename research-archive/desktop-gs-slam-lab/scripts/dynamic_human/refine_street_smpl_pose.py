"""Frozen-scene, target-person pose refinement with training RGB only.

Exploratory diagnostic from a completed30k checkpoint, not a fresh30k comparison.
"""
from pathlib import Path
import gc
import itertools
import json
import logging
import math
import os
import shutil
import time
import traceback
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from pytorch3d.transforms import quaternion_multiply, quaternion_invert, quaternion_to_axis_angle, axis_angle_to_quaternion
import train_street_deformable as helper
from resume_street_deformable import cuda_tree, equal_tree, sha

LAB = helper.LAB
SOURCE = LAB / 'outputs/dynamic-human-street-smpl-joint-011/articulated'
BASELINE = LAB / 'outputs/dynamic-human-street-joint-009/resume-10000-expandable'
OUT = LAB / 'outputs/dynamic-human-street-pose-refinement-014'
PARAMETERS = ('instances_trans', 'instances_quats', 'smpl_qauts')
UPDATES = 3000


def status(value, **fields):
    helper.write_json(OUT / 'state.json', dict(status=value, pid=os.getpid(), **fields))
    print(json.dumps(dict(status=value, pid=os.getpid(), **fields)), flush=True)


def training_regions(dataset, node, model_index):
    a = json.loads((helper.SCENE / 'instances/instances_info.json').read_text())['79']['frame_annotations']
    reference = np.linalg.inv(np.loadtxt(helper.SCENE / 'ego_pose/000.txt'))
    corners = np.array(list(itertools.product((-1., 1.), repeat=3)))
    result = []
    for index, frame in enumerate(a['frame_idx']):
        if frame in dataset.test_timesteps or not bool(node.instances_fv[frame, model_index]):
            continue
        transform = reference @ np.asarray(a['obj_to_world'][index])
        xyz = corners * np.asarray(a['box_size'][index])[None] / 2
        xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        for camera, cam in dataset.pixel_source.camera_data.items():
            inverse = np.linalg.inv(cam.cam_to_worlds[frame].numpy())
            camera_xyz = xyz @ inverse[:3, :3].T + inverse[:3, 3]
            if camera_xyz[:, 2].min() <= .1:
                continue
            p = camera_xyz @ cam.intrinsics[frame].numpy().T
            uv = p[:, :2] / p[:, 2:]
            low, high = np.floor(uv.min(0)).astype(int), np.ceil(uv.max(0)).astype(int)
            lo, hi = np.maximum(low, 0), np.minimum(high, [cam.WIDTH, cam.HEIGHT])
            width, height = hi - lo
            area = width * height
            if width < 12 or height < 24 or area < 512 or area < .5 * np.prod(high - low):
                continue
            image_id = frame * 7 + camera
            assert image_id in dataset.train_indices and image_id not in dataset.test_indices
            result.append(dict(frame=int(frame), camera=int(camera), image_id=int(image_id),
                box=[*lo.tolist(), *hi.tolist()], image_size=[int(cam.WIDTH), int(cam.HEIGHT)]))
    assert result and len({r['image_id'] for r in result}) == len(result)
    return result


def frozen_check(trainer, reference, model_index, active):
    actual = trainer.state_dict(only_model=True)
    for model_name, model_state in reference['models'].items():
        if model_name != 'SMPLNodes':
            equal_tree(model_state, actual['models'][model_name], 'frozen.' + model_name)
            continue
        for name, expected in model_state.items():
            current = actual['models'][model_name][name]
            if name not in PARAMETERS:
                equal_tree(expected, current, 'frozen.SMPLNodes.' + name)
            else:
                mask = torch.ones(expected.shape[:2], dtype=torch.bool)
                mask[active.cpu(), model_index] = False
                assert torch.equal(expected[mask], current.detach().cpu()[mask]), name
    for key in reference:
        if key != 'models':
            equal_tree(reference[key], actual[key], 'frozen.trainer.' + key)


@torch.no_grad()
def score_training(trainer, dataset, regions, label):
    trainer.set_eval()
    rows = []
    directory = OUT / 'training-evaluation' / label
    directory.mkdir(parents=True)
    for roi in regions:
        info, camera = dataset.full_image_set.get_image(roi['image_id'], 1.)
        info, camera = helper.to_cuda(info), helper.to_cuda(camera)
        result = trainer(info, camera, novel_view=True)
        prediction = result['rgb'].clamp(0, 1).cpu().numpy()
        target = info['pixels'].cpu().numpy()
        x0, y0, x1, y1 = roi['box']
        crop_target, crop_pred = target[y0:y1, x0:x1], prediction[y0:y1, x0:x1]
        rows.append(dict(**roi, **helper.score(crop_target, crop_pred), mae=float(np.abs(crop_target-crop_pred).mean())))
        if (roi['frame'], roi['camera']) in [(129,4),(131,4),(149,6),(151,6)]:
            stem = f"frame{roi['frame']:03d}-camera{roi['camera']}"
            helper.save_rgb(directory / (stem + '-prediction.png'), prediction)
            helper.save_rgb(directory / (stem + '-person79.png'), crop_pred)
            helper.save_rgb(OUT / 'training-ground-truth' / (stem + '.png'), target)
            helper.save_rgb(OUT / 'training-ground-truth' / (stem + '-person79.png'), crop_target)
        del info, camera, result
    result = dict(label=label, count=len(rows), mean_person_psnr_db=float(np.mean([r['psnr_db'] for r in rows])),
        mean_mae=float(np.mean([r['mae'] for r in rows])), rows=rows, source='training RGB only; not heldout validation')
    helper.write_json(directory / 'metrics.json', result)
    return result


def main():
    assert not (OUT / 'protocol.json').exists(), 'Preserve earlier attempts'
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True,garbage_collection_threshold:0.8'
    receipt = json.loads((SOURCE / 'completion.json').read_text())
    assert receipt['optimizer_updates'] == 30000 and len(receipt['views']) == 105
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, OUT / Path(__file__).name)
    cfg = OmegaConf.load(SOURCE / 'config.yaml')
    OmegaConf.save(cfg, OUT / 'config.yaml')
    protocol = dict(status='sealed_before_pose_refinement', source_updates=30000, additional_updates=UPDATES,
        source_checkpoint_sha256=sha(SOURCE / 'checkpoint-30000.pt'), script_sha256=sha(Path(__file__)),
        person=79, changed='Only target-person training-time root translations, root rotations and23joint rotations.',
        frozen='All Gaussian centers, appearance, opacity, scales, orientation, voxel skinning, background, other people, cameras and exposure. All heldout pose entries unchanged.',
        training_views='Projected annotated boxes of target79 on valid articulated training times; same visibility/size criteria as existing heldout ROI protocol.',
        rates=dict(translation_m=.005, root_quaternion=.001, joint_quaternion=.002),
        loss=dict(rgb_l1=.8, ssim=.2, translation_prior=.01, rotation_prior=.01, translation_delta_smooth=.05, rotation_delta_smooth=.01),
        hard_limits=dict(translation_from_source_m=.5, root_from_source_degrees=25, joint_from_source_degrees=45),
        discarded_preflight_updates=4, evaluation='All105original heldout images /435boxes; matched source reload verified; original neighbor interpolation.',
        boundaries=['Focused exploratory pose-optimization diagnostic after inspecting previous failures; not an untouched test benchmark.',
            'Additional3000updates, not same-budget comparison against30k source; no new data or heldout RGB supervision.',
            'Target boxes contain background and occluders; no silhouette or joint GT.',
            'Existing learned Gaussian offsets/appearance may already compensate pose error; failure here does not prove pose alignment irrelevant.',
            'No generic scene-wide completion claim from one person. No image enhancement.'])
    helper.write_json(OUT / 'protocol.json', protocol)
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(12_000_000_000 / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(79)
    helper.install_capture(helper.ArgoVerseCameraData)
    helper.install_training_only(helper.DrivingDataset)
    started = time.perf_counter()
    status('loading_dataset')
    dataset = helper.DrivingDataset(OmegaConf.load(BASELINE / 'config.yaml').data)
    assert len(dataset.train_image_set) == 987 and len(dataset.test_image_set) == 105
    trainer = helper.MultiTrainer(**cfg.trainer, num_timesteps=156,
        model_config=OmegaConf.create(OmegaConf.to_container(cfg.model)), num_train_images=987, num_full_images=1092,
        test_set_indices=dataset.test_timesteps, scene_aabb=dataset.aabb.reshape(2,3), device=torch.device('cuda'))
    saved = torch.load(SOURCE / 'checkpoint-30000.pt', map_location='cpu', weights_only=False, mmap=True)
    trainer.load_state_dict(cuda_tree(saved), load_only_model=True, strict=True)
    equal_tree(saved, trainer.state_dict(only_model=True), 'restored_source')
    trainer.completed_updates = 30000
    trainer.render_each_class = False
    rois = json.loads((SOURCE / 'person-rois.json').read_text())['regions']
    assert len(rois) == 435
    helper.write_json(OUT / 'person-rois.json', dict(regions=rois))
    status('verifying_source_render')
    helper.evaluate(trainer, dataset, rois, OUT, 'source-reload')
    reload_checks = []
    for frame,camera in [(130,4),(140,4),(150,6)]:
        name = f'frame{frame:03d}-camera{camera}-prediction.png'
        old = np.array(Image.open(SOURCE / 'evaluation/final' / name), dtype=np.int16)
        new = np.array(Image.open(OUT / 'evaluation/source-reload' / name), dtype=np.int16)
        error = int(np.abs(old-new).max())
        assert error <= 1
        reload_checks.append(dict(frame=frame, camera=camera, maximum_uint8_difference=error))
    node = trainer.models['SMPLNodes']
    initialization = json.loads((SOURCE / 'initialization.json').read_text())
    model_index = next(r['model_index'] for r in initialization['instances']['SMPLNodes'] if r['author_instance'] == 79)
    regions = training_regions(dataset, node, model_index)
    helper.write_json(OUT / 'training-rois.json', dict(regions=regions, source='Only training geometry; no optimization targets selected using heldout RGB'))
    for model in trainer.models.values():
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    active = node.instances_fv[:,model_index].clone()
    active[dataset.test_timesteps] = False
    assert int(active.sum()) == 90
    for name in PARAMETERS:
        getattr(node,name).requires_grad_(True)
    reference = {name:getattr(node,name).detach().clone() for name in PARAMETERS}
    active_indices = active.nonzero().flatten()
    pair_mask = active[1:] & active[:-1]
    rates = [.005,.001,.002]
    def optimizer_new():
        return torch.optim.Adam([dict(params=[getattr(node,n)],lr=r,name=n) for n,r in zip(PARAMETERS,rates)],eps=1e-15)
    def quaternion_change(name):
        current = F.normalize(getattr(node,name)[active,model_index],dim=-1)
        initial = F.normalize(reference[name][active,model_index],dim=-1)
        return current, initial
    def project():
        with torch.no_grad():
            displacement = node.instances_trans[active,model_index]-reference['instances_trans'][active,model_index]
            multiplier = (.5 / displacement.norm(dim=-1,keepdim=True).clamp_min(1e-9)).clamp(max=1.)
            node.instances_trans[active,model_index] = reference['instances_trans'][active,model_index] + displacement*multiplier
            for name,degrees in [('instances_quats',25),('smpl_qauts',45)]:
                current,initial=quaternion_change(name)
                relative=quaternion_multiply(quaternion_invert(initial),current)
                relative=torch.where(relative[...,:1]<0,-relative,relative)
                vector=quaternion_to_axis_angle(relative)
                scale=(math.radians(degrees)/vector.norm(dim=-1,keepdim=True).clamp_min(1e-9)).clamp(max=1.)
                limited=quaternion_multiply(initial,axis_angle_to_quaternion(vector*scale))
                getattr(node,name)[active,model_index] = limited
    def update(roi,optimizer):
        trainer.set_train()
        optimizer.zero_grad(set_to_none=True)
        info,cam=dataset.full_image_set.get_image(roi['image_id'],1.)
        assert roi['image_id'] in dataset.train_indices and roi['frame'] not in dataset.test_timesteps
        info,cam=helper.to_cuda(info),helper.to_cuda(cam)
        result=trainer(info,cam,novel_view=True)
        x0,y0,x1,y1=roi['box']
        predicted=result['rgb'][y0:y1,x0:x1]
        target=info['pixels'][y0:y1,x0:x1]
        l1=(predicted-target).abs().mean()
        ssim=1-trainer.ssim(target.permute(2,0,1)[None],predicted.permute(2,0,1)[None])
        delta=node.instances_trans[:,model_index]-reference['instances_trans'][:,model_index]
        prior_trans=delta[active].square().mean()
        smooth_trans=(delta[1:]-delta[:-1])[pair_mask].square().mean()
        prior_rot=0.
        smooth_rot=0.
        for name in ('instances_quats','smpl_qauts'):
            q,q0=quaternion_change(name)
            prior_rot=prior_rot+(1-(q*q0).sum(-1).square()).clamp_min(0).mean()
            all_q=F.normalize(getattr(node,name)[:,model_index],dim=-1)
            all_q0=F.normalize(reference[name][:,model_index],dim=-1)
            correction=quaternion_multiply(quaternion_invert(all_q0),all_q)
            dot=(correction[1:]*correction[:-1]).sum(-1)
            smooth_rot=smooth_rot+(1-dot.square())[pair_mask].clamp_min(0).mean()
        losses=dict(rgb=.8*l1,ssim=.2*ssim,translation_prior=.01*prior_trans,rotation_prior=.01*prior_rot,
            translation_smooth=.05*smooth_trans,rotation_smooth=.01*smooth_rot)
        assert all(torch.isfinite(v) for v in losses.values())
        sum(losses.values()).backward()
        gradients={}
        for name in PARAMETERS:
            parameter=getattr(node,name)
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            mask=torch.zeros(parameter.shape[:2],dtype=torch.bool,device=parameter.device)
            mask[active,model_index]=True
            parameter.grad[~mask]=0
            gradients[name]=float(parameter.grad.abs().max())
        optimizer.step()
        project()
        values={key:float(v.detach()) for key,v in losses.items()}
        del result,info,cam,losses,predicted,target
        return values,gradients
    status('discarded_preflight')
    optimizer=optimizer_new()
    probes=[]
    for frame,camera in [(129,4),(131,4),(149,6),(151,6)]:
        roi=next(r for r in regions if (r['frame'],r['camera'])==(frame,camera))
        losses,gradients=update(roi,optimizer)
        assert all(value>0 for value in gradients.values())
        probes.append(dict(frame=frame,camera=camera,losses=losses,gradients=gradients))
    with torch.no_grad():
        for name in PARAMETERS:getattr(node,name).copy_(reference[name])
    optimizer.zero_grad(set_to_none=True)
    equal_tree(saved,trainer.state_dict(only_model=True),'after_discarded_probe_rollback')
    optimizer=optimizer_new()
    helper.write_json(OUT / 'preflight.json',dict(status='finite_target_pose_gradients_and_exact_rollback_passed',
        probes=probes,reload_checks=reload_checks,training_regions=len(regions),model_index=model_index,
        active_training_times=90,checkpoint_unchanged=True))
    gc.collect();torch.cuda.empty_cache()
    status('scoring_training_before_refinement')
    before=score_training(trainer,dataset,regions,'before')
    rng=np.random.default_rng(79)
    order=[]
    log=(OUT/'training.jsonl').open('a',buffering=1)
    status('pose_refinement_running',updates=0,total=UPDATES,training_regions=len(regions))
    for step in range(UPDATES):
        if not order:order=rng.permutation(len(regions)).tolist()
        roi=regions[order.pop()]
        losses,gradients=update(roi,optimizer)
        if step%50==0 or step+1==UPDATES:
            row=dict(updates=step+1,image_id=roi['image_id'],losses=losses,gradients=gradients,
                seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            log.write(json.dumps(row)+'\n');helper.write_json(OUT/'progress.json',row)
            print(json.dumps(row),flush=True)
            torch.cuda.empty_cache()
        if (step+1)%500==0:
            frozen_check(trainer,saved,model_index,active)
            torch.save(dict(updates=step+1,parameters={name:getattr(node,name).detach().cpu() for name in PARAMETERS},
                optimizer=optimizer.state_dict()),OUT/f'pose-state-{step+1:04d}.pt')
    frozen_check(trainer,saved,model_index,active)
    helper.write_json(OUT/'frozen-state-audit.json',dict(status='all_nonpose_tensors_other_people_and_heldout_poses_exactly_unchanged',
        changed_parameters=list(PARAMETERS),person=79,training_time_mask=active.cpu().tolist(),additional_updates=UPDATES))
    status('evaluating_refined_training_and_heldout_views')
    after=score_training(trainer,dataset,regions,'after')
    trainer.completed_updates=30000+UPDATES
    final=helper.evaluate(trainer,dataset,rois,OUT,'final',complete=True)
    final.update(status='pose_only_refinement_and_all105_heldout_renders_finished_independent_audit_pending',
        source_updates=30000,additional_pose_updates=UPDATES,training_before=before['mean_person_psnr_db'],
        training_after=after['mean_person_psnr_db'],seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),quality_accepted=False)
    frozen_check(trainer,saved,model_index,active)
    # Complete local model for reproducible follow-up rendering. Never published.
    torch.save(trainer.state_dict(only_model=True),OUT/'checkpoint-refined.pt')
    helper.write_json(OUT/'completion.json',final)
    status('completed_independent_image_and_visual_review_pending',training_before=final['training_before'],training_after=final['training_after'])


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():status('failed_needs_inspection',error=repr(error))
        traceback.print_exc()
        raise
