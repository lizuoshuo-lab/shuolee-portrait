"""Full AV2 scene joint human-structure comparison, preserving the 30k baseline.

Same data, known camera/trajectory priors, photometric objectives and iteration
budget. Restores the author's SMPLNodes (including its own regularizers) for
eligible people. This is a representation-family comparison, not one-loss ablation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from omegaconf import OmegaConf

import train_street_deformable as helper
from audit_street_inputs import PoseReader
import io

LAB, REPO, SCENE = helper.LAB, helper.REPO, helper.SCENE
BASE = LAB / 'outputs/dynamic-human-street-smpl-joint-011'
CONTROL = LAB / 'outputs/dynamic-human-street-joint-009/baseline-bounded'
TEMPLATE = REPO / 'smpl_models/SMPL_NEUTRAL.pkl'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def configuration():
    cfg = OmegaConf.load(CONTROL / 'config.yaml')
    original = OmegaConf.to_container(cfg.model)
    smpl = OmegaConf.to_container(OmegaConf.load(REPO / 'configs/omnire.yaml').model.SMPLNodes)
    models = {}
    for name, item in original.items():
        models[name] = item
        if name == 'DeformableNodes':
            models['SMPLNodes'] = smpl
    cfg.model = OmegaConf.create(models)
    cfg.data.pixel_source.load_smpl = True
    assert 'CamPose' not in cfg.model and 'Affine' not in cfg.model
    for name in original:
        assert OmegaConf.to_container(cfg.model[name]) == original[name]
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--run-label', default='articulated')
    args = parser.parse_args()
    assert args.run_label.replace('-', '').isalnum()
    out = BASE / ('preflight' if args.preflight else args.run_label)
    assert not (out / 'protocol.json').exists(), 'Preserve previous attempts; use a new explicit version'
    if not args.preflight:
        probe = json.loads((BASE / 'preflight/completion.json').read_text())
        assert probe['status'] == 'full_scene_SMPL_joint_gradients_and_heldout_render_passed'
    conversion = json.loads((LAB / 'outputs/dynamic-human-smpl-010/template-conversion.json').read_text())
    assert sha(TEMPLATE) == conversion['normalized_sha256']
    node_check = json.loads((LAB / 'outputs/dynamic-human-smpl-010/smpl-node-smoke.json').read_text())
    assert node_check['status'].endswith('_passed')
    data_audit = json.loads((LAB / 'outputs/dynamic-human-street-005/training-only-initialization-audit.json').read_text())
    assert data_audit['adapter_sha256'] == sha(Path(__file__).with_name('street_training_only_adapter.py'))
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True,garbage_collection_threshold:0.8'
    out.mkdir(parents=True, exist_ok=True)
    cfg = configuration()
    OmegaConf.save(cfg, out / 'config.yaml')
    sources = {}
    for name in ('train_street_smpl.py', 'train_street_deformable.py', 'street_capture_time_adapter.py',
                 'street_training_only_adapter.py', 'audit_street_inputs.py'):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, out / name)
        sources[name] = sha(source)
    helper.write_json(out / 'protocol.json', dict(status='sealed_before_model_updates',
        preflight=args.preflight, updates=30000, seed=73, frames=[0, 155], cameras=list(range(7)),
        training_images=987, heldout_images=105, heldout_times=list(range(10, 156, 10)),
        configuration_sha256=sha(out / 'config.yaml'), scripts=sources, template_sha256=sha(TEMPLATE),
        control_configuration_sha256=sha(CONTROL / 'config.yaml'),
        initialization_audit_sha256=sha(LAB / 'outputs/dynamic-human-street-005/training-only-initialization-audit.json'),
        changed='Restore author SMPLNodes and load_smpl; eligible humans use articulated surface Gaussians. Retain DeformableNodes on annotated frames with missing body poses, with disjoint per-frame visibility, instead of dropping an entire track.',
        preserved='Same data split, 1/3 source resolution, camera capture poses, fixed camera/exposure, original RGB/SSIM/depth/sky losses, background/rigid/deformable configurations and 30000-update budget.',
        known_priors='Author estimated SMPL poses and shapes are available over the sequence, including heldout times; no claim that all upstream pose estimation excluded heldout RGB. Current RGB loss and LiDAR/color initialization use training times only.',
        allocator=dict(limit_bytes=12_000_000_000, expandable_segments=True, empty_cache_every=50),
        scope='Full156-frame7-camera scene, not cropped-person-only training.',
        evaluation='Original435 predeclared person boxes and all105 heldout images; actual full-frame and native-region PNGs.',
        limitations=['SMPL family changes geometry, topology and author regularizers, with explicit missing-pose fallback; not a single-factor or capacity-matched ablation.',
            'Initial mesh alignment visibly imperfect; training must demonstrate actual correction and clothing detail.',
            'Known calibrated cameras, boxes, trajectories and author pose priors; no motion forecasting.',
            'SMPL/body licenses retained locally; no model files published.']))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(12_000_000_000 / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(73)
    torch.cuda.manual_seed_all(73)
    np.random.seed(73)
    random.seed(73)
    helper.install_capture(helper.ArgoVerseCameraData)
    helper.install_training_only(helper.DrivingDataset)
    # Preserve the author loading/conversion code but decode only the audited pose input.
    import joblib
    original_load = joblib.load
    pose_path = SCENE / 'humanpose/smpl.pkl'
    source_poses = PoseReader(io.BytesIO(pose_path.read_bytes())).load()
    def load_pose(path, *a, **kw):
        assert Path(path).resolve() == pose_path.resolve(), path
        return source_poses
    joblib.load = load_pose
    started = time.perf_counter()
    try:
        dataset = helper.DrivingDataset(cfg.data)
    finally:
        joblib.load = original_load
    assert len(dataset.train_image_set) == 987 and len(dataset.test_image_set) == 105
    rois = helper.person_rois(dataset)
    assert rois == json.loads((CONTROL / 'person-rois.json').read_text())['regions']
    helper.write_json(out / 'person-rois.json', dict(regions=rois))
    trainer = helper.MultiTrainer(**cfg.trainer, num_timesteps=dataset.num_img_timesteps,
        model_config=OmegaConf.create(OmegaConf.to_container(cfg.model)), num_train_images=987,
        num_full_images=1092, test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.aabb.reshape(2, 3), device=torch.device('cuda'))
    mapping, smpl_seed_cache, coverage = {}, {}, []
    original_objects, original_smpl = dataset.get_init_objects, dataset.get_init_smpl_objects
    def smpl_seeds():
        if 'value' not in smpl_seed_cache:
            smpl_seed_cache['value'] = original_smpl(**cfg.model.SMPLNodes.init)
        return smpl_seed_cache['value']
    def record_objects(*a, **kw):
        kind = kw.get('cur_node_type', a[0] if a else None)
        if kind == 'DeformableNodes':
            kw['exclude_smpl'] = False
        result = original_objects(*a, **kw)
        if kind == 'DeformableNodes':
            articulated = smpl_seeds()
            for identity in list(result):
                if identity not in articulated:
                    continue
                annotated = result[identity]['frame_info'].clone()
                body = articulated[identity]['frame_info']
                assert not (body & ~annotated).any()
                fallback = annotated & ~body
                assert not (fallback & body).any()
                assert torch.equal(fallback | body, annotated)
                coverage.append(dict(author_instance=int(dataset.pixel_source.instances_true_id[identity]),
                    annotation_frames=int(annotated.sum()), articulated_frames=int(body.sum()),
                    fallback_frames=int(fallback.sum()), overlap_frames=0,
                    missing_pose_timesteps=torch.where(fallback)[0].tolist()))
                if fallback.any():
                    result[identity]['frame_info'] = fallback
                else:
                    del result[identity]
        mapping[kind] = [dict(model_index=i, dataset_index=int(k), author_instance=int(dataset.pixel_source.instances_true_id[k]),
            points=int(v['num_pts']), active_frames=int(v['frame_info'].sum())) for i, (k, v) in enumerate(result.items())]
        return result
    def record_smpl(*a, **kw):
        result = smpl_seeds()
        mapping['SMPLNodes'] = [dict(model_index=i, dataset_index=int(k), author_instance=int(dataset.pixel_source.instances_true_id[k]),
            valid_frames=int(v['frame_info'].sum()), points=6890) for i, (k, v) in enumerate(result.items())]
        return result
    dataset.get_init_objects, dataset.get_init_smpl_objects = record_objects, record_smpl
    trainer.init_gaussians_from_dataset(dataset)
    trainer.initialize_optimizer()
    assert any(row['author_instance'] == 79 for row in mapping['SMPLNodes'])
    main_coverage = next(row for row in coverage if row['author_instance'] == 79)
    assert main_coverage['annotation_frames'] == 139 and main_coverage['articulated_frames'] == 100
    assert main_coverage['fallback_frames'] == 39
    assert any(row['author_instance'] == 79 for row in mapping['DeformableNodes'])
    smpl_node = trainer.models['SMPLNodes']
    helper.write_json(out / 'initialization.json', dict(gaussians=trainer.get_gaussian_count(), instances=mapping,
        initialization_seconds=time.perf_counter()-started, lidar_points=dataset.lidar_source.num_points,
        allocated_bytes=torch.cuda.memory_allocated(), voxel_deformer=smpl_node.use_voxel_deformer,
        roi_count=len(rois), main_person_roi_count=sum(r['instance'] == 79 for r in rois),
        complementary_frame_coverage=coverage,
        coverage_boundary='Fallback and body masks are disjoint and union to annotations for listed seeded tracks. This establishes representation coverage, not continuity or visibility accuracy.'))
    steps = [0, 1001, 3001, 10001] if args.preflight else range(30000)
    # Fixed training views in which the target body is present; no heldout RGB probe.
    probe_images = [129*7+4, 131*7+4, 145*7+6, 151*7+6]
    log = (out / 'training.jsonl').open('a', buffering=1)
    for updates, step in enumerate(steps, 1):
        trainer.set_train()
        trainer.preprocess_per_train_step(step)
        trainer.optimizer_zero_grad()
        if args.preflight:
            info, cam = dataset.full_image_set.get_image(probe_images[updates-1], trainer._get_downscale_factor())
        else:
            info, cam = dataset.train_image_set.next(trainer._get_downscale_factor())
        image_id = int(info['img_idx'].flatten()[0])
        assert image_id in dataset.train_indices and image_id not in dataset.test_indices
        info, cam = helper.to_cuda(info), helper.to_cuda(cam)
        outputs = trainer(info, cam)
        trainer.update_visibility_filter()
        losses = trainer.compute_losses(outputs, info, cam)
        assert all(bool(torch.isfinite(value).all()) for value in losses.values())
        trainer.backward(losses)
        trainer.completed_updates = updates
        if not args.preflight:
            trainer.postprocess_per_train_step(step)
        if args.preflight or step % 50 == 0 or step == 29999:
            gradients = {}
            for name in ('instances_trans', 'instances_quats', 'smpl_qauts', '_means', '_features_dc'):
                gradient = getattr(smpl_node, name).grad
                gradients[name] = None if gradient is None else float(gradient.abs().max())
                if gradient is not None:
                    assert torch.isfinite(gradient).all(), name
            if args.preflight:
                assert gradients['instances_trans'] > 0 and gradients['smpl_qauts'] > 0, gradients
            row = dict(step=step, updates=updates, image_id=image_id, seconds=time.perf_counter()-started,
                losses={k: float(v.detach()) for k,v in losses.items()}, smpl_gradients=gradients,
                gaussians=trainer.get_gaussian_count(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            log.write(json.dumps(row, allow_nan=False)+'\n')
            helper.write_json(out / 'progress.json', row)
            print(json.dumps(row), flush=True)
        del outputs, losses, info, cam
        if step % 50 == 0 or args.preflight:
            torch.cuda.empty_cache()
        if not args.preflight and updates in (500, 3000, 10000, 15000, 20000, 25000, 30000):
            helper.checkpoint(trainer, out, updates)
            if updates in (500, 3000, 10000):
                helper.evaluate(trainer, dataset, rois, out, f'updates-{updates:05d}')
    if args.preflight:
        helper.evaluate(trainer, dataset, rois, out, 'discarded-smoke')
        result = dict(status='full_scene_SMPL_joint_gradients_and_heldout_render_passed',
            discarded_steps=list(steps), discarded_updates=4, quality_claim=False)
    else:
        result = helper.evaluate(trainer, dataset, rois, out, 'final', complete=True)
        result['status'] = '30000_updates_and_all105_heldout_images_rendered_pending_independent_visual_audit'
    result.update(seconds=time.perf_counter()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    helper.write_json(out / 'completion.json', result)
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
