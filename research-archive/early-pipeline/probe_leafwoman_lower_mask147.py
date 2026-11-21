"""Bounded lower-body containment control around the original final SMPLify-X stage."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import difflib
import inspect
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

import official_smplifyx_entry123 as entry
from review_full_sequence123 import bound_file, relocate
from run_leafwoman_front24145 import ASSET, CONTACTS, OFFICIAL, OUT, RUNTIME, SMPLX_ROOT, save_npz
from update_leafwoman129_state import atomic, read


OUT = OUT/'lower-mask147'
PARENT = OUT.parent
BASE = PARENT/'shared-cpu'
SOURCE_REVIEW = PARENT/'source-leg-visibility146'
LOWER_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)
UPPER_ANGLES = tuple(i for i in range(21) if i+1 not in LOWER_JOINTS)
LOWER_OBS = (10, 11, 13, 14, 19, 20, 21, 22, 23, 24)
W, H = 2704, 1520
MARGIN = 5
WORKERS = 4
# Source-only interior shoe patches, relative to the existing native review crops.
# They are for verification, and their whole crop band is held out from the mask loss.
SHOE = {1: (116, 226, 144, 245), 2: (93, 239, 111, 255)}


def prepare():
    assert not OUT.exists()
    base_plan = read(BASE/'plan.json')
    base = read(BASE/'result.json')
    assert base['successful'] and base['plan_sha256'] == entry.sha(BASE/'plan.json')
    assert all(entry.sha(path) == digest for path, digest in base_plan['model_sha256'].items())
    assert entry.sha(CONTACTS) == base_plan['contacts_sha256']
    assert [row['frame'] for row in base['frames']] == base_plan['source_frames']
    assert len(base['frames']) == 24 and not set(base_plan['source_frames']) & set(base_plan['retained_holdouts'])
    visibility = read(SOURCE_REVIEW/'visual-adjudication.json')
    assert visibility['reviewed_native_samples'] == 57 and not visibility['original_inputs_or_front_holdout_changed']
    assert visibility['early_shoe_samples_in_front_fit'] == [f'frame_{i:06d}' for i in SHOE]
    original = {row['frame']: row for row in read(CONTACTS)['records']}
    crop_review = read(SOURCE_REVIEW/'feet-crops.json')['origins']
    rows = []
    for row in base['frames']:
        frame = row['frame']
        raw = original[frame]
        assert raw['initial_detector_box'][3] == H and entry.sha(row['output']) == row['sha256']
        for key in ('body', 'mask'):
            bound_file(raw['paths'][key])
        assert entry.sha(relocate(raw['source_mask'])) == raw['source_mask_sha256']
        shoe = None
        fit_limit = H
        if int(frame[-6:]) in SHOE:
            i = int(frame[-6:])
            crop = crop_review[frame]
            x0, y0, x1, y1 = SHOE[i]
            shoe = dict(xyxy=[x0+crop[0], y0+crop[1], x1+crop[0], y1+crop[1]],
                        review_crop=crop, review_crop_sha256=entry.sha(SOURCE_REVIEW/f'foot-region-{i:02d}.png'))
            fit_limit = crop[1]
        rows.append(dict(frame=frame, baseline=row['output'], baseline_sha256=row['sha256'],
                         source=raw['paths']['body'], mask=raw['source_mask'], mask_sha256=raw['source_mask_sha256'],
                         fit_y_max_exclusive=fit_limit, shoe_witness=shoe))
    helper = Path(__file__).with_name('smplify_seeded_refine124.py')
    atomic(OUT/'plan.json', dict(updated=datetime.now(timezone.utc).isoformat(), runner_sha256=entry.sha(__file__),
        seed_adapter_sha256=entry.sha(helper), base_result_sha256=entry.sha(BASE/'result.json'),
        visibility_sha256=entry.sha(SOURCE_REVIEW/'visual-adjudication.json'), rows=rows,
        heldout_frames=base_plan['retained_holdouts'], W=W, H=H, margin_native_px=MARGIN,
        policy='Only project skinning-dominant lower vertices into confidently exterior pixels. No loss for uncovered cloth, withheld image rows, or outside raster.',
        loss='Original final-stage data weight squared times mean original GMoF on one lower-envelope distance. Keep original anatomical priors.',
        comparison='All24 from the same baseline, shared betas and fixed complete K and body placement. Original final-stage continuation with and without mask term.',
        fixed='Root, betas, upper body local angles, face, hands, camera; 4 single-thread CPU shards using existing runtime.',
        inspection='Report all24, common-domain counts, withheld shoe patch distances, source upper and ears. No GS handoff by metric alone.',
        original_solver_files_changed=False, training_started=False, quality_accepted=False))
    print('022_LOWER147_PLAN', len(rows), flush=True)


def read_mask(row):
    path = relocate(row['mask'])
    assert entry.sha(path) == row['mask_sha256']
    with Image.open(path) as im:
        mask = np.asarray(im.convert('L')) >= 128
    assert mask.shape == (H, W) and mask[-1].any()
    return mask


def distance(mask, limit):
    background = ~mask
    background[limit:] = False
    distance_px = cv2.distanceTransform(background.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.maximum(distance_px-MARGIN, 0)


def fit_shard(shard, plan):
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(SMPLX_ROOT))
    import smplx
    import smplify_seeded_refine124 as seeded

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    assert Path(smplx.__file__).is_relative_to(SMPLX_ROOT)
    assert entry.sha(seeded.__file__) == plan['seed_adapter_sha256']
    utils, _, _, _ = entry.load_official(OFFICIAL)
    mapping = utils.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True, use_face_contour=False)
    loader = entry.module(RUNTIME/'src/avatar_core/shuolee_vposer.py', '_lower147_vposer')
    loader.MODEL, loader.CODE = ASSET.parent.parent/'vposer/V02_05', RUNTIME/'third_party/human_body_prior'
    vposer, _, _ = loader.load_vposer('cpu')
    body = smplx.SMPLX(str(ASSET), batch_size=1, num_betas=10, num_expression_coeffs=10, use_pca=True,
        num_pca_comps=12, flat_hand_mean=False, use_face_contour=False, create_body_pose=False,
        create_transl=False, joint_mapper=utils.JointMapper(mapping), dtype=torch.float32).eval().requires_grad_(False)
    selected = torch.where(body.lbs_weights[:, LOWER_JOINTS].sum(dim=1) > .5)[0]
    upper = torch.tensor(UPPER_ANGLES)
    assert 1000 < selected.numel() < 7000

    class LowerDecoder:
        def __init__(self, seed):
            self.anchor = torch.from_numpy(seed['body_pose_axis_angle']).reshape(1,21,3)[:, upper]

        def decode(self, latent):
            pose = vposer.decode(latent)['pose_body'].reshape(1,21,3)
            return {'pose_body': pose.index_copy(1, upper, self.anchor)}

    source, prior_diff = seeded.source(OFFICIAL)
    changed = seeded.once(source, 'if opt_idx < 3:', 'if opt_idx != 4:')
    changed = seeded.once(changed, '                body_params = list(body_model.parameters())\n                body_params.append(camera.translation)',
                          '                body_params = []\n                camera.translation.requires_grad_(False)')
    hook = '                body_optimizer, body_create_graph = optim_factory.create_optimizer(\n'
    changed = seeded.once(changed, hook, '                assert len(final_params) == 1 and final_params[0] is pose_embedding\n'
                          "                kwargs['_stages'].append(opt_idx)\n"+hook)
    source_receipt = prior_diff + '\n' + ''.join(difflib.unified_diff(
        source.splitlines(True), changed.splitlines(True), fromfile='seeded-stage34', tofile='lower-only-final-stage'))
    entries = plan['rows'][shard::WORKERS]
    for i, row in enumerate(entries):
        dest = OUT/row['frame']
        dest.mkdir()
        atomic(OUT/f'shard-{shard}.json', dict(pid=os.getpid(), completed=i, active_frame=row['frame'], total=len(entries)))
        assert entry.sha(row['baseline']) == row['baseline_sha256']
        with np.load(row['baseline'], allow_pickle=False) as data:
            seed = dict(data)
        assert not seed['observed_native_keypoints'][:, LOWER_OBS, 2].any()
        decoder = LowerDecoder(seed)
        latent = torch.from_numpy(seed['pose_embedding']).requires_grad_()
        replay = decoder.decode(latent)['pose_body'].reshape(1,63)
        assert torch.max(torch.abs(replay.detach()-torch.from_numpy(seed['body_pose_axis_angle']))) < 2e-6
        field = torch.from_numpy(distance(read_mask(row), row['fit_y_max_exclusive']))[None,None]

        def config(root):
            config = entry.load_official_config(root)
            assert config['maxiters'] == 30 and config['optim_type'] == 'lbfgsls'
            return dict(config, _saved_seed=seed, _seed_pose_embedding=seed['pose_embedding'],
                        _seed_global_orient=seed['global_orient'], _seed_root_enabled=True, _stages=stages)

        def load(root):
            utils, prior, fitting, optimizer = entry.load_official(root)
            create_loss = fitting.create_loss

            class ContainmentLoss(fitting.SMPLifyLoss):
                def forward(self, output, camera, *args, **kwargs):
                    original = super().forward(output, camera, *args, **kwargs)
                    pixels = camera(output.vertices[:, selected])
                    grid = pixels/pixels.new_tensor([(W-1)/2, (H-1)/2])-1
                    residuals = F.grid_sample(field, grid.unsqueeze(1), align_corners=True).reshape(-1)
                    extra = self.robustifier(residuals).mean()*self.data_weight.square()
                    costs.append(float(extra.detach()))
                    return original+extra

            def factory(loss_type='smplify', **kwargs):
                return ContainmentLoss(**kwargs) if loss_type == 'smplify' else create_loss(loss_type=loss_type, **kwargs)

            fitting.create_loss = factory
            return utils, prior, fitting, optimizer

        # This resumed stage reads only the original raster dimensions from img.
        # The frozen observations and checked native mask carry the image evidence.
        image = np.broadcast_to(np.float32(0), (H,W,3))
        variants = {}
        for variant in ('original', 'containment'):
            stages, costs = [], []
            scope = dict(entry.__dict__, adapted_entry_source=lambda root: (changed, source_receipt),
                         load_official=load if variant == 'containment' else entry.load_official, load_official_config=config)
            exec(compile(inspect.getsource(entry.fit_one), '<original SMPLify-X final stage>', 'exec'), scope)
            arrays, _ = scope['fit_one'](official_root=OFFICIAL, image=image, keypoints=seed['observed_native_keypoints'],
                K_integer_centers=seed['K_integer_centers'], body_model=body, vposer2=decoder,
                result_path=dest/(variant+'.pkl'), shared_betas=seed['betas'], freeze_betas=True,
                device='cpu', diff_path=dest/'upstream-adapter.diff')
            assert stages == [4] and np.isfinite(costs).all() and bool(costs) == (variant == 'containment')
            for key in (*[name for name,_ in body.named_parameters()], 'camera_translation', 'K_integer_centers', 'faces'):
                assert np.array_equal(arrays[key], seed[key]), key
            assert np.array_equal(arrays['body_pose_axis_angle'].reshape(21,3)[list(UPPER_ANGLES)],
                                  seed['body_pose_axis_angle'].reshape(21,3)[list(UPPER_ANGLES)])
            assert (arrays['vertices_camera'][...,2] > .05).all()
            output = dest/(variant+'.npz')
            save_npz(output, **arrays, observed_native_keypoints=seed['observed_native_keypoints'])
            variants[variant] = dict(output=str(output), sha256=entry.sha(output), stages=stages, closure_calls=len(costs),
                first_containment_loss=costs[0] if costs else None, last_closure_containment_loss=costs[-1] if costs else None)
        atomic(dest/'receipt.json', dict(frame=row['frame'], variants=variants, fixed_parameters_equal=True))
    atomic(OUT/f'shard-{shard}.json', dict(pid=None, completed=len(entries), total=len(entries)))


def run_shard(shard, plan):
    with (OUT/f'shard-{shard}.log').open('x', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            fit_shard(shard, plan)


def run():
    plan_path = OUT/'plan.json'
    plan = read(plan_path)
    assert plan['runner_sha256'] == entry.sha(__file__)
    assert plan['base_result_sha256'] == entry.sha(BASE/'result.json')
    assert plan['visibility_sha256'] == entry.sha(SOURCE_REVIEW/'visual-adjudication.json')
    with (OUT/'started').open('x', encoding='utf-8') as lock:
        lock.write(str(os.getpid())+'\n')
    start = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(run_shard, range(WORKERS), [plan]*WORKERS))
        receipts = [read(OUT/row['frame']/'receipt.json') for row in plan['rows']]
        assert all(entry.sha(variant['output']) == variant['sha256'] for row in receipts for variant in row['variants'].values())
        atomic(OUT/'result.json', dict(plan_sha256=entry.sha(plan_path), frames=receipts,
            wall_seconds=time.monotonic()-start, complete=len(receipts)==24, quality_accepted=False))
        print('022_LOWER147_COMPLETE', len(receipts), flush=True)
    except Exception as error:
        atomic(OUT/'failure.json', dict(error=repr(error), complete=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run'))
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)()
