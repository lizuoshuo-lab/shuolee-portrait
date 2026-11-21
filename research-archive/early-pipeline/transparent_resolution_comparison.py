"""Controlled fresh MSV2 runs at native resolution and Lanczos 2x."""
from pathlib import Path
import argparse
import ast
import copy
import importlib.util
import json
import shutil
import sys

import numpy as np
from PIL import Image

BASE = Path('/opt/gs-slam-lab/msv2-demo-transparent-full-20261104')
OUT = Path('/mnt/d/00/记录/2026-11-04_MSV2_原始_2倍_4倍对比')
RUNS = {1: Path('/opt/gs-slam-lab/msv2-demo-transparent-native-20261104'),
        2: Path('/opt/gs-slam-lab/msv2-demo-transparent-interpolate2x-20261104')}
LABELS = {1: 'original', 2: 'interpolation2x'}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create(scale):
    ex = load('selected_transparent', BASE / 'experiment.py')
    assert ex.read(BASE / 'execution.json')['state'] == 'training_completed'
    run = RUNS[scale]
    assert not run.exists(), 'Fresh runs only; never overwrite an experiment'
    native = ex.read(BASE / 'native-inputs.json')['records']
    selected = ex.read(BASE / 'training-inputs.json')['records']
    assert len(native) == len(selected) == 59
    run.mkdir()
    record = OUT / LABELS[scale]
    shutil.copytree(BASE / 'source', run / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    unchanged = {}
    for name in ['config.json', 'initial-points.npz', 'native-inputs.json',
                 'source-manifest.json', 'transparent_hooks.py', 'frozen-baseline-diagnostic.json']:
        shutil.copy2(BASE / name, run / name)
        unchanged[name] = ex.sha(run / name)
        assert unchanged[name] == ex.sha(BASE / name)
    code = (BASE / 'experiment.py').read_text(encoding='utf-8')
    code = code.replace(str(BASE), str(run)).replace(str(ex.RECORD), str(record))
    code = code.replace('SwinIR 4x foreground RGB plus alpha contract',
                        f'{scale}x foreground RGB plus alpha contract')
    ast.parse(code)
    (run / 'experiment.py').write_text(code, encoding='utf-8')
    records = []
    checks = []
    for n, four in zip(native, selected):
        assert n['frame'] == four['frame'] and n['w2c'] == four['w2c']
        assert ex.sha(n['image']) == n['source_sha256']
        assert ex.sha(n['mask']) == n['mask_sha256']
        base_k = np.array(n['K'])
        four_k = base_k.copy(); four_k[:2] *= 4
        assert np.array_equal(four_k, np.array(four['K']))
        row = copy.deepcopy(n)
        if scale == 2:
            image = run / 'inputs/images' / n['name']
            mask = run / 'inputs/masks' / n['name']
            image.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            wh = [v * scale for v in n['wh']]
            with Image.open(n['image']) as im:
                assert list(im.size) == n['wh']
                im.convert('RGB').resize(wh, Image.Resampling.LANCZOS).save(image)
            with Image.open(n['mask']) as im:
                assert list(im.size) == n['wh']
                im.convert('L').resize(wh, Image.Resampling.BILINEAR).save(mask)
            k = base_k.copy(); k[:2] *= scale
            ki = k.copy(); ki[0, 2] -= .5; ki[1, 2] -= .5
            row.update(image=str(image), mask=str(mask), wh=wh, K=k.tolist(),
                       K_integer_centers=ki.tolist(), source_sha256=ex.sha(image),
                       mask_sha256=ex.sha(mask), native_image=n['image'], native_mask=n['mask'])
            for field in ['face_box_xyxy', 'initial_box_xyxy']:
                if field in row:
                    row[field] = (np.array(n[field]) * scale).tolist()
            row['other_person_boxes_xyxy'] = (np.array(n['other_person_boxes_xyxy']) * scale).tolist()
        row['input_method'] = 'original' if scale == 1 else 'Lanczos interpolation 2x'
        row['input_linear_scale'] = scale
        assert row['w2c'] == n['w2c']
        k = base_k.copy(); k[:2] *= scale
        assert np.array_equal(k, row['K'])
        ki = k.copy(); ki[0, 2] -= .5; ki[1, 2] -= .5
        assert np.array_equal(ki, row['K_integer_centers'])
        for field in ['image', 'mask']:
            with Image.open(row[field]) as im:
                assert list(im.size) == row['wh']
        checks.append(dict(frame=n['frame'], wh=row['wh'], image_sha256=row['source_sha256'],
            mask_sha256=row['mask_sha256'], native_image_sha256=n['source_sha256'],
            native_mask_sha256=n['mask_sha256'], extrinsics_unchanged=True,
            K_corner_scaling_verified=True, K_integer_centers_verified=True))
        records.append(row)
    ex.save(run / 'training-inputs.json', dict(records=records))
    ex.save(run / 'input-contract.json', dict(passed=True, frames=59, linear_scale=scale,
        original_sources_changed=False, RGB_resampling='none' if scale == 1 else 'Pillow Lanczos',
        mask_resampling='none' if scale == 1 else 'Pillow bilinear', checks=checks))
    plan = ex.read(BASE / 'plan.json')
    for key in ['swinir', 'diagnosis', 'failed_previous_gate', 'prior_trial', 'prior_trial_accepted',
                'protocol_correction', 'selected_configuration', 'pretraining_evidence']:
        plan.pop(key, None)
    plan.update(created=ex.now(), method=f'Transparent MSV2 {LABELS[scale]} fresh 18000-step run',
        human_direction='发给小润一份，然后再训练一版原始和一版插值两倍，msv2这三个原始，2，4做对比',
        inputs=f'59 original crops; RGB linear scale {scale}; masks at matching resolution',
        training_wh=records[0]['wh'], pixel_count_scale=scale**2, learned_superresolution=False,
        camera_policy=f'Original background w2c unchanged; pixel-corner K first two rows times {scale}; integer principal point is scaled corner minus 0.5',
        comparator=str(BASE), single_main_variable='Input sampling: native vs Lanczos 2x vs previously completed SwinIR 4x',
        hypothesis='Determine whether input enlargement improves the reconstructed source views and novel-view appearance with the selected transparent training contract fixed.',
        comparability='Same 59 observations, camera extrinsics, untrained initial point set, seed 6060, source code, optimizer and 18000-step schedule. Input resolution and interpolation/SR method differ.',
        fixed=['59 source frames', 'camera extrinsics', 'untrained 200000-point initialization',
               'seed 6060', 'foreground L1/SSIM plus balanced alpha BCE',
               'foreground-only depth sampling', '18000 labels / 17999 optimizer updates'],
        changes=[f'Input images and masks at linear scale {scale}; corresponding K scaling'],
        adapter_changes=['RUN and output paths', 'training input manifest', 'same full-resolution processing without half-size warmup'],
        inputs_reused=scale == 1, learned_gaussians_reused=False, old_trained_gaussians_loaded=False,
        optimizer_reinitialized=True, pretraining_evidence=str(run / 'rgba-probe.json'),
        known_baseline_limitations='Held object and original masks retained. Native training-view metrics are not held-out validation. SwinIR4x changes both resolution and predicted texture; cannot isolate resolution alone.',
        final_gate='All59 same native-resolution source cameras and12 same orbit cameras, gray/white/checker backgrounds, head/body/hands/feet inspected. No automatic quality promotion.')
    ex.save(run / 'plan.json', plan)
    ex.save(run / 'configuration-parity.json', dict(passed=True, baseline=str(BASE),
        unchanged_files=unchanged, trained_state_copied=False, fresh_optimizer=True,
        only_changes=['input method/resolution and matching K', 'paths and method description']))
    ex.save(run / 'adapter-manifest.json', {name: ex.sha(run / name)
        for name in ['experiment.py', 'transparent_hooks.py', 'source/msv2/train.py']})
    OUT.mkdir(exist_ok=True, parents=True)
    ex.save(OUT / 'comparison-plan.json', dict(created=ex.now(), methods=['original', 'Lanczos2x', 'SwinIR4x'],
        runs={str(k): str(v) for k, v in RUNS.items()} | {'4': str(BASE)},
        evaluation_reference='All59 original 476x960 images, masks and camera matrices',
        novel_views='Same12 existing orbit cameras; no ground truth',
        requested=True, from_scratch=True, unchanged_loss_and_schedule=True,
        four_x_is_learned_SR_and_two_x_is_interpolation=True))
    print('INPUT_CONTRACT_PASSED', scale, len(records), records[0]['wh'], flush=True)


def probe(scale):
    helper = load('transparent_probe', Path(__file__).with_name('transparent_msv2_demo.py'))
    helper.RUN = RUNS[scale]
    helper.probe()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['create', 'probe'])
    parser.add_argument('scale', type=int, choices=[1, 2])
    args = parser.parse_args()
    globals()[args.action](args.scale)
