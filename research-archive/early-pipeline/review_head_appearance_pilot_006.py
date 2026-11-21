"""Evaluate the 006 600-step appearance pilot at the frozen native and dolly cameras."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_face_photometric_006 as face
import review_head_appearance_pilot_006_holdouts as pilot_holdouts


pilot = pilot_holdouts.pilot
pin, read, require, sha = face.pin, face.read, face.require, face.sha
BASELINE_REVIEW_SHA = 'a15dfa5d45c12d2150f5f2fda158216b2ce25cec11c51f2db64a6f4a40a652cf'
PILOT_PREREG_SHA = '118d2b55625630d05379d5c079561c1f01159a6b457a80d811163fd0b14f4d7e'
HOLDOUT_VERIFIER_SHA = '258490b23702326f802012728a19b17dab4171901c6f10172bb6a4e73778e1b8'
OUT = pilot.OUT/'face-endpoint'


def preflight():
    pin(Path(pilot_holdouts.__file__), HOLDOUT_VERIFIER_SHA)
    pin(pilot.OUT/'preregistration.json', PILOT_PREREG_SHA)
    receipt, state = pilot_holdouts.preflight()
    if state is not None:
        result = read(pilot.OUT/'result.json')
        require(result['preregistration_sha256'] == PILOT_PREREG_SHA
                and result['acceptance_plan_sha256'] == pilot.ACCEPTANCE_SHA
                and result['locked_metrics_plan_sha256'] == pilot.face_trial.ACCEPTANCE_SHA
                and result['from_scratch'] is False and result['cold_video_clock'] is False,
                'Pilot terminal is outside the registered warm-start policy.')
    spec, rebuilt_rois = face.preflight()
    pin(face.OUT/'result.json', BASELINE_REVIEW_SHA)
    prior = read(face.OUT/'result.json')
    pin(face.OUT/'plan.json', prior['plan_sha256'])
    frozen = read(face.OUT/'plan.json')
    pin(Path(face.__file__), frozen['code_sha256'])
    require(prior['state'] == 'face_endpoint_review_complete' and prior['optimizer_updates'] == prior['refits'] == 0
            and spec['frame'] == pilot.face_trial.REFERENCE and spec['pose_index'] == 0 and spec['scale'] == 1.
            and all(frozen[key] == value for key,value in spec.items())
            and spec['baseline']['terminal_sha256'] == pilot.STOCK_RESULT,
            'Stock endpoint camera, fit or comparison changed.')
    baseline = read(face.OUT/'baseline.json')
    require(baseline == prior['reports']['baseline'], 'Stock baseline values do not match their frozen receipt.')
    pin(face.OUT/'baseline-panels.npz', baseline['panels_sha256'])
    pin(face.OUT/'rois.npz', frozen['roi_archive_sha256'])
    with np.load(face.OUT/'rois.npz', allow_pickle=False) as cache:
        rois = dict(cache)
    require(set(rois) == set(rebuilt_rois) and all(np.array_equal(value, rois[key]) for key,value in rebuilt_rois.items()),
            'Registered native, dolly or whole-eye masks changed.')
    plan = dict(frozen, candidate=None, stock_endpoint_review_sha256=BASELINE_REVIEW_SHA,
                stock_baseline_json_sha256=sha(face.OUT/'baseline.json'), pilot_acceptance_sha256=pilot.ACCEPTANCE_SHA,
                pilot_preregistration_sha256=PILOT_PREREG_SHA, pilot_holdout_verifier_sha256=HOLDOUT_VERIFIER_SHA,
                baseline_render_calls=0, evaluator_sha256=sha(Path(__file__)))
    if state is not None:
        plan['candidate'] = dict(run=str(pilot.OUT), project=str(pilot.PROJECT), steps=pilot.STEPS,
            terminal_sha256=receipt['pilot_result_sha256'], bound_manifest_sha256=receipt['pilot_bound_sha256'],
            changed_strict_head_points=receipt['changed_strict_head_points'], fit_updates=0)
    return plan, rois, baseline, state


def render(plan, rois, baseline, state):
    import torch
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    row = plan['source_row']
    pin(Path(row['image']), row['source_sha256'])
    with Image.open(row['image']) as image:
        source = np.asarray(image.convert('RGB'), dtype=np.float32)/255
    pin(pilot.OUT/'result.json', plan['candidate']['terminal_sha256'])
    reports, tiles = {}, {}
    with face.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            require(torch.cuda.device_count() == 1, 'Use one CUDA GPU.')
            state.rt.configure(state.asset.parents[1])
            from gaussian_renderer import render as raster

            model, manifest = load_bound_snapshot(state.run, state.snapshot, asset_path=state.asset)
            require(manifest['reference_frame_id'] == row['frame'] and manifest['reference_pose_index'] == 0,
                    'Pilot bound reference differs from the registered camera pose.')
            black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            model.select_mesh_by_timestep(0)
            for view, image_row, crop in (('native',row,plan['native_crop']), ('dolly',plan['near_row'],plan['dolly_crop'])):
                cam, camera_receipt = camera(image_row, 0, pose_index=0)
                require(camera_receipt == baseline['views'][view]['camera'], 'Registered native or dolly camera changed.')
                depth = model.get_xyz.new_tensor(image_row['w2c'][2])
                min_depth = float((model.get_xyz @ depth[:3] + depth[3]).min())
                require(min_depth > .2, 'Pilot intersects the stock renderer center cutoff.')
                pixels, composition = face.scale_pixels(raster, render_pose, cam, model, pipe, black, white, 0, 1.)
                info = dict(camera=camera_receipt, alpha=face.face_alpha(pixels[2],rois[view]),
                            composition_error=composition, min_center_depth_m=min_depth, appearance=None)
                if view == 'native':
                    info['appearance'] = face.appearance(pixels[0],source,rois[view])
                    info['eyes'] = {key: face.appearance(pixels[0],source,mask) for key,mask in rois.items() if key.startswith('eye_')}
                    with Image.open(state.snapshot/'reference-render.png') as saved:
                        delta = np.abs(face.original.uint8(pixels[0]).astype(np.int16)-np.asarray(saved.convert('RGB')))
                    require(delta.max() <= 1, 'Replay native RGB differs from the delivered pilot frame.')
                    info['delivered_native_max_uint8_delta'] = int(delta.max())
                x0,y0,x1,y1 = crop
                rgb = [face.original.uint8(layer[y0:y1,x0:x1]) for layer in pixels[:2]]
                alpha = face.original.uint8(pixels[2][y0:y1,x0:x1])
                tiles[view] = np.stack([*rgb,np.repeat(alpha[...,None],3,axis=2)])
                reports[view] = info
            require(model.optimizer is None, 'Evaluation changed optimization state.')
    return reports, tiles, manifest['geometry_parity']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check','render'))
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    plan, rois, baseline, state = preflight()
    if args.action == 'check':
        print(json.dumps(dict(state='pilot_face_inputs_verified', terminal_ready=state is not None,
              pilot=plan['candidate'], roi_pixels={k:int(v.sum()) for k,v in rois.items()}, gates=plan['face_gates'], gpu_calls=0)))
        return
    require(state is not None and not OUT.exists(), 'Wait for the completed pilot and use a fresh face review directory.')
    reports, tiles, parity = render(plan, rois, baseline, state)
    OUT.mkdir(exist_ok=False)
    for name, digest in (('baseline.json',plan['stock_baseline_json_sha256']), ('baseline-panels.npz',baseline['panels_sha256'])):
        pin(face.OUT/name, digest)
        shutil.copyfile(face.OUT/name, OUT/name)
        pin(OUT/name, digest)
    face.save(OUT/'plan.json', plan)
    np.savez_compressed(OUT/'candidate-panels.npz', **tiles)
    face.save(OUT/'candidate.json', dict(views=reports, model_parity=parity, optimizer_updates=0,
        pilot=plan['candidate'], panels_sha256=sha(OUT/'candidate-panels.npz')))
    face.merge(OUT, plan)


if __name__ == '__main__':
    main()
