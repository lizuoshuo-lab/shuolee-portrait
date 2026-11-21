"""Review the 016 common tail on the pre-registered native and dolly cameras."""

import argparse
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

import audit_unified_face_opacity as stock
import review_face_photometric_006 as face
from review_bound_alpha_ab import pin, read, require, sha


PLAN = face.PLAN.with_name('smpl-head-appearance-transfer-preregistration.json')
PLAN_SHA = 'd8d9ff902f1f0d312233d14076548e48554ed766f2ba29de1d6382ccf4dde55b'
PROJECT = stock.PROJECT.with_name('Shuolee-smpl-head-appearance-20261110')
COMMIT = '762d6d0b2a5db804d8d4de94c53779b2a5eb80b7'
RUN = face.LAB/'shuolee-smpl-head-appearance-016/run'
OUT = RUN.parent/'reference-face'


def inputs():
    pin(PLAN, PLAN_SHA)
    spec = read(PLAN)['subjects']['016']
    for key in ('baseline_terminal', 'fit', 'training_inputs', 'selection', 'prior_reference_face_receipt'):
        pin(Path(spec[key]['path']), spec[key]['sha256'])
    sys.path.insert(0, str(PROJECT/'src'))
    from shuolee_portrait import runtime

    require(Path(runtime.__file__).resolve().is_relative_to(PROJECT/'src'), 'Wrong active renderer checkout.')
    saved, pose, row, near, roi, near_roi, crop, near_crop, _, identity = stock.preflight('016')
    old = read(Path(spec['prior_reference_face_receipt']['path']))
    require(identity['reference'] == spec['reference_frame'] and pose == 0,
            'Unexpected reference pose.')
    with Image.open(row['image']) as image, Image.open(row['mask']) as matte:
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32)/255
        eyes, eye_sha = face.eye_regions(saved, row, rgb, np.asarray(matte.convert('L'))/255)
    require(all(old[key] == value for key, value in identity.items()),
            'Archived opacity plan identity differs.')
    views = dict(native=(row, roi, crop), dolly=(near, near_roi, near_crop))
    return spec, identity, views, rgb, eyes, eye_sha


def render(spec, identity, views, source, eyes, eye_sha):
    # Only the two production source revisions differ. Evaluate both snapshots
    # from the same completed run in the common stock renderer.
    sys.path.insert(0, str(PROJECT/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.head_appearance import snapshot_name
    from shuolee_portrait.training import camera, render_pose

    require(not OUT.exists(), 'Preserve the existing evaluation directory.')
    terminal = read(RUN/'result.json')
    wrapper = read(RUN.parent/'wrapper-terminal.json')
    pin(RUN/'result.json', wrapper['result_sha256'])
    require(wrapper['candidate_commit'] == COMMIT and wrapper['state'] == 'completed', 'Wrong transfer terminal.')
    for path, key in (('fit/fitted.npz', 'fit'), ('data/training-inputs.json', 'training_inputs'),
                      ('fit/selection.json', 'selection')):
        pin(RUN/path, spec[key]['sha256'])
    train = read(RUN/'gaussian-run/result.json')
    names = dict(baseline='step-010000', candidate=snapshot_name(train))
    require(train['head_appearance']['frozen_ply_bytes_verified'] is True and
            names['candidate'] == terminal['final_snapshot'] == 'step-010000-head-000600' and
            terminal['reference_frame_id'] == spec['reference_frame'], 'Expected the intact isolated tail.')
    plan = dict(identity, cross_person_plan_sha256=PLAN_SHA, snapshots=names,
                face_gates=dict(inner_face_dolly_alpha_below_095_absolute_improvement_min=.02,
                                inner_face_native_black_RGB_MAE_absolute_regression_max=.003),
                eye_observation_sha256=eye_sha, source_row=views['native'][0],
                candidate_terminal_sha256=wrapper['result_sha256'], formal_holdouts=[])
    rt.configure(face.ASSET.parents[1])
    from gaussian_renderer import render as raster

    OUT.mkdir()
    face.save(OUT/'plan.json', plan)
    with face.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            require(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
            for arm, name in names.items():
                snapshot = RUN/'gaussian-run'/name
                model, manifest = load_bound_snapshot(RUN, snapshot, asset_path=face.ASSET)
                require(manifest['reference_frame_id'] == spec['reference_frame'] and
                        manifest['reference_pose_index'] == 0 and model.optimizer is None,
                        'Replay pose or state differs.')
                reports, tiles = {}, {}
                for label, (row, roi, crop) in views.items():
                    cam, camera_receipt = camera(row, 0, pose_index=0)
                    model.select_mesh_by_timestep(0)
                    depth = model.get_xyz.new_tensor(row['w2c'][2])
                    require(float((model.get_xyz @ depth[:3] + depth[3]).min()) > .2,
                            'Dolly intersects the renderer center cutoff.')
                    pixels, composition = face.scale_pixels(raster, render_pose, cam, model, pipe, black, white, 0, 1.)
                    reports[label] = dict(camera=camera_receipt, alpha=face.face_alpha(pixels[2], roi),
                                          composition_error=composition)
                    if label == 'native':
                        reports[label].update(appearance=face.appearance(pixels[0], source, roi),
                                              eyes={k: face.appearance(pixels[0], source, v) for k, v in eyes.items()})
                    x0, y0, x1, y1 = crop
                    rgb = [stock.uint8(layer[y0:y1,x0:x1]) for layer in pixels[:2]]
                    alpha = stock.uint8(pixels[2][y0:y1,x0:x1])
                    tiles[label] = np.stack([*rgb, np.repeat(alpha[...,None], 3, axis=2)])
                np.savez_compressed(OUT/f'{arm}-panels.npz', **tiles)
                face.save(OUT/f'{arm}.json', dict(views=reports, snapshot=name, optimizer_updates=0,
                    panels_sha256=sha(OUT/f'{arm}-panels.npz')))
                del model
    face.merge(OUT, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'render'))
    args = parser.parse_args()
    data = inputs()
    if args.action == 'check':
        _, identity, views, _, eyes, _ = data
        print(json.dumps(dict(identity, roi_pixels={k: int(views[k][1].sum()) for k in views},
                              eye_pixels={k: int(v.sum()) for k, v in eyes.items()}, gpu_calls=0)))
    else:
        render(*data)


if __name__ == '__main__':
    main()
