"""Upper-opacity control at the exact saved native and close-up cameras."""

import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_unified_cold_views import ASSET, GPU_LEASE, LAB, save
from review_bound_alpha_ab import delivered, pin, read, require, sha
from review_bound_scale_coverage import face_alpha, interior, scale_pixels


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-unified-20261110')
FINAL_SHA = {
    '006': '18333c3592caa278dfdd9cb8c11ee3ec2b561e94688178587824b31208c6faf7',
    '016': 'a1042ccba8a4399a97497bedcd3c049fb15f698e434f746213e263340bebc696',
}


def preflight(subject):
    root = LAB/f'shuolee-smpl-unified-{subject}'
    run = root/'run'
    pin(run/'result.json', FINAL_SHA[subject])
    terminal = read(root/'wrapper-terminal.json')
    require(terminal['result_sha256'] == FINAL_SHA[subject] and terminal['state'] == 'completed',
            'Expected the completed unified terminal.')
    rows = read(run/'data/training-inputs.json')['records']
    selection = read(run/'fit/selection.json')
    sys.path.insert(0, str(PROJECT/'src'))
    saved = delivered(run, ASSET, rows, selection['retained_holdouts'])
    pose = saved.facts['reference_pose_index']
    require(pose == 0 and rows[0]['frame'] == terminal['reference_frame'], 'Reference mapping differs.')
    row = rows[0]
    pin(Path(row['image']), row['source_sha256'])
    pin(Path(row['mask']), row['mask_sha256'])
    comparison_path = root/'view-comparison/result.json'
    comparison = read(comparison_path)
    record = comparison['new']['records'][0]
    close = record['closeup']
    require(comparison['state'] == 'completed'
            and record['frame'] == row['frame'] and record['source_sha256'] == row['source_sha256']
            and close['kind'] == 'fixed_focal_dolly', 'Expected the saved reference dolly camera.')
    near_row = dict(row, w2c=close['camera']['w2c'])
    require(all(row[key] == close['camera'][key] for key in ('K','wh')), 'Dolly changed native intrinsics.')
    near_image = root/'view-comparison'/close['panel']['file']
    pin(near_image, close['panel']['sha256'])
    crop = close['panel']['crop_xyxy']
    observed = saved.fit['observed_native_keypoints'][pose, 67:118]
    joints = saved.fit['joints_camera'][pose, 67:118]
    good = (observed[:, 2] >= .5) & (joints[:, 2] > .2)
    K = np.asarray(row['K'])
    proj = joints[good] @ K.T
    residual = np.linalg.norm(proj[:, :2]/proj[:, 2:]-.5-observed[good, :2], axis=1)
    old_to_new = np.asarray(near_row['w2c']) @ np.linalg.inv(np.asarray(row['w2c']))
    near_joints = joints[good][residual <= 6.] @ old_to_new[:3,:3].T + old_to_new[:3,3]
    require(len(near_joints) >= 8 and (near_joints[:,2] > .2).all(), 'Insufficient reliable face landmarks.')
    near_xy = near_joints @ K.T
    near = interior(near_xy[:,:2]/near_xy[:,2:]-.5, row['wh'])
    native = interior(observed[good,:2], row['wh'])
    require(near is not None and native is not None, 'Invalid face landmark hull.')
    with Image.open(row['mask']) as mask:
        native &= np.asarray(mask.convert('L')) >= 253
    require(near.sum() >= 32 and native.sum() >= 32, 'Empty face interiors.')
    native_crop = [int(v) for v in row['face_box_xyxy']]
    identity = dict(subject=subject, reference=row['frame'], result_sha256=FINAL_SHA[subject],
                    comparison_sha256=sha(comparison_path), source_sha256=row['source_sha256'],
                    mask_sha256=row['mask_sha256'], plan_code_sha256=sha(Path(__file__)),
                    native_K=row['K'], native_w2c=row['w2c'], near_w2c=near_row['w2c'],
                    native_crop=native_crop, near_crop=crop, opacity_floor=.999, scale=1.,
                    roi='Eroded convex hull of the saved inner-face landmarks; dolly uses the native 6 px agreement gate.',
                    measured_dolly_rgb=False, optimizer_updates=0)
    return saved, pose, row, near_row, native, near, native_crop, crop, near_image, identity


def uint8(pixels):
    return np.rint(np.clip(pixels,0,1)*255).astype(np.uint8)


def sheet(path, crop, stock, high):
    x0,y0,x1,y1 = crop
    w,h = x1-x0,y1-y0
    canvas = Image.new('RGB',(3*w,2*(h+21)),'#181818')
    draw = ImageDraw.Draw(canvas)
    for i,(arm,pixels) in enumerate((('Stock',stock),('Opacity floor 0.999',high))):
        for j,(label,pixels_full) in enumerate(zip(('black','white','alpha'),pixels,strict=True)):
            x,y = j*w,i*(h+21)
            tile = uint8(pixels_full[y0:y1,x0:x1])
            canvas.paste(Image.fromarray(tile).convert('RGB'),(x,y+21))
            draw.text((x+4,y+4), f'{arm} / {label}', fill='white')
    canvas.save(path)
    return dict(file=path.name, sha256=sha(path))


def audit(subject, check):
    saved, pose, row, near_row, native, near, native_crop, crop, near_image, plan = preflight(subject)
    if check:
        return dict(plan, gpu_calls=0, native_roi_pixels=int(native.sum()), near_roi_pixels=int(near.sum()))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose
    import torch

    output = LAB/f'shuolee-smpl-unified-{subject}-face-opacity-audit'
    output.mkdir(exist_ok=False)
    save(output/'plan.json',plan)
    rt.configure(ASSET.parents[1])
    from gaussian_renderer import render

    with GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            require(torch.cuda.device_count() == 1, 'Expected one visible GPU.')
            model, parity = _load_model(saved.snapshot/'bound-local.ply',saved.snapshot,saved.fit,
                                        saved.mesh,saved.facts,ASSET)
            model.select_mesh_by_timestep(pose)
            points = model.get_xyz.clone()
            scales = model.get_scaling.clone()
            rotations = model.get_rotation.clone()
            opacity = model._opacity.clone()
            floor = math.log(999.)
            modified = int((opacity < floor).sum())
            pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
            black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
            rows = []
            for name, view, roi, box in (('native',row,native,native_crop),('dolly',near_row,near,crop)):
                cam, receipt = camera(view, 0, pose_index=pose)
                def raster():
                    return scale_pixels(render,render_pose,cam,model,pipe,black,white,pose,1.)
                stock,error = raster()
                if name == 'native':
                    replay = saved.replay['records'][0]
                    path = saved.snapshot/'posed-replay'/replay['image']
                    pin(path,replay['image_sha256'])
                    with Image.open(path) as archived:
                        expected = np.asarray(archived.convert('RGB'))
                    actual = uint8(stock[0])
                else:
                    x0,y0,x1,y1 = box
                    w,h = x1-x0,y1-y0
                    with Image.open(near_image) as archived:
                        expected = np.asarray(archived.crop((0,21,w,h+21)).convert('RGB'))
                    actual = uint8(stock[0][y0:y1,x0:x1])
                diff = int(np.abs(actual.astype(np.int16)-expected).max())
                require(diff <= 1, f'{name} stock camera does not match the saved image.')
                try:
                    model._opacity.clamp_min_(floor)
                    high,high_error = raster()
                finally:
                    model._opacity.copy_(opacity)
                restored,restore_error = raster()
                require(all(np.array_equal(a,b) for a,b in zip(stock,restored,strict=True))
                        and torch.equal(model._opacity,opacity) and torch.equal(model.get_xyz,points)
                        and torch.equal(model.get_scaling,scales) and torch.equal(model.get_rotation,rotations)
                        and model.optimizer is None, 'Control changed the frozen model or restoration differs.')
                stock_alpha, high_alpha = stock[2][roi], high[2][roi]
                holes = stock_alpha < .95
                rows.append(dict(camera=name, camera_receipt=receipt, archive_max_8bit=diff,
                                 composition_errors=[error,high_error,restore_error],
                                 stock=face_alpha(stock[2],roi), opacity_floor=face_alpha(high[2],roi),
                                 below_095_pixels=int(holes.sum()),
                                 recovered_095_pixels=int((holes & (high_alpha >= .95)).sum()),
                                 persistent_below_095_pixels=int((holes & (high_alpha < .95)).sum()),
                                 max_alpha_decrease_in_roi=float((stock_alpha-high_alpha).max()),
                                 sheet=sheet(output/f'{name}.png',box,stock,high)))
    result=dict(plan,state='completed',model_parity=parity,opacity_entries_raised=modified,
                rows=rows,restored_exact=True,quality_accepted=False,
                scope='Opacity-only upper control, no candidate model or change to the fitted surface.')
    save(output/'result.json',result)
    return dict(receipt=str(output/'result.json'), sha256=sha(output/'result.json'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=FINAL_SHA)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    print(json.dumps(audit(args.subject,args.check)))
