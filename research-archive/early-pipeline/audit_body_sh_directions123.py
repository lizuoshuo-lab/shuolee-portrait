"""Read-only SH export and direction-support audit of an immutable body snapshot.

No fitting, coefficient edits, source changes, or quality promotion. Controls are
diagnostic images, not replacement deliverables.
"""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from plyfile import PlyData
from export_demo_body_snapshot123 import ROOT, R, process
from shuolee_portrait import runtime as rt
from utils.sh_utils import eval_sh

VIEWS = [0, 9, 18, 27]


def paths(step):
    snap = ROOT/'gaussian-run'/f'step-{step:06d}'
    delivery = ROOT/'delivery'/f'step-{step:06d}'
    return snap, delivery, snap/'sh-direction-audit'


def plan(step):
    s, d, o = paths(step)
    o.mkdir(exist_ok=False)
    files = [s/'world.ply', s/'bound-local.ply', s/'binding.npz', s/'current-meshes.npz',
             s/'result.json', d/'point_cloud.ply', d/'novel/cameras.json', ROOT/'data/training-inputs.json',
             Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core/gaussian_renderer/__init__.py'),
             Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core/scene/smplx_gaussian_model.py')]
    rt.save(o/'plan.json', dict(created=rt.now(), step=step,
        hypotheses=['SH or coordinate conversion error in portable export',
                    'Novel-view SH extrapolation with camera-relative articulated fitting',
                    'Opacity/geometry holes independently remain'],
        inputs={str(p): rt.sha(p) for p in files}, points='all snapshot Gaussians',
        training_frames=59, orbit_views=36, render_control_views=VIEWS,
        controls=['Full SH3', 'SH0', 'Reference-direction baked RGB', 'White opacity'],
        intervention='Rendered colour only; identical geometry, camera and opacity',
        criteria=dict(sh_colour_export_max_abs=2e-5, reconstructed_xyz_max_abs_m=2e-6),
        cpu_budget_seconds=120, gpu_budget_seconds=90, optimization_steps=0,
        visibility_limit='Training ray set includes invisible points; nearest-angle gap is a lower bound on missing support, not measured supervision.',
        training_changed=False, model_publication=False, quality_accepted=False))
    print('SH_AUDIT_PLAN_REGISTERED', step, flush=True)


def checked(step):
    s, d, o = paths(step); p = rt.read(o/'plan.json')
    for f, h in p['inputs'].items(): assert rt.sha(f) == h, f
    return s, d, o, p


def ply(path):
    v = PlyData.read(path)['vertex'].data
    xyz = np.stack([v[k] for k in ('x', 'y', 'z')], -1).astype(np.float64)
    dc = np.stack([v[f'f_dc_{i}'] for i in range(3)], -1)[..., None]
    rest = np.stack([v[f'f_rest_{i}'] for i in range(45)], -1).reshape(-1, 3, 15)
    return xyz, np.concatenate([dc, rest], -1).astype(np.float64)


def unit(v):
    return v/np.sqrt(np.maximum(np.sum(v*v, axis=-1, keepdims=True), 1e-20))


def stats(v):
    return dict(median=float(np.median(v)), p90=float(np.quantile(v, .9)),
                p99=float(np.quantile(v, .99)), max=float(np.max(v)))


def angle(a, b):
    return np.rad2deg(np.arccos(np.clip(np.sum(a*b, axis=-1), -1, 1)))


def cpu(step):
    start = time.monotonic(); s, d, o, p = checked(step)
    assert not (o/'cpu-result.json').exists()
    source, coeff = ply(s/'world.ply'); portable, pc = ply(d/'point_cloud.ply')
    local, _ = ply(s/'bound-local.ply')
    mesh = np.load(s/'current-meshes.npz', allow_pickle=False)
    binding = np.load(s/'binding.npz', allow_pickle=False)['triangle_ids']
    rows = rt.read(ROOT/'data/training-inputs.json')['records']
    assert len(rows) == p['training_frames'] and list(mesh['frame_ids']) == [r['frame'] for r in rows]
    rays, local_rays = [], []
    xyz_error = None
    for i, (row, v) in enumerate(zip(rows, mesh['vertices_camera'])):
        E = np.array(row['w2c']); v = (v-E[:3, 3])@E[:3, :3]
        tri = v[mesh['faces']]
        e0 = unit(tri[:, 1]-tri[:, 0]); e1 = unit(np.cross(e0, tri[:, 2]-tri[:, 0]))
        e2 = -unit(np.cross(e1, e0)); orient = np.stack([e0, e1, e2], -1)[binding]
        scale = (.5*(np.linalg.norm(tri[:, 1]-tri[:, 0], axis=-1)+
                     abs(np.sum(e2*(tri[:, 2]-tri[:, 0]), axis=-1))))[binding]
        xyz = np.einsum('nij,nj->ni', orient, local)*scale[:, None]+tri.mean(1)[binding]
        if i == 0: xyz_error = float(abs(xyz-source).max())
        center = -E[:3, :3].T@E[:3, 3]
        ray = unit(xyz-center); rays.append(ray)
        local_rays.append(np.einsum('nji,nj->ni', orient, ray))
    assert xyz_error < p['criteria']['reconstructed_xyz_max_abs_m'], xyz_error
    rays = np.asarray(rays); local_rays = np.asarray(local_rays)
    records = []; parity = 0.
    for row in rt.read(d/'novel/cameras.json')['records']:
        E = np.array(row['w2c']); eye = -E[:3, :3].T@E[:3, 3]
        dest_ray = unit(portable-eye); source_ray = unit(source-eye@R)
        colour = eval_sh(3, coeff, source_ray)+.5
        error = float(abs(colour-(eval_sh(3, pc, dest_ray)+.5)).max()); parity = max(parity, error)
        best = np.max(np.einsum('fnc,nc->fn', rays, source_ray), axis=0)
        gap = np.rad2deg(np.arccos(np.clip(best, -1, 1)))
        records.append(dict(frame=row['frame'], export_sh_colour_max_abs=error,
            nearest_training_direction_gap_degrees=stats(gap),
            colour_channels_below_zero_fraction=float(np.mean(colour < 0)),
            colour_channels_above_one_fraction=float(np.mean(colour > 1)),
            sh_colour_minus_reference_abs=stats(abs(colour-(eval_sh(3, coeff, rays[0])+.5)))))
    assert parity < p['criteria']['sh_colour_export_max_abs'], parity
    rt.save(o/'cpu-result.json', dict(updated=rt.now(), step=step, points=len(source), frames=len(rows),
        reconstructed_world_xyz_max_abs_m=xyz_error, export_sh_all36_max_abs=parity,
        export_sh_all36_passed=True,
        all_extrinsics_identity_max_abs=float(max(abs(np.array(r['w2c'])-np.eye(4)).max() for r in rows)),
        world_training_direction_change_from_reference_degrees=stats(angle(rays, rays[0])),
        triangle_local_direction_change_from_reference_degrees=stats(angle(local_rays, local_rays[0])),
        records=records, seconds=time.monotonic()-start,
        limitation=p['visibility_limit'], quality_accepted=False, training_changed=False))
    print('SH_AUDIT_CPU_PASSED', parity, 'seconds', time.monotonic()-start, flush=True)


def replay(step, pid):
    assert process(pid)[1] == 'T'
    s, d, o, p = checked(step)
    assert rt.read(o/'cpu-result.json')['export_sh_all36_passed']
    assert not (o/'native-result.json').exists()
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera, image
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    model = GaussianModel(3); model.load_ply(str(d/'point_cloud.ply')); model.active_sh_degree = 3
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    xyz, coeff = ply(d/'point_cloud.ply')
    # Exported reference camera is origin zero in this snapshot; verify, don't guess.
    portable_camera = rt.read(d/'cameras.json')[0]; eye = np.array(portable_camera['position'])
    baked = torch.as_tensor(np.maximum(eval_sh(3, coeff, unit(xyz-eye))+.5, 0), dtype=torch.float32, device='cuda')
    cameras = rt.read(d/'novel/cameras.json')['records']; records = []
    with torch.no_grad():
        for index in VIEWS:
            row = cameras[index]; c, _ = camera(row, index)
            imgs = []; checks = dict(frame=row['frame'])
            for label, degree, override in [('full-sh3', 3, None), ('dc-sh0', 0, None),
                    ('baked-reference', 3, baked), ('opacity-white', 3, torch.ones_like(baked))]:
                model.active_sh_degree = degree
                raw = render(c, model, pipe, bg, override_color=override)['render']
                im = image(raw); im.save(o/f'{row["frame"]}_{label}.png'); imgs.append(im)
                if label == 'full-sh3':
                    expected = np.array(Image.open(d/'novel'/f'{row["frame"]}.png'))
                    checks['identical_full_sh_png'] = bool(np.array_equal(np.array(im), expected))
                    assert checks['identical_full_sh_png']
                if label == 'opacity-white': checks['opacity_mean'] = float(raw.mean())
            page = Image.new('RGB', (imgs[0].width*4, imgs[0].height), 'black')
            for j, im in enumerate(imgs): page.paste(im, (j*im.width, 0))
            page.save(o/f'{row["frame"]}_sh3_sh0_baked_alpha.png'); records.append(checks)
        assert process(pid)[1] == 'T'
        rt.save(o/'native-result.json', dict(updated=rt.now(), step=step, records=records,
            controls=p['controls'], geometry_camera_opacity_unchanged=True, optimizer_created=False,
            training_changed=False, quality_accepted=False))
    print('SH_AUDIT_NATIVE_COMPLETE', step, flush=True)


def run(step, pid):
    s, d, o, _ = checked(step)
    lock = open('/tmp/shuolee-body-export123.lock', 'a+'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd, state, identity = process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T', 'Z')
    start = time.monotonic()
    rt.save(o/'gpu-handoff.json', dict(started=rt.now(), pid=pid, command=cmd, identity=identity, state='suspending'))
    try:
        os.kill(pid, signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1] == 'T': break
            time.sleep(.05)
        assert process(pid)[1] == 'T'
        subprocess.run([sys.executable, '-B', __file__, 'replay', '--step', str(step), '--pid', str(pid)], check=True, timeout=90)
    finally:
        if process(pid)[2] == identity: os.kill(pid, signal.SIGCONT)
        r = rt.read(o/'gpu-handoff.json'); r.update(finished=rt.now(), seconds=time.monotonic()-start,
            state='resumed', trainer_after_state=process(pid)[1], timing_note='Read-only maintenance pause, not Gaussian compute time')
        rt.save(o/'gpu-handoff.json', r)
        print('TRAINER_RESUMED', pid, flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('mode', choices=['plan', 'cpu', 'run', 'replay'])
    a.add_argument('--step', type=int, required=True); a.add_argument('--pid', type=int)
    args = a.parse_args()
    if args.mode in ('plan', 'cpu'): globals()[args.mode](args.step)
    else: globals()[args.mode](args.step, args.pid)
