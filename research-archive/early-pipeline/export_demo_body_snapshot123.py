"""Read-only body snapshot export; reuse the existing rigid PLY/camera contract.

Training sources and optimizer state are never edited. Native export replay can
temporarily suspend the verified local trainer so GPU computation is serialized.
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

sys.path.insert(0, '/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')
from shuolee_portrait import runtime as rt
from shuolee_portrait.export import rotate_ply
from verify_demo_head_export123 import serialized_camera

ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/body-gaussian-upstream-offset')
R = np.diag([1., -1., -1.])


def paths(step):
    return ROOT/'gaussian-run'/f'step-{step:06d}', ROOT/'delivery'/f'step-{step:06d}'


def prepare(step):
    snap, out = paths(step)
    result = rt.read(snap/'result.json')
    parity = result['native_ply_parity']
    assert parity['mean_abs'] < 2e-6 and parity['max_abs'] < .005
    assert result['step'] == step and rt.sha(snap/'world.ply') == result['world_ply_sha256']
    row = rt.read(ROOT/'data/training-inputs.json')['records'][result['reference_slot']]
    assert row['frame'] == result['reference_frame']
    out.mkdir(parents=True, exist_ok=False)
    valid = rotate_ply(snap/'world.ply', out/'point_cloud.ply', R, np.zeros(3))
    E = np.array(row['w2c'], dtype=float)
    E[:3, :3] = E[:3, :3]@R.T
    c2w, K = np.linalg.inv(E), np.array(row['K'])
    cam = dict(id=0, img_name=row['frame'], width=row['wh'][0], height=row['wh'][1],
        rotation=c2w[:3, :3].tolist(), position=c2w[:3, 3].tolist(),
        fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        pixelConvention='pixel_corners', projectionModel='pinhole_export_assumption')
    rt.save(out/'cameras.json', [cam])
    serialized_camera(out, row, R)
    valid.update(step=step, source_sha256=result['world_ply_sha256'], reference_frame=row['frame'],
        coordinate_space='Reference body camera frame rotated to +Y preview; not measured gravity or independent metric calibration',
        active_sh_degree=result['active_sh_degree'], native_source_ply_parity=dict(parity, passed=True),
        native_rotated_render_parity='pending', format_passed=False, quality_accepted=False,
        viewer_ui_review='pending', run=ROOT.name)
    rt.save(out/'validation.json', valid)
    rt.save(out/'context.json', dict(label=f'demo全身 · SMPL-X绑定＋共享vertexoffset · {step:,}步',
        steps=step, referenceFrameId=row['frame'], referenceFrameIsPoseSnapshot=True, initialView=0,
        model=dict(sha256=valid['sha256']), world=valid['coordinate_space'],
        filter=dict(preBlurAmount=.3, blurAmount=0),
        rendererContract=dict(schema='shuolee-renderer-contract-v1', cameraProjection='pinhole',
            gaussianEvaluation='screen-space-2d', previewBackend='spark', scope='preview-only',
            trainingRenderer='GaussianAvatars diff-gaussian-rasterization full-K local SMPL-X adapter',
            nativeImageEquivalence='Native source and rotated export checked separately; Spark preview is not pixel-equivalence proof')))
    print('BODY_EXPORT_PREPARED', step, valid['points'], flush=True)


def process(pid):
    p = Path('/proc')/str(pid)
    cmd = (p/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    state = next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('State:'))
    start = (p/'stat').read_text().split()[21]
    return cmd, state, start


def replay(step, suspended_pid):
    cmd, state, _ = process(suspended_pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state == 'T'
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera, image
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    snap, out = paths(step)
    result, valid = rt.read(snap/'result.json'), rt.read(out/'validation.json')
    assert rt.sha(out/'point_cloud.ply') == valid['sha256']
    assert rt.sha(snap/'world.ply') == valid['source_sha256'] == result['world_ply_sha256']
    row = rt.read(ROOT/'data/training-inputs.json')['records'][result['reference_slot']]
    moved = serialized_camera(out, row, R)
    source, portable = GaussianModel(3), GaussianModel(3)
    source.load_ply(str(snap/'world.ply')); portable.load_ply(str(out/'point_cloud.ply'))
    source.active_sh_degree = portable.active_sh_degree = result['active_sh_degree']
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    with torch.no_grad():
        a, _ = camera(row, 0); b, proj = camera(moved, 0)
        ra = render(a, source, pipe, bg)['render']; rb = render(b, portable, pipe, bg)['render']
        diff = (ra-rb).abs()
        parity = dict(mean_abs=float(diff.mean()), max_abs=float(diff.max()))
        parity['passed'] = parity['mean_abs'] < 2e-6 and parity['max_abs'] < .005
        assert parity['passed'], parity
        image(rb).save(out/'reference-native.png')
        with np.load(snap/'current-meshes.npz', allow_pickle=False) as z:
            slot = result['reference_slot']
            assert str(z['frame_ids'][slot]) == row['frame']
            source_E = np.asarray(row['w2c'])
            vertices_world = (z['vertices_camera'][slot]-source_E[:3, 3])@source_E[:3, :3]
            vertices = vertices_world@R.T
        # Frame the complete body for synthetic orbit review. The recorded
        # reference camera and all exported model coordinates stay unchanged.
        center = (vertices.min(axis=0)+vertices.max(axis=0))*.5
        radius = max(float(np.ptp(vertices, axis=0).max())*1.8, .1)
        direction = -center/np.linalg.norm(center)
        novel = out/'novel'; novel.mkdir(exist_ok=False)
        records = []
        for i, angle in enumerate(np.linspace(0, 2*np.pi, 36, endpoint=False)):
            yaw = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            eye = center+radius*(yaw@direction)
            forward = center-eye; forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0, 1, 0]); right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            E = np.eye(4); E[:3, :3] = np.stack([right, down, forward]); E[:3, 3] = -E[:3, :3]@eye
            r = dict(frame=f'novel_{i:03d}', wh=[384, 512], K=[[450., 0, 192.], [0, 450., 256.], [0, 0, 1]], w2c=E.tolist(), synthetic=True)
            c, _ = camera(r, i)
            image(render(c, portable, pipe, bg)['render']).save(novel/(r['frame']+'.png'))
            records.append(r)
        rt.save(novel/'cameras.json', dict(records=records, ground_truth_available=False, synthetic=True))
        for start in (0, 12, 24):
            page = Image.new('RGB', (1536, 1536), 'black')
            for j in range(12):
                with Image.open(novel/f'novel_{start+j:03d}.png') as im:
                    page.paste(im, ((j%4)*384, (j//4)*512))
            page.save(novel/f'orbit_{start:02d}-{start+11:02d}.png')
        assert process(suspended_pid)[1] == 'T', 'Trainer resumed during export replay'
        valid.update(native_rotated_render_parity=parity, actual_cuda_projection=proj,
            serialized_camera_sha256=rt.sha(out/'cameras.json'), native_replay_uses_serialized_camera=True,
            format_passed=True, native_replay_updated=rt.now(), novel_synthetic_views=36,
            novel_quality_accepted=False, viewer_focus=center.tolist(),
            synthetic_orbit_framing='Body bounding-box center; radius 1.8 times maximum extent; recorded reference camera unchanged')
        rt.save(out/'validation.json', valid)
        print('BODY_EXPORT_NATIVE_REPLAY_PASSED', json.dumps(parity), flush=True)


def serialized_replay(step, pid):
    snap, out = paths(step)
    lock = open('/tmp/shuolee-body-export123.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd, state, identity = process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T', 'Z')
    before = rt.read(ROOT/'progress.json')
    assert before['pid'] == pid and before['state'] == 'training'
    start = time.monotonic()
    rt.save(out/'gpu-handoff.json', dict(started=rt.now(), trainer_pid=pid, trainer_command=cmd,
        trainer_identity=identity, step_before=before['step'], purpose='Read-only native export replay',
        training_sources_changed=False, optimizer_or_parameters_changed=False, state='suspending'))
    try:
        os.kill(pid, signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1] == 'T': break
            time.sleep(.05)
        assert process(pid)[1] == 'T'
        subprocess.run([sys.executable, '-B', __file__, 'replay', '--step', str(step), '--pid', str(pid)], check=True, timeout=90)
    finally:
        if process(pid)[2] == identity:
            os.kill(pid, signal.SIGCONT)
        record = rt.read(out/'gpu-handoff.json')
        record.update(finished=rt.now(), seconds=time.monotonic()-start, state='resumed',
            trainer_after_state=process(pid)[1], verification_passed=rt.read(out/'validation.json').get('format_passed', False),
            timing_note='This maintenance pause is not Gaussian compute time.')
        rt.save(out/'gpu-handoff.json', record)
        print('TRAINER_RESUMED', pid, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'verify', 'replay'])
    p.add_argument('--step', type=int, required=True); p.add_argument('--pid', type=int)
    args = p.parse_args()
    if args.mode == 'prepare': prepare(args.step)
    elif args.mode == 'verify': serialized_replay(args.step, args.pid)
    else: replay(args.step, args.pid)
