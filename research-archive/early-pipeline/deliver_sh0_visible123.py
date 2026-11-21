"""Make an explicit SH0 derivative of the approved-to-view 120k snapshot.

No optimization: retain every non-SH-rest float bit for bit. Native replay
compares the new file with the original file rendered at active SH degree zero.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np

PREFIX = Path('\\\\?\\C:/') if os.name == 'nt' else Path('/mnt/c')
T = PREFIX/'Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001'
V = T/'viewer'
ROOT = T/'shuolee/current/local-clothed-surface-123/demo-vhap-initialization/body-gaussian-upstream-offset'
OUT = ROOT/'delivery/step-120000/sh0-visible'
SOURCE_ID = 'local-surface-123-demo-body-smplx-offset-s120000'
ID = SOURCE_ID+'-sh0'
SOURCE = V/'models'/SOURCE_ID/'point_cloud.ply'
EXPECTED = 'd151f6d4f47a8ff4fd869ea34d6842f7e9663f421c46397dfba0a75de130a7ad'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def ply(p):
    raw = Path(p).read_bytes()
    match = re.search(br'end_header\r?\n', raw[:65536])
    assert match is not None
    header = raw[:match.end()].decode('ascii')
    assert 'format binary_little_endian 1.0' in header
    assert len(re.findall(r'^element ', header, re.M)) == 1
    count = int(re.search(r'element vertex (\d+)', header)[1])
    props = re.findall(r'^property (\S+) (\S+)', header, re.M)
    assert all(kind == 'float' for kind, _ in props)
    fields = [name for _, name in props]
    assert len(fields) == len(set(fields))
    data = np.frombuffer(raw[match.end():], dtype=np.dtype([(key, '<f4') for key in fields]))
    assert len(data) == count and len(raw) == match.end()+count*4*len(fields)
    return data, fields


def prepare():
    assert sha(SOURCE) == EXPECTED
    assert not (OUT/'plan.json').exists() and not (OUT/'point_cloud.ply').exists()
    OUT.mkdir(exist_ok=True)
    cameras_file = V/'models'/SOURCE_ID/'cameras.json'
    diagnostic_cameras = ROOT/'gaussian-run/step-120000/sh-direction-audit/user-side-view/viewer-diagnostic-cameras.json'
    save(OUT/'plan.json', dict(created=now(), operation='SH3 file to SH0 colour-only derivative',
        source_model=SOURCE_ID, source_sha256=EXPECTED, training_updates=0,
        native_replay_views=2, native_timeout_seconds=120,
        unchanged='All point positions, covariance scale/rotation, opacity, DC colour, normals, point order, reference frame and camera intrinsics/extrinsics',
        native_test='Original PLY at active_sh_degree=0 versus derivative PLY loaded as degree0',
        required_native_max_abs_error=0.0,
        cameras_sha256=sha(cameras_file), diagnostic_cameras_sha256=sha(diagnostic_cameras),
        quality_accepted=False, model_is_not_retrained=True))
    original, fields = ply(SOURCE)
    rest = [x for x in fields if x.startswith('f_rest_')]
    assert len(rest) == 45 and len(original) == 56884
    keep = [x for x in fields if x not in rest]
    expected = ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity',
                'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    assert keep == expected
    result = np.empty(len(original), dtype=[(name, '<f4') for name in keep])
    for name in keep:
        result[name] = original[name]
    header = ('ply\nformat binary_little_endian 1.0\ncomment SH0 derivative; geometry and DC preserved; no retraining\n'
              +f'element vertex {len(result)}\n'+''.join(f'property float {name}\n' for name in keep)+'end_header\n')
    (OUT/'point_cloud.ply').write_bytes(header.encode('ascii')+result.tobytes())
    reloaded, exported_fields = ply(OUT/'point_cloud.ply')
    assert exported_fields == keep
    same = {name: original[name].tobytes() == reloaded[name].tobytes() for name in keep}
    assert all(same.values()) and sha(SOURCE) == EXPECTED
    (OUT/'cameras.json').write_bytes(cameras_file.read_bytes())
    (OUT/'native-replay-cameras.json').write_bytes(diagnostic_cameras.read_bytes())
    context = read(ROOT/'delivery/step-120000/context.json')
    context.update(label='demo全身 · SH0稳定颜色 · 120,000步同几何', model=dict(sha256=sha(OUT/'point_cloud.ply')))
    save(OUT/'context.json', context)
    save(OUT/'field-validation.json', dict(updated=now(), points=len(original), source_sha256=EXPECTED,
        derivative_sha256=sha(OUT/'point_cloud.ply'), original_sh=3, derivative_sh=0,
        removed_fields=rest, fields_bitwise_equal=same, camera_bytes_unchanged=True,
        retrained=False, source_unchanged=True, quality_accepted=False))
    print('SH0_DERIVATIVE_FIELDS_AND_CAMERA_VERIFIED', flush=True)


def native():
    assert os.name != 'nt' and not (OUT/'native-result.json').exists()
    import fcntl
    lock = open('/tmp/shuolee-body-export123.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    sys.path[:0] = ['/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src',
                    '/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core']
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera, image
    from types import SimpleNamespace
    plan = read(OUT/'plan.json')
    check = read(OUT/'field-validation.json')
    assert sha(SOURCE) == EXPECTED and sha(OUT/'point_cloud.ply') == check['derivative_sha256']
    assert sha(OUT/'cameras.json') == plan['cameras_sha256']
    assert sha(OUT/'native-replay-cameras.json') == plan['diagnostic_cameras_sha256']
    save(OUT/'progress.json', dict(updated=now(), state='native_readback_running', pid=os.getpid(), optimization_updates=0))
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    original, derivative = GaussianModel(3), GaussianModel(0)
    original.load_ply(str(SOURCE)); derivative.load_ply(str(OUT/'point_cloud.ply'))
    original.active_sh_degree = derivative.active_sh_degree = 0
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    comparisons = []
    with torch.no_grad():
        for index, c in enumerate(read(OUT/'native-replay-cameras.json')):
            rotation, position = np.asarray(c['rotation']), np.asarray(c['position'])
            E = np.eye(4); E[:3,:3] = rotation.T; E[:3,3] = -rotation.T@position
            row = dict(frame=c['img_name'], wh=[c['width'],c['height']], w2c=E.tolist(),
                       K=[[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1]])
            cam, receipt = camera(row, index)
            a = render(cam,original,pipe,bg)['render']
            b = render(cam,derivative,pipe,bg)['render']
            delta = (a-b).abs()
            assert torch.equal(a,b), (c['img_name'],float(delta.max()))
            image(b).save(OUT/(c['img_name']+'_SH0.png'))
            comparisons.append(dict(frame=c['img_name'], width=c['width'], height=c['height'],
                mean_abs=float(delta.mean()),max_abs=float(delta.max()),bitwise_equal=True,
                synthetic=c.get('synthetic',False),projection=receipt))
    assert len(comparisons)==2 and original.optimizer is None and derivative.optimizer is None
    assert sha(SOURCE) == EXPECTED
    save(OUT/'native-result.json',dict(updated=now(), passed=True, seconds=time.monotonic()-started,
        source_sha256=EXPECTED, derivative_sha256=check['derivative_sha256'],views=comparisons,
        training_updates=0, quality_accepted=False, meaning='File readback parity only; not geometry quality or cross-view ground truth'))
    save(OUT/'progress.json',dict(updated=now(),state='native_readback_completed',pid=None,optimization_updates=0))
    print('SH0_NATIVE_READBACK_BITWISE_EQUAL_BOTH_VIEWS', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','native']); args=parser.parse_args()
    prepare() if args.mode=='prepare' else native()
