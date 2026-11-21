"""Recorded-camera, SHA-bound native Gaussian scene package and headless CLI.

This loader never estimates cameras, fills missing K, rotates a basis, or changes
ray-3D evaluation to the Spark screen-space preview. Metadata cannot choose code
or an import path. The locally installed, reviewed backend must match its record.
"""
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import inspect
import json
import re
import numpy as np

from native_gaussian_ply122 import read_standard_ply
from native_portrait_renderer122 import NativeCameraContract, render_native

SCHEMA = 'shuolee-native-scene-v1'
COORDINATES = {'world_up': '+Y', 'camera': 'OpenCV_w2c', 'pixels': 'pixel_corners',
               'gaussians': 'world', 'quaternion': 'wxyz_local_to_world',
               'SH': 'Graphdeco_real_SH_RGB_channel_major'}
RENDERER = {'with_ut': True, 'with_eval3d': True, 'packed': False, 'eps2d': 0.3,
            'tile_size': 16, 'near_plane': 0.01, 'far_plane': 1e10, 'radius_clip': 0,
            'render_mode': 'RGB', 'rasterize_mode': 'classic', 'background': [0, 0, 0],
            'ut_parameters': {'alpha': 0.1, 'beta': 2.0, 'kappa': 0.0,
                              'in_image_margin_factor': 0.1, 'require_all_sigma_points_valid': True},
            'sh_direction': 'world point minus camera center'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _object(pairs):
    obj = {}
    for key, val in pairs:
        if key in obj:
            raise ValueError('Duplicate JSON key: ' + key)
        obj[key] = val
    return obj


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=_object, parse_constant=reject)


def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(label + ': missing, extra or ambiguous fields')


def positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(label + ': positive integer required')


def array(value, shape, label):
    a = np.asarray(value)
    if a.dtype.kind not in 'ifu' or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(label + ': finite numeric matrix with exact shape required')
    return a.astype(np.float64)


def assert_constant(value, expected, label):
    # Preserve the boolean/numeric distinction in flags, not just Python ==.
    def types_match(a, b):
        if isinstance(b, dict):
            return isinstance(a, dict) and set(a) == set(b) and all(types_match(a[k], b[k]) for k in b)
        if isinstance(b, list):
            return isinstance(a, list) and len(a) == len(b) and all(types_match(x, y) for x, y in zip(a, b))
        if type(b) is bool:
            return type(a) is bool
        return not isinstance(a, bool) if isinstance(b, (int, float)) else type(a) is type(b)
    if value != expected or not types_match(value, expected):
        raise ValueError(label + ': unsupported convention or renderer; no fallback')


def checked_asset(root, descriptor):
    keys(descriptor, ['path', 'sha256'], 'asset')
    name = descriptor['path']
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('Asset path must be a local relative POSIX path')
    rel = PurePosixPath(name)
    if rel.is_absolute() or any(v in ('.', '..', '') for v in name.split('/')):
        raise ValueError('Asset path escapes or ambiguously addresses package')
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Asset must be a contained local file')
    expected = descriptor['sha256']
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected) or sha(path) != expected:
        raise ValueError('Asset SHA256 mismatch: ' + name)
    return path


def validate_camera(c):
    keys(c, ['id', 'image_name', 'width', 'height', 'K', 'w2c'], 'camera')
    if not isinstance(c['id'], str) or not c['id'] or not isinstance(c['image_name'], str) or not c['image_name']:
        raise ValueError('Camera ID and image identity must be nonempty strings')
    positive_int(c['width'], 'width'); positive_int(c['height'], 'height')
    k = array(c['K'], (3, 3), 'K'); v = array(c['w2c'], (4, 4), 'w2c')
    if k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0 or k[1, 0] != 0 or not np.array_equal(k[2], [0, 0, 1]):
        raise ValueError('Unsupported K; full focal lengths and principal point must be explicit')
    rot = v[:3, :3]
    if (not np.allclose(v[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rot @ rot.T, np.eye(3), atol=1e-5, rtol=0)
            or abs(np.linalg.det(rot) - 1) > 1e-5):
        raise ValueError('Expected rigid OpenCV w2c, without scale or reflection')


def runtime_record():
    import torch, gsplat
    from gsplat import rendering
    from gsplat.cuda import _wrapper
    import native_portrait_renderer122 as adapter
    import native_gaussian_ply122 as codec
    return {'name': 'gsplat', 'version': gsplat.__version__, 'torch': torch.__version__,
            'rendering_sha256': sha(rendering.__file__), 'wrapper_sha256': sha(_wrapper.__file__),
            'adapter_sha256': sha(adapter.__file__), 'codec_sha256': sha(codec.__file__)}


def validate_runtime(record):
    import gsplat
    from gsplat.cuda._wrapper import UnscentedTransformParameters, RollingShutterType
    if record != runtime_record() or record['version'] != '1.5.3' or record['torch'] != '2.7.1+cu128':
        raise ValueError('Backend/source/version mismatch; no substitution permitted')
    signature = inspect.signature(gsplat.rasterization)
    expected = {k: RENDERER[k] for k in ['near_plane', 'far_plane', 'radius_clip', 'tile_size', 'rasterize_mode']}
    expected.update(backgrounds=None, rolling_shutter=RollingShutterType.GLOBAL,
                    viewmats_rs=None, covars=None, sparse_grad=False, absgrad=False,
                    segmented=False, distributed=False, channel_chunk=32,
                    tangential_coeffs=None, thin_prism_coeffs=None, ftheta_coeffs=None)
    for name, val in expected.items():
        if signature.parameters[name].default != val:
            raise ValueError('Backend default changed: ' + name)
    assert_constant(asdict(UnscentedTransformParameters()), RENDERER['ut_parameters'], 'UT parameters')


@dataclass
class NativeScene:
    path: Path
    manifest: dict
    cameras: dict
    parameters: dict
    manifest_sha256: str

    def camera(self, camera_id):
        if camera_id not in self.cameras:
            raise ValueError('Recorded camera ID not found: ' + str(camera_id))
        return self.cameras[camera_id]

    def projection(self, name):
        if name not in self.manifest['projections']:
            raise ValueError('Projection not recorded in package: ' + str(name))
        return self.manifest['projections'][name]


def load_scene(path, *, check_runtime=True):
    path = Path(path).resolve(strict=True)
    manifest_sha = sha(path)
    m = read_json(path)
    keys(m, ['schema', 'scope', 'coordinates', 'input_domain', 'model', 'cameras',
             'backend', 'renderer', 'projections'], 'scene')
    if m['schema'] != SCHEMA or m['scope'] not in ('synthetic-control-only', 'person-reconstruction'):
        raise ValueError('Explicit supported schema and scope required')
    assert_constant(m['coordinates'], COORDINATES, 'coordinate system')
    assert_constant(m['renderer'], RENDERER, 'renderer settings')
    keys(m['backend'], ['name', 'version', 'torch', 'rendering_sha256', 'wrapper_sha256', 'adapter_sha256', 'codec_sha256'], 'backend')
    if m['backend']['name'] != 'gsplat' or m['backend']['version'] != '1.5.3' or m['backend']['torch'] != '2.7.1+cu128':
        raise ValueError('Unsupported pinned backend version')
    for key in ['rendering_sha256', 'wrapper_sha256', 'adapter_sha256', 'codec_sha256']:
        if not isinstance(m['backend'][key], str) or not re.fullmatch('[0-9a-f]{64}', m['backend'][key]):
            raise ValueError('Backend source hash required: ' + key)
    if not isinstance(m['projections'], dict) or not m['projections']:
        raise ValueError('Explicit projection record required')
    for name, projection in m['projections'].items():
        keys(projection, ['radial_coefficients'], 'projection')
        NativeCameraContract(name, m['input_domain'], m['coordinates']['pixels']).validate()
        radial = projection['radial_coefficients']
        if name == 'pinhole' and radial is not None:
            raise ValueError('Pinhole/rectified input accepts no distortion')
        if name == 'opencv_fisheye4':
            array(radial, (4,), 'KB4 coefficients')
    if m['scope'] == 'person-reconstruction' and (m['input_domain'] == 'synthetic_known_camera' or len(m['projections']) != 1):
        raise ValueError('A captured person has one explicit input projection, not alternative synthetic projections')
    keys(m['model'], ['asset', 'vertices', 'sh_degree'], 'model')
    positive_int(m['model']['vertices'], 'vertex count')
    if type(m['model']['sh_degree']) is not int or m['model']['sh_degree'] not in range(4):
        raise ValueError('Complete SH degree 0..3 required')
    model_path = checked_asset(path.parent, m['model']['asset'])
    camera_path = checked_asset(path.parent, m['cameras'])
    records = read_json(camera_path)
    if not isinstance(records, list) or not records:
        raise ValueError('Nonempty recorded camera list required')
    cameras = {}
    image_names = set()
    for camera in records:
        validate_camera(camera)
        if camera['id'] in cameras or camera['image_name'] in image_names:
            raise ValueError('Duplicate camera/frame identity')
        cameras[camera['id']] = camera; image_names.add(camera['image_name'])
    parameters = read_standard_ply(model_path)
    if parameters['sh_degree'] != m['model']['sh_degree'] or len(parameters['means']) != m['model']['vertices']:
        raise ValueError('PLY vertex/SH count disagrees with manifest')
    # Detect an asset changing during loading; render uses these loaded arrays.
    checked_asset(path.parent, m['model']['asset']); checked_asset(path.parent, m['cameras'])
    if sha(path) != manifest_sha:
        raise ValueError('Manifest changed during loading')
    if check_runtime:
        validate_runtime(m['backend'])
    return NativeScene(path, m, cameras, parameters, manifest_sha)


def render_recorded(scene, camera_id, projection_name):
    """Render exactly one recorded camera. No auto-framing or camera defaults."""
    camera = scene.camera(camera_id); projection = scene.projection(projection_name)
    validate_runtime(scene.manifest['backend'])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Native CUDA backend unavailable; no preview fallback')
    t = lambda v: torch.as_tensor(v, dtype=torch.float32, device='cuda')
    p = scene.parameters
    with torch.no_grad():
        rgb, alpha, meta = render_native(t(p['means']), t(p['quats']), t(p['scales']),
            t(p['opacities']), t(p['sh']), t([camera['w2c']]), t([camera['K']]),
            camera['width'], camera['height'],
            contract=NativeCameraContract(projection_name, scene.manifest['input_domain'],
                                          scene.manifest['coordinates']['pixels']),
            sh_degree=p['sh_degree'],
            radial_coeffs=t([projection['radial_coefficients']]) if projection['radial_coefficients'] is not None else None)
    torch.cuda.synchronize()
    if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
        raise RuntimeError('Nonfinite native output')
    return rgb[0].cpu().numpy(), alpha[0].cpu().numpy(), {'all_gaussians_visible': bool((meta['radii'] > 0).all()),
        'scene_sha256': scene.manifest_sha256, 'camera_id': camera_id, 'projection': projection_name,
        'native_interactive_UI': False, 'physical_calibration_validated': False}


def save_render(directory, rgb, alpha, receipt):
    from PIL import Image
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / 'native-rgb-alpha.npz', rgb=rgb, alpha=alpha)
    Image.fromarray((np.clip(rgb, 0, 1)*255).round().astype('uint8')).save(directory / 'rgb.png')
    Image.fromarray((np.clip(alpha[..., 0], 0, 1)*255).round().astype('uint8')).save(directory / 'alpha.png')
    receipt = dict(receipt, files={name: sha(directory / name) for name in ['native-rgb-alpha.npz', 'rgb.png', 'alpha.png']})
    (directory / 'render.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--camera', required=True, help='Exact recorded string ID')
    parser.add_argument('--projection', required=True, choices=['pinhole', 'opencv_fisheye4'])
    parser.add_argument('--output', type=Path, required=True, help='New output directory, never overwritten')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scene = load_scene(args.scene)
    rgb, alpha, receipt = render_recorded(scene, args.camera, args.projection)
    print(json.dumps(save_render(args.output, rgb, alpha, receipt), indent=2))


if __name__ == '__main__':
    main()
