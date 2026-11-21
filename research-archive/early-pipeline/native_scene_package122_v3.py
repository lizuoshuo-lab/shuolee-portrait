"""Exact native inference package plus standard PLY preview, full recorded K/E.

The v2 render kernel and pixel convention remain unchanged. V1/v2 packages and
their running UI services retain their original readers. This is not a trainer
checkpoint and contains no optimizer or permission to resume a person run.
"""
from pathlib import Path
import argparse
import re
import numpy as np
from native_scene_package122 import (COORDINATES, NativeScene, read_json, keys,
    checked_asset, validate_camera, assert_constant, array, save_render, sha)
from native_scene_package122_v2 import RENDERER, runtime_record, validate_runtime, write_json
from native_gaussian_ply122 import write_standard_ply, read_standard_ply
from native_portrait_renderer122_v2 import NativeCameraContractV2, render_native, REVISION
import native_linear_archive122 as archive

SCHEMA = 'shuolee-native-scene-v3-exact-arrays'


def io_record():
    return dict(loader=sha(__file__), linear_archive=sha(archive.__file__))


def load_scene(path, *, check_runtime=True):
    path = Path(path).resolve(strict=True)
    initial = sha(path)
    m = read_json(path)
    keys(m, ['schema', 'scope', 'coordinates', 'input_domain', 'model', 'cameras',
             'backend', 'renderer', 'projections', 'io_source_sha256'], 'scene-v3')
    if m['schema'] != SCHEMA or m['scope'] not in ('synthetic-control-only', 'person-reconstruction'):
        raise ValueError('Explicit exact-array schema and scope required')
    assert_constant(m['coordinates'], COORDINATES, 'coordinates')
    assert_constant(m['renderer'], RENDERER, 'renderer')
    assert_constant(m['io_source_sha256'], io_record(), 'native I/O implementation')
    keys(m['backend'], ['name', 'version', 'torch', 'adapter_revision', 'source_sha256'], 'backend')
    for k, v in dict(name='gsplat', version='1.5.3', torch='2.7.1+cu128', adapter_revision=REVISION).items():
        assert_constant(m['backend'][k], v, 'backend ' + k)
    keys(m['backend']['source_sha256'], ['adapter', 'loader', 'wrapper', 'codec', 'validation', 'camera_contract', 'kernel_tree'], 'sources')
    if any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in m['backend']['source_sha256'].values()):
        raise ValueError('Exact source hashes required')
    if not isinstance(m['projections'], dict) or not m['projections']:
        raise ValueError('Explicit recorded projection required')
    for name, pr in m['projections'].items():
        keys(pr, ['radial_coefficients'], 'projection')
        NativeCameraContractV2(name, m['input_domain'], m['coordinates']['pixels']).validate()
        if name == 'pinhole' and pr['radial_coefficients'] is not None:
            raise ValueError('No implicit distortion for processed pinhole inputs')
        if name == 'opencv_fisheye4':
            array(pr['radial_coefficients'], (4,), 'KB4')
    if m['scope'] == 'person-reconstruction' and (m['input_domain'] == 'synthetic_known_camera' or len(m['projections']) != 1):
        raise ValueError('One actual input domain required for person package')
    model = m['model']
    keys(model, ['native_asset', 'preview_asset', 'vertices', 'sh_degree', 'parameter_format', 'preview_equivalence'], 'model')
    if model['parameter_format'] != archive.FORMAT or model['preview_equivalence'] != 'must-be-measured-separately':
        raise ValueError('Native linear arrays and explicitly separate preview required')
    native_path = checked_asset(path.parent, model['native_asset'])
    checked_asset(path.parent, model['preview_asset'])
    camera_path = checked_asset(path.parent, m['cameras'])
    records = read_json(camera_path)
    if not isinstance(records, list) or not records:
        raise ValueError('Nonempty recorded cameras required')
    cameras = {}; names = set()
    for c in records:
        validate_camera(c)
        if c['id'] in cameras or c['image_name'] in names:
            raise ValueError('Duplicate camera identity')
        cameras[c['id']] = c; names.add(c['image_name'])
    p = archive.read(native_path, vertices=model['vertices'], degree=model['sh_degree'])
    p['sh_degree'] = model['sh_degree']
    for asset in (model['native_asset'], model['preview_asset'], m['cameras']):
        checked_asset(path.parent, asset)
    if initial != sha(path):
        raise ValueError('Manifest changed while loading')
    if check_runtime:
        validate_runtime(m['backend'])
    return NativeScene(path, m, cameras, p, initial)


def write_scene(path, parameters, cameras, *, contract, radial=None, scope):
    if not isinstance(contract, NativeCameraContractV2):
        raise ValueError('Explicit v2 native rendering contract required')
    contract.validate()
    if scope not in ('synthetic-control-only', 'person-reconstruction'):
        raise ValueError('Explicit scope required')
    n, degree = archive.validate(parameters)
    for c in cameras:
        validate_camera(c)
    path = Path(path); path.mkdir(exist_ok=False)
    backend = runtime_record(); validate_runtime(backend)
    archive.write(path / 'native-parameters.npz', parameters)
    write_standard_ply(path / 'point_cloud.ply', **parameters)
    write_json(path / 'cameras.json', cameras)
    asset = lambda name: dict(path=name, sha256=sha(path / name))
    m = dict(schema=SCHEMA, scope=scope, coordinates=COORDINATES, input_domain=contract.input_domain,
             model=dict(native_asset=asset('native-parameters.npz'), preview_asset=asset('point_cloud.ply'),
                        vertices=n, sh_degree=degree, parameter_format=archive.FORMAT,
                        preview_equivalence='must-be-measured-separately'),
             cameras=asset('cameras.json'), backend=backend, renderer=RENDERER,
             projections={contract.projection: dict(radial_coefficients=radial)}, io_source_sha256=io_record())
    write_json(path / 'scene.json', m)
    load_scene(path / 'scene.json')
    return path / 'scene.json'


def render_parameters(parameters, camera, *, contract, radial=None):
    import torch
    validate_camera(camera)
    t = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.float32, device='cuda')
    p = parameters
    degree = p.get('sh_degree', int(np.sqrt(p['sh'].shape[1]) - 1))
    with torch.no_grad():
        rgb, alpha, meta = render_native(t(p['means']), t(p['quats']), t(p['scales']), t(p['opacities']),
            t(p['sh']), t([camera['w2c']]), t([camera['K']]), camera['width'], camera['height'],
            contract=contract, sh_degree=degree, radial_coeffs=None if radial is None else t([radial]))
    torch.cuda.synchronize()
    if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
        raise RuntimeError('Nonfinite native response')
    return rgb[0].cpu().numpy(), alpha[0].cpu().numpy(), dict(
        visible_gaussians=int((meta['radii'] > 0).all(-1).sum()), adapter_revision=REVISION)


def render_recorded(scene, camera_id, projection_name):
    if scene.manifest['schema'] != SCHEMA:
        raise ValueError('Explicit exact-array package required')
    validate_runtime(scene.manifest['backend'])
    projection = scene.projection(projection_name)
    rgb, alpha, receipt = render_parameters(scene.parameters, scene.camera(camera_id),
        contract=NativeCameraContractV2(projection_name, scene.manifest['input_domain']),
        radial=projection['radial_coefficients'])
    receipt.update(scene_sha256=scene.manifest_sha256, camera_id=camera_id, projection=projection_name,
                   canonical_parameters='native-parameters.npz', physical_quality_accepted=False)
    return rgb, alpha, receipt


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--scene', required=True); a.add_argument('--camera', required=True)
    a.add_argument('--projection', required=True, choices=['pinhole', 'opencv_fisheye4'])
    a.add_argument('--output', required=True); args = a.parse_args()
    scene = load_scene(args.scene)
    rgb, alpha, receipt = render_recorded(scene, args.camera, args.projection)
    print(save_render(args.output, rgb, alpha, receipt))
