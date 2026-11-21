"""Portable SH0 replay of fixed SMPL-X rows and free canonical clothing rows."""

from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from plyfile import PlyData
import torch

from . import runtime as rt


SCHEMA = 'shuolee-drivable-clothing/1'
GARMENT_SCHEMA = 'shuolee-drivable-clothing/2'
ANCHORED_GARMENT_SCHEMA = 'shuolee-drivable-clothing/3'
MANIFEST = 'drive.json'
FIXED_PLY = 'fixed-bound.ply'
CLOTHING_PLY = 'clothing-canonical.ply'
MOTION = 'motion.npz'
GARMENT_GRAPH = 'garment-graph.npz'
GARMENT_ANCHORS = 'garment-anchors.npz'
REPLAY_GUIDE = 'README.md'
FILES = (FIXED_PLY, CLOTHING_PLY, MOTION, REPLAY_GUIDE)
PARAMETERS = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')
PLY_FIELDS = ('x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
              'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3')
MOTION_FIELDS = {'frame_ids', 'reference_index', 'vertices_world', 'faces', 'camera_K', 'camera_w2c',
                 'camera_wh', 'clothing_support_face_ids', 'clothing_stable_ids', 'next_clothing_stable_id'}
GRAPH_FIELDS = {'node_rest_xyz', 'node_centers', 'node_owner_faces', 'node_normal_neighbors',
                'graph_edges', 'row_node_ids', 'row_node_weights', 'face_groups'}

REPLAY_INSTRUCTIONS = '''# 回放可驱动人像

使用对应版本的 Shuolee SMPL 源码和 Linux NVIDIA CUDA 环境。Python 和原生高斯渲染器的安装见工程 README。
整份复制 `drive` 文件夹并保留文件名。它包含逐帧网格、原相机和两层高斯，加载时会核对 `drive.json` 中的 SHA-256。

进入工程目录并启用环境，将 `/path/to/drive` 换成解压后的目录。帧名从 `drive.json` 的 `frame_ids` 复制。

```bash
PYTHONPATH="$PWD/src" python - /path/to/drive frame_000027 pose.png <<'PY'
import sys
from types import SimpleNamespace

from PIL import Image
import torch
from shuolee_portrait.drivable_clothing_bundle import load_drive_bundle
from shuolee_portrait.training import camera
from gaussian_renderer import render

drive, frame, destination = sys.argv[1:]
scene, info = load_drive_bundle(drive)
slot = info['frame_ids'].index(frame)
scene.select_mesh_by_timestep(slot)
view, _ = camera(info['cameras'][slot], slot, pose_index=slot)
pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
with torch.no_grad():
    background = torch.zeros(3, device='cuda')
    rgb = render(view, scene, pipe, background)['render']
    pixels = (rgb.clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()
    Image.fromarray(pixels).save(destination)
PY
```

生成的 `pose.png` 使用这一帧的动作和拟合相机。普通静态查看器使用另行导出的标准 PLY 和相机包。
'''


def _need(condition, message):
    if not condition:
        raise ValueError(message)


class CombinedClothingScene:
    """Present both live point sets to the existing renderer in one depth sort."""

    active_sh_degree = max_sh_degree = 0

    def __init__(self, fixed, cloth):
        _need(fixed.max_sh_degree == fixed.active_sh_degree == 0 and
              cloth.max_sh_degree == cloth.active_sh_degree == 0,
              'The combined scene requires two SH0 point sets.')
        self.fixed, self.cloth = fixed, cloth

    def select_mesh_by_timestep(self, timestep):
        _need(isinstance(timestep, Integral) and not isinstance(timestep, bool) and
              self.fixed.num_timesteps == self.cloth.num_timesteps and
              0 <= timestep < self.fixed.num_timesteps, 'Unknown shared pose index.')
        self.fixed.select_mesh_by_timestep(int(timestep))
        self.cloth.select_mesh_by_timestep(int(timestep))

    def _combined(self, attribute):
        return torch.cat((getattr(self.fixed, attribute).detach(), getattr(self.cloth, attribute)), dim=0)

    @property
    def get_xyz(self):
        return self._combined('get_xyz')

    @property
    def get_scaling(self):
        return self._combined('get_scaling')

    @property
    def get_rotation(self):
        return self._combined('get_rotation')

    @property
    def get_features(self):
        return self._combined('get_features')

    @property
    def get_opacity(self):
        return self._combined('get_opacity')

    def get_covariance(self, scaling_modifier=1):
        if hasattr(self.cloth, 'graph_arrays'):
            return torch.cat((self.fixed.get_covariance(scaling_modifier).detach(),
                              self.cloth.get_covariance(scaling_modifier)), dim=0)
        return self.fixed.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _indices(value, name):
    array = _numpy(value)
    _need(array.dtype.kind in 'iu' and np.all(array >= 0), f'Invalid integer {name}.')
    result = array.astype(np.int64, copy=False)
    _need(np.all(result >= 0), f'Out-of-range integer {name}.')
    return result


def _validate_motion(motion):
    _need(set(motion) == MOTION_FIELDS, 'Missing or unexpected drive motion arrays.')
    frame, index = motion['frame_ids'], motion['reference_index']
    _need(frame.ndim == 1 and frame.dtype.kind == 'U' and len(frame) > 0,
          'Expected explicit named drive frames.')
    names = frame.tolist()
    _need(all(name.strip() for name in names) and len(set(names)) == len(names) and
          index.shape == () and index.dtype == np.int64 and 0 <= index.item() < len(names),
          'Duplicate drive frames or invalid reference pose.')
    vertices, faces = motion['vertices_world'], motion['faces']
    _need(vertices.ndim == 3 and vertices.shape[0] == len(names) and vertices.shape[1] >= 3 and
          vertices.shape[2] == 3 and vertices.dtype == np.float32 and np.isfinite(vertices).all() and
          faces.ndim == 2 and faces.shape[1] == 3 and len(faces) and faces.dtype == np.int64 and
          np.all((faces >= 0) & (faces < vertices.shape[1])) and
          not any(np.any(faces[:, a] == faces[:, b]) for a, b in ((0, 1), (0, 2), (1, 2))),
          'Invalid common-topology posed mesh.')
    support, ids = motion['clothing_support_face_ids'], motion['clothing_stable_ids']
    next_id = motion['next_clothing_stable_id']
    _need(support.ndim == 1 and len(support) and ids.shape == support.shape and
          ids.dtype == support.dtype == next_id.dtype == np.int64 and
          np.all((support >= 0) & (support < len(faces))) and
          ids[0] >= 0 and np.all(ids[1:] > ids[:-1]) and next_id.shape == () and next_id.item() > ids[-1],
          'Invalid inherited clothing support or stable ID lineage.')
    K, E, wh = (motion[name] for name in ('camera_K', 'camera_w2c', 'camera_wh'))
    _need(K.shape == (len(names), 3, 3) and E.shape == (len(names), 4, 4) and
          K.dtype == E.dtype == np.float64 and np.isfinite(K).all() and np.isfinite(E).all() and
          np.all(K[:, 2] == [0., 0., 1.]) and np.all(K[:, 0, 0] > 0) and np.all(K[:, 1, 1] > 0) and
          np.all(K[:, 0, 1] == 0) and np.all(K[:, 1, 0] == 0) and
          np.all(E[:, 3] == [0., 0., 0., 1.]) and wh.shape == (len(names), 2) and
          wh.dtype == np.int64 and np.all(wh > 0), 'Invalid native full-K camera rows.')
    R = E[:, :3, :3]
    _need(np.allclose(R @ R.swapaxes(1, 2), np.eye(3), rtol=0, atol=2e-5) and
          np.allclose(np.linalg.det(R), 1., rtol=0, atol=2e-5), 'Invalid native camera rotation.')


def _read_ply(path, triangle_count=None):
    ply = PlyData.read(path, mmap=False)
    _need(not ply.text and [element.name for element in ply.elements] == ['vertex'],
          'Expected a binary SH0 vertex-only PLY.')
    data = ply['vertex'].data
    fields = PLY_FIELDS if triangle_count is None else (*PLY_FIELDS, 'binding_0')
    _need(len(data) > 0 and data.dtype.names == fields and
          all(data[name].dtype.kind == 'f' and data[name].dtype.itemsize == 4 and
              np.isfinite(data[name]).all() for name in fields), 'Invalid native SH0 point fields.')
    _need(np.all(np.maximum.reduce([abs(data[f'rot_{axis}']) for axis in range(4)]) > 0),
          'Zero rotation in native SH0 rows.')
    if triangle_count is None:
        return len(data), None
    face_ids = data['binding_0']
    _need(np.all((face_ids >= 0) & (face_ids < triangle_count) & (face_ids == np.floor(face_ids))),
          'Invalid fixed-row triangle identity.')
    return len(data), face_ids.astype(np.int64)


def save_drive_bundle(directory, fixed, cloth, *, vertices_sequence, faces, reference_frame,
                      support_face_ids, frame_ids, cameras):
    """Save the historical rigid triangle-support inference bundle."""
    _need(not hasattr(cloth, 'graph_arrays'), 'Save a learned graph with save_garment_drive_bundle.')
    return _save_drive_bundle(directory, fixed, cloth, vertices_sequence, faces, reference_frame,
                              support_face_ids, frame_ids, cameras, graph=None)


def save_garment_drive_bundle(directory, fixed, cloth, *, vertices_sequence, faces, reference_frame,
                              support_face_ids, frame_ids, cameras):
    """Save the learned garment graph and native poses for inference replay."""
    graph = cloth.graph_arrays()
    _validate_graph_arrays(graph)
    anchors = cloth.anchor_arrays() if hasattr(cloth, 'anchor_arrays') else None
    if anchors is not None:
        _validate_anchor_arrays(anchors, graph)
    return _save_drive_bundle(directory, fixed, cloth, vertices_sequence, faces, reference_frame,
                              support_face_ids, frame_ids, cameras, graph=graph, anchors=anchors)


def _validate_graph_arrays(graph):
    _need(set(graph) == GRAPH_FIELDS and all(isinstance(array, np.ndarray) and array.dtype.kind in 'iuf'
          and np.isfinite(array).all() for array in graph.values()), 'Invalid garment graph archive arrays.')


def _validate_anchor_arrays(anchors, graph):
    rows, nodes = len(graph['row_node_ids']), len(graph['node_centers'])
    expected = {
        'anchor_reference_normal': ((rows, 3), np.float32),
        'anchor_trusted': ((rows,), np.bool_),
        'anchor_normal_raw': ((rows, 1), np.float32),
        'anchor_tangent_raw': ((nodes, 3), np.float32),
    }
    _need(set(anchors) == set(expected) and all(
        isinstance(anchors[name], np.ndarray) and anchors[name].shape == shape
        and anchors[name].dtype == dtype and np.isfinite(anchors[name]).all()
        for name, (shape, dtype) in expected.items()), 'Invalid garment anchor archive arrays.')


def _save_drive_bundle(directory, fixed, cloth, vertices_sequence, faces, reference_frame,
                       support_face_ids, frame_ids, cameras, *, graph, anchors=None):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    frame_ids, cameras = tuple(frame_ids), tuple(cameras)
    _need(all(isinstance(name, str) for name in frame_ids), 'Frame identities must be exact strings.')
    if isinstance(reference_frame, str):
        reference_frame = frame_ids.index(reference_frame)
    _need(isinstance(reference_frame, Integral) and not isinstance(reference_frame, bool),
          'Choose an exact reference frame identity or index.')
    _need(len(frame_ids) == len(cameras) and all(
        row['frame'] == frame and row.get('pose_index', index) == index and
        row.get('pixel_convention', 'pixel_corners') == 'pixel_corners'
        for index, (frame, row) in enumerate(zip(frame_ids, cameras, strict=True))),
        'Named camera and motion indices differ.')
    vertices, faces = _numpy(vertices_sequence), _indices(faces, 'faces')
    support = _indices(support_face_ids, 'support faces')
    _need(np.array_equal(vertices, _numpy(fixed.vertices_sequence)) and
          np.array_equal(vertices, _numpy(cloth.vertices_sequence)) and
          np.array_equal(faces, _numpy(fixed.faces)) and np.array_equal(faces, _numpy(cloth.faces)) and
          np.array_equal(support, _numpy(cloth.support_face_ids)) and cloth.reference_frame == reference_frame,
          'Fixed and clothing rows must share exactly the saved posed mesh and support.')
    motion = dict(frame_ids=np.asarray(frame_ids, dtype=np.str_), reference_index=np.int64(reference_frame),
                  vertices_world=vertices, faces=faces, clothing_support_face_ids=support,
                  clothing_stable_ids=_indices(cloth.densification_ids, 'stable IDs'),
                  next_clothing_stable_id=_indices(cloth.next_densification_id, 'next stable ID'),
                  camera_K=np.asarray([row['K'] for row in cameras], dtype=np.float64),
                  camera_w2c=np.asarray([row['w2c'] for row in cameras], dtype=np.float64),
                  camera_wh=_indices([row['wh'] for row in cameras], 'camera dimensions'))
    _validate_motion(motion)
    CombinedClothingScene(fixed, cloth)
    _need(fixed.binding is not None and cloth.binding is None and
          all(not getattr(fixed, name).requires_grad for name in PARAMETERS),
          'The fixed rows must be frozen and the canonical clothing rows must be unbound.')
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.drive-', dir=directory.parent) as temporary:
        staging = Path(temporary)/'bundle'
        staging.mkdir()
        fixed.save_ply(str(staging/FIXED_PLY))
        cloth.save_ply(str(staging/CLOTHING_PLY))
        fixed_points, binding = _read_ply(staging/FIXED_PLY, len(faces))
        cloth_points, _ = _read_ply(staging/CLOTHING_PLY)
        _need(np.array_equal(binding, _numpy(fixed.binding)) and len(support) == cloth_points,
              'The native PLY changed the fixed binding or canonical row count.')
        np.savez_compressed(staging/MOTION, **motion)
        files = FILES
        guide = REPLAY_INSTRUCTIONS
        if graph is not None:
            _need(len(graph['row_node_ids']) == cloth_points and len(graph['face_groups']) == len(faces),
                  'Garment graph rows or faces do not match the native bundle.')
            np.savez_compressed(staging/GARMENT_GRAPH, **graph)
            files = (*files, GARMENT_GRAPH)
            guide += '\n`garment-graph.npz` 保存学到的衣物图。此包用于回放。续训所需的优化器状态由训练检查点保存。\n'
            if anchors is None:
                guide = guide.replace('compute_cov3D_python=False', 'compute_cov3D_python=True')
        if anchors is not None:
            np.savez_compressed(staging/GARMENT_ANCHORS, **anchors)
            files = (*files, GARMENT_ANCHORS)
            guide += '\n`garment-anchors.npz` 保存参考法向、可信支持掩码及训练得到的法向和切向参数。\n'
        (staging/REPLAY_GUIDE).write_text(guide, encoding='utf-8')
        schema = SCHEMA
        if graph is not None:
            schema = GARMENT_SCHEMA if anchors is None else ANCHORED_GARMENT_SCHEMA
        manifest = dict(schema=schema, sh_degree=0,
                        fixed_points=fixed_points, clothing_points=cloth_points,
                        frame_ids=list(frame_ids), reference_frame=frame_ids[reference_frame],
                        reference_index=int(reference_frame), pixel_convention='pixel_corners',
                        pose_coordinates='shared fitted world, metres', fixed_ply_coordinates='triangle local',
                        clothing_ply_coordinates='reference world', support_policy='motion only',
                        replay_guide=REPLAY_GUIDE,
                        files_sha256={name: rt.sha(staging/name) for name in files})
        if graph is not None:
            manifest.update(garment_graph=GARMENT_GRAPH, artifact_role='inference_replay', optimizer_state_included=False,
                            clothing_ply_coordinates='base reference world before learned garment graph')
        if anchors is not None:
            manifest.update(garment_anchors=GARMENT_ANCHORS,
                            clothing_ply_coordinates='base reference world before garment graph and anchors')
        rt.save(staging/MANIFEST, manifest)
        staging.rename(directory)
    return manifest


@torch.no_grad()
def load_drive_bundle(directory, *, fixed_model=None, cloth_model=None, device='cuda'):
    """Verify the native files and rebuild both point sets without fitting or assets."""
    from contextlib import nullcontext

    directory = Path(directory)
    manifest = rt.read(directory/MANIFEST)
    hashes = manifest['files_sha256']
    schema = manifest['schema']
    anchored = schema == ANCHORED_GARMENT_SCHEMA
    garment = schema in (GARMENT_SCHEMA, ANCHORED_GARMENT_SCHEMA)
    files = (*FILES, GARMENT_GRAPH) if garment else FILES
    if anchored:
        files = (*files, GARMENT_ANCHORS)
    _need(schema in (SCHEMA, GARMENT_SCHEMA, ANCHORED_GARMENT_SCHEMA)
          and manifest['sh_degree'] == 0 and set(hashes) == set(files) and
          manifest['pixel_convention'] == 'pixel_corners' and manifest['replay_guide'] == REPLAY_GUIDE and
          all(rt.sha(directory/name) == hashes[name] for name in files), 'Drive bundle identity differs.')
    if garment:
        _need(manifest['garment_graph'] == GARMENT_GRAPH and manifest['artifact_role'] == 'inference_replay' and
              manifest['optimizer_state_included'] is False, 'A garment drive bundle is inference only.')
        with np.load(directory/GARMENT_GRAPH, allow_pickle=False) as archive:
            graph = dict(archive)
        _validate_graph_arrays(graph)
    if anchored:
        _need(manifest['garment_anchors'] == GARMENT_ANCHORS, 'The garment anchor file differs from the schema.')
        with np.load(directory/GARMENT_ANCHORS, allow_pickle=False) as archive:
            anchors = dict(archive)
        _validate_anchor_arrays(anchors, graph)
    with np.load(directory/MOTION, allow_pickle=False) as archive:
        motion = dict(archive)
    _validate_motion(motion)
    vertices, faces = motion['vertices_world'], motion['faces']
    names, index = motion['frame_ids'].tolist(), motion['reference_index'].item()
    fixed_points, binding = _read_ply(directory/FIXED_PLY, len(faces))
    cloth_points, _ = _read_ply(directory/CLOTHING_PLY)
    _need(names == manifest['frame_ids'] and index == manifest['reference_index'] and
          names[index] == manifest['reference_frame'] and fixed_points == manifest['fixed_points'] and
          cloth_points == manifest['clothing_points'] == len(motion['clothing_support_face_ids']),
          'Native point counts or named motion differ from the drive manifest.')
    if fixed_model is None:
        from scene.smplx_gaussian_model import SMPLXGaussianModel
        fixed_model = SMPLXGaussianModel(0)
    if cloth_model is None:
        if anchored:
            from .anchored_garment_surface import SurfaceAnchoredGarmentModel
            cloth_model = SurfaceAnchoredGarmentModel(0)
        elif garment:
            from .garment_surface import GarmentSurfaceModel
            cloth_model = GarmentSurfaceModel(0)
        else:
            from scene.clothing_gaussian_model import ClothingGaussianModel
            cloth_model = ClothingGaussianModel(0)
    _need(garment == hasattr(cloth_model, 'graph_arrays'), 'The clothing model does not match the drive schema.')
    _need(anchored == hasattr(cloth_model, 'anchor_arrays'), 'The clothing model does not match the anchor schema.')
    _need(all(model.optimizer is None and model.max_sh_degree == 0 and model.binding is None and
              len(model._xyz) == 0 for model in (fixed_model, cloth_model)),
          'Drive replay requires fresh SH0 models with no optimizer or prior rows.')
    device = torch.device(device)
    with torch.cuda.device(device) if device.type == 'cuda' else nullcontext():
        fixed_model.load_ply(str(directory/FIXED_PLY))
        cloth_model.load_ply(str(directory/CLOTHING_PLY))
        point_device = fixed_model._xyz.device
        _need(point_device == cloth_model._xyz.device and point_device.type == device.type and
              (device.index is None or point_device.index == device.index) and cloth_model.binding is None and
              np.array_equal(_numpy(fixed_model.binding), binding), 'Native loader changed device or binding.')
        fixed_model.vertices_sequence = torch.as_tensor(vertices, device=point_device)
        fixed_model.faces = torch.as_tensor(faces, device=point_device)
        fixed_model.num_timesteps = len(names)
        fixed_model.binding_counter = torch.bincount(fixed_model.binding.long(), minlength=len(faces)).to(torch.int32)
        cloth_model.configure_triangle_support(fixed_model.vertices_sequence, fixed_model.faces, index,
            motion['clothing_support_face_ids'], stable_ids=motion['clothing_stable_ids'],
            next_stable_id=motion['next_clothing_stable_id'].item())
        if garment:
            cloth_model.configure_garment_surface(graph=graph)
            restored = cloth_model.graph_arrays()
            _need(set(restored) == set(graph) and all(restored[name].dtype == value.dtype and
                  np.array_equal(restored[name], value) for name, value in graph.items()),
                  'The garment model did not recover the exact learned graph arrays.')
            cloth_model.node_centers.requires_grad_(False)
        if anchored:
            cloth_model.configure_surface_anchors(anchors=anchors)
            restored = cloth_model.anchor_arrays()
            _need(set(restored) == set(anchors) and all(restored[name].dtype == value.dtype and
                  np.array_equal(restored[name], value) for name, value in anchors.items()),
                  'The garment model did not recover the exact anchor arrays.')
            cloth_model.anchor_normal_raw.requires_grad_(False)
            cloth_model.anchor_tangent_raw.requires_grad_(False)
        for model in (fixed_model, cloth_model):
            for name in PARAMETERS:
                getattr(model, name).requires_grad_(False)
        scene = CombinedClothingScene(fixed_model, cloth_model)
        scene.select_mesh_by_timestep(index)
    cameras = [dict(frame=frame, pose_index=pose, pixel_convention='pixel_corners',
                    K=motion['camera_K'][pose].tolist(), w2c=motion['camera_w2c'][pose].tolist(),
                    wh=motion['camera_wh'][pose].tolist()) for pose, frame in enumerate(names)]
    return scene, dict(manifest, cameras=cameras)
