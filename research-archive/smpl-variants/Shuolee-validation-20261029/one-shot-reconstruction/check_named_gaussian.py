"""CPU validation of the actual saved roster and export, without optimization."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

project = Path('/mnt/d/Projects/Shuolee')
source = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')
base = Path('/mnt/d/Projects/Shuolee-validation-20261029/one-shot-reconstruction')
historical = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle/full94-shared-shape')
sys.path.insert(0, str(project/'src'))
from shuolee_portrait import training, export, runtime as rt
from shuolee_portrait.camera_exchange import write_camera_exchange

started = time.monotonic()
destination = base/'official-gaussian-cpu-check'
destination.mkdir(exist_ok=False)
def load_reference(filename):
    name = 'shuolee_portrait._reference_'+filename
    spec = importlib.util.spec_from_file_location(name, source/'shuolee_portrait'/(filename+'.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
reference_training, reference_export = load_reference('training'), load_reference('export')
paths = [source/'shuolee_portrait/training.py', source/'shuolee_portrait/export.py',
         Path(training.__file__), Path(export.__file__)]
hashes = {str(p): rt.sha(p) for p in paths}
assert hashes[str(paths[0])] == 'ea5520249cc931e9d63e6ec59d62bcd6ac33fff52ca787d4ae0c950f61830d99'
assert hashes[str(paths[1])] == '4da488438b5fcfb40f6a028d5bab9952c223a522f3446d9a7ba6344d5ead1520'

def nodes(path):
    return {n.name:n for n in ast.parse(path.read_text(encoding='utf-8')).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
def dump(node): return ast.dump(node, include_attributes=False)
old, new = nodes(paths[0]), nodes(paths[2])
functions = ['camera', 'named_pose_layout', 'render_pose', 'reference_ply_comparison',
             'foreground_loss', 'image', 'Views']
exact = {n: dump(old[n]) == dump(new[n]) for n in functions}
old_loop = copy.deepcopy(next(n for n in ast.walk(old['_train_static'])
                             if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step'))
class RemoveResearchReview(ast.NodeTransformer):
    def visit_If(self, node):
        if any(isinstance(n, ast.Name) and n.id == 'checkpoint_review' for n in ast.walk(node.test)):
            return None
        return self.generic_visit(node)
old_loop = RemoveResearchReview().visit(old_loop)
new_loop = next(n for n in ast.walk(new['_train_static']) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'step')
exact['entire_optimization_loop_without_research_review_callback'] = dump(old_loop) == dump(new_loop)
oe, ne = nodes(paths[1]), nodes(paths[3])
exact.update({n:dump(oe[n]) == dump(ne[n]) for n in ['reference_pose_index', 'rigid_rotation64']})
class IgnoreMmap(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'PlyData' and node.func.attr == 'read':
            node.keywords = [k for k in node.keywords if k.arg != 'mmap']
        return self.generic_visit(node)
exact['rotate_ply_except_existing_Windows_mmap_flag'] = dump(IgnoreMmap().visit(copy.deepcopy(oe['rotate_ply']))) == dump(IgnoreMmap().visit(copy.deepcopy(ne['rotate_ply'])))
assert all(exact.values()), exact
core = ['avatar_core/scene/smplx_gaussian_model.py', 'avatar_core/scene/gaussian_model.py',
        'avatar_core/scene/cameras.py', 'avatar_core/gaussian_renderer/__init__.py',
        'avatar_core/arguments/__init__.py']
core_exact = {n:rt.sha(source/n) == rt.sha(project/'src'/n) for n in core}
assert all(core_exact.values())

assembled = base/'official-named-assembly'
verified = training.verify_official_fit(assembled)
with np.load(assembled/'fit/fitted.npz', allow_pickle=False) as f: fit = dict(f)
rows = rt.read(assembled/'data/training-inputs.json')['records']
layout = training.named_pose_layout(fit, rows)
assert layout == reference_training.named_pose_layout(fit, rows)
camera_exact = True; max_projection_error = 0.
for i, row in enumerate(rows):
    a, ra = training.camera(row, i, 899, pose_index=layout['pose_indices'][i])
    b, rb = reference_training.camera(row, i, 899, pose_index=layout['pose_indices'][i])
    camera_exact &= ra == rb and np.array_equal(a.full_proj_transform.numpy(), b.full_proj_transform.numpy())
    assert a.timestep == layout['pose_indices'][i]
    max_projection_error = max(max_projection_error, ra['projection_error_px'])
assert camera_exact

train_result = rt.read(historical/'gaussian-run/result.json')
snapshot = historical/'gaussian-run'/train_result['final_snapshot']
parity = rt.read(snapshot/'native-ply-parity.json')
slot, reference = export.reference_pose_index(fit, parity)
assert (slot, reference) == reference_export.reference_pose_index(fit, parity)
assert slot == layout['reference_pose_index'] and reference == layout['reference_frame_id']
validation = rt.read(historical/'export/validation.json')
R, origin = np.asarray(validation['rotation']), np.asarray(validation['origin_world'])
world_ply = snapshot/'world.ply'; original_ply = historical/'export/point_cloud.ply'
source_hashes = {str(p):rt.sha(p) for p in [world_ply, original_ply]}
report = export.rotate_ply(world_ply, destination/'point_cloud.ply', R, origin)
assert report['sha256'] == source_hashes[str(original_ply)], 'CPU transformed PLY bytes differ'
target_rows=[]
for row in rows:
    E=np.asarray(row['w2c']); mapped=E.copy()
    mapped[:3,:3]=E[:3,:3]@R.T; mapped[:3,3]=E[:3,:3]@origin+E[:3,3]
    target_rows.append(dict(frame=row['frame'], source_frame_index=row['source_frame_index'],
        time_seconds=row['time_seconds'], wh=row['wh'], K=row['K'], w2c=mapped.tolist()))
old_cameras = rt.read(historical/'export/cameras.json')
assert target_rows == old_cameras['records'], 'Reference camera transforms changed'
write_camera_exchange(destination, target_rows, world_to_viewer=report)
for path, digest in {**hashes, **source_hashes}.items(): assert rt.sha(path) == digest, path
result = dict(state='passed', frames=len(rows), reference_frame=reference, reference_slot=slot,
    AST_equivalence=exact, core_bytes_identical=core_exact, actual_official_receipts_verified=True,
    training_layout_exact=True, all_camera_matrices_exact=camera_exact,
    max_projection_error_px=max_projection_error, export_PLY_bytes_exact=True,
    exported_camera_transforms_exact=True, output_sha256=report['sha256'],
    output=str(destination/'point_cloud.ply'), source_hashes=hashes,
    originals_unchanged=True, GPU_calls=0, fitting_runs=0, gaussian_optimizer_steps=0,
    quality_accepted=False, server_verified=False, end_to_end_verified=False,
    seconds=time.monotonic()-started)
rt.save(destination/'result.json', result)
print(json.dumps(result, indent=2))
