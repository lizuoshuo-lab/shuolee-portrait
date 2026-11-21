"""Independent frozen SOURCE/CPU review; never runs author or GPU rendering.

One completed GT-free map is read on CPU. SciPy reconstructs the declared
world gauge action without author functions. It is not a renderer result.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'scripts/check_gaussian_global_gauge_renderer.py'
AUTHOR_SHA = 'f525ffec8f78d0ccd80ac0f565bd04a204092fbf86ab51b3940c47c1bbb3e86f'
PREP = ROOT / 'outputs/gaussian-global-gauge-renderer-cpu-preparation-002'
PROTOCOL_SHA = '732abff449b619a36b2b907630c26b0f7b36ff9a7834c8456cc5a2b71317939f'
OUTPUT = ROOT / 'outputs/gaussian-global-gauge-renderer-independent-source-review-002'


def require(condition, message):
    if not condition: raise AssertionError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2, allow_nan=False)


def record(array):
    if isinstance(array, torch.Tensor): array = array.detach().cpu().contiguous().numpy()
    a = np.ascontiguousarray(array)
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'raw_bytes': a.nbytes, 'raw_bytes_sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def project(mean, covariance, K):
    x, y, z = mean.T
    uv = np.column_stack([K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]])
    J = np.zeros((len(mean), 2, 3))
    J[:, 0, 0] = K[0, 0] / z; J[:, 1, 1] = K[1, 1] / z
    J[:, 0, 2] = -K[0, 0] * x / z ** 2; J[:, 1, 2] = -K[1, 1] * y / z ** 2
    return uv, np.einsum('nai,nij,nbj->nab', J, covariance, J)


def source_review(text, protocol):
    compile(text, str(AUTHOR), 'exec'); tree = ast.parse(text)
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    config_node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in n.targets))
    # FRAME_ID is fixed548; evaluate literals only, never execute author definitions.
    copied = ast.fix_missing_locations(ast.NodeTransformer().visit(config_node.value))
    class ReplaceFrame(ast.NodeTransformer):
        def visit_Name(self, node):
            require(node.id == 'FRAME_ID', 'Only one known literal CONFIG symbol'); return ast.copy_location(ast.Constant(548), node)
    copied = ReplaceFrame().visit(copied)
    require(ast.literal_eval(copied) == protocol['configuration'], 'Exact author CONFIG AST equals reviewed protocol')
    execute = ast.unparse(funcs['execute'])
    raster_calls = [n for n in ast.walk(funcs['execute']) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'rasterization']
    draw_loops = [n for n in ast.walk(funcs['execute']) if isinstance(n, ast.For) and ast.unparse(n.iter) == 'enumerate(pairs)']
    count_controls = [n for n in ast.walk(funcs['execute']) if isinstance(n, ast.Compare) and ast.unparse(n.left) == "state['render_calls']" and len(n.comparators) == 1 and isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value == 2]
    require(len(raster_calls) == len(draw_loops) == len(count_controls) == 1, 'Exactly one draw expression within declared two-arm loop and no retry')
    pair_loops = [n for n in ast.walk(funcs['execute']) if isinstance(n, ast.For) and isinstance(n.iter, ast.Tuple) and [ast.unparse(t) for t in n.iter.elts] == ['(params, w)', '(updated, wnew)']]
    require(len(pair_loops) == 1, 'Two declared complete-map/saved-camera pairs')
    call = raster_calls[0]
    for position, name in [(2, 'scales'), (3, 'opacity'), (4, 'RGB')]:
        require(ast.unparse(call.args[position]) == f"shared_constants['{name}']", 'Both draws share the exact same activated CUDA constant tensor')
    require('separately_activated_gauged' in execute and 'GPU map parameter bytes mutated by rendering' in execute and 'Camera or K bytes mutated' in execute and 'Shared activated constant bytes mutated' in execute, 'Actual activated/map/camera before-after guards retained')
    require(execute.index('deadline()') < execute.index('from gsplat import rasterization') and execute.index('first_render_start = deadline()') < execute.index('rasterization('), 'Cutoff before backend initialization and first actual draw')
    require('except BaseException as error:' in execute and "failed_global_gauge_render_control_preserved_no_retry" in execute and 'np.savez_compressed' in execute, 'Raw predictions and original failed comparisons retained without retry')
    require("w.numpy().astype(np.float64) @ inverse" in ast.unparse(funcs['transformed_inputs']), 'Saved w2c uses world-right inverseDelta action')
    compare = ast.unparse(funcs['compare'])
    require('support0 & support1' in compare and 'covered0 & covered1' in compare and 'v is not None' in compare, 'Both positive-common ED masks; empty masks cannot pass')
    return {'status': 'passed_complete_manual_and_AST_SOURCE_review', 'source_functions_read': list(funcs),
            'GPU_draw_expressions': len(raster_calls), 'predeclared_actual_draw_count': 2,
            'active_world_left_quaternion_order': 'Hamilton wxyz qDelta product rawqOld; renderer then normalizes',
            'renderer_inputs': 'All23838 Gaussians, exact saved548 float32 W, fixed160x120 K; shared CUDA activated RGB/scales/opacity; RGB+ED blackbackground near.1 far6.',
            'cutoff_and_failures': 'Late backend-init and first-draw deadline guards; started two-arm control finishes and retains failure without retries.',
            'manual_scope': 'No GT/sensor observations, optimizer, fitting, map selection, per-anchor repair or SH transport. Original arm is frozen-map render, not observed RGB.'}


def main():
    OUTPUT.mkdir(exist_ok=False); (OUTPUT / 'review_source.py').write_bytes(Path(__file__).read_bytes()); (OUTPUT / 'author_source.py').write_bytes(AUTHOR.read_bytes())
    try:
        require(sha(AUTHOR) == AUTHOR_SHA and sha(PREP / 'protocol.json') == PROTOCOL_SHA and sha(PREP / 'source_script.py') == AUTHOR_SHA, 'Frozen author SOURCE/protocol/snapshot')
        protocol = read(PREP / 'protocol.json'); cpu = read(PREP / 'cpu_checks.json'); require(cpu == protocol['CPU_checks'], 'Actual CPU checks duplicate protocol exactly')
        cfg = protocol['configuration']; require(protocol['status'] == 'prepared_CPU_geometry_passed_GPU_not_authorized' and protocol['expected_render_calls'] == 2 and protocol['GPU_render_calls'] == protocol['GT_reads'] == protocol['optimizer_steps'] == 0 and cfg['frame_id'] == 548 and cfg['width'] == 160 and cfg['height'] == 120, 'SOURCE-only two-draw fixed548 protocol')
        locks = {AUTHOR: AUTHOR_SHA, PREP / 'protocol.json': PROTOCOL_SHA, PREP / 'source_script.py': AUTHOR_SHA, PREP / 'cpu_checks.json': sha(PREP / 'cpu_checks.json')}
        for rel, digest in protocol['source_sha256'].items(): require(sha(ROOT / rel) == digest, 'Frozen source reference'); locks[ROOT / rel] = digest
        for path, digest in protocol['gsplat_python_source_sha256'].items(): require(sha(Path(path)) == digest, 'Frozen gsplat Python rendering/wrapper source'); locks[Path(path)] = digest
        require(len(cpu['all_checks']) == 9 and all(row['passed'] and row['actual'] <= row['limit'] == cfg['CPU_limits'][row['name']] for row in cpu['all_checks']) and cpu['author_gpu_render_calls'] == 0 and cpu['CUDA_initialized'] is False, 'Author actual9 CPU predicates predeclared and no GPU')
        structure = source_review(AUTHOR.read_text(encoding='utf-8'), protocol)
        fixed = protocol['fixed_input']; checkpoint = ROOT / fixed['checkpoint']; require(sha(checkpoint) == fixed['checkpoint_sha256'], 'Only reviewed completed causal checkpoint')
        locks[checkpoint] = fixed['checkpoint_sha256']; saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(saved['status'] == 'completed' and saved['ground_truth_included'] is False and saved['current_local_index'] == 548 and saved['frame_ids_with_stored_pose'] == list(range(549)), 'Complete GT-free chronological map, no new GT matrix input')
        params = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in saved['parameters'].items()}
        require(set(params) == {'means', 'quats', 'log_scales', 'color_logits', 'opacity_logits'} and len(params['means']) == 23838 and all(t.dtype == torch.float32 and torch.isfinite(t).all() and not t.requires_grad for t in params.values()), 'All23838 finite frozen constant-RGB Gaussians')
        W = saved['estimated_w2c'][548].detach().cpu().contiguous().clone(); K = saved['K'].detach().cpu().contiguous().clone()
        require({name: record(t) for name, t in params.items()} == fixed['parameters'] and record(W) == fixed['exact_saved_W'] and record(K) == fixed['fixed_K'], 'Every original parameter/K/exact548camera raw byte record independently matches')
        activated = {'scales': params['log_scales'].exp(), 'RGB': params['color_logits'].sigmoid(), 'opacity': params['opacity_logits'].sigmoid()}
        require({name: record(t) for name, t in activated.items()} == fixed['activated_constant_RGB_opacity_scales'], 'Actual CPU activation raw bytes independently match reviewed records')
        rd = Rotation.from_euler('xyz', cfg['correction_rotation_xyz_degrees_RzRyRx'], degrees=True)
        D = np.eye(4); D[:3, :3] = rd.as_matrix(); D[:3, 3] = cfg['correction_translation_m']
        Wnew = (W.numpy().astype(np.float64) @ np.linalg.inv(D)).astype(np.float32)
        means = params['means'].numpy().astype(np.float64); new_means = (rd.apply(means) + D[:3, 3]).astype(np.float32)
        old_q = params['quats'].numpy().astype(np.float64)
        delta_q = rd.as_quat()[[3, 0, 1, 2]]; w, x, y, z = delta_q
        # Independent left-Hamilton matrix route; no author quaternion helper.
        L = np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])
        new_q = (old_q @ L.T).astype(np.float32)
        rotation_old = Rotation.from_quat(old_q[:, [1, 2, 3, 0]]).as_matrix()
        rotation_new = Rotation.from_quat(new_q[:, [1, 2, 3, 0]]).as_matrix()
        scale2 = activated['scales'].numpy().astype(np.float64) ** 2
        sigma_old = np.einsum('nij,nj,nkj->nik', rotation_old, scale2, rotation_old)
        sigma_new = np.einsum('nij,nj,nkj->nik', rotation_new, scale2, rotation_new)
        W64, N64 = W.numpy().astype(np.float64), Wnew.astype(np.float64)
        before_p = means @ W64[:3, :3].T + W64[:3, 3]; after_p = new_means.astype(np.float64) @ N64[:3, :3].T + N64[:3, 3]
        before_S = W64[:3, :3] @ sigma_old @ W64[:3, :3].T; after_S = N64[:3, :3] @ sigma_new @ N64[:3, :3].T
        low, high = cfg['CPU_projection_depth_margin_m']; common = (before_p[:, 2] > low) & (before_p[:, 2] < high) & (after_p[:, 2] > low) & (after_p[:, 2] < high)
        require(common.any(), 'Nonempty independent well-conditioned projection')
        before_u, before_U = project(before_p[common], before_S[common], K.numpy()[0]); after_u, after_U = project(after_p[common], after_S[common], K.numpy()[0])
        residuals = {'camera_mean_max_abs_m': float(np.max(np.abs(after_p - before_p))), 'camera_covariance_max_abs_m2': float(np.max(np.abs(after_S - before_S))), 'quaternion_covariance_max_abs_m2': float(np.max(np.abs(sigma_new - D[:3, :3] @ sigma_old @ D[:3, :3].T))), 'pinhole_mean_max_abs_pixel': float(np.max(np.abs(after_u - before_u))), 'pinhole_covariance_max_abs_pixel2': float(np.max(np.abs(after_U - before_U)))}
        require(all(v <= cfg['CPU_limits'][name] for name, v in residuals.items()), 'Independent actual-map SciPy/float32 geometry meets unchanged predeclared CPU bounds')
        transformed_hash_matches = {'means': record(new_means) == fixed['transformed_parameters']['means'], 'quats': record(new_q) == fixed['transformed_parameters']['quats'], 'Wnew': record(Wnew) == fixed['transformed_W']}
        omitted_error = float(np.max(np.abs(N64[:3, :3] @ sigma_old @ N64[:3, :3].T - before_S)))
        wrong_W = np.linalg.inv(D) @ W64; wrong_p = new_means.astype(np.float64) @ wrong_W[:3, :3].T + wrong_W[:3, 3]
        wrong_error = float(np.max(np.abs(wrong_p - before_p)))
        require(omitted_error > 1e-8 and wrong_error > .001, 'Independent full-map anisotropic and left/right negative witnesses remain physically effective')
        require(all(sha(path) == digest for path, digest in locks.items()) and not torch.cuda.is_initialized(), 'All consumed files unchanged and CUDA uninitialized')
        result = {'status': 'passed_independent_global_gauge_renderer_SOURCE_and_CPU_review_no_GPU', 'review_source_sha256': sha(Path(__file__)), 'author_source_sha256': AUTHOR_SHA, 'author_protocol_sha256': PROTOCOL_SHA, 'author_cpu_checks_sha256': sha(PREP / 'cpu_checks.json'), 'source_review': structure,
            'independent_actual_checkpoint_CPU_witness': {'point_count': 23838, 'saved_frame_id': 548, 'original_parameter_K_camera_records_byte_exact': True, 'CPU_activated_RGB_opacity_scales_records_byte_exact': True, 'SciPy_independent_transformed_payload_hashes_equal_protocol': transformed_hash_matches, 'well_conditioned_point_count': int(common.sum()), 'residuals': residuals, 'negative_omit_covariance_max_abs_m2': omitted_error, 'negative_wrong_camera_max_abs_m': wrong_error},
            'consumed_file_hash_records': {str(p): d for p, d in locks.items()}, 'consumed_files_unchanged': len(locks), 'author_functions_executed': 0, 'GPU_render_calls': 0, 'optimizer_steps': 0, 'new_GT_inputs': 0, 'CUDA_initialized': False,
            'limitations': ['CPU proof and SOURCE review do not imply actual renderer numerical pass; exactly2 GPU draws still require root authorization.', 'CPU and CUDA exp/sigmoid activation can differ by numerical rounding; author deliberately shares exact CUDA activated tensors within its pair, and records their GPU bytes.', 'World-gauge control uses the exact slightly nonorthogonal saved float32 W; algebra-only generic inverse does not replace or repair the rendering camera.', 'Only one global rigid action, fixed one view and constantRGB map are covered; no per-anchor repair/loopclosure, localization accuracy, GT image comparison or SH transport.', 'Python gsplat rendering/wrapper hashes/version are bound; no claim of independently hashing every compiled CUDA binary.', 'All positive common ED masks must be nonempty; mask XOR and predeclared failure handling prevent silently declaring invariance by dropping unsupported pixels.']}
        write(OUTPUT / 'metrics.json', result)
        print(json.dumps({'status': result['status'], 'metrics_sha256': sha(OUTPUT / 'metrics.json'), 'review_source_sha256': result['review_source_sha256'], 'point_count': 23838, 'GPU_render_calls': 0}))
    except BaseException as error:
        write(OUTPUT / 'failure.json', {'status': 'independent_source_CPU_review_failure_retained', 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'author_functions_executed': 0, 'GPU_render_calls': 0}); raise


if __name__ == '__main__': main()
