"""CPU cut-count/resource evidence for the actual unoptimized City006 tree.

Uses pinned official CPU camera math, the frozen CPU size predicate and the
exact evaluator count AST. No renderer, CUDA initialization, quality metrics,
process control, compact-leaf substitution or sealed-source mutation.
"""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import traceback
import numpy as np

sys.dont_write_bytecode = True
LAB = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE = LAB / 'outputs/large-scene-gs-001'
OUTROOT = BASE / 'independent-evaluation'
NATIVE = BASE / 'official-hierarchy/small-city-hierarchy-006'
REPO = LAB / 'repos/hierarchical-3d-gaussians'
CORE = LAB / 'scripts/large_scene_gs/evaluate_official_hierarchy.py'
TOPOLOGY = LAB / 'scripts/large_scene_gs/evaluate_hierarchy.py'
CODEC = NATIVE / 'helper-snapshots/read_hierarchy.py'
LEDGER = OUTROOT / 'city-anchor-lineage-driver-CPU-006/audit_city_hierarchy_lineage_006_posebytes_v2.py'
BINDER = OUTROOT / 'small-city006-unoptimized-merge-binding-CPU-001/report.json'
CANONICAL = OUTROOT / 'small-city-local-coarse-003-flat-heldout-001/independent-inputs/evaluated-inputs.npz'
CAMERA_PROTOCOL = CANONICAL.parent.parent / 'protocol.json'
MODEL = NATIVE / 'merged-unoptimized.hier'
RECEIPT = NATIVE / 'merge-unoptimized-outcome.json'
PIN = {
    CORE: '672533b6aef0f79d64a77976055ce747d05c3f226646e576cb913b0c405af1ec',
    TOPOLOGY: '1d3f6aa22c687a237e06992436dec8f77c2d158f38939bc275b200dc450a077e',
    CODEC: '583c96f50b6f486d73bbbbaf9faf053ed15d486a7c7b48e15ef8e3ebc0964538',
    LEDGER: 'aa4d620778231f1b9c405d95b4c6a04cf2af734eab5a8941817a0cfa0b408bb0',
    BINDER: '758516eff1941284108363e274409b461952e643f8b6751bb72ed9efd93f2d42',
    CANONICAL: '60ab56f705c6db347b00f821b08a44e49c0d2dca4f0f06e9a978b9c73c8bce25',
    MODEL: 'e92827534bef94f0f2e213ce31508570c6f6da5f24a59cf62da20793d2276e78',
    RECEIPT: '427461e8056146354c82541b0802de4111bde47de878c00b2df837b6ca7ff6a8',
    REPO / 'utils/graphics_utils.py': 'e0527b80d8356b7e15381c9ad6adeee3200eff9a5ca793179d1cfd647cdfd2e6',
}
SETTINGS = [('leaves', None), ('tau1', 1.), ('tau3', 3.), ('tau5', 5.), ('tau6', 6.), ('tau15', 15.)]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def function_ast(path, names):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in selected} == set(names), 'Pinned AST functions unavailable')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    require(args.run.replace('-', '').isalnum() and len(args.run) <= 100, 'Invalid new CPU run name')
    output = OUTROOT / args.run
    output.mkdir(exist_ok=False)
    began = time.perf_counter()
    outcome = {'status': 'running_CPU_counts_only', 'started_utc': datetime.now(timezone.utc).isoformat(),
               'CUDA_initialized': False, 'GPU_renderer_started': False}
    save(output / 'outcome.json', outcome)
    try:
        protected = {str(p): sha(p) for p in PIN}
        require(all(protected[str(p)] == expected for p, expected in PIN.items()), 'Pinned actual source or model changed')
        protected[str(CAMERA_PROTOCOL)] = sha(CAMERA_PROTOCOL)
        binder = read(BINDER)
        receipt = read(RECEIPT)
        protocol = read(CAMERA_PROTOCOL)
        require(binder['actual_merged_model_bound'] and binder['all_actual_bound_bytes_unchanged'], 'Actual unoptimized CPU binding missing')
        require(receipt['status'] == 'completed' and receipt['exit_code'] == receipt['actual_child_poll_exit_code'] == 0, 'Real Merger terminal missing')
        require(receipt['source_kind'] == 'unoptimized' and receipt['skybox_in_tree'] == 0, 'Wrong tree or unexpected Sky tail')
        codec = load('count117_frozen_codec', CODEC)
        ledger = load('count117_vector_map', LEDGER)
        h, numeric = ledger.mapped(MODEL, codec, 0)
        require(h.summary() == receipt['hierarchy'] == binder['actual_merged_counts'], 'Decoded tree differs from bound real Merger')
        require(len(h.positions) == 7000396 and len(h.nodes) == 7000393, 'Unexpected actual G/node count')
        leaves = int(h.nodes[:, 3].sum(dtype=np.int64))
        require(leaves == 4643252, 'Actual leaf count changed')
        topology = SimpleNamespace(h=h, depth=h.nodes[:, 0].astype(np.int64), parent=h.nodes[:, 1].astype(np.int64),
                                   leaves=h.nodes[:, 3].astype(np.int64), merged=h.nodes[:, 4].astype(np.int64))
        require(topology.parent[0] == -1 and np.all(topology.parent[1:] >= 0), 'Root/parent schema differs')
        core_functions = function_ast(CORE, ['official_camera_math_CPU', 'make_official_camera_CPU', 'memory_plan', 'threshold_scope'])
        camera_scope = {'np': np, 'ast': ast, 'math': math, 'REPO': REPO, 'SimpleNamespace': SimpleNamespace,
                        'require': require, 'POST_RANGE': (.005, .1), 'tau_to_threshold': codec.tau_to_threshold}
        exec(compile(ast.Module(body=core_functions, type_ignores=[]), str(CORE), 'exec'), camera_scope)
        parsed_topology = ast.parse(TOPOLOGY.read_text(encoding='utf-8'))
        topology_class = next(n for n in parsed_topology.body if isinstance(n, ast.ClassDef) and n.name == 'Topology')
        count_ast = next(n for n in topology_class.body if isinstance(n, ast.FunctionDef) and n.name == 'reference_counts')
        cached = {}

        def same_view_sizes(actual_h, camera):
            require(actual_h is h and np.array_equal(np.asarray(camera, np.float32), cached['center']), 'CPU size cache used for different tree/camera')
            return cached['sizes']

        count_scope = {'np': np, 'official_size_proxy_cpu': same_view_sizes}
        exec(compile(ast.Module(body=[count_ast], type_ignores=[]), str(TOPOLOGY), 'exec'), count_scope)
        # This imports Torch for the exact pinned CPU camera matrices only.
        import torch
        require(not torch.cuda.is_initialized(), 'CPU resource audit unexpectedly owns a CUDA context')
        torch.set_num_threads(1)
        official_math = camera_scope['official_camera_math_CPU']()
        with np.load(CANONICAL, allow_pickle=False) as z:
            names = z['names'].astype(str).tolist()
            matrices_k, matrices_c2w = z['K'], z['c2w']
        require(len(names) == len(set(names)) == 117 and names == protocol['test_names'], 'Canonical117 names/order differ')
        require(matrices_k.shape == (117, 3, 3) and matrices_c2w.shape == (117, 4, 4), 'Canonical camera shapes differ')
        rows = protocol['views']
        require(len(rows) == 117 and [row['name'] for row in rows] == names, 'Canonical protocol camera order differs')
        records = []
        for j, name in enumerate(names):
            require(np.array_equal(matrices_k[j], np.asarray(rows[j]['K'])) and
                    np.array_equal(matrices_c2w[j], np.asarray(rows[j]['C2W'])), 'Canonical K/C2W values differ')
            width, height = rows[j]['derived_dimensions']
            require((width, height) == (256, 172), 'Wrong fixed r4 output dimensions')
            camera = camera_scope['make_official_camera_CPU'](matrices_k[j], matrices_c2w[j], width, height, name, official_math)
            center = camera.camera_center.numpy().copy()
            sizes = codec.official_size_proxy_cpu(h, center)
            require(not np.isnan(sizes).any() and np.all(sizes >= 0), 'Invalid CPU size proxy')
            cached.update(center=np.asarray(center, np.float32), sizes=sizes)
            fx = camera.image_width / (2 * math.tan(camera.FoVx * .5))
            for setting, tau in SETTINGS:
                if tau is None:
                    count = leaves
                else:
                    theta = codec.tau_to_threshold(tau, fx)
                    counts, returned_sizes = count_scope['reference_counts'](topology, center, theta)
                    require(returned_sizes is sizes and np.all(counts >= 0), 'Count predicate/cache invariant differs')
                    count = int(counts.sum(dtype=np.int64))
                    del counts
                require(0 < count <= len(h.positions), 'Actual CPU selection count outside real tree')
                records.append({'name': name, 'camera_index': j, 'setting': setting,
                                'CPU_predicate_selected_tree_G': count, 'one_shared_Sky_G_for_future_render': 100000,
                                'CPU_predicted_total_render_G_including_Sky': count + 100000,
                                'camera_center_float32': center.tolist(), 'threshold_scope': camera_scope['threshold_scope'](tau, fx)})
            cached.clear()
            del sizes, camera
            if (j + 1) % 10 == 0 or j + 1 == 117:
                print(json.dumps({'CPU_views_done': j + 1, 'CPU_views_requested': 117, 'CUDA_initialized': False}), flush=True)
        statistics = []
        for setting, tau in SETTINGS:
            values = np.asarray([r['CPU_predicate_selected_tree_G'] for r in records if r['setting'] == setting], np.int64)
            require(len(values) == 117, 'Missing actual CPU camera/setting counts')
            maximum = int(values.max())
            statistics.append({'setting': setting, 'min_selected_tree_G': int(values.min()),
                               'median_selected_tree_G': float(np.median(values)), 'mean_selected_tree_G': float(values.mean()),
                               'max_selected_tree_G': maximum, 'maximum_camera_names': [names[k] for k in np.flatnonzero(values == maximum)],
                               'known_tensor_memory_plan_at_actual_CPU_max': camera_scope['memory_plan'](len(h.positions) + 100000,
                                        len(h.nodes), leaves, maximum, 100000, 117)})
        require(len(records) == 702, 'Only one real unoptimized tree expected: 117x6 CPU counts, not 1404 rendered frames')
        require(not torch.cuda.is_initialized(), 'CPU camera audit initialized CUDA')
        require(all(sha(p) == expected for p, expected in protected.items()), 'Read-only actual bound bytes changed while counted')
        report = {'status': 'completed_actual_unoptimized_City006_117x6_CPU_cut_counts_and_known_tensor_resource_accounting',
                  'UTC': datetime.now(timezone.utc).isoformat(), 'CPU_time_seconds': time.perf_counter() - began,
                  'source_sha256': sha(__file__), 'protected_source_model_and_input_SHA_before_after': protected,
                  'source_AST_SHA': {'count_method': hashlib.sha256(ast.dump(count_ast, include_attributes=False).encode()).hexdigest(),
                                     'core_functions': hashlib.sha256(ast.dump(ast.Module(body=core_functions, type_ignores=[]), include_attributes=False).encode()).hexdigest()},
                  'actual_model_counts': h.summary(), 'independent_numeric_storage': numeric, 'camera_count': 117,
                  'CPU_count_records': records, 'setting_statistics': statistics, 'CPU_camera_math_uses_pinned_Torch_CPU_only': True,
                  'size_proxy_cached_once_per_view_for_six_settings': True,
                  'native_GPU_cut_counts_verified': False, 'actual_rendered_frames': 0, 'actual_City1404_frames': 0,
                  'full_formalpost_or_paired_merge_completion_claimed': False, 'CUDA_initialized': False,
                  'native_workspace_upper_bound': None, 'actual_GPU_memory_peak_or_fit_guaranteed': False,
                  'compact_leaf_enabled_or_streaming': False,
                  'scope': 'Actual immutable unoptimized e928 tree and canonical117 K/C2W only. Exact frozen CPU predicate yields expected counts, not native GPU selection validation or image-quality/time measurements. Fresh native-workspace/free-device gates still required; post-optimized tree may differ.',
                  'old_sources_reports_models_or_process_state_modified': False}
        save(output / 'report.json', report)
        outcome.update(status=report['status'], report_sha256=sha(output / 'report.json'),
                       ended_utc=report['UTC'], elapsed_seconds=time.perf_counter() - began)
        save(output / 'outcome.json', outcome)
        print(json.dumps({'status': outcome['status'], 'report_sha256': outcome['report_sha256'],
                          'statistics': [{k: row[k] for k in ['setting', 'min_selected_tree_G', 'max_selected_tree_G']} for row in statistics],
                          'CUDA_initialized': False, 'actual_rendered_frames': 0}), flush=True)
    except Exception as error:
        torch_module = sys.modules.get('torch')
        initialized = bool(torch_module is not None and torch_module.cuda.is_initialized())
        outcome.update(status='failed_preserved_unoptimized_City117_CPU_resource_count', error=str(error),
                       traceback=traceback.format_exc(), CUDA_initialized=initialized, elapsed_seconds=time.perf_counter() - began)
        save(output / 'outcome.json', outcome)
        raise


if __name__ == '__main__':
    main()
