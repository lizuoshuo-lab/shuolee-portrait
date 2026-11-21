"""CPU-only actual original-exposure lineage and native anchor-set classification.

Uses immutable official workflow005 CPU helpers. Never infers anchors==leaves,
never changes models/inputs, never uses a pilot post tree as a formal result.
The actual Truck anchor reference validates a source contract, not City progress.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.dont_write_bytecode = True
LAB = Path(__file__).resolve().parents[2]
BASE = LAB / 'outputs/large-scene-gs-001'
LOCK = BASE / 'official-hierarchy/small-city-hierarchy-005'
POST_WRAPPER = LAB / 'scripts/large_scene_gs/run_small_city_post_preflight_001.py'
POST_SHA = '11a68bc78bdf58a521934bf941f17bd0790b60a58f3354b2c458ee0c11b43a75'
CONFIG_SHA = '55c7dd48fb06b07340fd0a49e611fd6d4334dc98253bcc45aa7ac2261b6fdcea'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def set_record(indices):
    values = np.asarray(indices, dtype='<i8')
    return dict(count=len(values), sorted_int64_le_sha256=hashlib.sha256(values.tobytes()).hexdigest())


def anchor_classification(h, anchors):
    """Check source writer's whole-node ranges and descendant closure, not leaf equality."""
    p = len(h.positions)
    require(len(np.unique(anchors)) == len(anchors), 'Duplicate native anchor indices')
    anchored = np.zeros(p, dtype=np.bool_)
    anchored[anchors] = True
    leaf = np.zeros(p, dtype=np.bool_)
    nodes = h.nodes
    for node in nodes:
        start, count = int(node[2]), int(node[3])
        leaf[start:start + count] = True
    require(int(leaf.sum()) == int(nodes[:, 3].sum()), 'Exclusive leaf storage contract differs')
    prefix = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(anchored, dtype=np.int64)))
    starts = nodes[:, 2].astype(np.int64)
    owned_counts = nodes[:, 3].astype(np.int64) + nodes[:, 4].astype(np.int64)
    in_node = prefix[starts + owned_counts] - prefix[starts]
    require(np.all((in_node == 0) | (in_node == owned_counts)), 'Anchors contain only part of an owned node range')
    node_anchored = (owned_counts > 0) & (in_node == owned_counts)
    # A zero-G node has no observable anchor membership. Propagate an anchored
    # positive-G ancestor through it; any positive-G descendant must be anchored.
    stack, visited = [(0, False)], 0
    while stack:
        index, above = stack.pop()
        if above and owned_counts[index] > 0:
            require(bool(node_anchored[index]), 'Anchored node has an unanchored positive-G descendant')
        below = above or bool(node_anchored[index])
        start, count = int(nodes[index, 5]), int(nodes[index, 6])
        stack.extend((j, below) for j in range(start, start + count))
        visited += 1
    require(visited == len(nodes), 'Anchor closure traversal missed nodes')
    a_and_l = np.flatnonzero(anchored & leaf)
    l_minus_a = np.flatnonzero(leaf & ~anchored)
    a_minus_l = np.flatnonzero(anchored & ~leaf)
    return dict(status='actual_native_anchor_node_range_and_descendant_contract_passed',
        gaussian_count=p, node_count=len(nodes), native_anchor_count=len(anchors),
        actual_leaf_count=int(leaf.sum()), anchor_intersection_leaf=set_record(a_and_l),
        leaf_minus_anchor=set_record(l_minus_a), anchor_minus_leaf=set_record(a_minus_l),
        anchors_unique=True, whole_owned_node_ranges=True, anchored_descendant_closure=True,
        anchors_equal_leaves=bool(np.array_equal(anchored, leaf)),
        assumption='Actual classification only. Anchors may contain summary Gaussians; no universal all-leaf freeze claim.'), leaf


def leaf_changes(before, after, leaf):
    require(len(after.positions) == len(before.positions) + 100000, 'Expected one post sky tail')
    result = {}
    for field in ('positions', 'rotations', 'log_scales', 'opacities', 'sh'):
        a, b = getattr(before, field), getattr(after, field)
        changed_rows, max_abs, total = 0, 0., 0
        for start in range(0, len(leaf), 65536):
            end = min(len(leaf), start + 65536)
            selected = leaf[start:end]
            old = np.asarray(a[start:end][selected], dtype=np.float32)
            new = np.asarray(b[start:end][selected], dtype=np.float32)
            require(np.isfinite(old).all() and np.isfinite(new).all(), 'Nonfinite actual leaf parameters')
            if not len(old):
                continue
            unequal = old != new
            changed_rows += int(np.any(unequal.reshape(len(old), -1), axis=1).sum())
            max_abs = max(max_abs, float(np.max(np.abs(old - new))))
            total += len(old)
        result[field] = dict(actual_leaf_count=total, changed_leaf_rows=changed_rows,
                             max_abs_decoded_float32_difference=max_abs, decoded_values_equal=changed_rows == 0)
    return dict(fields=result, all_leaf_decoded_parameters_equal=all(x['decoded_values_equal'] for x in result.values()),
                scope='Actual chunk endpoints only; not equality of consolidated trees, all RGB frames, or final leaf topology.')


def modules():
    require(digest(POST_WRAPPER) == POST_SHA and digest(LOCK / 'configuration.json') == CONFIG_SHA,
            'Fixed CPU wrapper/formal005 configuration changed')
    wrapper = load('_city_external_lineage_fixed_cpu', POST_WRAPPER)
    return wrapper.modules()


def actual_city(driver, codec, helper, training, config, manifest):
    # Reject missing real outputs before decoding any alleged tree. No plan or
    # actual64 pilot is acceptable as an I15000 formal post dependency.
    required = [LOCK / 'trained_chunks' / chunk / name for chunk in config['chunk_runs']
                for name in ('creator-outcome.json', 'post-outcome.json', 'hierarchy.hier',
                             'hierarchy.hier_opt', 'anchors.bin', 'exposure.json')]
    require(all(p.is_file() for p in required), 'Four actual formal005 Creator/post trees and receipts unavailable; refuse City completion')
    coarse, coarse_receipt, _ = driver.completed_training(config['coarse_run'], 'coarse', None, config, manifest, training)
    protected = {str(p): digest(p) for p in required}
    result = []
    for chunk, run in config['chunk_runs'].items():
        saved, chunk_receipt, chunk_config = driver.completed_training(run, 'chunk', chunk, config, manifest, training)
        directory = LOCK / 'trained_chunks' / chunk
        creator, post = read(directory / 'creator-outcome.json'), read(directory / 'post-outcome.json')
        require(creator['status'] == post['status'] == 'completed' and creator['exit_code'] == post['exit_code'] == 0,
                'Formal Creator/post actual exit0 audit incomplete')
        require(creator['source_kind'] == 'unoptimized' and post['source_kind'] == 'post_optimized', 'Wrong real source kind')
        require(creator['configuration_sha256'] == post['configuration_sha256'] == CONFIG_SHA, 'Formal receipt config differs')
        require(creator['source_training_receipt'] == chunk_receipt, 'Original full chunk lineage differs')
        require(creator['input_model']['sha256'] == chunk_receipt['model_parameter_sha256'], 'Creator original model differs')
        require(post['requested_iterations'] == 15000 and post['optimizer_update_calls'] == 14999, 'Formal15000/14999 budget differs')
        expected = driver.post_command(directory, BASE / 'official-inputs' / config['input'] / 'chunks' / chunk, coarse, manifest, 15000)
        require(post['command_argv'] == expected, 'Actual formal post argv differs')
        original_exp = BASE / 'official-training' / run / 'model/exposure.json'
        original_model_audit = BASE / 'official-training' / run / 'model-audit.json'
        protected[str(original_exp)] = digest(original_exp)
        protected[str(original_model_audit)] = digest(original_model_audit)
        exposure = driver.exposure_audit(original_exp, chunk_config['training_names'], manifest['test_names'])
        exposure_hash = exposure['sha256']
        require(exposure_hash == read(original_model_audit)['exposure']['sha256'] == creator['source_exposure']['sha256']
                == creator['copied_exposure_sha256'] == digest(directory / 'exposure.json') == post['frozen_exposure']['sha256'],
                'Original trained exposure→Creator source/copy→post frozen SHA edge differs')
        require(post['exposure_optimizer_updates'] == 0 and post['depth_loss_present_in_unchanged_post'] is False,
                'Official post depth/exposure condition differs')
        require(digest(directory / 'hierarchy.hier') == creator['hierarchy_sha256'] and
                digest(directory / 'hierarchy.hier_opt') == post['hierarchy_sha256'] and
                digest(directory / 'anchors.bin') == creator['anchor_sha256'], 'Formal actual tree/anchor hash differs')
        before = codec.read_hierarchy(directory / 'hierarchy.hier')
        after = codec.read_hierarchy(directory / 'hierarchy.hier_opt', expected_unowned_tail=100000)
        anchors = codec.read_anchors(directory / 'anchors.bin', len(before.positions))
        classification, leaf = anchor_classification(before, anchors)
        freeze = helper.post_freeze_audit(before, after, anchors, coarse, 100000)
        result.append(dict(chunk=chunk, run=run, original_full_training_receipt=chunk_receipt,
            original_exposure=exposure, original_creator_post_exposure_hash_exact=True,
            creator_tree_sha256=creator['hierarchy_sha256'], post_tree_sha256=post['hierarchy_sha256'],
            actual_anchor_classification=classification, actual_anchor_and_one_sky_freeze=freeze,
            actual_pre_post_leaf_parameters=leaf_changes(before, after, leaf)))
    require(all(digest(path) == sha for path, sha in protected.items()), 'Read-only actual source changed during CPU audit')
    return dict(status='completed_actual_four_city_formal_source_exposure_anchor_CPU_ledger',
        chunks=result, original_actual_coarse_model_sha256=coarse_receipt['model_parameter_sha256'],
        protected_sources=protected, all_read_sources_unchanged=True, City_formal_chunk_post_lineage_complete=True,
        City_consolidated_or_1404_quality_complete=False, CUDA_initialized=False, torch_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['reference-truck', 'audit-city'], required=True)
    parser.add_argument('--run', help='New immutable output identifier; use a separate namespace for missing-model controls')
    args = parser.parse_args()
    run = args.run or ('city-anchor-source-contract-truck-reference-CPU-001' if args.stage == 'reference-truck'
                       else 'small-city-formal-post-lineage-CPU-001')
    require(run.replace('-', '').isalnum() and len(run) <= 110, 'Invalid output identifier')
    output = BASE / 'independent-evaluation' / run
    output.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    write(output / 'outcome.json', dict(status='running_CPU_only', started_utc=start, stage=args.stage, CUDA_initialized=False))
    try:
        driver, codec, helper, training, config, manifest = modules()
        if args.stage == 'reference-truck':
            source = BASE / 'official-hierarchy/truck-official-hierarchy-001/trained_chunks/0_0'
            paths = [source / n for n in ('creator-outcome.json', 'hierarchy.hier', 'anchors.bin')]
            protected = {str(p): digest(p) for p in paths}
            creator = read(source / 'creator-outcome.json')
            require(creator['status'] == 'completed' and creator['exit_code'] == 0, 'Actual Truck native Creator receipt incomplete')
            require(creator['hierarchy_sha256'] == digest(source / 'hierarchy.hier') and
                    creator['anchor_sha256'] == digest(source / 'anchors.bin'), 'Actual reference tree/anchors differ')
            h = codec.read_hierarchy(source / 'hierarchy.hier')
            anchors = codec.read_anchors(source / 'anchors.bin', len(h.positions))
            classification, _ = anchor_classification(h, anchors)
            require(all(digest(path) == sha for path, sha in protected.items()), 'Historical Truck sources changed')
            report = dict(status='completed_actual_Truck_native_anchor_contract_reference_only',
                classification=classification, protected_Truck_sources=protected, protected_sources_unchanged=True,
                City_Creator_or_post_completed=False, CUDA_initialized=False, torch_imported=False)
        else:
            report = actual_city(driver, codec, helper, training, config, manifest)
        require('torch' not in sys.modules, 'CPU ledger must not import torch')
        report.update(started_utc=start, completed_utc=datetime.now(timezone.utc).isoformat(),
            source_script_sha256=digest(Path(__file__)), fixed_formal005_configuration_sha256=CONFIG_SHA)
        write(output / 'report.json', report)
        shutil.copyfile(Path(__file__), output / 'source-snapshot.py')
        write(output / 'outcome.json', dict(status=report['status'], started_utc=start, exit_code=0, CUDA_initialized=False,
            report_sha256=digest(output / 'report.json'), completed_utc=report['completed_utc']))
        print(json.dumps(dict(status=report['status'], output=str(output), CUDA_initialized=False)))
    except Exception as error:
        write(output / 'outcome.json', dict(status='failed_preserved_actual_CPU_ledger', started_utc=start, stage=args.stage,
            error=str(error), exit_code=1, CUDA_initialized=False, ended_utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    main()
