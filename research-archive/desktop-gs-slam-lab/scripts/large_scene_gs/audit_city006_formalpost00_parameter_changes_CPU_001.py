"""Read-only endpoint changes on the actual original post00 tree, excluding Sky.

The prior independently completed physical audit and measured Creator A/L ledger
are immutable dependencies. This supplement counts stored endpoint differences;
it does not count optimizer calls or infer geometry/image-quality improvement.
"""
from pathlib import Path
import argparse, hashlib, json, struct, sys, traceback
from datetime import datetime, timezone
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE = ROOT/'outputs/large-scene-gs-001'
LOCK = BASE/'official-hierarchy/small-city-hierarchy-006'
DIR = LOCK/'trained_chunks/0_0'
PHYSICAL = BASE/'recovery-audit/city006-first-formalpost00-independent-completion-CPU-002/completion-audit.json'
LEDGER = BASE/'independent-evaluation/small-city006-all4-creator-anchor-source-exposure-CPU-001/report.json'
BEFORE_SHA = '9b3e8ad0901601410e7e73d902cdb448051258d582e41574f6b0fd98f7a30cd7'
AFTER_SHA = 'a387578911cc4d630e1105659010de4af7bd7b7add3d21be483e73243cd712a6'
FIELDS = ('positions', 'rotations', 'log_scales', 'opacities', 'sh')
UNITS = {'positions': 'native world coordinates, unchanged original scene scale',
         'rotations': 'raw stored quaternion components; no renormalization or angular-distance interpretation',
         'log_scales': 'natural logarithms of scale; raw stored coordinates, not scale differences',
         'opacities': 'hierarchical stored abs-kappa activation, unbounded above; not sigmoid probability',
         'sh': 'stored spherical-harmonic coefficient components, native color basis'}

def require(ok, message):
    if not ok: raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def map_fields(path, expected_tail):
    """Narrow mode=r writer-schema map; prior dd061 validates full tree/numerics.

    Offsets are identical to frozen codec583c/ledgeraa4d compressed schema.
    Only header and total file size are checked here; no repeated tree/all5 audit.
    """
    with path.open('rb') as f:
        signed = struct.unpack('<i', f.read(4))[0]
        require(signed < 0, 'Actual compressed writer required')
        count = -signed
        require(count > expected_tail >= 0, 'Explicit tree/Sky tail bounds')
        offset = 4+124*count
        f.seek(offset)
        nodes = struct.unpack('<i', f.read(4))[0]
    require(nodes > 0 and path.stat().st_size == offset+4+36*nodes, 'Actual compressed payload length')
    schema = [('positions', '<f4', 4, (count, 3)),
              ('rotations', '<f2', 4+12*count, (count, 4)),
              ('log_scales', '<f2', 4+20*count, (count, 3)),
              ('opacities', '<f2', 4+26*count, (count,)),
              ('sh', '<f2', 4+28*count, (count, 16, 3))]
    return count, nodes, {name: np.memmap(path, dtype=dtype, mode='r', offset=off, shape=shape)
                          for name, dtype, off, shape in schema}

def measured_set(ids, record, count):
    ids = np.asarray(ids, dtype='<i8')
    require(ids.ndim == 1 and len(ids) == record['count'], 'Actual A/L count differs from measured ledger')
    require(not len(ids) or (0 <= ids[0] and ids[-1] < count and np.all(ids[1:] > ids[:-1])), 'A/L IDs not sorted/unique/in original tree')
    require(hashlib.sha256(ids.tobytes()).hexdigest() == record['sorted_int64_le_sha256'], 'Actual A/L ID hash differs')
    mask = np.zeros(count, np.bool_); mask[ids] = True
    return mask

def changes(before, after, group_masks, count):
    records = {name: {'rows': int(mask.sum()), 'fields': {}} for name, mask in group_masks.items()}
    any_value_change = np.zeros(count, np.bool_)
    any_bit_change = np.zeros(count, np.bool_)
    for field in FIELDS:
        old, new = before[field], after[field]
        width = int(np.prod(old.shape[1:])) if old.ndim > 1 else 1
        field_records = {name: {'total_rows': record['rows'], 'total_scalar_values': record['rows']*width,
                               'changed_rows': 0, 'changed_scalar_values': 0, 'max_abs_decoded_difference': 0.0,
                               'float32_bit_changed_rows': 0, 'float32_bit_changed_scalar_values': 0,
                               'first_changed_model_local_IDs': []}
                         for name, record in records.items()}
        for start in range(0, count, 65536):
            end = min(count, start+65536)  # Both endpoints and masks stop before appended Sky.
            x = np.ascontiguousarray(old[start:end], dtype=np.float32).reshape(end-start, width)
            y = np.ascontiguousarray(new[start:end], dtype=np.float32).reshape(end-start, width)
            require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite compared endpoint '+field)
            different = x != y
            bit_different = x.view(np.uint32) != y.view(np.uint32)
            row_different = different.any(axis=1); row_bit_different = bit_different.any(axis=1)
            any_value_change[start:end] |= row_different
            any_bit_change[start:end] |= row_bit_different
            delta = np.abs(x.astype(np.float64)-y.astype(np.float64))
            for name, mask in group_masks.items():
                selected = mask[start:end]; stats = field_records[name]
                if not selected.any(): continue
                stats['changed_rows'] += int(row_different[selected].sum())
                stats['changed_scalar_values'] += int(different[selected].sum())
                stats['float32_bit_changed_rows'] += int(row_bit_different[selected].sum())
                stats['float32_bit_changed_scalar_values'] += int(bit_different[selected].sum())
                stats['max_abs_decoded_difference'] = max(stats['max_abs_decoded_difference'], float(delta[selected].max()))
                need = 16-len(stats['first_changed_model_local_IDs'])
                if need > 0:
                    stats['first_changed_model_local_IDs'].extend((np.flatnonzero(selected & row_different)[:need]+start).tolist())
        for name, record in records.items():
            record['fields'][field] = field_records[name]
    for name, mask in group_masks.items():
        records[name]['changed_rows_in_any_of_five_fields'] = int(any_value_change[mask].sum())
        records[name]['float32_bit_changed_rows_in_any_of_five_fields'] = int(any_bit_change[mask].sum())
    return records

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--audit-run', required=True); args = p.parse_args()
    require(args.audit_run.replace('-', '').isalnum() and len(args.audit_run) <= 110, 'Unique safe namespace required')
    out = BASE/'recovery-audit'/args.audit_run; out.mkdir(parents=True, exist_ok=False)
    (out/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    try:
        pins = {PHYSICAL: 'dd0611011d19665a44bbd9a8aebe9817951246a9614e7fbfab46c2ec6ab7d462',
                LEDGER: '17be8b334c09f3dfbe91ba5e56d0e8107b674b379bc8977dc7447c12d572f774',
                DIR/'hierarchy.hier': BEFORE_SHA, DIR/'hierarchy.hier_opt': AFTER_SHA,
                DIR/'creator-outcome.json': '7920acf67814ab13c2525344defe8d20357e11538ae3e9900b1d8e9d5d4f9853',
                DIR/'post-outcome.json': '37f2b3826e7210451f5d64ac00790da9578c4da9c96476e9a884409f2e5169c5',
                LOCK/'helper-snapshots/read_hierarchy.py': '583c96f50b6f486d73bbbbaf9faf053ed15d486a7c7b48e15ef8e3ebc0964538'}
        physical = read(PHYSICAL); ledger = read(LEDGER)
        classification = ledger['records'][0]['actual_anchor_classification']
        npz_path = Path(classification['actual_set_npz_path'])
        pins[npz_path] = classification['actual_set_npz_sha256']
        pins[Path(__file__)] = sha(__file__)
        protected = {str(path): digest for path, digest in pins.items()}
        for path, digest in pins.items(): require(sha(path) == digest, 'Immutable evidence changed '+str(path))
        require(physical['status'] == 'actual_first15000_formalpost00_physical_source_and_owned_terminal_CPU_audit_passed', 'Prior actual full15000 physical audit missing')
        require(physical['protected_source_and_output_before_after'][str(DIR/'hierarchy.hier')] == BEFORE_SHA and
                physical['protected_source_and_output_before_after'][str(DIR/'hierarchy.hier_opt')] == AFTER_SHA, 'Actual audited endpoint identity differs')
        creator, post = read(DIR/'creator-outcome.json'), read(DIR/'post-outcome.json')
        require(creator['status'] == post['status'] == 'completed' and creator['exit_code'] == post['exit_code'] == post['actual_child_poll_exit_code'] == 0,
                'Actual full source/final terminal receipts required')
        require(post['requested_iterations'] == 15000 and post['optimizer_call_count_source_derived'] is True and post['optimizer_update_calls'] == 14999,
                'Full15000 original official source-derived14999 budget required')
        require(post['source_kind'] == 'post_optimized' and post['training_origin'] == creator['training_origin'] and post['formal_observation_reclassification'] is False,
                'Selected origin/reclassification changed')
        count, nodes, before = map_fields(DIR/'hierarchy.hier', 0)
        after_count, after_nodes, after = map_fields(DIR/'hierarchy.hier_opt', 100000)
        require(count == classification['G'] == 4332649 and nodes == classification['nodes'] and after_nodes == nodes and after_count == count+100000,
                'Original tree count/exact post single Sky contract differs')
        with np.load(npz_path, allow_pickle=False) as sets:
            anchored = measured_set(sets['anchors_sorted'], classification['A'], count)
            leaf = measured_set(sets['leaves_sorted'], classification['L'], count)
            for key, label in [('anchor_intersection_leaf', 'A_intersection_L'), ('leaf_minus_anchor', 'L_minus_A'), ('anchor_minus_leaf', 'A_minus_L')]:
                measured_set(sets[key], classification[label], count)
        masks = {'A_all': anchored, 'L_all': leaf, 'anchor_leaf_A_intersection_L': anchored & leaf,
                 'anchor_internal_A_minus_L': anchored & ~leaf, 'nonAnchor_leaf_L_minus_A': ~anchored & leaf,
                 'nonAnchor_internal': ~anchored & ~leaf}
        require(int(masks['anchor_internal_A_minus_L'].sum()) == 451 and int(masks['nonAnchor_leaf_L_minus_A'].sum()) == 0 and
                int(masks['nonAnchor_internal'].sum()) == 1354085, 'Measured storage classification differs')
        records = changes(before, after, masks, count)
        require(records['A_all']['changed_rows_in_any_of_five_fields'] == 0, 'Actual frozen anchors changed')
        for path, digest in pins.items(): require(sha(path) == digest, 'Immutable evidence changed during read '+str(path))
        require('torch' not in sys.modules, 'Torch imported by CPU endpoint audit')
        result = {'status': 'actual_post00_full15000_original_tree_parameter_endpoint_changes_measured_CPU',
                  'UTC': datetime.now(timezone.utc).isoformat(), 'chunk': '0_0', 'before_sha256': BEFORE_SHA, 'after_sha256': AFTER_SHA,
                  'original_tree_G': count, 'after_G_including_Sky': after_count, 'excluded_shared_Sky_rows': 100000,
                  'original_tree_node_count': nodes, 'measured_A_count': int(anchored.sum()), 'measured_L_count': int(leaf.sum()),
                  'group_statistics': records, 'field_coordinate_scope': UNITS,
                  'groups_overlap_scope': 'A_all and L_all are aggregate views; the remaining four A/L intersections partition original tree IDs. Do not sum overlapping groups.',
                  'evidence_scope': 'Raw decoded float32 stored endpoint differences after native compressed half expansion; no sigmoid/exp/quaternion normalization applied. Signed-zero bit differences are measured separately.',
                  'limits': 'Changes show endpoint parameters differ after post, not which iteration updated them, number of optimizer calls, transient/sub-storage-ULP/cancelled updates, geometry or image-quality improvement. Original tree only; appended100kSky excluded.',
                  'full_budget': {'requested_iterations': 15000, 'Gaussian_call_count_source_derived': 14999, 'Gaussian_call_count_instrumented': False},
                  'training_lineage_kind': physical['training_lineage_kind'], 'observation_reclassified_formal': False,
                  'protected_files_before_and_after': protected, 'all_read_bytes_unchanged': True,
                  'broad_all5_or_physical_numeric_audit_repeated': False, 'Torch_imported': False, 'GPU_control_or_process_query_performed': False,
                  'global_GPU_free_claimed': False, 'old_sources_or_outputs_modified': False}
        save(out/'parameter-change-audit.json', result)
        print(json.dumps({'status': result['status'], 'report': str(out/'parameter-change-audit.json'), 'SHA': sha(out/'parameter-change-audit.json'),
                          'nonAnchor_internal_changed_rows': records['nonAnchor_internal']['changed_rows_in_any_of_five_fields'],
                          'frozen_A_changed_rows': records['A_all']['changed_rows_in_any_of_five_fields']}))
    except Exception as error:
        save(out/'audit-failure.json', {'status': 'CPU_endpoint_audit_failed_not_training_failure', 'error_type': type(error).__name__,
                                      'error': str(error), 'traceback': traceback.format_exc(), 'GPU_or_process_control': False})
        raise

if __name__ == '__main__': main()
