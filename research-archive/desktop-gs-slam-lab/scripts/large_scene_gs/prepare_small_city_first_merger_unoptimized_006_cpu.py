"""CPU-only actual four-Creator input/source plan for the first unoptimized Merger.

This is not a dispatch gate. A current authorized post may be live. No native
execution, global GPU-empty claim, retry, model/output write, or source change.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE = ROOT / 'outputs/large-scene-gs-001'
OUT = BASE / 'recovery-audit/stage006-first-merger-unoptimized-CPU-001'
ENTRY = ROOT / 'scripts/large_scene_gs/006-supervision-revision-002/supervise_small_city_hierarchy_stage_006.py'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    entry = load('_first_unoptimized_merger_actual_revision002_CPU', ENTRY)
    supervisor, runtime = entry.configure()
    driver, config, binding, manifest, _ = supervisor.fixed_sources('merge-unoptimized')
    driver.execution_source_gate()  # Source bytes only; never the current global GPU-owner gate.
    helper, codec, training, _ = driver.frozen_helpers(supervisor.LOCK)
    archives = supervisor.read(supervisor.LOCK / 'plan.json')['camera_archives']
    rows = []
    for chunk in supervisor.CHUNKS:
        directory = supervisor.LOCK / 'trained_chunks' / chunk
        receipt = driver.creator_gate(chunk, config, manifest, training, codec, helper, archives)
        model = directory / 'hierarchy.hier'
        h = codec.read_hierarchy(model)
        rows.append({'chunk':chunk, 'source_kind':'unoptimized', 'training_source_kind':receipt['training_source_kind'],
            'formal_training_run':receipt['training_origin']['formal_training_run'],
            'actual_training_parameter_sha256':receipt['input_model']['sha256'], 'path':str(model), 'sha256':sha(model),
            'actual_storage_bytes':model.stat().st_size, 'actual_decoded_summary':h.summary(),
            'actual_creator_receipt_path':str(directory / 'creator-outcome.json'), 'actual_creator_receipt_sha256':sha(directory / 'creator-outcome.json'),
            'actual_creator_native_birth':receipt['actual_child_birth'], 'actual_creator_handle_poll_exit_code':receipt['actual_child_poll_exit_code'],
            'anchor_count':receipt['anchor_count'], 'exposure_sha256':receipt['copied_exposure_sha256'],
            'center':next(c['center'] for c in manifest['chunks'] if c['name'] == chunk), 'numeric_audit':receipt['numeric_audit']})
    assert rows[-1]['actual_training_parameter_sha256'] == '367bef34e584616daeda938e2e0b5826a8e4bbf30d13b4d987db32c9f72abcb4'
    assert rows[-1]['training_source_kind'] == 'observed_local_official_training' and rows[-1]['formal_training_run'] is False
    variant_path = BASE / 'hierarchy-build/merger-unoptimized/manifest.json'
    assert sha(variant_path) == config['unoptimized_filename_variant_manifest_sha256']
    variant = supervisor.read(variant_path)
    exe = Path(variant['executable']['path'])
    assert sha(exe) == variant['executable']['sha256']
    destination = supervisor.LOCK / 'merged-unoptimized.hier'
    for path in [destination, supervisor.LOCK / 'merge-unoptimized-outcome.json', supervisor.LOCK / 'merge-unoptimized-stage']:
        assert not path.exists(), 'Actual Merger target exists; no overwrite/retry'
    inputs = BASE / 'official-inputs/small-city-training-002'
    argv = [str(exe), str(supervisor.LOCK / 'trained_chunks'), '0', str(inputs / 'chunks'), str(destination), *supervisor.CHUNKS]
    snapshot = supervisor.process_snapshot()
    gaussians = sum(r['actual_decoded_summary']['gaussian_count'] for r in rows)
    nodes = sum(r['actual_decoded_summary']['node_count'] for r in rows)
    result = {'status':'CPU_actual_all4_original_Creator_inputs_and_filename_only_merger_source_prepared_not_dispatched',
        'observed_utc':supervisor.now(), 'preparation_source_sha256':sha(__file__), 'runtime_controller_identity':runtime,
        'fixed_native_driver_sha256':supervisor.DRIVER_SHA, 'fixed_configuration_sha256':supervisor.CONFIG_SHA,
        'fixed_CPU_plan_sha256':supervisor.PLAN_SHA, 'training_lineage_binding_sha256':supervisor.BINDING_SHA,
        'actual_four_original_Creator_inputs':rows, 'actual_filename_only_variant_manifest_path':str(variant_path),
        'actual_filename_only_variant_manifest_sha256':sha(variant_path), 'actual_merger_executable_sha256':sha(exe),
        'native_merger_command_argv':argv, 'chunk_order':list(supervisor.CHUNKS), 'source_kind':'unoptimized',
        'ownership':config['merge_ownership'], 'formalpost_or_pilot_tree_accepted_as_input':False,
        'future_output_policy':'Actual merged-unoptimized.hier, real native exit0 receipt, decoded counts/ownership/tail0; append own100k coarse sky exactly once only during future rendering.',
        'input_storage_and_decoded_baseline':{'actual_compressed_input_bytes':sum(r['actual_storage_bytes'] for r in rows),
            'actual_sum_input_Gaussians':gaussians,'actual_sum_input_nodes':nodes,
            'input_float32_Gaussian_fields_bytes':gaussians * 59 * 4,'input_int32_node_fields_bytes':nodes * 7 * 4,
            'input_float32_box_fields_bytes':nodes * 6 * 4,
            'scope':'Input representation only; no assumption of merged counts, full GPU residency, transform/interpolation/screen/native unknown workspaces, or streaming.'},
        'system_drive_free_bytes':shutil.disk_usage(ROOT).free, 'actual_process_snapshot':snapshot,
        'observed_current_GPU_stage_processes':supervisor.busy_processes(snapshot),
        'global_GPU_empty_claimed':False, 'current_formalpost00_may_be_live':True,
        'mandatory_root_future_dispatch_gates':['Actual fullpost00 terminal with matching independent physical audit and source provenance',
            'Fresh same source/all4 original input SHA and current 80GiB reserve','Fresh actual GPU-stage/observed descendant release and unique atomic006 ownership lock',
            'One explicit root merge-unoptimized stage only; no retry or automatic next stage'],
        'paired_fullpost_merge_and_117x12_evaluation_remain_required':True,'GPU_started':False,'native_executed':False,'next_stage_dispatched':False}
    path = OUT / 'merger-input-source-plan.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'path':str(path),'sha256':sha(path),'native_merger_command_argv':argv},ensure_ascii=False))


if __name__ == '__main__':
    main()
