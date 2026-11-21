"""Prepare an independent 006 workflow from the immutable 005 snapshot; never execute it."""
from pathlib import Path
import hashlib

ROOT = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
old = ROOT / 'outputs/large-scene-gs-001/official-hierarchy/small-city-hierarchy-005/driver-source.py'
assert hashlib.sha256(old.read_bytes()).hexdigest() == '0e51b5c51d0742a20315d02c37355aaae7e9f1d905069a1f6594da0ee06d7f84'
s = old.read_text(encoding='utf-8')
s = s.replace('ROOT = Path(__file__).resolve().parents[2]', "ROOT = Path(r'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab')")
s = s.replace('DRIVER_VERSION = 4', 'DRIVER_VERSION = 5')
start = s.index('def completed_training(')
end = s.index('\ndef exposure_audit(', start)
s = s[:start] + '''BINDING_PATH = OUTPUT / 'official-hierarchy/small-city-hierarchy-006/binding-audits/lineage006-complete-cpu-001/lineage-binding.json'
BINDING_SHA = '1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c'
BINDER_PATH = ROOT / 'scripts/large_scene_gs/bind_small_city_lineage_006.py'
BINDER_SNAPSHOT = BINDING_PATH.parent / 'binder-source.py'
BINDER_SHA = 'c72fea79bb3b083ead1dfa58e9cf7e7d0134842d764dd3c5482162448dc4db9e'
FIXED_RUN = 'small-city-hierarchy-006'
ORDER = ('0_0', '0_1', '1_0', '1_1')


def fixed_sources():
    """CPU-only immutable mixed-lineage contract; never authorizes CUDA."""
    for p, expected in [(BINDING_PATH, BINDING_SHA), (BINDER_PATH, BINDER_SHA), (BINDER_SNAPSHOT, BINDER_SHA)]:
        if digest(p) != expected:
            raise RuntimeError('Actual006 binder source or lineage receipt changed')
    b = read(BINDING_PATH)
    if (b['status'] != 'actual_complete_mixed_local_training_lineage_CPU_bound'
            or b['formal_observation_reclassification'] is not False
            or b['training_lineage_kind'] != 'three_legacy_local_plus_one_observed_local_official_training'
            or [r['chunk'] for r in b['ordered_chunks']] != list(ORDER)
            or b['ordered_chunks'][-1]['training_source_kind'] != 'observed_local_official_training'
            or b['ordered_chunks'][-1]['formal_training_run'] is not False):
        raise RuntimeError('006 actual mixed training origin contract differs')
    binder = load_module('_smallcity006_actual_CPU_binder', BINDER_SNAPSHOT)
    binder.ROOT, binder.BASE = ROOT, OUTPUT
    # This module is loaded from its audit snapshot, so restore all actual path globals.
    binder.LOCK = OUTPUT / 'official-hierarchy/small-city-hierarchy-006'
    binder.OLD = OUTPUT / 'official-hierarchy/small-city-hierarchy-005'
    binder.INPUT = OUTPUT / 'official-inputs/small-city-training-002'
    binder.OBS = OUTPUT / 'official-training-observation' / binder.OBS_RUN
    binder.SUP = OUTPUT / 'official-training-observation-supervision' / binder.OBS_RUN
    binder.PLAN = OUTPUT / 'official-training-observation-plans' / (binder.OBS_RUN + '-v3')
    binder.COARSE = OUTPUT / 'official-training' / binder.COARSE_RUN
    binder.LEGACY = {c: OUTPUT / 'official-training' / ('small-city-chunk' + c.replace('_', '') + '-i30000-r4-003') for c in ORDER[:3]}
    if binder.sources() != b['source_pins']:
        raise RuntimeError('Current actual006 source pins differ from bound bytes')
    proofs = {}
    _, contract = binder.observed_terminal(proofs)
    binder.observed_sources(contract)
    if proofs['observed_terminal_accounting'] != b['observed_terminal']:
        raise RuntimeError('Observed actual save/step/skip accounting changed')
    return binder, b


def lineage_gate(config, manifest, training, codec):
    """Recheck all five saved parameter arrays, raw finite/activations/Sky/exposures/source."""
    binder, binding = fixed_sources()
    if (config['training_lineage_binding_path'] != str(BINDING_PATH)
            or config['training_lineage_binding_sha256'] != BINDING_SHA
            or config['training_lineage_kind'] != binding['training_lineage_kind']
            or config['input_manifest_sha256'] != binder.PINS['manifest']):
        raise RuntimeError('006 configuration lost actual mixed training binding')
    # Exact pre-existing model receipt identities are independently recomputed rather than trusted.
    refs = [binding['coarse']] + binding['ordered_chunks']
    actual = []
    workflow = sys.modules[__name__]
    for ref in refs:
        directory = Path(ref['training_directory'])
        item = binder.bind_one(directory, ref['stage'], ref['chunk'], manifest, workflow, training, codec,
                               observed=ref['training_source_kind'] == 'observed_local_official_training')
        if item != ref:
            raise RuntimeError('Actual saved model/source/config/finite/Sky/exposure differs: ' + str(ref['chunk']))
        actual.append(item)
    input_split_audit(binder.INPUT, manifest)
    binder.fresh_terminal()  # CPU actual process observation, no CUDA device query.
    return {'coarse': actual[0], 'ordered_chunks': actual[1:],
            'training_lineage_binding_path': str(BINDING_PATH), 'training_lineage_binding_sha256': BINDING_SHA,
            'all_five_saved_model_bindings_rechecked': True, 'GPU_started': False}


def completed_training(run_name, stage, chunk, config, manifest, training):
    """Compatibility interface with explicit origin, not formal-status relabelling."""
    _, binding = fixed_sources()
    ref = binding['coarse'] if stage == 'coarse' else next(r for r in binding['ordered_chunks'] if r['chunk'] == chunk)
    directory = Path(ref['training_directory'])
    if directory.name != run_name or stage != ref['stage']:
        raise RuntimeError('006 run mapping differs from immutable actual lineage')
    for p, expected in [(directory / 'outcome.json', ref['training_receipt']['sha256']),
                        (directory / 'configuration.json', ref['configuration']['sha256']),
                        (directory / 'model-audit.json', ref['model_audit_sha256']),
                        (Path(ref['model']['path']), ref['model']['sha256'])]:
        if digest(p) != expected:
            raise RuntimeError('Bound training model or receipt changed')
    return Path(ref['model']['path']).parent, read(directory / 'outcome.json'), read(directory / 'configuration.json')

''' + s[end:]
s = s.replace("default='small-city-hierarchy-002'", "default=FIXED_RUN")
s = s.replace("default='small-city-coarse-i30000-r4-002'", "default='small-city-coarse-i30000-r4-003'")
s = s.replace("    if args.post_iterations < 2:\n        raise ValueError('At least two post iterations')", "    if args.post_iterations != 15000:\n        raise ValueError('006 formal post budget is exactly15000; short64 uses independent post006 namespace')\n    if args.run != FIXED_RUN or args.input != 'small-city-training-002':\n        raise RuntimeError('006 driver is locked to its independent006 namespace and002 input')")
s = s.replace("    inputs = OUTPUT / 'official-inputs' / args.input", "    _, actual_binding = fixed_sources()\n    expected_mappings = {r['chunk']: Path(r['training_directory']).name for r in actual_binding['ordered_chunks']}\n    if mappings != expected_mappings or args.coarse_run != Path(actual_binding['coarse']['training_directory']).name:\n        raise RuntimeError('006 CLI mappings differ from actual completed mixed lineage')\n    inputs = OUTPUT / 'official-inputs' / args.input")
s = s.replace("    fx = camera_archives[next(iter(chunks))]['original_fx_fy'][0] / 4", "    if list(chunks) != list(ORDER):\n        raise RuntimeError('Original chunk order changed')\n    fx = camera_archives[next(iter(chunks))]['original_fx_fy'][0] / 4")
s = s.replace("              'official_main_commit': COMMIT", "              'training_lineage_binding_path': str(BINDING_PATH), 'training_lineage_binding_sha256': BINDING_SHA,\n              'training_lineage_kind': actual_binding['training_lineage_kind'], 'formal_observation_reclassification': False,\n              'training_origin_records': actual_binding['ordered_chunks'],\n              'post_optimizer_call_accounting': '14999 derived from unchanged unconditional post step before finalI15000 save/return; not instrumented per-Gaussian nonzero updates',\n              'official_main_commit': COMMIT")
s = s.replace("    write(config_path, config)\n    shutil.copyfile", "    lineage = lineage_gate(config, manifest, training, codec)\n    write(config_path, config)\n    shutil.copyfile")
s = s.replace("Path('<future-completed-local-coarse>')", "Path(actual_binding['coarse']['model']['path']).parent")
s = s.replace("'policy': 'Future complete local>=30000,r4 runs with same manifest; neither preflight nor author pretrained'", "'policy': 'Actual3legacy+1observed complete finite local30k,r4 models; origin remains distinct, no formal relabelling', 'all_five_actual_bindings': lineage")
s = s.replace("'validation_scope': 'CPU source/hash, five actual sparse/0/test.txt splits and official CPU Reader receipt, exact train-only pose archives, CLI parse; runtime/model dependencies pending, no native Creator/post/Merger executed'", "'validation_scope': 'CPU actual all5 saved model/source/finite/Sky/exposure binding, five splits, train-only camera archives and CLI; no native Creator/post/Merger executed',\n                'training_lineage_binding_path': str(BINDING_PATH), 'training_lineage_binding_sha256': BINDING_SHA,\n                'creator_command_previews': {c: [str(EXES / 'GaussianHierarchyCreator.exe'), str(Path(r['model']['path']).parent / 'point_cloud.ply'), str(root / 'creator-camera-archives' / c), str(root / 'trained_chunks' / c), str(Path(actual_binding['coarse']['model']['path']).parent)] for c,r in zip(ORDER, actual_binding['ordered_chunks'])}")
s = s.replace("'actual_trained_dependencies_present': False", "'actual_trained_dependencies_present': True, 'all_five_bindings_rechecked': True, 'training_lineage_binding_sha256': BINDING_SHA")
s = s.replace("        exposure = OUTPUT / 'official-training' / mappings[args.chunk] / 'model/exposure.json'", "        training_origin = next(r for r in actual_binding['ordered_chunks'] if r['chunk'] == args.chunk)\n        exposure = Path(training_origin['exposure']['path'])")
s = s.replace("source_training_receipt=training_receipt, source_exposure=exposures,", "source_training_receipt=training_receipt, source_exposure=exposures,\n                       training_origin=training_origin, training_source_kind=training_origin['training_source_kind'],\n                       formal_observation_reclassification=False,")
s = s.replace("                       optimizer_update_calls=args.post_iterations-1", "                       source_training_receipt=creator['source_training_receipt'], input_model=creator['input_model'],\n                       training_origin=creator['training_origin'], training_source_kind=creator['training_source_kind'],\n                       formal_observation_reclassification=False, optimizer_call_count_source_derived=True,\n                       optimizer_update_calls=args.post_iterations-1")
s = s.replace("                       configuration_sha256=digest(config_path))", "                       configuration_sha256=digest(config_path),\n                       training_lineage_binding_path=str(BINDING_PATH), training_lineage_binding_sha256=BINDING_SHA)")
s = s.replace("numeric_audit=helper.hierarchy_numeric_audit(h), configuration_sha256=digest(config_path))", "numeric_audit=helper.hierarchy_numeric_audit(h), configuration_sha256=digest(config_path),\n                       training_lineage_binding_path=str(BINDING_PATH), training_lineage_binding_sha256=BINDING_SHA)")
target = ROOT / 'scripts/large_scene_gs/run_small_city_hierarchy_006.py'
assert not target.exists(), 'Preserve existing006 source if any'
compile(s, str(target), 'exec')
target.write_text(s, encoding='utf-8')
print(target)
print(hashlib.sha256(target.read_bytes()).hexdigest())
