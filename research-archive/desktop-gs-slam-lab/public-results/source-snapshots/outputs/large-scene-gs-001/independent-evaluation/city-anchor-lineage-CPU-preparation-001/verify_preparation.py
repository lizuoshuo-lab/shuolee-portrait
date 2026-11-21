from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

LAB = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE = LAB / 'outputs/large-scene-gs-001'
HERE = Path(__file__).resolve().parent
SOURCE = LAB / 'scripts/large_scene_gs/audit_city_hierarchy_lineage_001.py'
spec = importlib.util.spec_from_file_location('_root_cpu_lineage_integrity', SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert not (HERE / 'CPU-verification.json').exists()
assert module.digest(SOURCE) == 'f52e19946b0df7efb6a4ad69bcfdbc844979ad76fe9e530761c1f0e292c1ed2b'
driver, codec, helper, training, config, manifest = module.modules()
tree = BASE / 'official-hierarchy/truck-official-hierarchy-001/trained_chunks/0_0'
protected = {str(tree / n):module.digest(tree / n) for n in ('hierarchy.hier', 'anchors.bin', 'creator-outcome.json')}
h = codec.read_hierarchy(tree / 'hierarchy.hier')
anchors = codec.read_anchors(tree / 'anchors.bin', len(h.positions))
original, _ = module.anchor_classification(h, anchors)
controls = []
for name, altered in [('duplicate_actual_anchor', np.concatenate((anchors, anchors[:1]))),
                       ('anchor_actual_root_without_its_internal_descendants', np.concatenate((anchors,
                          np.arange(int(h.nodes[0,2]), int(h.nodes[0,2]+h.nodes[0,3]+h.nodes[0,4]), dtype=np.int32))))]:
    try:
        module.anchor_classification(h, altered)
    except RuntimeError as e:
        controls.append(dict(control=name, rejected=True, message=str(e), in_memory_only=True, original_tree_unchanged=True))
    else:
        raise RuntimeError('Corrupted native source contract was accepted: ' + name)
missing = module.read(BASE / 'independent-evaluation/small-city-formal-post-lineage-missing-model-control-CPU-001/outcome.json')
assert missing['status'] == 'failed_preserved_actual_CPU_ledger' and missing['exit_code'] == 1
assert 'Four actual formal005 Creator/post trees' in missing['error'] and missing['CUDA_initialized'] is False
assert all(module.digest(p) == sha for p,sha in protected.items())
assert 'torch' not in sys.modules
report = dict(status='CPU_preparation_actual_native_reference_and_integrity_refusals_passed',
    observed_utc=datetime.now(timezone.utc).isoformat(), source_sha256=module.digest(SOURCE),
    actual_native_Truck_reference=original, actual_missing_City_dependency_control=missing,
    actual_anchor_corruption_controls=controls, protected_Truck_sources_unchanged=True,
    formal005_source_config_unchanged=module.digest(module.LOCK / 'configuration.json') == module.CONFIG_SHA,
    actual_City_Creator_or_post_ledger_complete=False, CUDA_initialized=False, torch_imported=False,
    future_actual_command=[str(LAB / '.venv-large-gs/Scripts/python.exe'), '-X', 'utf8', str(SOURCE),
                           '--stage', 'audit-city', '--run', 'small-city-formal-post-lineage-CPU-001'])
module.write(HERE / 'CPU-verification.json', report)
print(json.dumps(dict(status=report['status'], controls=len(controls), actualCityLedgerComplete=False)))
