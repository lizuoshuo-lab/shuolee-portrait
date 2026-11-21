"""Thin CPU-supervision revision over immutable006; only decision change is ISO parsing.

The baseline source SHA remains factual. This actual entry, parser, and their
source seal are recorded separately in every runtime receipt and active lock.
Native/training algorithms,006 sources/config/results, budgets and dispatch
policy are unchanged. Default CPU plan/check; one GPU stage needs --execute.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE = ROOT / 'outputs/large-scene-gs-001'
BASELINE = ROOT / 'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_006.py'
BASELINE_SHA = '3db6b7ee81d0b7df592609509f8dd565e146179d399f4ca8fbee9baf65e85cf7'
PARSER = Path(__file__).parent / 'parse_windows_creation.py'
PARSER_SHA = '68cdbccd72881203236481f839defa01ad0d9387c45c8fc7cae869d363d6c63a'
SUPERVISION = BASE / 'official-hierarchy-stage-supervision-006-revision-002'
SEAL = SUPERVISION / 'source-lock-r2-final.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def identity():
    value = json.loads(SEAL.read_text(encoding='utf-8-sig'))
    expected_paths = {'immutable_baseline': BASELINE, 'actual_runtime_entry': Path(__file__).resolve(), 'actual_time_parser': PARSER.resolve()}
    if value['revision'] != '006-supervision-revision-002' or value['decision_logic_overrides'] != ['creation_ms']:
        raise RuntimeError('Revision source seal contract changed')
    for name, path in expected_paths.items():
        record = value['sources'][name]
        if Path(record['path']).resolve() != path.resolve() or sha(path) != record['sha256']:
            raise RuntimeError('Actual composite runtime source differs: ' + name)
    if sha(BASELINE) != BASELINE_SHA or sha(PARSER) != PARSER_SHA:
        raise RuntimeError('Frozen baseline or independently reviewed parser changed')
    return {'revision': value['revision'], 'source_seal_path': str(SEAL), 'source_seal_sha256': sha(SEAL),
            'immutable_baseline_source_sha256': BASELINE_SHA, 'actual_runtime_entry_path': str(Path(__file__).resolve()),
            'actual_runtime_entry_sha256': sha(__file__), 'actual_parser_path': str(PARSER.resolve()),
            'actual_parser_sha256': PARSER_SHA, 'decision_logic_overrides': ['creation_ms'],
            'recording_only_hooks': ['process_birth_runtime_source_supplement', 'receipt_runtime_source_supplement'],
            'native_algorithm_or_original006_source_modified': False}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure():
    runtime = identity()
    module = load('_immutable006_supervisor_with_explicit_runtime_revision002', BASELINE)
    creation = load('_revision002_explicit_zoned_CIM_parser', PARSER)
    module.creation_ms = creation.creation_ms
    # Native006 ownership guard expects the same shared active lock and baseline source.
    # Only the new controller receipt directory changes; the old namespace is preserved.
    module.SUPERVISION = SUPERVISION
    old_identity = module.current_process_identity
    old_write = module.write

    def current_identity_with_provenance():
        birth, argv = old_identity()
        birth['runtime_control_revision'] = runtime
        return birth, argv

    def record_runtime_supplement(path, data):
        if identity() != runtime:
            raise RuntimeError('Composite controller source changed during supervision')
        path = Path(path)
        if path.name == 'configuration.json':
            supplement = SUPERVISION / 'runtime-source-supplements' / path.parent.name
            supplement.mkdir(parents=True, exist_ok=True)
            metadata = supplement / 'runtime-control-revision.json'
            if metadata.exists():
                if json.loads(metadata.read_text(encoding='utf-8-sig')) != runtime:
                    raise RuntimeError('Existing CPU plan runtime source supplement changed')
            else:
                with metadata.open('x', encoding='utf-8') as stream:
                    stream.write(json.dumps(runtime, ensure_ascii=False, indent=2) + '\n')
                for source, target in [(Path(__file__), 'actual-runtime-entry-source.py'), (PARSER, 'actual-time-parser-source.py')]:
                    with (supplement / target).open('xb') as stream:
                        stream.write(source.read_bytes())
        if path.name in {'supervisor-outcome.json', 'dispatch-audit.json', 'active-stage.lock.json'}:
            data['runtime_control_revision'] = runtime
        old_write(path, data)

    module.current_process_identity = current_identity_with_provenance
    module.write = record_runtime_supplement
    return module, runtime


def main():
    module, runtime = configure()
    module.main()
    if '--check-dispatch' in sys.argv:
        print(json.dumps({'status': 'actual_revision002_CPU_check_source_supplement', 'runtime_control_revision': runtime,
                          'GPU_started': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
