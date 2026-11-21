"""CPU provenance gates for versioned004 serial handoff; no CUDA or mutation."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
GATE_DIR = BASE / 'official-hierarchy/small-city-hierarchy-005'
GATE_SHA = '0e51b5c51d0742a20315d02c37355aaae7e9f1d905069a1f6594da0ee06d7f84'
CONFIG_SHA = '55c7dd48fb06b07340fd0a49e611fd6d4334dc98253bcc45aa7ac2261b6fdcea'
INPUT_SHA = '2dd5c04122e290d27934a224921b2506b7955a34d801a0482e0408af536de3d5'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def import_fixed_gate():
    source, config_path = GATE_DIR / 'driver-source.py', GATE_DIR / 'configuration.json'
    input_path = BASE / 'official-inputs/small-city-training-002/manifest.json'
    if sha(source) != GATE_SHA or sha(config_path) != CONFIG_SHA or sha(input_path) != INPUT_SHA:
        raise ValueError('Frozen005 gate/config or002 input manifest changed')
    spec = importlib.util.spec_from_file_location('_fixed005_completion_gate_for004', source)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.ROOT, gate.OUTPUT, gate.REPO = ROOT, BASE, ROOT / 'repos/hierarchical-3d-gaussians'
    return gate, read(config_path), read(input_path)


def completed_gate(run, stage, chunk):
    gate, config, manifest = import_fixed_gate()
    saved, receipt, source_config = gate.completed_training(run, stage, chunk, config, manifest, None)
    expected_trainer = gate.REPO / ('train_coarse.py' if stage == 'coarse' else 'train_single.py')
    argv = source_config['command_argv']
    if Path(argv[0]).resolve() != (ROOT / '.venv-large-gs/Scripts/python.exe').resolve() or argv[1] != '-u' or Path(argv[2]).resolve() != expected_trainer.resolve():
        raise ValueError('Actual command argv interpreter/trainer differs from selected unchanged stage')
    for flag, expected in [('--iterations', str(source_config['requested_iterations'])), ('-r', '4'),
                           ('-s', source_config['effective_official_arguments']['source_path']), ('-m', str(BASE / 'official-training' / run / 'model'))]:
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != expected:
            raise ValueError(f'Actual command argv differs from effective/final stage: {flag}')
    audit_path = BASE / 'official-training' / run / 'model-audit.json'
    audit = read(audit_path)
    parameter = Path(audit['parameter_path']).resolve()
    if parameter.parent != saved.resolve() or sha(parameter) != receipt['model_parameter_sha256'] or Path(receipt['model_audit']).resolve() != audit_path.resolve():
        raise ValueError('Actual completed receipt/model-audit/final parameter path and hash differ')
    return {'status': 'fixed005_completed_training_gate_and_actual_argv_provenance_passed',
            'run': run, 'stage': stage, 'chunk': chunk, 'saved_directory': str(saved),
            'gate_source_sha256': GATE_SHA, 'gate_configuration_sha256': CONFIG_SHA,
            'input_manifest_sha256': INPUT_SHA, 'stage_outcome_sha256': sha(BASE / 'official-training' / run / 'outcome.json'),
            'source_config_sha256': sha(BASE / 'official-training' / run / 'configuration.json'),
            'model_audit_sha256': sha(audit_path), 'model_parameter_sha256': sha(parameter),
            'exit_code': receipt['exit_code'], 'requested_iterations': receipt['requested_iterations'],
            'optimizer_update_calls': receipt['optimizer_update_calls'], 'training_image_updates': receipt['training_image_updates'],
            'source_changed': source_config['source_changed'], 'trainer_source_sha256': source_config['trainer_source_sha256'],
            'coarse_run': source_config.get('coarse_run'), 'actual_command_argv': argv,
            'CUDA_initialized': False, 'torch_imported': 'torch' in sys.modules}


def creation_ms(value):
    if isinstance(value, str) and value.startswith('/Date('):
        return int(re.match(r'/Date\((-?\d+)', value).group(1))
    normalized = re.sub(r'(\.\d{6})\d+(?=Z|[+-]\d\d:\d\d|$)', r'\1', str(value)).replace('Z', '+00:00')
    instant = datetime.fromisoformat(normalized)
    return int(instant.timestamp() * 1000)


def live_binding(run, launch, supervisor, processes, parser, config):
    by_pid = {p['ProcessId']: p for p in processes}
    shim_pid, sup_pid = launch['start_process_pid'], supervisor['supervisor_pid']
    if shim_pid not in by_pid or sup_pid not in by_pid:
        raise ValueError('Exact launch shim and original supervisor must both be live for adoption')
    shim, sup = by_pid[shim_pid], by_pid[sup_pid]
    if creation_ms(shim['CreationDate']) != creation_ms(launch['start_process_creation_utc']) or sup['ParentProcessId'] != shim_pid:
        raise ValueError('Launch shim creation time or supervisor parent binding differs')
    original = None
    journal = BASE / 'official-training-supervision' / run / 'process-observations.jsonl'
    with journal.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            original = next((p for p in row.get('actual_processes', []) if p['ProcessId'] == sup_pid), None)
            if original is not None:
                break
    if original is None or creation_ms(sup['CreationDate']) != creation_ms(original['CreationDate']):
        raise ValueError('Supervisor PID creation time differs from original live observation')
    expected = {'--stage': 'chunk', '--run': run, '--input': 'small-city-training-002', '--chunk': config['chunk'],
                '--coarse-run': 'small-city-coarse-i30000-r4-003', '--iterations': '30000', '--resolution': '4', '--exposure': 'official'}
    for process in (shim, sup):
        argv = parser(process.get('CommandLine') or '')
        if not any(Path(argument).resolve() == (ROOT / 'scripts/large_scene_gs/supervise_small_city_stage.py').resolve() for argument in argv if argument.lower().endswith('.py')):
            raise ValueError('Exact supervisor script argument missing')
        for flag, value in expected.items():
            if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != value:
                raise ValueError(f'Exact live supervisor argv differs: {flag}')
    trainers = []
    for process in processes:
        argv = parser(process.get('CommandLine') or '')
        if 'train_single.py' not in {Path(argument).name.lower() for argument in argv[1:]}:
            continue
        if argv[1:] != config['command_argv'][1:]:
            raise ValueError('A live official trainer actual argv differs from the adopted stage')
        chain, cursor, seen = [], process['ProcessId'], set()
        while cursor in by_pid and cursor not in seen:
            seen.add(cursor)
            chain.append(cursor)
            if cursor == sup_pid:
                break
            cursor = by_pid[cursor]['ParentProcessId']
        if not chain or chain[-1] != sup_pid:
            raise ValueError('Exact live trainer is outside the adopted supervisor descendant chain')
        trainers.append({'process': process, 'descendant_chain_to_supervisor': chain, 'actual_argv': argv})
    if not trainers:
        raise ValueError('Adoption requires a real exact-argv descendant trainer')
    return {'status': 'actual_live_stage_shim_supervisor_creation_and_trainer_descendant_binding_passed',
            'observed_utc': datetime.now(timezone.utc).isoformat(), 'run': run, 'actual_launch_shim': shim,
            'actual_supervisor': sup, 'original_supervisor_creation_reference': original,
            'trainer_bindings': trainers, 'launch_creation_precision': 'WindowsPowerShell CIM JSON milliseconds; launch timestamp compared at matching millisecond precision',
            'GPU_restarted': False, 'CUDA_initialized_by_this_audit': False}
