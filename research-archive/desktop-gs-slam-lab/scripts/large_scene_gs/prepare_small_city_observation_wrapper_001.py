"""CPU generator: preserve the original stage and make an explicitly diagnostic wrapper."""
import difflib
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'scripts/large_scene_gs'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = SCRIPTS / 'run_small_city_stage.py'
    target = SCRIPTS / 'run_small_city_observation_001.py'
    if sha(original) != 'dead1eab49ff455d3da04bfbd97d99570ecbb5235e9523caab457c5ab87bf704':
        raise ValueError('Original isolated stage source changed')
    if target.exists():
        raise ValueError('Preserve existing generated observation wrapper')
    text = original.read_text(encoding='utf-8')
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('Expected exactly one original wrapper splice')
        text = text.replace(old, new)
    replace('    args = parser.parse_args()\n', '''    args = parser.parse_args()
    if (args.stage, args.run, args.input, args.chunk, args.coarse_run, args.iterations, args.resolution, args.exposure, args.port) != (
            'chunk', 'small-city-chunk11-observation-i30000-r4-001', 'small-city-training-002', '1_1',
            'small-city-coarse-i30000-r4-003', 30000, 4, 'official', 6024):
        raise ValueError('Only the separately authorized full-budget observation contract is supported')
''')
    replace('    out = runs / args.run\n', "    out = BASE / 'official-training-observation' / args.run\n")
    observer_sha = sha(SCRIPTS / 'observe_small_city_train_single_001.py')
    replace("    configuration = {'official_commit': COMMIT, 'official_source_files': manifest['official_source_files'],\n", f'''    observer = ROOT / 'scripts/large_scene_gs/observe_small_city_train_single_001.py'
    if sha(observer) != '{observer_sha}':
        raise ValueError('Frozen observation driver source changed')
    adam = REPO / 'scene/OurAdam.py'
    adam_sha = 'fb552b3c31f07ef541dd71a2ac2b8fef72833423a5ca7d9fdf4f17cc22e1c9ba'
    canonical_adam = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{{COMMIT}}:scene/OurAdam.py'])
    if sha(adam) != adam_sha or adam.read_bytes().replace(b'\\r\\n', b'\\n') != canonical_adam.replace(b'\\r\\n', b'\\n'):
        raise ValueError('Original official OurAdam bytes changed')
    official_command = command[:]
    baseline_path = BASE / 'official-training/small-city-chunk11-i30000-r4-003/configuration.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    expected_command = baseline['command_argv'][:]
    expected_command[expected_command.index('-m') + 1] = str(out / 'model')
    if official_command != expected_command:
        raise ValueError('Observed original official argv differs beyond isolated output model path')
    command = [str(TRAINING_PYTHON), '-u', str(observer), '--execute', '--directory', str(out / 'observer'), '--', *official_command[2:]]
    source_files = manifest['official_source_files'] + [{{'path': str(adam), 'relative_path': 'scene/OurAdam.py', 'sha256': adam_sha}}]
    configuration = {{'official_commit': COMMIT, 'official_source_files': source_files,
                     'original_official_command_argv': official_command, 'actual_observation_child_argv': command,
                     'observer_source': str(observer), 'observer_source_sha256': sha(observer), 'OurAdam_source_sha256': adam_sha,
                     'diagnostic_scope': 'Read-only observation hooks, not formal completion or reproduction timing',
                     'formal_training_run': False, 'seed': 0, 'seed_source': 'Unchanged official safe_state(args.quiet)',
                     'baseline_failed_configuration': str(baseline_path), 'baseline_failed_configuration_sha256': sha(baseline_path),
''')
    replace("        plan = inputs / 'execution-plans' / args.run\n", "        plan = BASE / 'official-training-observation-plans' / args.run\n")
    replace("    source_dir = out / 'official-source'\n    for item in manifest['official_source_files']:\n", "    (out / 'observer').mkdir()\n    shutil.copyfile(observer, out / 'observer-source.py')\n    source_dir = out / 'official-source'\n    for item in source_files:\n")
    replace("        outcome.update(exit_code=code, status='failed')\n        if code == 0:\n", '''        observed = json.loads((out / 'observer/observer-outcome.json').read_text(encoding='utf-8'))
        outcome.update(exit_code=code, status='failed', observer_outcome=observed,
                       diagnostic_observation_only=True, formal_completed_model_claimed=False)
        if code == 86 and observed['status'] == 'diagnostic_stopped_at_first_observed_nonfinite':
            outcome.update(status='diagnostic_stopped_first_observed_nonfinite_no_repair')
        if code == 0:
            if observed['status'] != 'official_full30000_observation_returned_no_detected_nonfinite_not_formal' or observed['original_optimizer_call_count'] != 29999:
                raise ValueError('Actual instrumented original step count/full observation return failed')
''')
    replace("            outcome.update(status='completed', saved_model_directory=str(saved), gaussian_count=audit['gaussian_count'],\n", "            outcome.update(status='completed_observation_only_final_finite_not_formal', saved_model_directory=str(saved), gaussian_count=audit['gaussian_count'],\n")
    replace("update_count_scope='Derived from unchanged official loop/control flow plus exit0 and final snapshot, not optimizer instrumentation',", "update_count_scope='Original Adam bound calls instrumented externally, one unchanged original call per observation; 29999 calls required',")
    replace("memory_scope='Official logged torch CUDA allocated peak only; reserved/total unavailable',", "memory_scope='Torch peak includes read-only observation temporaries/synchronization; not an equivalent memory or timing benchmark of failed003',")
    replace('    finally:\n        outcome.update(completed_utc=', '''    finally:
        changed = any(sha(item['path']) != item['sha256'] for item in source_files) or sha(observer) != configuration['observer_source_sha256']
        configuration['source_changed'] = changed
        save(out / 'configuration.json', configuration)
        outcome['source_changed'] = changed
        if changed:
            outcome.update(status='failed_source_changed_no_formal_claim')
        outcome.update(completed_utc=''')
    replace("    if outcome['status'] != 'completed':\n", "    if outcome['status'] != 'completed_observation_only_final_finite_not_formal':\n")
    text = '"""Independent full-budget observation wrapper derived from the preserved original stage.\nAll official input/model validation is retained; actual child argv and observation-only scope are explicit.\n"""\n' + text[text.index('import argparse'):]
    target.write_text(text, encoding='utf-8', newline='\n')
    review = ROOT / 'outputs/large-scene-gs-001/recovery-audit/small-city-observation001-wrapper-preparation-CPU-001'
    review.mkdir(parents=True, exist_ok=False)
    (review / 'wrapper-vs-original.diff').write_text(''.join(difflib.unified_diff(original.read_text(encoding='utf-8').splitlines(True), text.splitlines(True), fromfile=str(original), tofile=str(target))), encoding='utf-8')
    print(f'CPU generated {target}; SHA256={sha(target)}; observer SHA256={observer_sha}; no Torch/CUDA import')


if __name__ == '__main__':
    main()
