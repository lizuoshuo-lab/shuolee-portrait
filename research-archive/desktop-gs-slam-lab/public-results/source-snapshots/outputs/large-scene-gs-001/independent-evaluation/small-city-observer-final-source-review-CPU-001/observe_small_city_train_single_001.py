"""External observation-only hooks; the official trainer and Adam sources stay unchanged.

The normal invocation only prints the contract. CUDA execution requires --execute
and a prepared isolated stage. No observer operation writes model/gradient/state,
draws a random number, changes a hyperparameter, or alters optimizer arguments.
GPU-to-CPU reads synchronize and can affect wall time; bit-exact rerun is not promised.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'repos/hierarchical-3d-gaussians'
SUSPECT = 1377084
EXPECTED_SCAFFOLD = 1835543
EXPECTED_INITIAL_G = 2065281
ORACLE = ROOT / 'outputs/large-scene-gs-001/official-training-observation-plans/small-city-initial-suspect-oracle-001/expected-raw.json'
ORACLE_SHA = '3fc724de94b8609f443ec63e9cdb6705a2083ca4785298af57d208064899c236'
OUR_ADAM_SHA = 'fb552b3c31f07ef541dd71a2ac2b8fef72833423a5ca7d9fdf4f17cc22e1c9ba'
TRAINER_SHA = '2dc8d237519e9467c6756b8d0ecbbcf38a615ef9135914f15d6c348584cea839'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


class ObservationDetected(RuntimeError):
    pass


def encoded(number):
    number = float(number)
    return number if math.isfinite(number) else ('NaN' if math.isnan(number) else ('+Infinity' if number > 0 else '-Infinity'))


def tensor_record(tensor):
    if tensor is None:
        return None
    # Clone the CPU result: on CPU detach().cpu() alone may still alias the original.
    array = tensor.detach().cpu().clone().numpy()
    values = array.reshape(-1)
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'values': [encoded(x) for x in values],
            'uint32_hex': [format(int(x), '08x') for x in array.view('uint32').reshape(-1)] if array.dtype.name == 'float32' else None,
            'all_finite': bool(all(math.isfinite(float(x)) for x in values))}


def nonfinite_paths(snapshot):
    found = []
    def visit(value, prefix):
        if isinstance(value, dict):
            if value.get('all_finite') is False:
                found.append(prefix)
                return
            for key, item in value.items():
                visit(item, f'{prefix}.{key}' if prefix else key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f'{prefix}[{index}]')
    visit(snapshot, '')
    return found


class ReadOnlyObserver:
    def __init__(self, directory, row=SUSPECT, whole_every=100, expected_scaffold=EXPECTED_SCAFFOLD, expected_G=EXPECTED_INITIAL_G):
        self.directory, self.row, self.whole_every = directory, row, whole_every
        self.expected_scaffold, self.expected_G = expected_scaffold, expected_G
        self.ring = deque(maxlen=8)
        self.optimizer_calls = 0
        self.events = 0
        self.model = None
        self.last_after = None
        self.first_failure = None
        self.started = time.perf_counter()

    def context(self):
        frame = inspect.currentframe().f_back
        try:
            while frame is not None:
                if (Path(frame.f_code.co_filename).resolve() == REPO / 'train_single.py' and frame.f_code.co_name == 'training'):
                    local = frame.f_locals
                    camera = local.get('viewpoint_cam')
                    result = {'iteration': int(local.get('iteration', 0)),
                              'official_training_frame_line': frame.f_lineno,
                              'camera': {key: getattr(camera, key, None) for key in ('image_name', 'uid', 'colmap_id', 'image_width', 'image_height', 'depth_reliable')} if camera is not None else None}
                    return result
                frame = frame.f_back
        finally:
            del frame
        return {'iteration': None, 'camera': None, 'official_training_frame_line': None}

    def capture(self, model):
        groups = {}
        for group in model.optimizer.param_groups:
            if len(group['params']) != 1:
                raise RuntimeError('Expected unchanged official one-Parameter group')
            p = group['params'][0]  # Current p: densification/reset replaces Parameters.
            if len(p) <= self.row:
                raise RuntimeError('Protected suspect prefix disappeared')
            state = model.optimizer.state.get(p, {})  # Never lazy-create optimizer state.
            state_rows = {}
            for key, value in state.items():
                if hasattr(value, 'detach'):
                    selected = value if value.numel() == 1 else value[self.row]
                    state_rows[key] = tensor_record(selected)
                else:
                    state_rows[key] = value
            groups[group['name']] = {
                'parameter': tensor_record(p[self.row]), 'gradient': tensor_record(p.grad[self.row]) if p.grad is not None else None,
                'Adam_state': state_rows, 'parameter_shape': list(p.shape), 'parameter_object_id': id(p),
                'lr': float(group['lr']), 'eps': float(group['eps']), 'betas': list(group['betas']),
                'weight_decay': float(group['weight_decay']), 'amsgrad': bool(group['amsgrad']),
                'maximize': bool(group['maximize']), 'capturable': bool(group['capturable']), 'foreach': group['foreach'],
            }
        return {'row_zero_based': self.row, 'G': len(model.optimizer.param_groups[0]['params'][0]),
                'skybox_points': model.skybox_points, 'scaffold_points': model.scaffold_points,
                'active_SH_degree': model.active_sh_degree, 'groups': groups}

    def whole_finite(self, model):
        import torch
        checks = {}
        for group in model.optimizer.param_groups:
            p = group['params'][0]
            finite = torch.isfinite(p.detach())
            ok = bool(finite.all().item())
            checks[group['name']] = {'all_finite': ok, 'shape': list(p.shape)}
            if not ok:
                bad = (~finite).nonzero()
                checks[group['name']]['nonfinite_scalar_count'] = len(bad)
                checks[group['name']]['first_20_indices'] = bad[:20].detach().cpu().clone().tolist()
            del finite
        return checks

    def append(self, value):
        with (self.directory / 'observation-journal.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')

    def fail(self, phase, before, after, context, extra=None):
        if self.first_failure is None:
            event = {'status': 'first_observed_nonfinite_diagnostic_stop_no_repair', 'observed_utc': utc(),
                     'phase': phase, 'context': context, 'before': before, 'after': after,
                     'before_nonfinite_paths': nonfinite_paths(before) if before else [],
                     'after_nonfinite_paths': nonfinite_paths(after) if after else [],
                     'previous_observed_after': self.last_after, 'recent_read_only_events': list(self.ring),
                     'original_optimizer_call_count': self.optimizer_calls, 'extra': extra,
                     'no_tensor_gradient_state_RNG_or_hyperparameter_writes_by_observer': True,
                     'first_observed_scope': 'Per-suspect reads at each hooked boundary; full-model raw finite sampled every100 optimizer calls. Not a continuous kernel trace.',
                     'CUDA_observation_overhead_scope': 'Detach/CPU copies synchronize; observation is not a timing or bitexact deterministic benchmark.'}
            path = self.directory / 'first-nonfinite.json'
            save(path, event)
            self.first_failure = str(path)
        raise ObservationDetected(f'Observed nonfinite at {phase}; original data preserved')

    def initialized(self, model):
        self.model = model
        before = self.capture(model)
        if before['scaffold_points'] != self.expected_scaffold or before['G'] != self.expected_G or not (100000 <= self.row < before['scaffold_points']):
            raise RuntimeError('Actual initial protected-prefix contract differs')
        if sha(ORACLE) != ORACLE_SHA:
            raise RuntimeError('Frozen exact initial raw identity oracle changed')
        oracle = json.loads(ORACLE.read_text(encoding='utf-8'))
        if (self.row != oracle['suspect_selected_prefix_row_zero_based'] or
                sha(oracle['actual_coarse_PLY']) != oracle['actual_coarse_PLY_sha256'] or
                oracle['actual_coarse_PLY_sha256'] != 'abd4f6af0463dce35648182a66d2ae3d6893417901f3885c5be1eb98368f2e31'):
            raise RuntimeError('Initial oracle does not bind actual unchanged coarse and suspect row')
        identity = {name: {'shape_equal': before['groups'][name]['parameter']['shape'] == expected['shape'],
                           'float32_bits_equal': before['groups'][name]['parameter']['uint32_hex'] == expected['uint32_hex'],
                           'expected_raw_group_sha256': expected['raw_group_bytes_sha256']}
                    for name, expected in oracle['groups'].items()}
        matched = set(before['groups']) == set(oracle['groups']) and all(g['shape_equal'] and g['float32_bits_equal'] for g in identity.values())
        save(self.directory / 'initial-raw6group-identity.json', {'observed_utc': utc(), 'actual_match_passed': matched,
             'oracle_path': str(ORACLE), 'oracle_sha256': ORACLE_SHA, 'original_coarse_row': 1778538,
             'suspect_selected_prefix_row': self.row, 'group_checks': identity,
             'semantics': 'Raw float32 exact. No sigmoid/exp/quaternion normalization; SH1 copied and SH3 higher rows zero.'})
        if not matched:
            raise RuntimeError('Actual initialized suspect raw6group identity differs from completed coarse oracle')
        whole = self.whole_finite(model)
        save(self.directory / 'initial-suspect-and-whole-finite.json', {'observed_utc': utc(), 'suspect': before, 'whole': whole})
        if nonfinite_paths(before) or nonfinite_paths(whole):
            self.fail('initialization', None, before, self.context(), {'whole': whole})
        self.last_after = before

    def wrap_step(self, model, original):
        def observed_step(*args, **kwargs):
            context = self.context()
            before = self.capture(model)
            relevant = args[0] if args else kwargs['relevant']
            relevant_count = int(relevant.numel())
            selected = bool((relevant == self.row).any().item()) if relevant_count else None
            invocation = {'relevant_object_id': id(relevant), 'relevant_numel': relevant_count,
                          'suspect_in_relevant': selected, 'original_branch': 'dense_fallback' if relevant_count == 0 else 'indexed_sparse_rows'}
            # Persist bad pre-step evidence before forwarding; call the original exactly once.
            if nonfinite_paths(before):
                save(self.directory / 'first-bad-pre-step-before-original.json', {'context': context, 'before': before, 'invocation': invocation})
            returned = original(*args, **kwargs)
            self.optimizer_calls += 1
            after = self.capture(model)
            entry = {'phase': 'Adam.step', 'context': context, 'before': before, 'after': after, 'invocation': invocation,
                     'original_step_forwarded_exactly_once': True, 'original_optimizer_call_count': self.optimizer_calls}
            self.ring.append(entry)
            self.append({'observed_utc': utc(), 'context': context, 'phase': 'Adam.step', 'call': self.optimizer_calls,
                         'before_nonfinite_paths': nonfinite_paths(before), 'after_nonfinite_paths': nonfinite_paths(after),
                         'invocation': invocation, 'G': after['G']})
            if nonfinite_paths(before) or nonfinite_paths(after):
                self.fail('Adam.step_before_after', before, after, context, invocation)
            if self.optimizer_calls % self.whole_every == 0:
                whole = self.whole_finite(model)
                self.append({'observed_utc': utc(), 'phase': 'whole_raw_parameter_finite_sample', 'context': context,
                             'original_optimizer_call_count': self.optimizer_calls, 'whole': whole})
                save(self.directory / 'latest-suspect-before-after.json', entry)
                if nonfinite_paths(whole):
                    self.fail('whole_parameter_sample_after_original_step', before, after, context, {'whole': whole})
            self.last_after = after
            return returned
        return observed_step

    def wrap_mutator(self, label, original):
        def observed(model, *args, **kwargs):
            if self.model is not model:
                return original(model, *args, **kwargs)
            before, context = self.capture(model), self.context()
            returned = original(model, *args, **kwargs)
            after = self.capture(model)
            entry = {'phase': label, 'context': context, 'before': before, 'after': after, 'original_call_forwarded_exactly_once': True}
            self.ring.append(entry)
            self.append({'observed_utc': utc(), 'phase': label, 'context': context,
                         'before_nonfinite_paths': nonfinite_paths(before), 'after_nonfinite_paths': nonfinite_paths(after)})
            if nonfinite_paths(before) or nonfinite_paths(after):
                self.fail(label, before, after, context)
            self.last_after = after
            return returned
        return observed

    def before_save(self, scene, iteration):
        model = scene.gaussians
        snapshot, whole = self.capture(model), self.whole_finite(model)
        save(self.directory / f'pre-official-save-{iteration}-finite.json', {'observed_utc': utc(), 'iteration': iteration, 'suspect': snapshot, 'whole': whole})
        if nonfinite_paths(snapshot) or nonfinite_paths(whole):
            self.fail('before_unchanged_official_save', self.last_after, snapshot, self.context(), {'whole': whole})


def execute(official_argv, directory):
    if any(directory.iterdir()):
        raise RuntimeError('Existing observer artifacts; refusing a repeated execute')
    configuration = json.loads((directory.parent / 'configuration.json').read_text(encoding='utf-8'))
    if (configuration['original_official_command_argv'][2:] != official_argv
            or configuration['observer_source_sha256'] != sha(Path(__file__))
            or configuration['requested_iterations'] != 30000 or configuration['resolution_pixels'] != [256, 172]
            or configuration['chunk'] != '1_1' or configuration['input_manifest_sha256'] != '2dd5c04122e290d27934a224921b2506b7955a34d801a0482e0408af536de3d5'
            or configuration['coarse_run'] != 'small-city-coarse-i30000-r4-003'
            or configuration['exposure_mode'] != 'official'):
        raise RuntimeError('Actual observation stage/argv/source/input/default contract differs')
    for item in configuration['official_source_files']:
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('Pinned actual official source changed')
    if sha(REPO / 'train_single.py') != TRAINER_SHA or sha(REPO / 'scene/OurAdam.py') != OUR_ADAM_SHA:
        raise RuntimeError('Pinned official trainer/Adam source changed')
    if str(Path(official_argv[0]).resolve()) != str(REPO / 'train_single.py'):
        raise RuntimeError('Only the pinned train_single is supported')
    import torch
    sys.path.insert(0, str(REPO))
    from scene.gaussian_model import GaussianModel
    from scene import Scene
    from scene.OurAdam import Adam
    observer = ReadOnlyObserver(directory)
    original_setup = GaussianModel.training_setup
    original_reset, original_density = GaussianModel.reset_opacity, GaussianModel.densify_and_prune
    original_save = Scene.save
    def observed_setup(model, *args, **kwargs):
        returned = original_setup(model, *args, **kwargs)
        if type(model.optimizer) is not Adam or model.optimizer.step.__func__ is not Adam.step:
            raise RuntimeError('Actual original OurAdam bound step differs')
        observer.initialized(model)
        model.optimizer.step = observer.wrap_step(model, model.optimizer.step)
        return returned
    def observed_save(scene, iteration, *args, **kwargs):
        observer.before_save(scene, iteration)
        return original_save(scene, iteration, *args, **kwargs)
    GaussianModel.training_setup = observed_setup
    GaussianModel.reset_opacity = observer.wrap_mutator('GaussianModel.reset_opacity', original_reset)
    GaussianModel.densify_and_prune = observer.wrap_mutator('GaussianModel.densify_and_prune', original_density)
    Scene.save = observed_save
    source = {'driver_sha256': sha(Path(__file__)), 'trainer_sha256': sha(REPO / 'train_single.py'),
              'OurAdam_sha256': sha(REPO / 'scene/OurAdam.py'), 'original_setup_source': inspect.getsourcefile(original_setup),
              'original_Adam_step_decorator_source': inspect.getsourcefile(Adam.step),
              'original_Adam_step_unwrapped_source': inspect.getsourcefile(inspect.unwrap(Adam.step)),
              'original_Adam_step_signature': str(inspect.signature(Adam.step)),
              'official_sys_argv': official_argv, 'runpy_run_name': '__main__', 'safe_state_seed': 0,
              'exact_initial_raw_identity_oracle': str(ORACLE), 'exact_initial_raw_identity_oracle_sha256': ORACLE_SHA,
              'suspect_row_zero_based': SUSPECT, 'whole_raw_finite_interval_optimizer_calls': 100,
              'original_parameters_gradients_optimizer_state_hyperparameters_RNG_not_modified_by_observer': True,
              'observation_only_not_formal_completion': True}
    save(directory / 'runtime-hook-contract.json', source)
    status, code = 'unexpected_observation_driver_failure', 1
    old_argv = sys.argv
    try:
        sys.argv = official_argv
        # Temporary __main__.__file__ is the official file. Windows workers rerun
        # it as __mp_main__, where the trainer's main guard is false.
        runpy.run_path(str(REPO / 'train_single.py'), run_name='__main__')
        if observer.optimizer_calls != 29999:
            raise RuntimeError('Actual instrumented original optimizer-call count differs from I-1')
        status, code = 'official_full30000_observation_returned_no_detected_nonfinite_not_formal', 0
    except ObservationDetected as error:
        status, code = 'diagnostic_stopped_at_first_observed_nonfinite', 86
        source['diagnostic_error'] = str(error)
    except Exception as error:
        source['unexpected_error_type'], source['unexpected_error'] = type(error).__name__, str(error)
        raise
    finally:
        sys.argv = old_argv
        GaussianModel.training_setup, GaussianModel.reset_opacity, GaussianModel.densify_and_prune = original_setup, original_reset, original_density
        Scene.save = original_save
        save(directory / 'observer-outcome.json', {'status': status, 'actual_exit_code': code, 'observed_utc': utc(),
             'original_optimizer_call_count': observer.optimizer_calls, 'first_nonfinite_event': observer.first_failure,
             'elapsed_seconds': time.perf_counter() - observer.started, 'runtime_hook_contract': source,
             'full_budget_requested': 30000, 'formal_completed_model_claimed': False,
             'CUDA_initialized': torch.cuda.is_initialized(), 'no_repair_no_retry_no_RNG_sampling_or_LR_changes': True})
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--directory', type=Path)
    parser.add_argument('official_argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'CPU_contract_only_no_torch_import_no_CUDA', 'suspect_row': SUSPECT,
                          'requested_budget': 30000, 'our_Adam_sha256': OUR_ADAM_SHA, 'GPU_execution_started': False}))
        return
    if args.directory is None or not args.directory.is_dir():
        raise RuntimeError('Prepared isolated observation directory required')
    official_argv = args.official_argv[1:] if args.official_argv[:1] == ['--'] else args.official_argv
    raise SystemExit(execute(official_argv, args.directory))


if __name__ == '__main__':
    main()
