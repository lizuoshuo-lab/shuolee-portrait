"""One permitted frame32 replay of the portable entry, never a sequence job."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import fcntl
import json
import sys
import time
import traceback

import numpy as np
from PIL import Image

project = Path('/mnt/d/Projects/Shuolee')
sys.path.insert(0, str(project/'src'))
from shuolee_portrait import official_smplifyx as entry

read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
evidence = Path(__file__).parent
source = Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/official_smplifyx_entry123.py')
score_source = source.with_name('smplify_score_contract123.py')
base = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control')
root = base/'official-first-cycle/full94-shared-shape'
official = base/'official'
assets = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
reference_path = root/'reviews/official-topology-full-fit/cpu_parent_indices/winner-camera-space.npz'
output = evidence/'official-frame32-replay'
assert not output.exists(), 'This finite replay must not be restarted.'
assert entry.sha(source) == '9dd537a60bffb00263214c6e27a3a132136a8b588662596f83681c05569e213f'


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body
            if isinstance(node, ast.FunctionDef)}


original, portable = functions(source), functions(entry.__file__)
core_names = ['need', 'sha', 'save', 'module', 'load_official_config',
              'adapted_entry_source', 'load_official', 'fit_one']
assert all(original[name] == portable[name] for name in core_names)
for name, function in functions(score_source).items():
    assert portable[name] == function
entry.adapted_entry_source(official)
provenance = read(root/'input-provenance.json')
input_path = next(Path(p) for p in provenance['input_files'] if p.endswith('/data/inputs.json'))
obs_path = next(Path(p) for p in provenance['input_files'] if p.endswith('/data/observations.npz'))
for path in (input_path, obs_path):
    assert entry.sha(path) == provenance['input_files'][str(path)]
rows = read(input_path)['records']
ids = [row['frame'] for row in rows]
slot = ids.index('frame_000032')
row = rows[slot]
with np.load(obs_path, allow_pickle=False) as z:
    assert z['frame_ids'].astype(str).tolist() == ids
    targets, confidence_meta = entry.adapt_rtmw_observations(z['keypoints'], z['scores'],
                                                           z['boxes'], z['flip_disagreement'])
with np.load(root/'fit/fitted.npz', allow_pickle=False) as z:
    fit_slot = z['frame_ids'].astype(str).tolist().index('frame_000032')
    assert np.array_equal(targets[slot], z['observed_native_keypoints'][fit_slot])
    shared = z['shared_betas'].copy()
    K = z['K_integer_centers'][fit_slot].copy()
assert np.array_equal(K, row['K_integer_centers'])
assert entry.sha(row['image']) == row['source_sha256']
with Image.open(row['image']) as image:
    assert list(image.size) == row['wh']
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)/255
with np.load(reference_path, allow_pickle=False) as z:
    expected = {name: value.copy() for name, value in z.items()}
source_files = [Path(entry.__file__), project/'src/shuolee_portrait/smplifyx_runtime.py',
                source, score_source, input_path, obs_path, reference_path]
before_hashes = {str(path): entry.sha(path) for path in source_files}
output.mkdir()
entry.save(output/'preflight.json', dict(core_functions_exact=core_names,
    confidence_adapter_functions_exact=True, original_roster_targets_exact=True,
    input_hashes=before_hashes, frame_id=row['frame'], image_sha256=row['source_sha256'],
    confidence_contract=confidence_meta, single_frame_only=True))

import torch
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
stages = []
original_load = entry.load_official
started_fit = False
try:
    with open('/tmp/shuolee-native122-gpu.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        body, vposer, model_meta = entry.create_models(official_root=official, assets=assets)
        assert body.parents.device.type == 'cpu'
        assert not hasattr(body, 'body_pose') and not hasattr(body, 'transl')
        assert all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in body.parameters())

        def instrumented_load(path):
            modules = original_load(path)
            fitting = modules[2]
            original_monitor = fitting.FittingMonitor.run_fitting
            def monitor(self, optimizer, closure, *args, **kwargs):
                count = 0
                def counted(*args, **kwargs):
                    nonlocal count
                    count += 1
                    return closure(*args, **kwargs)
                try:
                    return original_monitor(self, optimizer, counted, *args, **kwargs)
                finally:
                    stages.append(dict(stage=len(stages), closure_evaluations=count))
                    entry.save(output/'stages.json', stages)
            fitting.FittingMonitor.run_fitting = monitor
            return modules

        entry.load_official = instrumented_load
        torch.manual_seed(6032)
        torch.cuda.manual_seed_all(6032)
        started = time.monotonic()
        entry.save(output/'progress.json', dict(state='one_official_frame_running',
            started=datetime.now(timezone.utc).isoformat(), frame_id=row['frame']))
        started_fit = True
        arrays, report = entry.fit_frame(official_root=official, image=rgb,
            keypoints=targets[slot:slot+1], K_integer_centers=K, body_model=body,
            vposer2=vposer, result_path=output/'official-result.pkl', shared_betas=shared,
            freeze_betas=True, device='cuda', diff_path=output/'input-output-adapter.diff')
        torch.cuda.synchronize()
        seconds = time.monotonic()-started
        np.savez_compressed(output/'winner-camera-space.npz', **arrays)
        entry.save(output/'official-result.json', report)
        assert set(arrays) == set(expected)
        equal = {name: np.array_equal(arrays[name], expected[name]) for name in arrays}
        errors = {name: float(np.max(np.abs(arrays[name].astype(np.float64)-expected[name].astype(np.float64))))
                  for name in arrays}
        unchanged = all(entry.sha(Path(path)) == digest for path, digest in before_hashes.items())
        result = dict(state='passed' if all(equal.values()) else 'equivalence_failed',
            frame_id=row['frame'], all_fields_exact=all(equal.values()), fields_exact=equal,
            max_abs_difference=errors, loss=report['winning_loss'],
            closure_evaluations=sum(s['closure_evaluations'] for s in stages),
            stages=stages, seconds=seconds, source_files_unchanged=unchanged,
            torch=torch.__version__, gpu=torch.cuda.get_device_name(),
            core_function_count_exact=len(core_names), runtime_hook_connected=True,
            five_stages=report['five_stages'], translation_applied_once=report['translation_applied_once'],
            full_K_integer_centers=report['full_K_integer_centers'], gvhmr_used=report['gvhmr_used'],
            runtime_optimization=report['runtime_optimization'], full_fit_runs=1,
            sequence_refit=False, gaussian_training_started=False, quality_accepted=False,
            video_to_export_connected=False, server_verified=False,
            model_construction=model_meta, reference_path=str(reference_path))
        entry.save(output/'result.json', result)
        assert all(equal.values()) and unchanged
        assert result['closure_evaluations'] == 1357
        entry.save(output/'progress.json', dict(state='complete', full_fit_runs=1))
        print(json.dumps({name: result[name] for name in ['state', 'all_fields_exact', 'loss',
            'closure_evaluations', 'seconds', 'runtime_hook_connected', 'full_fit_runs']}), flush=True)
except BaseException as error:
    entry.save(output/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
        fitting_started=started_fit, retry_authorized=False, original_results_preserved=True))
    raise
finally:
    entry.load_official = original_load
