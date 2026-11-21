"""Compare complete default/threaded FFprobe output on the three remaining native videos."""
import argparse
import ast
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter


LAB = Path('/opt/gs-slam-lab')
BASE = Path('/mnt/d/Projects/Shuolee-smpl-sixworker-opacity')
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-ffprobe-threads')
PREPROCESS = 'src/shuolee_portrait/preprocessing.py'
BASE_SHA = 'da83836694a884debec91b262e87163d68b451e197e5d27f5c37701d60249452'
BASE_COMMIT = '213ad16c035ead820947549f2783d6f7cea1f771'
INPUTS = {
    '001': ('shuolee-smpl-coverage-001', '6405f0e9b144287a4a2f6349d1d5d5b5c4748ffc00039b505b22d66518e88499',
            'b3874422f301f7b05c30210b5adf7ed4c351b3a5b0c9ccd5445078073a6797d0'),
    '005': ('shuolee-smpl-coverage-005', 'd9b693de3f5e4fe0476ef054b19ea6cc25a31e47793d00bc01d538c70d352673',
            'cbcba2dea98ec174e4c42cb96eb6862f910ce36d9ba4db3550d38579daf328f7'),
    '008': ('shuolee-smpl-visibility-008', '1c9f671f046b659d980240ebe8326aa1e7dd58b6241a49192c3043b8f02fc77f',
            '71c1c9d2d9e7a169bcf27714fb174c77cc8668f749eed9a15cb47876f79c4207'),
}
ORDER = ('original', 'threads4', 'threads4', 'original')
LEASE = Path('/tmp/shuolee-native122-gpu.lock')

sys.path.insert(0, str(PROJECT/'src'))
import numpy as np
from shuolee_portrait import runtime as rt
from shuolee_portrait.preprocessing import sampling_indices
from shuolee_portrait.video119 import validate_video_stream


def require(ok, message):
    if not ok:
        raise ValueError(message)


def probe_argv(text):
    function, = [node for node in ast.parse(text).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'extract_dual_resolution']
    assign, = [node for node in function.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'probe' for t in node.targets)]
    call = assign.value
    require(ast.unparse(call.func) == 'subprocess.run' and ast.unparse(call.args[0].elts[-1]) == 'str(video)',
            'Unexpected source FFprobe invocation.')
    return [ast.literal_eval(value) for value in call.args[0].elts[:-1]]


def prepare(output):
    require(not output.exists() and output.parent == LAB, 'Use a fresh isolated WSL lab output.')
    original, candidate = BASE/PREPROCESS, PROJECT/PREPROCESS
    require(rt.sha(original) == BASE_SHA, 'Base extraction source changed.')
    a, b = [path.read_text(encoding='utf-8') for path in (original, candidate)]
    needle = "probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',"
    replacement = "probe = subprocess.run(['ffprobe', '-threads', '4', '-v', 'error', '-select_streams', 'v:0',"
    require(a.count(needle) == 1 and a.replace(needle, replacement) == b,
            'The candidate must differ only in FFprobe decoder threads.')
    original_cmd, candidate_cmd = probe_argv(a), probe_argv(b)
    require(candidate_cmd == [original_cmd[0], '-threads', '4', *original_cmd[1:]],
            'The candidate FFprobe input option changed.')
    sources = {}
    for subject, (name, run_sha, decoded_sha) in INPUTS.items():
        folder = LAB/name/'run'
        run_path, decoded_path = folder/'run.json', folder/'decoded.json'
        require(rt.sha(run_path) == run_sha and rt.sha(decoded_path) == decoded_sha,
                f'Archived source identities changed: {subject}')
        run, decoded = rt.read(run_path), rt.read(decoded_path)
        path = Path(run['video'])
        require(path.is_file() and run['video_sha256'] == decoded['video_sha256']
                and decoded['decoder'] == 'FFmpeg noautorotate, native pixels'
                and decoded['sample_fps'] == 3 and decoded['source_indices'] ==
                    [r['source_frame_index'] for r in decoded['records']], f'Wrong native video roster: {subject}')
        sources[subject] = (path, decoded)
    plan = dict(state='pinned_native_ffprobe_inputs', base_commit=BASE_COMMIT, base_preprocessing_sha256=BASE_SHA,
                candidate_preprocessing_sha256=rt.sha(candidate), script_sha256=rt.sha(Path(__file__)),
                commands=dict(original=original_cmd, threads4=candidate_cmd), timing_order=list(ORDER),
                sources={subject: dict(video=str(path), video_sha256=decoded['video_sha256'],
                    native_wh=decoded['preprocessing']['source_wh'], decoded_total=decoded['decoded_total'],
                    sample_count=len(decoded['records']), run_json_sha256=INPUTS[subject][1],
                    decoded_json_sha256=INPUTS[subject][2]) for subject, (path, decoded) in sources.items()},
                gpu_calls=0, fit_started=False, gaussian_training_started=False, cold_to_export_measured=False)
    return sources, plan


def validate_pts(stdout, decoded):
    probe = json.loads(stdout)
    require(len(probe['streams']) == 1, 'Expected one video stream.')
    stream = probe['streams'][0]
    validate_video_stream(stream)
    require([int(stream['width']), int(stream['height'])] == decoded['preprocessing']['source_wh'],
            'The native stream dimensions differ.')
    pts = np.asarray([float(row['best_effort_timestamp_time']) for row in probe['frames']])
    targets, ids = sampling_indices(pts, decoded['sample_fps'])
    require(len(pts) == decoded['decoded_total'] and ids.tolist() == decoded['source_indices']
            and all(pts[index] == row['time_seconds'] and target == row['target_time_seconds']
                    for index, target, row in zip(ids, targets, decoded['records'], strict=True)),
            'Decoded or selected timestamps changed.')
    return probe


def run(sources, plan, output):
    output.mkdir()
    rt.save(output/'preregistration.json', plan)
    all_records = {}
    for subject, (video, decoded) in sources.items():
        require(rt.sha(video) == decoded['video_sha256'], f'Original video changed: {subject}')
        expected = None
        rows = []
        for i, name in enumerate(ORDER):
            start = perf_counter()
            proc = subprocess.run([*plan['commands'][name], str(video)], capture_output=True)
            elapsed = perf_counter()-start
            path = output/f'{subject}-{i+1}-{name}.json'
            path.write_bytes(proc.stdout)
            if proc.returncode:
                (output/f'{subject}-{i+1}-{name}.stderr.txt').write_bytes(proc.stderr)
                raise RuntimeError(f'FFprobe failed with code {proc.returncode}: {subject}/{name}')
            probe = validate_pts(proc.stdout, decoded)
            if expected is None:
                expected = (proc.stdout, probe)
            require(proc.stdout == expected[0] and probe['frames'] == expected[1]['frames']
                    and probe['streams'] == expected[1]['streams'], f'Complete FFprobe output differs: {subject}/{name}')
            rows.append(dict(arm=name, seconds=elapsed, exact_full_stdout=True,
                             stdout_file=path.name, stdout_sha256=sha256(proc.stdout).hexdigest()))
        require(rt.sha(video) == decoded['video_sha256'], f'Video changed during probe: {subject}')
        times = {name: median(row['seconds'] for row in rows if row['arm'] == name)
                 for name in ('original', 'threads4')}
        all_records[subject] = dict(samples=rows, original_median_seconds=times['original'],
            threads4_median_seconds=times['threads4'], median_saved_seconds=times['original']-times['threads4'],
            exact_archived_3fps_pts=True, codec=expected[1]['streams'][0]['codec_name'])
        print(json.dumps(dict(subject=subject, **all_records[subject])), flush=True)
    result = dict(state='exact_cross_person_ffprobe_validation', subjects=all_records,
        version=subprocess.check_output(['ffprobe', '-version'], text=True).splitlines()[0],
        preregistration_sha256=rt.sha(output/'preregistration.json'), gpu_calls=0, fit_started=False,
        gaussian_training_started=False, cold_to_export_measured=False)
    rt.save(output/'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=LAB/'shuolee-ffprobe-threads-crossperson-001005008')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    sources, plan = prepare(output)
    if args.preflight:
        print(json.dumps(plan))
        return
    with LEASE.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(sources, plan, output)


if __name__ == '__main__':
    main()
