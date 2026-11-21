"""Prepare then run the registered three-frame FAN observation-only control.

Dataset paths and selection live here and in the registration, never in the
generic semantic_face module. Preparation does not load a checkpoint or infer.
"""
import os
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                  NUMEXPR_NUM_THREADS='4', PYTHONNOUSERSITE='1')
import argparse
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import unittest

PROJECT = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
DEFAULT_PLAN = ROOT/'smplifyx-reference-control/framewise-canonical-route/shared-three-frame/final-state-diagnosis/fan-observation-plan.json'
REGISTERED_PLAN_SHA256 = '7cf7996df518e1b2b4fe3741dd7716e665c13d516a70b590a24aed56e62aa15b'
sys.path.insert(0, str(PROJECT/'src'))
from shuolee_portrait import semantic_face as face


def now():
    return datetime.now(timezone.utc).isoformat()


def local_path(value):
    value = str(value).replace('\\', '/')
    prefix = '//wsl.localhost/Ubuntu-20.04'
    if value.startswith(prefix):
        return Path(value[len(prefix):])
    if len(value) > 2 and value[1:3] == ':/':
        return Path('/mnt/'+value[0].lower()+'/'+value[3:])
    return Path(value)


def localized(item):
    value = dict(item)
    value['path'] = str(local_path(value['path']))
    return value


def frozen_registration(plan_path):
    face.need(face.sha(plan_path) == REGISTERED_PLAN_SHA256, 'Preregistered plan bytes changed; stop before preparation.')
    plan = face.read(plan_path)
    face.need(plan['state'] == 'preregistered_not_executed', 'Unexpected source registration state.')
    inventory_desc = localized(plan['source_inventory'])
    inventory = face.read(face.verify(inventory_desc))
    items = [inventory_desc, face.descriptor(plan_path)]
    for key in ('immutable_sources', 'immutable_FAN_sources'):
        items.extend(localized(x) for x in plan[key])
    items.extend(localized(x) for x in inventory['key_source_files'])
    items.extend(localized(x) for x in inventory['source_evidence_files'])
    items.extend([localized(plan['immutable_model']), localized(plan['static51_mapping']['template'])])
    unique = {}
    for item in items:
        face.verify(item)
        old = unique.setdefault(item['path'], item)
        face.need(old['sha256'] == item['sha256'], 'Conflicting frozen hashes.')
    return plan, inventory, list(unique.values())


def implementation_files():
    return [face.descriptor(Path(face.__file__), 'generic_semantic_face_module'),
            face.descriptor(PROJECT/'tests/test_semantic_face.py', 'narrow_CPU_unit_tests'),
            face.descriptor(Path(__file__), 'dataset_specific_prepare_run_wrapper')]


def run_unit_tests():
    path = PROJECT/'tests/test_semantic_face.py'
    spec = importlib.util.spec_from_file_location('_test_semantic_face', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    face.need(result.wasSuccessful() and not result.skipped, stream.getvalue())
    return dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                skipped=len(result.skipped), log=stream.getvalue(), network_forwards=0, checkpoint_loads=0)


def prepare(plan_path):
    output = plan_path.parent/'fan-observation'
    face.need(not output.exists(), 'Preparation already exists; inspect it instead of overwriting.')
    plan, inventory, pinned = frozen_registration(plan_path)
    implementation = implementation_files()
    source_by_role = {x.get('role'): localized(x) for x in plan['immutable_sources']}
    deps = [localized(x) for x in plan['immutable_FAN_sources']]
    utils_path = next(x['path'] for x in deps if x['role'] == 'installed_crop_heatmap_decoder_inverse_and_flip')
    dependencies = dict(files=deps, utils_path=utils_path, package_version='1.4.1',
                        root=str(Path(utils_path).parents[1]))
    frames = []
    for original in plan['immutable_frames']:
        frames.append(dict(slot=original['slot'], frame=original['frame'],
            source_frame_index=original['source_frame_index'], time_seconds=original['time_seconds'],
            wh=original['wh'], source_sha256=original['source_sha256'],
            image=dict(path=str(local_path(original['image_windows'])), sha256=original['source_sha256']),
            roi_native_xyxy=original['roi_native_xyxy'], crop_center=original['frozen_crop_center'],
            crop_scale=original['frozen_crop_scale']))
    baseline = dict(source_by_role['frozen_RTMw_targets_no_replacement'], coordinates_key='keypoints')
    bindings_path = local_path(plan['static51_mapping']['binding_manifest'])
    bound_binding = next(localized(x) for x in inventory['source_evidence_files'] if x['role'] == 'exact_51_surface_anchor_manifest')
    face.need(Path(bound_binding['path']) == bindings_path, 'Binding manifest path differs.')
    manifest = dict(schema=face.SCHEMA, camera_model='rectified_pinhole', coordinate_domain=face.PIXELS,
        second_undistort=False, frames=frames, pinned_files=pinned+implementation,
        checkpoint=localized(plan['immutable_model']), dependencies=dependencies,
        frames_source=source_by_role['three_frame_source_frame_ROI_camera_manifest_preserved'],
        frames_rows_key='records', roi_key='face_box_xyxy', review=source_by_role['frozen_semantic_visibility_and_FIT_HELD'],
        baseline=baseline, bindings=bound_binding, template=localized(plan['static51_mapping']['template']),
        budget=dict(device='cpu', threads=4, seconds=180, max_forwards=6),
        source_registration=face.descriptor(plan_path), purpose='observation_only_no_fitting',
        prior_review_target_coordinates='Frozen RTMW review targets remain unchanged. FAN values are separate observations, not a replacement reviewed fitting manifest.')
    with face.offline_only():
        bundle = face.validate_manifest(manifest)
        tests = run_unit_tests()
        controls = face.synthetic_controls(face.load_utils(dependencies))
    for item in pinned+implementation:
        face.verify(item)
    face.need([r['source_frame_index'] for r in frames] == [0, 155, 310] and len(frames) == 3,
              'Registered three-frame selection changed.')
    review_counts = {key: 0 for key in ('observed', 'FIT', 'HELD', 'unknown')}
    for row in bundle['review']['records']:
        if row['visibility'] == 'observed':
            review_counts['observed'] += 1
            review_counts[row['split']] += 1
        elif row['visibility'] == 'unknown':
            review_counts['unknown'] += 1
    face.need(review_counts == dict(observed=34, FIT=27, HELD=7, unknown=119), 'Frozen review population changed.')
    output.mkdir()
    face.save(output/'execution-manifest.json', manifest)
    shutil.copyfile(manifest['review']['path'], output/'frozen-review-snapshot.json')
    face.need(face.sha(output/'frozen-review-snapshot.json') == manifest['review']['sha256'], 'Review copy differs.')
    face.save(output/'synthetic-checks.json', dict(unit_tests=tests, installed_decoder_controls=controls,
        binding_triangle_vertex_barycentric_entries_checked=51, actual_native_images_hash_and_RGB_checked=3,
        source_files_unchanged=True, checkpoint_loads=0, network_forwards=0, GPU_tasks=0))
    face.save(output/'prepare-environment.json', face.software_versions())
    face.save(output/'implementation-receipt.json', dict(created=now(), source_registration=face.descriptor(plan_path),
        execution_manifest=face.descriptor(output/'execution-manifest.json'), code=implementation,
        frozen_sources=pinned, synthetic_checks=face.descriptor(output/'synthetic-checks.json'),
        frozen_review_snapshot=face.descriptor(output/'frozen-review-snapshot.json'),
        prior_plan_mutated=False, production_fitter_or_optimizer_edits=False, real_inference_started=False))
    receipt = dict(state='prepared_CPU_synthetic_checks_passed_real_inference_not_started', created=now(),
        frames=3, source_frames=[0,155,310], frozen_review_counts=review_counts,
        unit_tests=tests['tests'], installed_controls=controls, network_forwards=0, checkpoint_loads=0,
        optimizer_steps=0, gaussian_steps=0, quality_accepted=False,
        implementation_receipt=face.descriptor(output/'implementation-receipt.json'))
    face.save(output/'prepare.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def native_review(manifest, output):
    """Exact native crops; no rescaling, original files remain untouched."""
    import numpy as np
    from PIL import Image, ImageDraw
    review = face.read(manifest['review']['path'])['records']
    with np.load(output/'observations.npz', allow_pickle=False) as z:
        data = {key: z[key] for key in ('rtmw_native68', 'original_continuous', 'flip_restored_continuous', 'sum_continuous')}
    index = []
    for i, row in enumerate(manifest['frames']):
        rgb = face.read_rgb(row)
        roi = np.asarray(row['roi_native_xyxy'])
        x, y = np.floor((roi[:2]+roi[2:])/2-96).astype(int)
        bounds = [int(x), int(y), int(x+192), int(y+192)]
        raw = Image.fromarray(rgb).crop(bounds)
        visible = {q['ibug68_index']:q['split'] for q in review if q['frame']==row['frame'] and q['visibility']=='observed'}
        panels = [raw.copy()]
        for key in ('rtmw_native68', 'original_continuous', 'flip_restored_continuous', 'sum_continuous'):
            panel = raw.copy(); draw = ImageDraw.Draw(panel)
            for j in range(17,68):
                u,v = data[key][i,j]-[x,y]
                color = '#0080ff' if visible.get(j)=='FIT' else ('#ef7c00' if visible.get(j)=='HELD' else '#888888')
                draw.ellipse((u-1.5,v-1.5,u+1.5,v+1.5), outline=color, width=1)
            panels.append(panel)
        sheet = Image.new('RGB', (192*5,192))
        for col,panel in enumerate(panels):
            sheet.paste(panel, (col*192,0))
        path = output/f"slot{row['slot']:06d}-native-review.png"
        sheet.save(path)
        raw_path = output/f"slot{row['slot']:06d}-native-raw.png"; raw.save(raw_path)
        index.append(dict(slot=row['slot'], frame=row['frame'], source_frame_index=row['source_frame_index'],
            source_sha256=row['source_sha256'], native_crop_xyxy=bounds, figure=face.descriptor(path), raw=face.descriptor(raw_path)))
    face.save(output/'native-review-index.json', dict(rows=index,
        columns=['raw','unchanged_RTMW','FAN_original','FAN_flip_restored','FAN_heatmap_sum'],
        colors=dict(FIT='blue', HELD='orange', UNKNOWN='gray'), native_scale_unchanged=True,
        burned_labels=False, physical_accuracy_established=False, review_not_reassigned=True))


def run(plan_path):
    output = plan_path.parent/'fan-observation'
    prepared = face.read(output/'prepare.json')
    face.need(prepared['network_forwards'] == 0, 'Invalid preparation state.')
    receipt = face.read(face.verify(prepared['implementation_receipt']))
    for item in receipt['code']+receipt['frozen_sources']+[receipt['synthetic_checks'], receipt['frozen_review_snapshot']]:
        face.verify(item)
    manifest = face.read(face.verify(receipt['execution_manifest']))
    face.need(not (output/'started.json').exists(), 'A run was already attempted; inspect preserved outputs, do not retry.')
    current = dict(state='running', started=now(), pid=os.getpid(), network_forward_attempts=0,
                   network_forwards_completed=0, optimizer_steps=0, gaussian_steps=0, quality_accepted=False)
    face.save(output/'started.json', dict(current, implementation_receipt=prepared['implementation_receipt']))
    def progress(values):
        current.update(values, updated=now())
        face.save(output/'progress.json', current)
        print(json.dumps(current), flush=True)
    def timeout(*args):
        raise TimeoutError('Registered 180-second FAN observation budget exhausted.')
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 180.)
    try:
        with face.offline_only():
            result = face.run_observations(manifest, output, progress)
            native_review(manifest, output)
        for item in receipt['code']+receipt['frozen_sources']:
            face.verify(item)
        face.save(output/'source-preservation.json', dict(state='all_registered_sources_unchanged',
            files=receipt['code']+receipt['frozen_sources'], frozen_review_snapshot=face.descriptor(output/'frozen-review-snapshot.json'),
            no_target_replacement=True, no_new_fit_manifest=True))
        result.update(completed=now(), registration=face.descriptor(plan_path),
                      observations=face.descriptor(output/'observations.npz'),
                      native_review=face.descriptor(output/'native-review-index.json'))
        face.save(output/'result.json', result)
        progress(dict(state=result['state']))
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        current.update(state='failed_no_retry', error=repr(exc), traceback=traceback.format_exc(),
                       completed=now(), physical_truth=False, fitting_admitted=False)
        face.save(output/'failure.json', current)
        face.save(output/'progress.json', current)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.plan)
    else:
        run(args.plan)
