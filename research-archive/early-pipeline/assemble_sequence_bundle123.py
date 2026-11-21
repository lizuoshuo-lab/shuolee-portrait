"""CPU bridge from a completed sequence fit to the existing FramePose contract.

No body model, fitter, renderer or Torch is imported. ``inspect`` may read an
initial snapshot; only ``assemble`` can create a Bundle, and it requires the
hash-bound final fit and a terminal result from the same run. Source files are
never edited. Geometry and normalized cameras are copied, never rescaled.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import sys

import numpy as np


SCHEMA = 'shuolee-sequence-fit-to-frame-pose/1'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def io_path(path):
    path = Path(path).resolve()
    if os.name == 'nt' and not str(path).startswith('\\\\'):
        return Path('\\\\?\\' + str(path))
    return path


def sha(path):
    h = hashlib.sha256()
    with io_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    value = json.loads(io_path(path).read_text(encoding='utf-8-sig'))
    need(isinstance(value, dict), 'Expected JSON object: ' + str(path))
    return value


def write_new(path, value):
    with io_path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def descriptor(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def resolver(path_maps=()):
    """Explicit relocation only; no inferred camera or image transformations."""
    pairs = sorted([(str(a).replace('\\', '/').rstrip('/'), str(b).rstrip('/\\')) for a, b in path_maps],
                   key=lambda item: len(item[0]), reverse=True)

    def resolve(value, base=None):
        text = str(value)
        comparable = text.replace('\\', '/')
        for old, new in pairs:
            if comparable == old or comparable.startswith(old + '/'):
                text = new + comparable[len(old):]
                break
        p = Path(text)
        if not p.is_absolute():
            need(base is not None, 'Unresolved source path; provide an explicit path map: ' + text)
            p = Path(base) / p
        return p.resolve()
    return resolve


def _modules(project_root):
    src = Path(project_root).resolve() / 'src'
    need(src.is_dir(), 'Missing existing project src directory.')
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from shuolee_portrait.sequence_input_adapter import load_sequence_inputs
    from shuolee_portrait.camera_world_units import normalize_camera_w2c
    from shuolee_portrait.frame_pose_contract import load_frame_pose_bundle
    return load_sequence_inputs, normalize_camera_w2c, load_frame_pose_bundle


def _numeric(array, shape, name):
    a = np.asarray(array)
    need(a.shape == shape and a.dtype.kind in 'iuf' and np.isfinite(a).all(), 'Invalid ' + name)
    return a


def _strings(array, shape, name):
    a = np.asarray(array)
    need(a.shape == shape and a.dtype.kind in 'US', 'Invalid ' + name)
    values = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in a.ravel()]
    need(all(v.strip() for v in values), 'Empty ' + name)
    return values


def _roundtrip_bound(camera, world, E):
    """Check independent float32 snapshot rounding, not physical pose accuracy."""
    need(camera.dtype in (np.float32, np.float64) and world.dtype in (np.float32, np.float64),
         'Only float32/float64 snapshot vertices are supported.')
    c, w = camera.astype(np.float64), world.astype(np.float64)
    R, t = E[:, :3, :3], E[:, :3, 3]
    expected = np.einsum('nvi,nij->nvj', c - t[:, None], R)
    # Round-to-nearest uncertainty of the two independently stored arrays.
    dc = .5 * np.abs(np.spacing(np.abs(camera))).astype(np.float64)
    dw = .5 * np.abs(np.spacing(np.abs(world))).astype(np.float64)
    gamma = 16 * np.finfo(np.float64).eps
    bound = np.einsum('nvi,nij->nvj', dc, np.abs(R)) + dw
    bound += gamma * np.einsum('nvi,nij->nvj', np.abs(c) + np.abs(t[:, None]), np.abs(R))
    error = np.abs(expected - w)
    need(np.all(error <= bound), 'Stored camera/world vertices violate the same normalized E (beyond storage rounding).')
    return dict(max_abs_model_units=float(error.max()), max_storage_rounding_bound=float(bound.max()),
                physical_accuracy_test=False, coordinate_conversion_applied_to_output=False)


def inspect_snapshot(snapshot, *, fit_plan, source_run, sequence_id, project_root, path_maps=()):
    """Read-only input validation, including initial.npz; creates no Bundle."""
    need(isinstance(sequence_id, str) and bool(sequence_id.strip()), 'Explicit sequence_id is required.')
    resolve = resolver(path_maps)
    snapshot, fit_plan, source_run = [Path(p).resolve() for p in (snapshot, fit_plan, source_run)]
    load_inputs, normalize, _ = _modules(project_root)
    plan = read(fit_plan)
    need(plan.get('mode') == 'one_full_sequence_research_adaptation', 'Wrong fit-plan role.')
    manifest_path = resolve(plan['inputs'], fit_plan.parent)
    need(plan.get('source_sha256', {}).get(plan['inputs']) == sha(manifest_path),
         'Input manifest is not hash-bound by this fit plan.')
    manifest = read(manifest_path)
    bundle = load_inputs(io_path(manifest_path), source_path_resolver=lambda p: str(io_path(resolve(p, manifest_path.parent))),
                         decode_images=True)
    need(len(bundle.frame_ids) == plan['frames'], 'Plan and actual full roster count differ.')
    need(bundle.reference_frame_id == plan['reference_frame_id'], 'Reference identity differs from plan.')
    need(dict(bundle.camera_length_contract) == plan['camera_length_contract'], 'Plan/input camera unit declarations differ.')
    frame_path = resolve(manifest['frames']['path'], manifest_path.parent)
    rows = read(frame_path)['records']
    row_by_id = {str(r['frame']): r for r in rows}
    run = read(source_run)
    # Bind the historical acquisition receipt to its actual extraction tree.
    # This checks file identity/provenance, not a fresh video-to-PNG decode.
    need(frame_path.is_relative_to(source_run.parent), 'Source run does not own this extraction roster.')
    for row in rows:
        for field in ('image', 'mask'):
            need(resolve(row[field], frame_path.parent).is_relative_to(source_run.parent),
                 'Source run does not own the frame/mask paths.')
    video = resolve(run['video'], source_run.parent)
    video_hash = sha(video)
    need(video_hash == run['video_sha256'], 'Original video content differs from source-run provenance.')
    if isinstance(run.get('subject'), dict) and 'video_sha256' in run['subject']:
        need(run['subject']['video_sha256'] == video_hash, 'Source-run subject/video identity differs.')
    subject = run.get('subject', {})
    need(subject.get('selected_frames') == len(rows) and
         all(row['wh'] == subject.get('source_wh') for row in rows) and
         all(0 <= row['source_frame_index'] < subject.get('source_frames', 0) for row in rows),
         'Source-run video roster dimensions/count/index range differ.')
    with np.load(io_path(snapshot), allow_pickle=False) as z:
        fit = {k: z[k].copy() for k in z.files}
    n = len(bundle.frame_ids)
    ids = _strings(fit['frame_ids'], (n,), 'snapshot frame_ids')
    need(len(set(ids)) == n and set(ids) == set(bundle.frame_ids), 'Snapshot must cover the exact named roster once.')
    index = {name: i for i, name in enumerate(bundle.frame_ids)}
    order = np.array([index[name] for name in ids], dtype=np.int64)
    source_indices = np.asarray(fit['source_frames'])
    need(source_indices.dtype.kind in 'iu' and np.array_equal(source_indices, bundle.source_frames[order]),
         'Snapshot source-video indices do not match named frames.')
    for key, row_key in [('image_sha256', 'source_sha256'), ('mask_sha256', 'mask_sha256')]:
        actual = _strings(fit[key], (n,), key)
        need(actual == [row_by_id[name][row_key] for name in ids], 'Snapshot ' + key + ' identity differs.')
    K = _numeric(fit['K_integer_centers'], (n, 3, 3), 'snapshot K')
    E = _numeric(fit['w2c'], (n, 4, 4), 'normalized snapshot w2c')
    source_E = _numeric(fit['source_w2c'], (n, 4, 4), 'source snapshot w2c')
    need(np.array_equal(K, bundle.K_integer_centers[order]), 'Snapshot K differs from native full K.')
    need(np.array_equal(source_E, bundle.source_w2c[order]), 'Snapshot source E differs from bound input.')
    units = json.loads(_strings(fit['camera_length_contract_json'], (), 'camera unit metadata')[0])
    need(units == dict(bundle.camera_length_contract), 'Snapshot camera unit metadata differs.')
    # Validation only. Output copies fit.w2c verbatim; it is already normalized.
    normalized = normalize(source_E, units)
    need(np.array_equal(E, normalized.w2c_meters), 'Snapshot E violates its declared source-unit conversion.')
    ref = _strings(fit['reference_frame_id'], (), 'snapshot reference')[0]
    need(ref == bundle.reference_frame_id, 'Snapshot named reference differs.')
    beta = _numeric(fit['shared_betas'], (1, 10), 'one shared beta')
    need(np.array_equal(_numeric(fit['betas'], (1, 10), 'fit beta'), beta),
         'Duplicate fit beta representations disagree; no per-frame identity accepted.')
    world = np.asarray(fit['vertices_world'])
    need(world.ndim == 3 and world.shape[0] == n and world.shape[1] >= 3 and world.shape[2] == 3,
         'Invalid framewise vertices shape.')
    _numeric(world, world.shape, 'world vertices')
    camera = _numeric(fit['vertices_camera'], world.shape, 'camera vertices')
    faces = np.asarray(fit['faces'])
    need(faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype.kind in 'iu' and len(faces) > 0 and
         np.all(faces >= 0) and np.all(faces < world.shape[1]) and
         np.all(faces[:, 0] != faces[:, 1]) and np.all(faces[:, 0] != faces[:, 2]) and np.all(faces[:, 1] != faces[:, 2]),
         'Invalid common topology.')
    roundtrip = _roundtrip_bound(camera, world, E)
    bound = {(str(Path(p).resolve()), h) for p, h in bundle.bound_files}
    for p in (snapshot, fit_plan, source_run, video):
        bound.add((str(p), sha(p)))
    sources = [dict(path=p, sha256=h) for p, h in sorted(bound)]
    report = dict(schema=SCHEMA, inspected_at=datetime.now(timezone.utc).isoformat(),
        snapshot=descriptor(snapshot), snapshot_role='initial_not_final' if snapshot.name == 'initial.npz' else 'unbound_snapshot',
        complete_named_roster=True, frame_count=n, reference_frame_id=ref, sequence_id=sequence_id,
        source_video=dict(path=str(video), sha256=video_hash, file_bytes_verified=True, frames_redecoded=False),
        source_video_frame_association='Historical acquisition run/tree and recorded source indices; not freshly decoded equivalence',
        inputs=descriptor(manifest_path), extraction_roster=descriptor(frame_path), fit_plan=descriptor(fit_plan), source_run=descriptor(source_run),
        shared_beta_shape=list(beta.shape), vertices_per_frame=world.shape[1], common_faces=len(faces),
        world_camera_roundtrip=roundtrip, upstream_camera_length_contract=units,
        output_camera_contract=dict(translation_unit='meters', meters_per_camera_unit=1., basis=units['basis'],
             evidence='Already normalized model-coordinate geometry and fit.w2c, copied verbatim. Upstream declaration retained separately; no measured physical scale implied.'),
        output_camera_scale_applications=0, pixel_conversion_applications=0,
        input_face_review=dict(active_fit=bundle.report()['active_face_fit'], active_held=bundle.report()['active_face_held'],
             missing_inactive=bundle.report()['face_records_missing_inactive']),
        sources=sources, geometry_recomputed=False, body_model_forwards=0, GPU_tasks=0,
        visual_review_passed=False, quality_accepted=False, physical_calibration_validated=False,
        static_scene_placement_validated=False, training_admitted=False, bundle_created=False,
        remaining_assembly_fields=['sequence_id and verified video identity must be added to manifest, roster, and poses',
            'fit contains duplicate betas/shared_betas; Bundle must contain shared_betas only',
            'final result/progress and final fit hash must be available; initial snapshot cannot be promoted'])
    bundle.assert_unchanged()
    return report, fit, rows, resolve


def _check_sources(sources):
    for item in sources:
        need(sha(item['path']) == item['sha256'], 'Source changed during assembly: ' + item['path'])


def assemble_final_bundle(run_dir, output_dir, *, source_run, sequence_id, project_root, path_maps=()):
    """Create a new local Bundle after final fit completion; never publish it."""
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    need(not io_path(output_dir).exists(), 'Output must be new; preserve previous bundles and evidence.')
    snapshot, plan_path = run_dir / 'fit.npz', run_dir / 'plan.json'
    result_path, progress_path = run_dir / 'result.json', run_dir / 'progress.json'
    result, progress = read(result_path), read(progress_path)
    need(result.get('state') in ('schedule_complete', 'budget_ended') and progress.get('state') == result['state']
         and progress.get('pid') is None, 'No terminal nonfailed final fit is available.')
    need(result.get('all_named_frames_exported') is True and result.get('fresh_optimizer') is True and
         result.get('old_optimized_fit_loaded') is False and result.get('all_K_E_unchanged') is True,
         'Final fit receipt does not establish the required source/fresh-state contract.')
    need(result.get('optimizer', {}).get('evaluations', 0) > 0, 'No actual fitting evaluations recorded; initial is not final.')
    need(result.get('output_fit_sha256') == sha(snapshot), 'Final result is not bound to this fit.npz.')
    need(result.get('plan_sha256') == sha(plan_path), 'Final result is not bound to this fit plan.')
    report, fit, rows, resolve = inspect_snapshot(snapshot, fit_plan=plan_path, source_run=source_run,
        sequence_id=sequence_id, project_root=project_root, path_maps=path_maps)
    need(result['frames'] == report['frame_count'] and result['reference_frame_id'] == report['reference_frame_id'] and
         result['camera_length_contract'] == report['upstream_camera_length_contract'], 'Final receipt roster/reference/units differ.')
    sources = report['sources'] + [descriptor(result_path), descriptor(progress_path)]
    _check_sources(sources)
    ids = fit['frame_ids'].astype(str).tolist()
    slots = {name: i for i, name in enumerate(ids)}
    video_hash = report['source_video']['sha256']
    records = []
    frame_base = Path(report['extraction_roster']['path']).parent
    for row in rows:
        name = str(row['frame']); i = slots[name]
        records.append(dict(frame=name, source_frame=int(fit['source_frames'][i]), sequence_id=sequence_id,
            video_sha256=video_hash, wh=row['wh'], image=str(resolve(row['image'], frame_base)), source_sha256=row['source_sha256'],
            mask=str(resolve(row['mask'], frame_base)), mask_sha256=row['mask_sha256'],
            K_integer_centers=fit['K_integer_centers'][i].tolist(), w2c=fit['w2c'][i].tolist()))
    io_path(output_dir).mkdir(parents=True, exist_ok=False)
    try:
        poses = {key: fit[key] for key in ('frame_ids', 'source_frames', 'image_sha256', 'mask_sha256',
                  'vertices_world', 'faces', 'shared_betas', 'K_integer_centers', 'w2c')}
        poses.update(sequence_id=np.asarray(sequence_id), video_sha256=np.asarray(video_hash))
        with io_path(output_dir / 'poses.npz').open('xb') as stream:
            np.savez_compressed(stream, **poses)
        write_new(output_dir / 'frames.json', dict(records=records))
        report.update(snapshot_role='final_fit_receipt_bound', sources=sources, optimizer_state=result['state'],
                      numerical_converged=result['optimizer'].get('numerical_converged', False),
                      final_snapshot_verified=True, remaining_assembly_fields=[],
                      bundle_loader_alone_does_not_recheck_video_or_fit=True)
        write_new(output_dir / 'provenance.json', report)
        manifest = dict(schema='shuolee-frame-pose/1', purpose='real', sequence_id=sequence_id,
            video_sha256=video_hash, coordinate_system='opencv_world_meters', pixel_domain='rectified_pinhole',
            pixel_convention='integer_centers', reference_frame_id=report['reference_frame_id'],
            frames=dict(path='frames.json', sha256=sha(output_dir / 'frames.json'), rows_key='records'),
            poses=dict(path='poses.npz', sha256=sha(output_dir / 'poses.npz')),
            bridge_provenance=dict(path='provenance.json', sha256=sha(output_dir / 'provenance.json')),
            camera_length_contract=report['output_camera_contract'],
            upstream_camera_length_contract=report['upstream_camera_length_contract'],
            physical_scale_measured=False, training_admitted=False, quality_accepted=False)
        # Manifest appears last so partial files cannot masquerade as a Bundle.
        _check_sources(sources)
        write_new(output_dir / 'manifest.json', manifest)
        bundle, verified = load_assembled_bundle(output_dir / 'manifest.json', project_root=project_root)
        write_new(output_dir / 'assembly-result.json', verified)
        return bundle, verified
    except BaseException:
        # Retain incomplete artifacts as evidence; no deletion or auto-resume.
        if not io_path(output_dir / 'assembly-failed.json').exists():
            import traceback
            write_new(output_dir / 'assembly-failed.json', dict(state='failed', traceback=traceback.format_exc(), training_admitted=False))
        raise


def load_assembled_bundle(manifest_path, *, project_root):
    """Recheck bridge provenance as well as the existing Bundle contract."""
    manifest_path = Path(manifest_path).resolve()
    need(not io_path(manifest_path.parent / 'assembly-failed.json').exists(), 'Assembly has a retained failure receipt.')
    m = read(manifest_path); desc = m['bridge_provenance']
    provenance = manifest_path.parent / desc['path']
    need(sha(provenance) == desc['sha256'], 'Bridge provenance hash mismatch.')
    evidence = read(provenance)
    need(evidence['snapshot_role'] == 'final_fit_receipt_bound' and evidence['final_snapshot_verified'] is True,
         'Only a final fit assembly may be loaded.')
    need(m['sequence_id'] == evidence['sequence_id'] and m['video_sha256'] == evidence['source_video']['sha256'] and
         m['camera_length_contract'] == evidence['output_camera_contract'], 'Bridge/manifest identity or units differ.')
    _check_sources(evidence['sources'])
    _, _, loader = _modules(project_root)
    bundle = loader(io_path(manifest_path))
    report = dict(schema=SCHEMA, manifest=descriptor(manifest_path), bundle_contract=bundle.report(),
        source_video_verified_by_bridge=True, final_fit_source_verified=True,
        sources=evidence['sources'], frames=len(bundle._frames), reference_frame_id=bundle.reference_frame_id,
        output_camera_scale_applications=0, geometry_recomputed=False, bundle_created=True,
        physical_scale_measured=False, static_scene_placement_validated=False,
        visual_review_passed=False, quality_accepted=False, training_admitted=False, GPU_tasks=0, body_model_forwards=0)
    return bundle, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW')
    sub = parser.add_subparsers(dest='mode', required=True)
    inspect = sub.add_parser('inspect', help='Read only, including initial snapshots; no Bundle is created.')
    inspect.add_argument('--snapshot', required=True); inspect.add_argument('--fit-plan', required=True)
    inspect.add_argument('--source-run', required=True); inspect.add_argument('--sequence-id', required=True)
    inspect.add_argument('--report')
    assemble = sub.add_parser('assemble', help='Assemble only a completed final fit.npz with terminal receipts.')
    assemble.add_argument('--run-dir', required=True); assemble.add_argument('--output-dir', required=True)
    assemble.add_argument('--source-run', required=True); assemble.add_argument('--sequence-id', required=True)
    verify = sub.add_parser('verify'); verify.add_argument('--manifest', required=True)
    args = parser.parse_args()
    maps = []
    for item in args.path_map:
        need('=' in item, 'path-map must be OLD=NEW')
        maps.append(tuple(item.split('=', 1)))
    if args.mode == 'inspect':
        report, _, _, _ = inspect_snapshot(args.snapshot, fit_plan=args.fit_plan, source_run=args.source_run,
            sequence_id=args.sequence_id, project_root=args.project_root, path_maps=maps)
        if args.report: write_new(args.report, report)
    elif args.mode == 'assemble':
        _, report = assemble_final_bundle(args.run_dir, args.output_dir, source_run=args.source_run,
            sequence_id=args.sequence_id, project_root=args.project_root, path_maps=maps)
    else:
        _, report = load_assembled_bundle(args.manifest, project_root=args.project_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
