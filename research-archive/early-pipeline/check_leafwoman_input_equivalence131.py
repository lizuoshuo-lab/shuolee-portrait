"""CPU-only prepared-input equality, 131 versus frozen 129 and 130.

No prepare, feature extraction, tensor/CUDA import, rendering, or admission.
This verifies an existing experiment's observations, not its reconstruction.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
RUNS = {
    '129': ROOT / 'leafwoman-half12-paired18k129',
    '130': ROOT / 'leafwoman-half12-dcbox1000-130',
    '131': ROOT / 'leafwoman-half12-dcbox-safe131',
}
EXPECTED_FRAMES = [4, 7, 10, 13, 14, 18, 23, 24, 27, 33, 34, 39]
ROW_SEMANTICS = (
    'frame', 'name', 'source_frame_index', 'time_seconds',
    'target_time_seconds', 'sampling_error_seconds', 'source_sha256',
    'mask_sha256', 'validity_sha256', 'wh', 'K', 'K_integer_centers', 'w2c',
    'official_prediction_slot', 'observation_weight', 'camera_kind',
    'camera_registration', 'mask_kind', 'initial_box_xyxy',
    'other_person_boxes_xyxy', 'face_box_xyxy', 'face_region_source',
    'face_region_is_observed', 'visible_eyes_open',
    'source_joint_group_supported', 'source_pose_episode', 'foot_out_of_frame',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run():
    assert os.name != 'nt' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = RUNS['131'] / 'root-input-equivalence.json'
    assert not output.exists(), output
    reads = {}

    def bind(path):
        path = Path(path)
        value = sha(path)
        assert str(path) not in reads or reads[str(path)] == value
        reads[str(path)] = value
        return value

    def read(path):
        bind(path)
        return json.loads(Path(path).read_text(encoding='utf-8'))

    review_path = RUNS['131'] / 'validity-review.json'
    original_review = read(review_path)
    assert original_review['actual_visual_review'] is False
    assert original_review['training_admitted'] is False
    manifests = {key: read(root / 'data/training-inputs.json') for key, root in RUNS.items()}
    mappings = {key: read(root / 'pixel-mapping.json') for key, root in RUNS.items()}
    for root in RUNS.values():
        bind(root / 'plan.json')
    bind(Path(__file__))

    def check_row(row):
        images = {}
        for field, hashfield in (('image', 'source_sha256'), ('mask', 'mask_sha256'), ('validity', 'validity_sha256')):
            images[field] = bind(row[field])
            assert images[field] == row[hashfield], (row['frame'], field)
            with Image.open(row[field]) as image:
                assert list(image.size) == row['wh']
                if field == 'validity':
                    assert np.all(np.asarray(image) == 255), (row['frame'], field)
        k = np.asarray(row['K'], dtype=np.float64)
        ki = np.asarray(row['K_integer_centers'], dtype=np.float64)
        expected_corner_k = ki.copy()
        expected_corner_k[:2, 2] += 0.5
        assert k.shape == (3, 3) and np.array_equal(k, expected_corner_k)
        assert np.array_equal(np.asarray(row['w2c']), np.eye(4))
        return images

    for label in RUNS:
        assert len(manifests[label]['records']) == 12
        assert len(mappings[label]['records']) == 12
        frames = [int(row['frame'].split('_')[-1]) for row in manifests[label]['records']]
        assert frames == EXPECTED_FRAMES, (label, frames)
        assert [row['frame'] for row in mappings[label]['records']] == [row['frame'] for row in manifests[label]['records']]
    records = []
    for slot in range(12):
        prepared = {label: manifests[label]['records'][slot] for label in RUNS}
        mapping = {label: mappings[label]['records'][slot] for label in RUNS}
        native = {label: mapping[label]['source'] for label in RUNS}
        hashes = {}
        for domain, rows in (('prepared_ROI2x', prepared), ('original_native', native)):
            by_run = {label: check_row(row) for label, row in rows.items()}
            assert by_run['129'] == by_run['130'] == by_run['131']
            for key in ROW_SEMANTICS:
                values = [row[key] for row in rows.values()]
                assert values[0] == values[1] == values[2], (domain, prepared['131']['frame'], key)
            hashes[domain] = by_run['131']
        normalized_mapping = [{key: value for key, value in row.items() if key != 'source'} for row in mapping.values()]
        assert normalized_mapping[0] == normalized_mapping[1] == normalized_mapping[2]
        m = mapping['131']
        assert m['derived_wh'] == prepared['131']['wh'] and m['scale'] == 2
        left, top, right, bottom = m['crop_xyxy']
        assert m['derived_wh'] == [2 * (right - left), 2 * (bottom - top)]
        affine = np.asarray(m['integer_center_transform'], dtype=np.float64)
        assert np.array_equal(affine, [[2., 0., 0.5 - 2 * left], [0., 2., 0.5 - 2 * top], [0., 0., 1.]])
        mapped_k = affine @ np.asarray(native['131']['K_integer_centers'], dtype=np.float64)
        recorded_k = np.asarray(prepared['131']['K_integer_centers'], dtype=np.float64)
        # Fixed machine-roundoff bound for a matrix identity, not a pixel-quality gate.
        roundoff = 64 * np.finfo(np.float64).eps * max(1., float(abs(recorded_k).max()))
        assert float(abs(mapped_k - recorded_k).max()) <= roundoff
        records.append(dict(frame=prepared['131']['frame'], wh=prepared['131']['wh'], native_wh=native['131']['wh'],
            same_129_130_131_file_hashes=hashes, exact_camera_and_PTS_semantics=True,
            exact_pixel_mapping_semantics=True, crop_xyxy=m['crop_xyxy'], scale=m['scale']))

    archives = []
    for filename in ('fitted.npz', 'surface-cache.npz'):
        paths = {label: root / 'fit' / filename for label, root in RUNS.items()}
        archive_hashes = {label: bind(path) for label, path in paths.items()}
        loaded = {label: np.load(path, allow_pickle=False) for label, path in paths.items()}
        try:
            keys = sorted(loaded['129'].files)
            assert keys == sorted(loaded['130'].files) == sorted(loaded['131'].files)
            arrays = []
            for key in keys:
                values = [np.asarray(loaded[label][key]) for label in RUNS]
                assert not values[0].dtype.hasobject
                assert all(value.dtype == values[0].dtype and value.shape == values[0].shape for value in values)
                raw = [value.tobytes(order='C') for value in values]
                assert raw[0] == raw[1] == raw[2], (filename, key)
                arrays.append(dict(key=key, dtype=str(values[0].dtype), shape=list(values[0].shape),
                    identical_array_bytes_sha256=hashlib.sha256(raw[0]).hexdigest()))
            archives.append(dict(file=filename, archive_sha256=archive_hashes, array_count=len(keys),
                all_keys_dtypes_shapes_bytes_identical=True, arrays=arrays))
        finally:
            for archive in loaded.values():
                archive.close()

    for path, digest in reads.items():
        assert sha(path) == digest, path
    result = dict(updated=datetime.now(timezone.utc).isoformat(), schema='leafwoman-prepared-input-equivalence131-v1',
        state='CPU_input_equivalence_verified', helper=str(Path(__file__).resolve()), helper_sha256=reads[str(Path(__file__))],
        runs={key: str(value) for key, value in RUNS.items()}, frames=EXPECTED_FRAMES, records=records,
        fit_archives=archives, all12_native_and_prepared_RGB_mask_validity_SHA_verified_equal=True,
        full_raster_validity_all255=True, WH_full_K_integer_corner_w2c_PTS_exact=True,
        pixel_mapping_semantics_exact=True, all_read_files_SHA_unchanged=reads,
        validity_review_preserved=dict(path=str(review_path), sha256=reads[str(review_path)],
            original_actual_visual_review=False, original_training_admitted=False, modified=False),
        scope='Input equality only. ROI2x creates derived pixels, not new optical detail. K remains conditional; identity w2c is body-relative, not a validated static world trajectory.',
        training_admitted=False, quality_accepted=False, GPU_started=False, prepare_repeated=False,
        raw_mask_camera_fit_model_or_training_modified=False)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(dict(path=str(output), sha256=sha(output), helper_sha256=sha(__file__),
        frames=EXPECTED_FRAMES, native_and_prepared_files_equal=True, fit_array_counts=[x['array_count'] for x in archives],
        validity_review_preserved=True, training_admitted=False, quality_accepted=False), indent=2))


if __name__ == '__main__':
    run()
