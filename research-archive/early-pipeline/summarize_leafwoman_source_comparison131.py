"""Read-only CPU comparison of frozen 1000-update source-view snapshots.

Callable summarize(baseline, candidate, output); no prepare, render, or training.
Uses exactly the 130 mask/MinFilter(11)/quantized black-background RGB contract.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from plyfile import PlyData

ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
BASELINE = ROOT / 'leafwoman-half12-paired18k129'
CANDIDATE = ROOT / 'leafwoman-half12-dcbox-safe131'
REGIONS = ('full_raster_black_composite', 'mask', 'mask_eroded_5_derived_pixels')
LABELS = ('baseline1291000', 'candidate1311000')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(baseline, candidate, output):
    assert os.name != 'nt' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    baseline, candidate, output = map(Path, (baseline, candidate, output))
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

    bind(__file__)
    equivalence_path = candidate / 'root-input-equivalence.json'
    equivalence = read(equivalence_path)
    assert equivalence['all12_native_and_prepared_RGB_mask_validity_SHA_verified_equal']
    assert equivalence['WH_full_K_integer_corner_w2c_PTS_exact']
    assert equivalence['pixel_mapping_semantics_exact']
    for archive in equivalence['fit_archives']:
        assert archive['all_keys_dtypes_shapes_bytes_identical']
        for label, root in (('129', baseline), ('131', candidate)):
            assert bind(root / 'fit' / archive['file']) == archive['archive_sha256'][label]
    runs = dict(zip(LABELS, (baseline, candidate)))
    inputs = {label: read(root / 'data/training-inputs.json')['records'] for label, root in runs.items()}
    mappings = {label: read(root / 'pixel-mapping.json')['records'] for label, root in runs.items()}
    snapshots = {label: root / 'gaussian-run/step-001000' for label, root in runs.items()}
    results = {label: read(snapshot / 'result.json') for label, snapshot in snapshots.items()}
    for label in LABELS:
        assert len(inputs[label]) == len(mappings[label]) == 12
        result = results[label]
        assert result['step'] == result['optimizer_updates'] == result['global_iteration'] == 1000
        assert result['frames_rendered'] == 12 and result['active_sh_degree'] == 0
        assert result['checkpoint_phase'] == 'after_density_adam_zero_grad'
    assert [row['frame'] for row in inputs[LABELS[0]]] == [row['frame'] for row in inputs[LABELS[1]]]
    aggregate = {region: {label: dict(pixels=0, rgb_abs_sum=0., **({} if region == REGIONS[0] else
        dict(alpha_count_below_0_5=0, alpha_count_below_0_95=0))) for label in LABELS} for region in REGIONS}
    records = []
    for slot in range(12):
        left, right = [inputs[label][slot] for label in LABELS]
        for field in ('frame', 'wh', 'K', 'K_integer_centers', 'w2c', 'source_sha256', 'mask_sha256', 'validity_sha256'):
            assert left[field] == right[field], (slot, field)
        ma, mb = [mappings[label][slot] for label in LABELS]
        for field in ('frame', 'crop_xyxy', 'scale', 'integer_center_transform', 'derived_wh'):
            assert ma[field] == mb[field], (slot, field)
        for row in (left, right):
            for field, hashfield in (('image', 'source_sha256'), ('mask', 'mask_sha256'), ('validity', 'validity_sha256')):
                assert bind(row[field]) == row[hashfield]
            with Image.open(row['validity']) as image:
                assert np.all(np.asarray(image) == 255)
        with Image.open(left['image']) as image:
            source = np.asarray(image.convert('RGB'), dtype=np.uint8).astype(np.float32) / 255.
        with Image.open(left['mask']) as image:
            mask_image = image.convert('L')
        mask = np.asarray(mask_image, dtype=np.uint8)
        target = source * (mask.astype(np.float32) / 255.)[..., None]
        regions = {REGIONS[0]: np.ones(mask.shape, dtype=bool), REGIONS[1]: mask > 0,
            REGIONS[2]: np.asarray(mask_image.filter(ImageFilter.MinFilter(11))) > 0}
        frame = left['frame']
        measured = {}
        for label in LABELS:
            snapshot = snapshots[label]
            metadata_path = snapshot / (frame + '_cumulative-alpha.json')
            metadata = read(metadata_path)
            npy = snapshot / (frame + '_cumulative-alpha.npy')
            png = snapshot / (frame + '_cumulative-alpha.png')
            assert metadata['frame'] == frame and metadata['mask_sha256'] == left['mask_sha256']
            assert bind(npy) == metadata['alpha_npy_sha256']
            assert bind(png) == metadata['display_png_sha256']
            alpha = np.load(npy, allow_pickle=False)
            assert alpha.dtype == np.float32 and alpha.shape == mask.shape and np.isfinite(alpha).all()
            with Image.open(png) as image:
                display = np.asarray(image, dtype=np.uint8)
            assert np.array_equal(display, np.rint(np.clip(alpha, 0., 1.) * 255.).astype(np.uint8))
            rgb_path = snapshot / (frame + '_gaussian.png')
            rgb_sha = bind(rgb_path)
            with Image.open(rgb_path) as image:
                prediction = np.asarray(image.convert('RGB'), dtype=np.uint8).astype(np.float32) / 255.
            assert prediction.shape == target.shape
            error = np.abs(prediction - target)
            values = {}
            for region, selected in regions.items():
                pixels = int(selected.sum())
                total = float(error[selected].sum(dtype=np.float64))
                row = dict(pixels=pixels, rgb_abs_sum=total, quantized_RGB_L1=total / (3 * pixels))
                if region != REGIONS[0]:
                    chosen = alpha[selected]
                    count5 = int((chosen < .5).sum())
                    count95 = int((chosen < .95).sum())
                    reported = metadata['regions'][region]
                    assert reported['pixels'] == pixels
                    assert reported['fraction_below_0_5'] == count5 / pixels
                    assert reported['fraction_below_0_95'] == count95 / pixels
                    row.update(alpha_count_below_0_5=count5, alpha_count_below_0_95=count95,
                        alpha_fraction_below_0_5=count5 / pixels, alpha_fraction_below_0_95=count95 / pixels)
                for key in aggregate[region][label]:
                    aggregate[region][label][key] += row[key]
                values[region] = row
            measured[label] = dict(alpha_json_sha256=reads[str(metadata_path)], alpha_npy_sha256=reads[str(npy)],
                alpha_png_sha256=reads[str(png)], Gaussian_RGB_PNG_sha256=rgb_sha,
                raw_float32_display_and_reported_statistics_verified=True, regions=values)
        records.append(dict(frame=frame, derived_wh=left['wh'], identical_source_RGB_sha256=left['source_sha256'],
            identical_mask_sha256=left['mask_sha256'], identical_validity_sha256=left['validity_sha256'], models=measured))
    for region in REGIONS:
        for label in LABELS:
            row = aggregate[region][label]
            row['quantized_RGB_L1'] = row['rgb_abs_sum'] / (3 * row['pixels'])
            if region != REGIONS[0]:
                for suffix in ('0_5', '0_95'):
                    row['alpha_fraction_below_' + suffix] = row['alpha_count_below_' + suffix] / row['pixels']
        keys = ['quantized_RGB_L1'] + ([] if region == REGIONS[0] else ['alpha_fraction_below_0_5', 'alpha_fraction_below_0_95'])
        aggregate[region]['candidate_minus_baseline'] = {key: aggregate[region][LABELS[1]][key] - aggregate[region][LABELS[0]][key] for key in keys}
    colors = {}
    for label in LABELS:
        ply = snapshots[label] / 'world.ply'
        assert bind(ply) == results[label]['world_ply_sha256']
        vertices = PlyData.read(ply)['vertex'].data
        assert len(vertices) == results[label]['points']
        dc = np.stack([vertices['f_dc_' + str(i)] for i in range(3)], axis=-1).astype(np.float32)
        rgb = dc * np.float32(.28209479177387814) + np.float32(.5)
        colors[label] = dict(points=len(vertices), world_ply_sha256=reads[str(ply)],
            DC_min=float(dc.min()), DC_max=float(dc.max()), SH0_decoded_RGB_min_float32=float(rgb.min()),
            SH0_decoded_RGB_max_float32=float(rgb.max()), points_any_RGB_below0=int((rgb < 0).any(axis=-1).sum()),
            points_any_RGB_above1=int((rgb > 1).any(axis=-1).sum()),
            points_outside_bounds_beyond_float32_tolerance_1e_7=int(((rgb < -1e-7) | (rgb > 1 + 1e-7)).any(axis=-1).sum()))
    projection = read(candidate / 'dc-projection-1000-receipt.json')
    assert projection['original_Adam_steps'] == projection['DC_projection_steps'] == 1000
    assert projection['complete_checkpoint_after_projection'] and projection['Adam_moments_preserved']
    for path, digest in reads.items():
        assert sha(path) == digest, path
    result = dict(updated=datetime.now(timezone.utc).isoformat(), schema='leafwoman-frozen-source-comparison131-v1',
        scope='Only1291000 versus1311000, each1000 actualoptimizerupdates; no18k/ceiling control.',
        helper_sha256=reads[str(Path(__file__))], source_roots={label: str(root) for label, root in runs.items()},
        input_equivalence_receipt=dict(path=str(equivalence_path), sha256=reads[str(equivalence_path)]),
        snapshot_contract={label: results[label] for label in LABELS}, records=records,
        area_weighted_aggregate=aggregate, point_SH0_color_ranges=colors, DC_projection_receipt=projection,
        alpha_contract='Rawfloat32 cumulative all-layer1-T with unchanged native sorting/truncation, not per-point/front-surface opacity or physical geometry truth. Mask=m>0; diagnostic interior=PIL MinFilter(11)>0,5 derived pixels/2.5native atROI2x. Masks unchanged.',
        RGB_contract='QuantizedRGB PNG/255 versus identical sourceRGB/255 times sourceMask/255 with black elsewhere. Mean absolute error per RGBchannel, area weighted separately over full derived raster/mask/interior. Not native float paired-background training loss, SSIM, unseen-view quality or physical geometry truth.',
        uncertainty='One fresh configuration comparison; no repeated-CUDA variance control. Small metrics differences do not establish robust improvement or quality acceptance; actual image review is separate.',
        numeric_boundary_note='1e-7 is only a float32 color-boundary reporting tolerance, not an image/geometry acceptance threshold.',
        all_read_files_SHA_unchanged=reads, source_fit_mask_camera_model_or_training_modified=False,
        GPU_started=False, rerendered=False, quality_accepted=False)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(dict(path=str(output), sha256=sha(output), helper_sha256=sha(__file__),
        frames=len(records), aggregate=aggregate, SH0=colors), indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--candidate', type=Path, default=CANDIDATE)
    parser.add_argument('--output', type=Path, default=CANDIDATE / 'gaussian-run/step-001000/summary131.json')
    args = parser.parse_args()
    summarize(args.baseline, args.candidate, args.output)
