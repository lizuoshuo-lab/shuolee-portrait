"""Compare R5 and R6 on the exact native 016 f051 right-ear audit pixels."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_smpl_delivery_round5_ear016 as ear


ROUND5 = ear.ROUND4.with_name('smpl-delivery-round5')
ROUND5_RECEIPT_SHA = '6db51a1c18534fa9b7965383d6162107c69ccf3e0335fa32cb05ebafca31cf5d'
ROUND5_EAR_SHA = '217ad79b38c697cba5d639baf27dbbd5e1cd09005b3042e17e52e9887f741fbb'
ROUND5_EAR_TOOL_SHA = 'c2e57cec61b95726043c0af18dec71d967d6ef2438a3a98a59d4c5a130f0e3ed'
OLD_ROW = 'round5_9013c26'


def measure(review=None):
    ear.pin(ear.__file__, ROUND5_EAR_TOOL_SHA)
    audit = ear.read(ear.pin(ear.ROUND4 / 'ear-016-source-audit.json', ear.EAR_SHA))
    prior = ear.read(ear.pin(ROUND5 / 'receipt.json', ROUND5_RECEIPT_SHA))
    previous = prior['subjects']['016']
    previous_review = ear.pin(previous['review'], previous['review_sha256'])
    previous_result = ear.pin(previous['result'], previous['result_sha256'])
    old_receipt = ear.read(previous_review)
    baseline = ear.read(ear.pin(ROUND5 / 'ear-016-reevaluation.json', ROUND5_EAR_SHA))
    ear.require(prior['ear_016']['sha256'] == ROUND5_EAR_SHA and baseline['state'] == 'remeasured'
                and old_receipt['state'] == 'round5_delivery_views_complete' and old_receipt['subject'] == '016'
                and baseline['round5']['review_sha256'] == previous['review_sha256'], 'The published R5 ear baseline differs.')
    old_native = ear.camera_record(old_receipt['new'], audit)
    old_mesh_path, old_mesh = ear.posed_mesh(previous_result, audit, old_native)
    roi, identity = audit['roi'], audit['identity']
    image = Image.new('1', tuple(roi['wh']))
    ImageDraw.Draw(image).polygon([tuple(xy) for xy in roi['polygon_xy_native_integer_centers_inclusive']], fill=1)
    raster = np.asarray(image, dtype=np.bool_)
    y, x = np.nonzero(raster)
    xy = np.column_stack((x, y))
    roi_sha = hashlib.sha256(raster.tobytes(order='C')).hexdigest()
    ear.require(len(xy) == roi['pixels'] == 574 and roi_sha == roi['roi_sha256'] == baseline['roi']['sha256'],
                'The frozen native ear polygon differs.')
    image_pin, matte_pin = identity['source_rgb'], identity['score_mask']
    crop = baseline['roi']['crop_xyxy']
    with Image.open(ear.pin(image_pin['path'], image_pin['sha256'])) as rgb, Image.open(ear.pin(matte_pin['path'], matte_pin['sha256'])) as matte:
        ear.require(rgb.size == matte.size == tuple(roi['wh']) and np.all(np.asarray(matte.convert('L'))[y, x] == 255),
                    'The original native RGB or ear matte support changed.')
        source = np.asarray(rgb.convert('RGB').crop(crop))
    local = xy - crop[:2]
    ear.require(np.all(local.min(0) >= 0) and np.all(local.max(0) < np.asarray(source.shape[:2])[::-1]), 'The crop does not contain the ROI.')
    old_path, old_strips = ear.panel(previous_review, old_receipt, source, crop, [ear.OLD_ROW, OLD_ROW])
    slot = old_native['native_camera']['pose_index']
    owners = ear.mesh_coverage(old_mesh['vertices_world'][slot], old_mesh['faces'], old_native['native_camera'], xy)
    on_mesh = owners >= 0
    ear.require(int((~on_mesh).sum()) == audit['mesh_support']['no_nearest_triangle_pixels'] == 267
                and np.unique(owners[on_mesh]).tolist() == audit['mesh_support']['distinct_right_ear_owner_faces'],
                'Round-five nearest ear-mesh support differs from frozen audit.')
    old_metrics = ear.metrics(old_strips[-1], source, local, on_mesh, previous_review, old_native)
    ear.require(old_metrics == baseline['round5']['metrics'] and ear.sha(old_path) == baseline['round5']['panel_sha256']
                and ear.sha(old_mesh_path) == baseline['round5']['posed_mesh_sha256'], 'Published round-five measurements no longer reproduce.')
    result = dict(schema='shuolee-smpl-round6-ear016-remeasurement-v1', state='baseline_verified', subject='016', frame=ear.FRAME,
        roi=baseline['roi'], source_frame_index=baseline['source_frame_index'], source_sha256=image_pin['sha256'],
        mask_sha256=matte_pin['sha256'], native_camera=old_native['native_camera'], round5=baseline['round5'],
        round5_ear_receipt_sha256=ROUND5_EAR_SHA, optimizer_updates=0, gpu_calls=0, quality_accepted=False)
    if review is None:
        return result

    receipt = ear.read(review)
    plan = ear.read(ear.pin(review.parent / 'plan.json', receipt['plan_sha256']))
    commit = plan['new_commit']
    ear.require(receipt['state'] == 'round6_delivery_views_complete' and receipt['subject'] == '016'
                and receipt['optimizer_updates'] == receipt['refits'] == 0 and re.fullmatch('[0-9a-f]{40}', commit)
                and Path(plan['baseline_review']) == previous_review and plan['baseline_review_sha256'] == previous['review_sha256'],
                'Use the completed matched R6 review and its registered R5 baseline.')
    native = ear.camera_record(receipt['new'], audit)
    ear.require(native['native_camera'] == old_native['native_camera'] and native['pose_sha256'] == old_native['pose_sha256'],
                'The registered R6 native camera or fit pose changed.')
    new_result = ear.pin(Path(plan['new']['run']) / 'result.json', plan['new']['terminal_sha256'])
    mesh_path, mesh = ear.posed_mesh(new_result, audit, native)
    ear.require(old_mesh.keys() == mesh.keys() and all(old_mesh[k].shape == mesh[k].shape
                and old_mesh[k].dtype == mesh[k].dtype and old_mesh[k].tobytes() == mesh[k].tobytes() for k in old_mesh),
                'The R6 posed mesh differs bitwise from R5.')
    candidate_path, strips = ear.panel(review, receipt, source, crop, [OLD_ROW, f'round6_{commit[:7]}'])
    ear.require(np.array_equal(old_strips[-1], strips[0]), 'The matched first row differs from the published R5 PNG.')
    metrics = ear.metrics(strips[1], source, local, on_mesh, review, native)
    result.update(state='remeasured', frozen_posed_mesh_arrays_bitwise_equal=True,
        round6=dict(commit=commit, panel=str(candidate_path), panel_sha256=ear.sha(candidate_path),
                    review_sha256=ear.sha(review), posed_mesh_sha256=ear.sha(mesh_path), metrics=metrics),
        delta_round6_minus_round5={region: {key: metrics[region][key] - old_metrics[region][key]
            for key in ('alpha_u8_lt_166','alpha_u8_lt_243','black_rgb_mae_01')}
            for region in ('all_roi','no_mesh','right_ear_owner_mesh')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--review', type=Path)
    mode.add_argument('--check-baseline', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.dumps(measure(args.review), indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as output:
            output.write(data)
    print(data, end='')


if __name__ == '__main__':
    main()
