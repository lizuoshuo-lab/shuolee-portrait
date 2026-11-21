"""Measure frozen ear and static hotspot regions in the official lossless R7 comparisons."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_smpl_delivery_round5_ear016 as native


ROOT = Path(__file__).resolve().parents[1] / 'outputs'
R6 = ROOT / 'smpl-delivery-round6'
R7 = ROOT / 'smpl-delivery-round7'
R6_SHA = 'a728539832d89b4f4d11b644d1d2398f1eedb157f13e021eaf2234e3b5658c1c'
EAR_SHA = 'de0b15c4505a2a42b8bc450c1800066b220cb5332fae93ae3288c2d85e0884d2'
CAUSAL_SHA = '8d44e129dad6bfdacdb38dcb1c794723c930d4739b26e450df10f6f314a73939'
COMMIT = 'dd99e967e9c8f146b99a45df0855f3f7f4bade78'
OLD_LABEL, NEW_LABEL = 'round6_8c722b9', f'round7_{COMMIT[:7]}'
LAB = Path('/opt/gs-slam-lab')
require, pin, read, sha, choose = native.require, native.pin, native.read, native.sha, native.choose


def png(path, expected):
    path = pin(path, expected)
    with Image.open(path) as image:
        require(image.format == 'PNG', 'A registered panel must be lossless PNG.')
        return np.asarray(image.convert('RGB'))


def same_bytes(old, new):
    return old.keys() == new.keys() and all(old[k].shape == new[k].shape and old[k].dtype == new[k].dtype
                                             and old[k].tobytes() == new[k].tobytes() for k in old)


def endpoints(subject, baseline_only):
    published = read(pin(R6/'receipt.json', R6_SHA))['subjects'][subject]
    old_review = pin(published['review'], published['review_sha256'])
    old_run = pin(published['result'], published['result_sha256'])
    old = read(old_review)
    old_plan = read(pin(old_review.parent/'plan.json', old['plan_sha256']))
    require(old['subject'] == subject and old['state'] == 'round6_delivery_views_complete', 'R6 review is incomplete.')
    if baseline_only:
        return old_review, old, old_plan, old_run, None
    root = LAB/f'shuolee-smpl-delivery-round7-{subject}'
    review = root/'view-comparison/result.json'
    new = read(review)
    plan = read(pin(review.parent/'plan.json', new['plan_sha256']))
    require(new['subject'] == subject and new['state'] == 'round7_delivery_views_complete'
            and new['optimizer_updates'] == new['refits'] == 0 and plan['new_commit'] == COMMIT
            and plan['baseline_review_sha256'] == sha(old_review) and Path(plan['baseline_review']) == old_review,
            'Use the exact completed R7 reviewer and pinned R6 endpoint.')
    run = pin(Path(plan['new']['run'])/'result.json', plan['new']['terminal_sha256'])
    require(run.parent == root/'run' and read(run)['final_snapshot'] == 'step-010000', 'Unexpected R7 terminal.')
    return old_review, old, old_plan, old_run, (review, new, plan, run)


def frozen_replay(path):
    result = read(path)
    replay_path = pin(path.parent/result['bound_replay'], result['bound_replay_sha256'])
    replay = read(replay_path)
    mesh = pin(replay_path.parent/replay['mesh_file'], replay['mesh_sha256'])
    rows = pin(path.parent/'data/training-inputs.json', replay['training_inputs_sha256'])
    cameras = pin(path.parent/'gaussian-run/training-cameras.json', replay['training_camera_sha256'])
    require(replay['state'] == 'completed' and replay['optimizer_updates'] == 0, 'Frozen replay incomplete.')
    identities = [(v['frame'],v['pose_index'],v['source_frame_index'],v['source_sha256'],v['mask_sha256'],v['camera'])
                  for v in replay['records']]
    return dict(mesh=mesh, mesh_sha256=replay['mesh_sha256'], rows_sha256=sha(rows), cameras_sha256=sha(cameras),
                fit_sha256=sha(path.parent/'fit/fitted.npz'), holdouts=replay['retained_holdouts'], records=identities)


def verify_replay(old, new):
    old, new = frozen_replay(old), frozen_replay(new)
    require(all(old[k] == new[k] for k in ('rows_sha256','cameras_sha256','fit_sha256','holdouts','records')),
            'R7 changed source pixels, fit, training masks, cameras, or retained holdouts.')
    with np.load(old['mesh'], allow_pickle=False) as a, np.load(new['mesh'], allow_pickle=False) as b:
        require(same_bytes(dict(a),dict(b)), 'R7 posed replay geometry differs bitwise.')
    return dict(frozen_geometry_bitwise_equal=True, old_mesh_sha256=old['mesh_sha256'], new_mesh_sha256=new['mesh_sha256'],
                fit_sha256=old['fit_sha256'], rows_sha256=old['rows_sha256'], cameras_sha256=old['cameras_sha256'],
                replay_frames=len(old['records']), retained_holdouts=old['holdouts'])


def ear(baseline_only):
    audit = read(pin(native.ROUND4/'ear-016-source-audit.json', native.EAR_SHA))
    baseline = read(pin(R6/'ear-016-reevaluation.json', EAR_SHA))
    old_path, old, _, old_run, candidate = endpoints('016', baseline_only)
    old_native = native.camera_record(old['new'], audit)
    old_mesh_path, old_mesh = native.posed_mesh(old_run, audit, old_native)
    coords, identity = audit['roi'], audit['identity']
    roi = Image.new('1', tuple(coords['wh']))
    ImageDraw.Draw(roi).polygon([tuple(p) for p in coords['polygon_xy_native_integer_centers_inclusive']], fill=1)
    raster = np.asarray(roi, dtype=np.bool_)
    y,x = np.nonzero(raster)
    roi_sha = hashlib.sha256(raster.tobytes(order='C')).hexdigest()
    require(len(x) == coords['pixels'] == 574 and roi_sha == coords['roi_sha256'] == baseline['roi']['sha256'],
            'Frozen ear polygon differs.')
    crop = baseline['roi']['crop_xyxy']
    source_id, mask_id = identity['source_rgb'], identity['score_mask']
    with Image.open(pin(source_id['path'],source_id['sha256'])) as raw, Image.open(pin(mask_id['path'],mask_id['sha256'])) as matte:
        require(raw.size == matte.size == tuple(coords['wh']) and np.all(np.asarray(matte.convert('L'))[y,x] == 255),
                'Native ear source or matte support differs.')
        source = np.asarray(raw.convert('RGB').crop(crop))
    xy = np.column_stack((x,y))
    local = xy-crop[:2]
    require(np.all(local.min(0)>=0) and np.all(local.max(0)<np.asarray(source.shape[:2])[::-1]), 'Ear ROI crosses crop.')
    owners = native.mesh_coverage(old_mesh['vertices_world'][old_native['native_camera']['pose_index']], old_mesh['faces'],
                                  old_native['native_camera'], xy)
    on_mesh = owners >= 0
    require(int((~on_mesh).sum()) == audit['mesh_support']['no_nearest_triangle_pixels'] == 267
            and np.unique(owners[on_mesh]).tolist() == audit['mesh_support']['distinct_right_ear_owner_faces'], 'Ear mesh support differs.')
    pane, strips = native.panel(old_path,old,source,crop,['round5_9013c26',OLD_LABEL])
    prior_metrics = native.metrics(strips[-1],source,local,on_mesh,old_path,old_native)
    require(prior_metrics == baseline['round6']['metrics'] and sha(pane) == baseline['round6']['panel_sha256']
            and sha(old_mesh_path) == baseline['round6']['posed_mesh_sha256'], 'Published R6 ear scores do not reproduce.')
    out = dict(schema='shuolee-smpl-round7-ear016-remeasurement-v1',state='baseline_verified',subject='016',frame=native.FRAME,
        roi=baseline['roi'],source_sha256=source_id['sha256'],mask_sha256=mask_id['sha256'],native_camera=old_native['native_camera'],
        round6=baseline['round6'],gpu_calls=0,optimizer_updates=0,quality_accepted=False)
    if baseline_only:
        return out
    path, new, _, run = candidate
    rec = native.camera_record(new['new'],audit)
    require(rec['native_camera'] == old_native['native_camera'] and rec['pose_sha256'] == old_native['pose_sha256'], 'R7 ear pose changed.')
    geometry = verify_replay(old_run,run)
    panel, rows = native.panel(path,new,source,crop,[OLD_LABEL,NEW_LABEL])
    require(np.array_equal(rows[0],strips[-1]), 'R7 first row differs from pinned R6 ear pixels.')
    metrics = native.metrics(rows[1],source,local,on_mesh,path,rec)
    out.update(state='remeasured',**geometry,round7=dict(panel=str(panel),panel_sha256=sha(panel),review_sha256=sha(path),metrics=metrics),
        delta_round7_minus_round6={region:{key:metrics[region][key]-prior_metrics[region][key]
            for key in ('alpha_u8_lt_166','alpha_u8_lt_243','black_rgb_mae_01')} for region in ('all_roi','no_mesh','right_ear_owner_mesh')})
    return out


def mask(polygons, size):
    image = Image.new('1', tuple(size))
    painter = ImageDraw.Draw(image)
    for polygon in polygons:
        painter.polygon([tuple(p) for p in polygon],fill=1)
    return np.asarray(image, dtype=bool)


def static_metrics(before, after, hot, control):
    def values(data, region):
        alpha = data[2][region].astype(float)/255.
        return dict(alpha_normalized_mean=float(alpha.mean()),alpha_sum=float(alpha.sum()),alpha_at_least_half_pixels=int((alpha >= .5).sum()))

    def difference(region):
        change = after[2][region].astype(int)-before[2][region].astype(int)
        return dict(pixels=int(region.sum()),round6=values(before,region),round7=values(after,region),
            alpha_delta_normalized_mean=float(change.mean()/255.),alpha_abs_delta_normalized_mean=float(abs(change).mean()/255.),
            black_rgb_mae_01=float(abs(after[0][region].astype(int)-before[0][region].astype(int)).mean()/255.),
            white_rgb_mae_01=float(abs(after[1][region].astype(int)-before[1][region].astype(int)).mean()/255.),
            alpha_gain_ge_32_codes=int((change>=32).sum()),alpha_loss_ge_32_codes=int((change<=-32).sum()))

    y,x = np.nonzero(hot|control)
    context = np.zeros_like(hot)
    context[max(0,y.min()-24):y.max()+25,max(0,x.min()-24):x.max()+25] = True
    return dict(hotspot=difference(hot),nearby_control=difference(control),
                adjacent_context_excluding_hotspot_and_control=difference(context&~(hot|control)))


def hotspot(baseline_only):
    causal = read(pin(R7/'causal-diagnostic/receipt.json', CAUSAL_SHA))
    require(causal['state'] == 'completed' and causal['delivery_sha256'] == R6_SHA, 'R6 causal diagnostic differs.')
    out = dict(schema='shuolee-smpl-round7-static-hotspot-comparison-v1',state='baseline_verified' if baseline_only else 'remeasured',
        r6_causal_receipt_sha256=CAUSAL_SHA,round7_commit=COMMIT,gpu_calls=0,optimizer_updates=0,quality_accepted=False,subjects={},
        scope='Synthetic R6/R7 same-camera comparisons. No source RGB truth for these angles. ROI gains or losses do not establish disappearance, relocation, or global appearance quality.')
    for subject, item in causal['subjects'].items():
        evidence = read(pin(item['path'],item['sha256']))
        old_path, old, old_plan, old_run, candidate = endpoints(subject,baseline_only)
        view = item['view']
        original = old['static_render_receipts']['round6'][view]
        crop = old_plan['static_views'][view]['crop_xyxy']
        w,h = np.diff(np.asarray(crop).reshape(2,2),axis=0)[0]
        require(evidence['review_sha256'] == sha(old_path) and evidence['archived_png_sha256'] == original['sha256']
                and evidence['crop'] == crop and evidence['camera'] == old_plan['static_views'][view]['camera'], 'R6 causal source camera or crop differs.')
        source = png(old_path.parent/original['file'],original['sha256'])
        require(source.shape == (h,3*w,3), 'Frozen static panel dimensions differ.')
        hot = mask(evidence['hotspot_poly_crop_local'],(w,h))
        control = mask(evidence['control_poly_crop_local'],(w,h))
        require(int(hot.sum()) == evidence['hotspot_pixels'] and int(control.sum()) == evidence['control_pixels']
                and not (hot&control).any(), 'Locked causal polygons differ.')

        def split(row):
            black,white,alpha = np.split(row,3,axis=1)
            require(np.array_equal(alpha[...,0],alpha[...,1]) and np.array_equal(alpha[...,0],alpha[...,2]),'Alpha panel is not grayscale.')
            return black,white,alpha[...,0]

        before = split(source)
        expected = evidence['interventions']['hide_top']['hotspot']['alpha_sum_before']
        require(abs(before[2][hot].sum()/255.-expected) <= .5*hot.sum()/255.+1e-4,
                'Pinned R6 hotspot does not reproduce causal float alpha within PNG quantization.')
        entry = dict(view=view,hotspot_poly_crop_local=evidence['hotspot_poly_crop_local'],
            control_poly_crop_local=evidence['control_poly_crop_local'],round6_png_sha256=original['sha256'],
            round6_causal_subject_sha256=item['sha256'],crop_xyxy=crop)
        if not baseline_only:
            path,new,plan,run = candidate
            rendered = new['static_render_receipts']['round7'][view]
            panel = choose(new['panels'],view=view,scale=1.0)
            require(panel['columns'] == ['black','white','alpha'] and panel['rows'] == [OLD_LABEL,NEW_LABEL]
                    and panel['header_pixels'] == 0 and panel['crop_xyxy'] == crop and rendered['camera'] == original['camera']
                    and rendered['source_rgb_compared'] is False and rendered['static_ply_sha256'] == read(run)['sha256'],
                    'The matched static camera, crop, or PLY changed.')
            require(plan['static_views'][view] == old_plan['static_views'][view], 'Registered static orbit changed.')
            candidate_png = png(path.parent/rendered['file'],rendered['sha256'])
            combined = png(path.parent/panel['file'],panel['sha256'])
            require(combined.shape == (2*h,3*w,3) and np.array_equal(combined[:h],source)
                    and np.array_equal(combined[h:],candidate_png),'Matched rows differ from their directly rendered PNGs.')
            entry.update(**verify_replay(old_run,run),round7_png_sha256=rendered['sha256'],matched_panel_sha256=panel['sha256'],
                         metrics=static_metrics(before,split(candidate_png),hot,control))
        out['subjects'][subject] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=('ear','hotspot'))
    parser.add_argument('--check-baseline',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    data = (ear if args.part == 'ear' else hotspot)(args.check_baseline)
    text = json.dumps(data,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x',encoding='utf-8') as f:
            f.write(text)
    print(text,end='')


if __name__ == '__main__':
    main()
