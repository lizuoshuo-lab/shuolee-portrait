"""Fit the 021 candidate pool with the existing independent SMPLify-X worker."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import time
import types

import numpy as np
from PIL import Image

import official_smplifyx_entry123 as entry
import run_leafwoman_front24_cpu145 as original
from review_full_sequence123 import relocate
from run_official_sequence123 import save_npz
from smplify_score_contract123 import relative_confidence
from update_leafwoman129_state import ROOT, atomic, read


NATIVE = ROOT/'seated021-native148'
OUT = ROOT/'seated021-fit148'
CONTRACT = NATIVE/'source-admission.json'
ARCHIVE = OUT/'fit-observations.npz'
CONTACTS = OUT/'contacts.json'
WORKERS = 4
SOURCE_IDX = np.r_[np.arange(23), np.arange(91,133), np.arange(40,91)]
TARGET_IDX = np.r_[entry.COCO, np.arange(19,25), np.arange(25,67), np.arange(67,118)]


def prepare():
    assert not OUT.exists()
    native = read(NATIVE/'result.json')
    source = read(CONTRACT)
    assert source['video_sha256'] == native['video_sha256'] and source['actual_source_and_observation_review']
    assert (source['front_view_target'],source['full_circle_target']) == (16,30)
    assert not source['rear_coverage_observed'] and source['observations_sha256'] == native['observations_sha256']
    assert entry.sha(relocate(read(NATIVE/'plan.json')['source_video'])) == native['video_sha256']
    data = relocate(native['data_root'])
    inputs, obs = data/'data/inputs.json', data/'data/observations.npz'
    assert entry.sha(inputs) == native['observation_manifest_sha256'] and entry.sha(obs) == native['observations_sha256']
    rows = read(inputs)['records']
    assert len(rows) == 59
    names = [row['frame'] for row in rows]
    K = np.asarray(rows[0]['K_integer_centers'])
    assert all(row['wh']==[2704,1520] and np.array_equal(row['K_integer_centers'],K) for row in rows)
    with np.load(obs,allow_pickle=False) as z:
        assert z['frame_ids'].tolist() == names
        points, raw_scores, boxes, flip = [z[k] for k in ('keypoints','scores','boxes','flip_disagreement')]
    scores = raw_scores.copy()
    for item in source['source_visibility_exclusions']:
        start, stop = item['sample_index_inclusive']
        scores[start-1:stop, item['coco133_indices']] = 0
    assert np.array_equal(points.shape,(59,133,2)) and (scores<=raw_scores).all()
    # Reserve one fixed interior sample in four different temporal strata first.
    bins = np.array_split(np.arange(len(rows)),16)
    hold = [int(bins[b][len(bins[b])//2]) for b in (2,6,10,14)]
    candidates, anchors = [], []
    for i, section in enumerate(bins):
        usable = [int(j) for j in section if j not in hold]
        assert len(usable) >= 2
        choices = [usable[0],usable[-1]]
        candidates.extend(choices)
        if i % 2 == 1:
            reliability = scores[choices,:23].sum(axis=1)/(1+flip[choices,:23].mean(axis=1))
            anchors.append(choices[int(np.argmax(reliability))])
    assert len(candidates)==32 and len(anchors)==8 and not set(candidates)&set(hold)
    confidence, q90 = relative_confidence(scores[candidates],boxes[candidates],flip[candidates])
    observed = np.zeros((len(candidates),118,3),dtype=np.float32)
    observed[:,TARGET_IDX,:2] = points[candidates][:,SOURCE_IDX]
    observed[:,TARGET_IDX,2] = confidence[:,SOURCE_IDX]
    assert np.array_equal(observed[:,TARGET_IDX,:2],points[candidates][:,SOURCE_IDX].astype(np.float32))
    for i in candidates+hold:
        row = rows[i]
        assert entry.sha(relocate(row['image']))==row['source_sha256'] and entry.sha(relocate(row['mask']))==row['mask_sha256']
    records = []
    for i in candidates:
        row = rows[i]
        rgb = dict(path=str(relocate(row['image'])),sha256=row['source_sha256'],
                   native_integer_crop=True,resized=False,crop_original_xyxy=[0,0,2704,1520])
        records.append(dict(row, paths={'body':rgb}))
    OUT.mkdir()
    save_npz(ARCHIVE,frame_ids=np.asarray([names[i] for i in candidates]),observed_native_keypoints=observed,
             K_integer_centers=K, confidence_q90=q90)
    atomic(CONTACTS,dict(records=records))
    pins = (original.ASSET,original.SMPLX_ROOT/'smplx/body_models.py',original.SMPLX_ROOT/'smplx/lbs.py',
            original.RUNTIME/'src/avatar_core/shuolee_vposer.py')
    atomic(OUT/'plan.json',dict(updated=datetime.now(timezone.utc).isoformat(),
        runner_sha256=entry.sha(__file__), worker_sha256=entry.sha(original.__file__), fit_entry_sha256=entry.sha(entry.__file__),
        model_sha256={str(path):entry.sha(path) for path in pins}, video_sha256=native['video_sha256'],
        source_admission_sha256=entry.sha(CONTRACT), source_obs_sha256=entry.sha(obs),
        baseline_observation_sha256=entry.sha(ARCHIVE),contacts_sha256=entry.sha(CONTACTS),
        source_frames=[names[i] for i in candidates], native_indices=[row['source_frame_index'] for row in records],
        shape_anchors=[names[i] for i in anchors], retained_holdouts=[names[i] for i in hold],
        K_integer_centers=K.tolist(), conditional_K=True, source_view_bins=[[names[j] for j in b] for b in bins],
        candidate_policy='First and last unreserved sample in each of 16 chronological bins. Selected training count will depend on fit and angular review.',
        shape_policy='One RTMW-reliability anchor in each odd chronological bin; original independent free fits, componentwise median beta, then one fixed-shape original fit per candidate.',
        review_before_Gaussian=True, half_orbit_target=16, full_orbit_target=30, rear_coverage_observed=False,
        original_inputs_unchanged=True, gaussian_started=False,quality_accepted=False))
    print('VIDEO21_PLAN',len(candidates),len(anchors),len(hold),flush=True)


def fit_shard(folder, shard):
    records = read(CONTACTS)['records']
    by_index = {r['source_frame_index']:r['paths']['body'] for r in records}

    def decoded_sources(indices, crop, reverse):
        assert not reverse and crop == (0,0,2704,1520)
        for index in indices:
            row = by_index[index]
            assert entry.sha(row['path']) == row['sha256']
            with Image.open(row['path']) as im:
                yield index, np.asarray(im.convert('RGB'))

    scope = dict(original.__dict__,__file__=__file__,ARCHIVE=ARCHIVE,CONTACTS=CONTACTS,decode=decoded_sources)
    scope['fit_frames'] = types.FunctionType(original.fit_frames.__code__,scope)
    types.FunctionType(original.fit_shard.__code__,scope)(folder,shard)


def stage(name, plan, indices, shape):
    path = OUT/name
    assert not path.exists()
    atomic(path/'plan.json',dict(plan,shared_shape=shape,shards=[indices[i::WORKERS] for i in range(WORKERS)]))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS) as workers:
        list(workers.map(fit_shard,[path]*WORKERS,range(WORKERS)))
    rows = [read(path/plan['source_frames'][i]/'receipt.json') for i in indices]
    assert all(entry.sha(r['output'])==r['sha256'] for r in rows)
    atomic(path/'result.json',dict(frames=rows,seconds=time.monotonic()-started,
        plan_sha256=entry.sha(path/'plan.json'),quality_accepted=False))
    print('VIDEO21_FIT_STAGE',name,len(rows),flush=True)
    return rows


def run():
    plan = read(OUT/'plan.json')
    assert plan['runner_sha256']==entry.sha(__file__) and plan['worker_sha256']==entry.sha(original.__file__)
    assert plan['fit_entry_sha256']==entry.sha(entry.__file__) and plan['source_admission_sha256']==entry.sha(CONTRACT)
    assert all(entry.sha(path)==digest for path,digest in plan['model_sha256'].items())
    anchor_indices = [plan['source_frames'].index(name) for name in plan['shape_anchors']]
    started = time.monotonic()
    anchors = stage('shape-anchors',plan,anchor_indices,None)
    betas = []
    for r in anchors:
        with np.load(r['output'],allow_pickle=False) as z:
            betas.append(z['betas'])
    shape = np.median(np.stack(betas),axis=0).astype(np.float32)
    assert shape.shape==(1,10) and np.isfinite(shape).all()
    fits = stage('shared-cpu',plan,list(range(len(plan['source_frames']))),dict(vector=shape.tolist()))
    atomic(OUT/'result.json',dict(frames=fits,anchor_count=len(anchors),fixed_shape=shape.tolist(),
        wall_seconds=time.monotonic()-started,plan_sha256=entry.sha(OUT/'plan.json'),fitting_complete=True,
        original_worker_and_solver_reused=True,gaussian_started=False,quality_accepted=False))
    print('VIDEO21_FIT_COMPLETE',len(fits),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run'))
    (prepare if parser.parse_args().mode=='prepare' else run)()
