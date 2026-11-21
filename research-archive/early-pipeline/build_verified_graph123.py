"""One frozen94-frame track input repair, no camera pose optimization/training."""
from pathlib import Path
from functools import lru_cache
from collections import Counter
import os
import time
import json
import traceback
import cv2
import numpy as np
import audit_smplify_observation_contract123 as a
from shuolee_portrait.fixedk_correspondence import build_verified_graph
from probe_fixedk_track_edges123 import serial

OUT = a.REF/'solver-track-membership/verified-full-graph'

def main():
    assert not OUT.exists(), 'Once-only graph repair output exists; inspect, do not restart.'
    OUT.mkdir()
    previous=a.read(OUT.parent/'preunion-geometry-probe/result.json')
    assert previous['held_observations_used_for_E']==0 and not previous['K_changed']
    raw_path=a.ROOT/'camera/appearance-tracks.json';raw=a.read(raw_path)
    decoded=a.read(a.ROOT/'decoded.json')['records'];n=len(decoded)
    assert n==94
    K=np.asarray(a.read(a.ROOT/'intrinsic-contract.json')['raw_K_integer'])
    inputs=[raw_path,a.ROOT/'decoded.json',a.ROOT/'intrinsic-contract.json',
            a.PROJECT/'src/shuolee_portrait/fixedk_correspondence.py',
            a.PROJECT/'src/shuolee_portrait/material_tracks.py',
            a.ROOT/'camera/solver-output.npz']
    inputs += [a.ROOT/f'camera/features-{s:03d}.npz' for s in range(n)]
    hashes={str(p):a.sha(p) for p in inputs}
    a.save(OUT/'plan.json',dict(registered=a.now(),hypothesis=
        'The frozen camera initializer joins all mutual appearance matches before triangulation, allowing physically false transitive families. '
        'Repair its input by fixed-K FIT-only pair verification and the existing three-view cycle builder, with no relabelling of parent families.',
        distinction_from_previous='The seven-pair assay independently tested geometry and cycle checks. This full input applies their combination, so any future change cannot be attributed to K alone.',
        frame_count=n, scheduled_pairs=len(raw['pairs']), all_original_frames_required=True,
        original_appearance_parents=len(raw['records']),
        settings=dict(threshold_px=1.5,minimum_FIT_pairs=30,seed=118,descriptor_ratio=.75,max_ransac_iterations=10000),
        budget=dict(cpu_seconds=240,new_features=0,new_global_camera_candidates=0,new_person_fits=0,new_gaussian_steps=0),
        old_data='Reuse original native feature observations only. The old solver-output is hash protected and never loaded as an initializer.',
        splits='All child tracks inherit the original whole-family split. Keep all raw HELD families for subsequent validation, including false or ambiguous ones flagged separately.',
        stop='Insufficient pair models cannot add edges. No threshold fallback; do not register/train until all-frame support and physical review are examined.',
        source_hashes=hashes))
    a.save(OUT/'progress.json',dict(state='running',pid=os.getpid(),started=a.now(),pairs_completed=0))
    began=time.monotonic();cv2.setNumThreads(4)
    @lru_cache(maxsize=94)
    def features(s):
        return dict(np.load(a.ROOT/f'camera/features-{s:03d}.npz'))
    def progress(done,total):
        if time.monotonic()-began>240:raise RuntimeError('240-second graph input budget exhausted; no retry.')
        if done%25==0 or done==total:
            a.save(OUT/'progress.json',dict(state='running',pid=os.getpid(),pairs_completed=done,pairs_total=total,seconds=time.monotonic()-began))
    result=build_verified_graph(raw['records'],raw['pairs'],features,{s:K for s in range(n)},progress=progress)
    counts=np.zeros((n,2),int)
    parents={t['id']:t for t in raw['records']}
    for tr in result['records']:
        parent=parents[tr['parent_id']]
        assert tr['held']==parent['held'] and set(map(tuple,tr['nodes']))<=set(map(tuple,parent['nodes']))
        for s,_ in tr['nodes']:counts[s,int(tr['held'])]+=1
    by_status=dict(Counter(p['model']['status'] for p in result['pairs']))
    for p,h in hashes.items():assert a.sha(Path(p))==h
    a.save(OUT/'appearance-tracks.json',dict(records=result['records'],raw_parent_source=str(raw_path),
        raw_parent_source_sha256=hashes[str(raw_path)],partition='inherited_whole_raw_appearance_family',
        original_held_tracks_preserved_for_validation=True,physical_identity_verified=False,
        pairs=raw['pairs'],pipeline='fixedK_FIT_only_pair_geometry_then_canonical_three_view_cycles'))
    a.save(OUT/'pair-evidence.json',serial(dict(pairs=result['pairs'],accepted_edges=result['accepted_edges'])))
    summary=dict(updated=a.now(),frames=n,pair_count=len(raw['pairs']),pair_statuses=by_status,
        raw_parent_count=len(raw['records']),candidate_subtracks=len(result['records']),
        FIT_subtracks=sum(not t['held'] for t in result['records']),HELD_subtracks=sum(t['held'] for t in result['records']),
        per_frame=[dict(slot=s,source_frame=decoded[s]['source_frame_index'],FIT=int(counts[s,0]),HELD=int(counts[s,1])) for s in range(n)],
        minimum_FIT_per_frame=int(counts[:,0].min()),minimum_HELD_per_frame=int(counts[:,1].min()),
        all_frames_have_50_FIT_observations=bool(counts[:,0].min()>=50),
        source_hashes_unchanged=True,splits_inherited_without_rehash=True,
        held_coordinates_used_for_pair_estimation=False,physical_identity_review_complete=False,
        cycle_report=result['cycle_report'],old_poses_loaded=False,K_changed=False,
        new_global_cameras=0,new_gaussian_steps=0,quality_accepted=False,seconds=time.monotonic()-began)
    a.save(OUT/'result.json',summary)
    a.save(OUT/'progress.json',dict(state='complete',pid=None,finished=a.now()))
    print(json.dumps({k:v for k,v in summary.items() if k!='per_frame'}))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            a.save(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),auto_retry=False))
            a.save(OUT/'progress.json',dict(state='failed',pid=None,finished=a.now()))
        raise
