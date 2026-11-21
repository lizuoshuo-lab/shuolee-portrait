"""Run the existing fixed-K background SfM once on 021 short tracks."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import numpy as np

import global_fixedk121 as original
from update_leafwoman129_state import atomic, read, sha


def main(track_dir):
    track_dir = track_dir.resolve()
    source = read(track_dir/'result.json')
    track_plan = read(track_dir/'plan.json')
    archive = track_dir/'tracks.npz'
    assert source['plan_sha256'] == sha(track_dir/'plan.json')
    assert source['tracks_sha256'] == sha(archive)
    with np.load(archive, allow_pickle=False) as data:
        frames, uv, families, split = (data[k] for k in ('frames', 'uv', 'family', 'split'))
    frames = frames.tolist()
    assert frames == track_plan['frames'] and not set(frames) & set(track_plan['retained_portrait_holdouts'])
    assert set(families[split==0]).isdisjoint(families[split==1])
    assert uv.shape == (len(frames), len(split), 2)
    assert np.isfinite(uv[..., 0]).sum() == np.isfinite(uv[..., 1]).sum()
    tracks = []
    for idx in np.flatnonzero(split >= 0):
        by_frame = {int(i):dict(uv=uv[i, idx].tolist()) for i in np.flatnonzero(np.isfinite(uv[:, idx]).all(axis=1))}
        tracks.append(dict(track_id=int(idx), by_frame=by_frame,
            regions={i:'background_static_support_candidate' for i in by_frame},
            split='fit' if split[idx]==0 else 'heldout', camera_fit_eligible=bool(split[idx]==0)))
    inputs = dict(records=[dict(frame=f, wh=track_plan['wh']) for f in frames],
        pairs=[dict(a=a,b=b,frames=[frames[a],frames[b]]) for a in range(len(frames)-1)
               for b in range(a+1,min(a+3,len(frames)))])
    out = track_dir.with_name('fixedk-background155')
    assert not out.exists()
    subject = 'seated021'
    hashes = {str(path):sha(path) for path in
        (Path(__file__), Path(original.__file__), track_dir/'plan.json', track_dir/'result.json', archive)}
    plan = dict(updated=datetime.now(timezone.utc).isoformat(),
        cpu_seconds_per_subject=420, subject=subject,
        subjects={subject:dict(K_integer_centers=track_plan['fixed_K'])},
        material_preflight={subject:dict(decision='allow_background_diagnostic_only')},
        tracks_sha256=source['tracks_sha256'], input_hashes=hashes,
        experiment_frames=frames, candidate_frames=track_plan['candidate_frames'],
        retained_portrait_holdouts=track_plan['retained_portrait_holdouts'],
        hypothesis='Short-interval background families support all 55 cameras in a common gauge and predict independent material families.',
        uses_existing_relative_PnP_BA_and_LOO=True, fixed_intrinsics=True, K_fitted=False,
        no_prior_camera_trajectory_or_person_points=True, no_interpolation=True,
        review='Diagnose on full source pixels with whole-track held-out leave-one-view-out, then inspect material identity and per-frame coverage before body fitting.',
        physical_camera_calibration_accepted=False, gaussian_started=False)
    atomic(out/'global-fixedk-plan.json', plan)
    atomic(out/'launch.json', dict(pid=os.getpid(), command=[sys.executable,*sys.argv],
        plan_sha256=sha(out/'global-fixedk-plan.json'), stage='background_only'))
    (out/subject).mkdir()
    original.ROOT = out
    original.R = out
    original.input_data = lambda name: (inputs, tracks, hashes)
    original.apply_material_gate = lambda name, observations, table, protocol: None
    print('FIXEDK_BACKGROUND_REGISTERED', len(frames), source['fit_tracks'], source['heldout_tracks'], flush=True)
    try:
        result = original.one(subject, plan)
        assert all(sha(path)==digest for path,digest in hashes.items())
        saved = out/subject/'global-fixedk/result.json'
        assert result['plan_sha256'] == sha(out/'global-fixedk-plan.json')
        atomic(out/'process-exit.json', dict(exit_code=0, result_sha256=sha(saved)))
    except BaseException as error:
        atomic(out/'process-exit.json', dict(exit_code=1, error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('track_dir', type=Path)
    main(parser.parse_args().track_dir)
