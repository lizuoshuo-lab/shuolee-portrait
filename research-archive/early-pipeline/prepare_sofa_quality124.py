"""Select from one stopped sofa run's committed fits; never resume its solver."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import audit_gaussianavatars_params123 as a
from official_smplifyx_entry123 import COCO
from select_quality_frames124 import POLICY, select
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster

ROOT = a.REC/'cross-subject-sh0-124'
RUN = ROOT/'sofa-temporal-warmstart-v2'
SRC = ROOT/'sofa-input'
OUT = ROOT/'quality-first-eight124/sofa'


def rotations(axis_angles):
    result = []
    for aa in np.asarray(axis_angles).reshape(-1, 3):
        theta = float(np.linalg.norm(aa))
        if theta < 1e-12:
            result.append(np.eye(3)); continue
        x, y, z = aa/theta
        cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        result.append(np.eye(3)+np.sin(theta)*cross+(1-np.cos(theta))*(cross@cross))
    return np.stack(result)


def main():
    assert not a.lp(OUT/'shortlist.json').exists()
    progress = a.read(RUN/'temporal-progress.json')
    assert progress['state'] == 'stopped_or_failed'
    ids = sorted(r['frame'] for r in progress['records'])
    assert len(ids) == len(set(ids)) == 47
    rows = a.read(SRC/'data/inputs.json')['records']; byid = {r['frame']: r for r in rows}
    selection = a.read(SRC/'selection.json'); cached = {r['frame']:r for r in selection['candidates']}
    with np.load(a.lp(SRC/'data/observations.npz'), allow_pickle=False) as z:
        conf, _ = relative_confidence(z['scores'], z['boxes'], z['flip_disagreement'])
        obs = np.zeros((len(rows), 118, 3), np.float32)
        dst = np.r_[COCO, np.arange(19, 25), np.arange(25, 67), np.arange(67, 118)]
        source = np.r_[np.arange(23), np.arange(91, 133), np.arange(40, 91)]
        obs[:, dst, :2] = z['keypoints'][:, source]; obs[:, dst, 2] = conf[:, source]
    for i, row in enumerate(rows):
        obs[i:i+1] = mask_outside_raster(obs[i:i+1], row['wh'])[0]
    targets = dict(zip([r['frame'] for r in rows], obs))
    fits, manifest = {}, []
    for name in ids:
        path = RUN/'official-fit/frames'/name/'winner-camera-space.npz'
        receipt_path = path.parent/'result.json'; receipt = a.read(receipt_path)
        with np.load(a.lp(path), allow_pickle=False) as z: f = dict(z)
        assert str(f['frame_id']) == name
        assert str(f['image_sha256']) == byid[name]['source_sha256'] == receipt['source_image_sha256']
        assert str(f['mask_sha256']) == byid[name]['mask_sha256']
        assert np.array_equal(f['K_integer_centers'], byid[name]['K_integer_centers'])
        assert np.isfinite(f['joints_projected_native']).all()
        fits[name] = f
        manifest.append(dict(frame=name,path=str(path),sha256=a.sha(path),receipt_sha256=a.sha(receipt_path)))
    assert all(np.array_equal(fits[ids[0]]['betas'], fits[n]['betas']) for n in ids)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    a.save(OUT/'source-manifest.json', dict(run=str(RUN), records=manifest,
        committed_frames=47, total_source_frames=56, absent_frames=sorted(set(byid)-set(ids)),
        source_set_sha256=digest, mixed_experiments=False))
    # Numerical parity with already rendered same-run evidence checks the
    # observation mapping; it does not imply that the fitted pose is correct.
    checked = 0
    for rel in ['checkpoint-review/result.json', 'progress-review-029/result.json']:
        for r in a.read(RUN/rel)['records']:
            e = np.linalg.norm(fits[r['frame']]['joints_projected_native'][0]-targets[r['frame']][:,:2],axis=-1)
            assert np.allclose(e, r['all_118_error_px'], atol=1e-4, rtol=0)
            checked += 1
    assert checked == 29
    flags = {name:[] for name in ids}; transitions = []
    for prev, curr in zip(ids[:-1], ids[1:]):
        if int(curr[-6:])-int(prev[-6:]) != 1: continue
        ra, rb = (rotations(fits[n]['body_pose_axis_angle']) for n in [prev,curr])
        angle = np.degrees(np.arccos(np.clip((np.trace(ra.transpose(0,2,1)@rb,axis1=1,axis2=2)-1)/2,-1,1)))
        dt = byid[curr]['time_seconds']-byid[prev]['time_seconds']; assert dt > 0
        speed = float(angle.max()/dt)
        transitions.append(dict(previous=prev,current=curr,max_joint_speed_deg_s=speed))
        if speed > POLICY['joint_speed_deg_per_second']:
            for n in [prev,curr]: flags[n].append('body_pose_jump_over_existing135deg_s')
    rr = np.stack([fits[n]['w2c'][:3,:3]@fits[n]['root_rotation'][0] for n in ids])
    yaw = np.degrees(np.unwrap(np.arctan2(rr[:,0,2],rr[:,2,2])))
    records = []
    for n, angle in zip(ids, yaw):
        f= fits[n]; target=targets[n]; row=byid[n]; h=row['initial_box_xyxy'][3]-row['initial_box_xyxy'][1]
        e = np.linalg.norm(f['joints_projected_native'][0]-target[:,:2],axis=-1); m={}; ff=list(flags[n])
        for name, slots in [('body',COCO),('face',np.arange(67,118)),('feet',np.arange(19,25))]:
            w=target[slots,2]**2; assert w.sum() > 0 and h > 0
            m[name+'_weighted_rms_px']=float(np.sqrt(np.sum(w*e[slots]**2)/w.sum()))
            m[name+'_supported_max_over_height']=float(e[slots][w>0].max()/h)
            if name in ('body','feet') and m[name+'_supported_max_over_height'] > POLICY[name+'_max_over_height']:
                ff.append(name+'_reprojection_over_existing3percent_height')
        if n in selection['review_only_frames']: ff.append('preregistered_Gaussian_review_only_never_train')
        records.append(dict(frame=n,time_seconds=row['time_seconds'],source_fit_sha256=digest,
            source_image_sha256=row['source_sha256'],yaw_proxy_deg=float(angle),
            face_sharpness=cached[n]['sharpness'],body_confidence=float(target[COCO,2].mean()),screen_flags=ff,**m))
    result=select(records,admitted_only=False)
    result.update(updated=datetime.now(timezone.utc).isoformat(),source_run=str(RUN),
        source_manifest=str(OUT/'source-manifest.json'),source_fit_sha256=digest,
        source_fit_is_coherent_partial_set=True,source_frames=56,available_frames=47,
        evidence_parity_frames=checked,review_only_frames=selection['review_only_frames'],
        no_new_fit=True,source_solver_not_resumed=True,all56_not_required=True,
        numerical_flags_are_not_physical_truth=True)
    a.save(OUT/'plan.json',dict(policy=POLICY,source_run=str(RUN),
        source_manifest_sha256=a.sha(OUT/'source-manifest.json'),new_fit_calls=0,
        parameter_changes=False,one_coherent_run=True,selector_sha256=a.sha(Path(__file__).with_name('select_quality_frames124.py'))))
    a.save(OUT/'transitions.json',transitions); a.save(OUT/'shortlist.json',result)
    print(dict(subject='sofa',available=47,shortlist=result['selected_frames'],
        survivors=sum(r['eligible'] for r in result['records']),parity_frames=checked,new_fit_calls=0))


if __name__ == '__main__': main()
