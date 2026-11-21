"""One reviewed selector-contract repair on the completed seven-view archive."""
import re,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import audit_gaussianavatars_params123 as a
import screen_official_body_contract124 as screen
from quality_budget124 import select_budget

ROOT=a.REC/'cross-subject-sh0-124'
FIT=ROOT/'grass-screened-seven124'
AUDIT=ROOT/'selected-body-point-contract124'
OUT=ROOT/'official-body-screen124'


def main():
    assert not OUT.exists()
    started=time.monotonic();stamp=datetime.now(timezone.utc).isoformat()
    audit=a.read(AUDIT/'result.json');assert audit['all_seven_observations_bit_identical']
    official=a.REC/'smplifyx-reference-control/official'
    config=(official/'cfg_files/fit_smplx.yaml').read_text(encoding='utf-8')
    entry=Path(__file__).parent/'official_smplifyx_entry123.py'
    assert re.search(r'joints_to_ign:\s+- 1\s+- 9\s+- 12',config)
    assert 'jw[:,[1,9,12]] = 0' in entry.read_text(encoding='utf-8')
    frozen=[Path(__file__),Path(screen.__file__),entry,official/'cfg_files/fit_smplx.yaml',
            AUDIT/'result.json',FIT/'selected-fit.npz',FIT/'selection-reviewed.json',FIT/'visual-review.json']
    hashes={str(p):a.sha(p) for p in frozen}
    plan=dict(updated=stamp,state='registered',source_hashes=hashes,
        causal_change='Only the original-body reprojection gate uses the executed nonzero joint-weight support. Ignored hip detector residuals remain explicitly reported and require source/pose review; no raw point, fitting parameter, threshold or ranking weight changes.',
        original_problem='The screen treated even original-loss-ignored, clothing-occluded hip hypotheses as exact veto points. Source audit and originalcfg/entry agree on their status.',
        scope='Completed grass seven-view archive only, samefitSHA. Reusable module fails closed for unknown weight contracts; other people need source verification before application.',
        risks=['A wrong hip/torso pose can be missed by numeric support alone; actual source, knees/feet and synthetic sides remain mandatory.','A detector-supported point is still not independently measured3Dtruth.'],
        forbidden=['No outcome-tuned3percent threshold','No forced8frame count','No per-frame solver mixing','No Gaussian launch from this screen alone'],
        budget=dict(CPU_seconds=120,fit_calls=0,detector_calls=0))
    a.save(OUT/'plan.json',plan)
    checks=screen.verify()
    observations=[]
    notes={
        'frame_000004':'The left shoulder is covered by the loose sleeve/hair; the raised right elbow lies within a loose sleeve. Neither detector centre is an independently visible bone centre. Model-vs-detector shoulder/elbow discrepancies remain; keep the supported-point failure.',
        'frame_000012':'Same covered shoulder and dress-occluded hip. The shoulder mismatch is real relative to the detector, but its exact anatomical centre is not physically localized. Keep the shoulder gate.',
        'frame_000058':'The sole numeric failure is the left hip hidden under the dress. Its detector hypothesis cannot be read as an observed surface landmark. Native body/face/feet and all four model sides were reviewed; raised hand stays by the chest, arms and legs have no gross new disconnection. Precise hip, finger and shoe shape remain uncertain.',
        'frame_000070':'Raised right elbow remains displaced toward the outer sleeve/cape relative to the detector. Both hips are hidden by the dress. Keep the supported-elbow failure; no manual elbow relocation.'}
    for r in audit['records']:
        assert a.sha(r['image'])==r['image_sha256']
        observations.append(dict(frame=r['frame'],actual_native_image_review=True,image=r['image'],sha256=r['image_sha256'],
            finding=notes[r['frame']],joint_centres_physically_localized=False,original_coordinates_unchanged=True))
    visual=dict(updated=stamp,state='mapping_and_native_body_source_reviewed',records=observations,
        mapping_error_found=False,all_seven_targets_bit_identical=True,
        finding='Original-side labels and2D-to-SMPLXjoint indices agree. All observed hip centres are hidden by clothing; the current3percent all-body gate confuses ignored detector hypotheses with measured anatomy. Shoulder/elbow disagreement remains, so correcting this contract cannot resolve all side views.',
        no_new_detector_or_fit=True,independent_3d_truth=False)
    a.save(AUDIT/'visual-review.json',visual);a.save(AUDIT/'conclusion.json',visual)
    old=a.read(FIT/'selection-reviewed.json');vr=a.read(FIT/'visual-review.json')
    evidence={r['frame']:r for r in a.read(FIT/'review/result.json')['records']}
    decisions={f:dict(v) for f,v in vr['decisions'].items()}
    decisions['frame_000058']=dict(status='accepted',depth_reviewed=True,
        original_ignored_joints_reviewed=True,reason=notes['frame_000058'],
        evidence=evidence['frame_000058']['images'],evidence_sha256=evidence['frame_000058']['images']['sides']['sha256'],
        acceptance_scope='Coarse visible pose only; not exact hidden hip/hand geometry or subpixel face acceptance.')
    byid={r['frame']:r for r in old['records']}
    with np.load(FIT/'selected-fit.npz') as z:
        ids=z['frame_ids'].astype(str).tolist();obs=z['observed_native_keypoints'].copy();proj=z['joints_projected_native'].copy()
    weights=np.ones(118);weights[list(screen.IGNORED)]=0
    records=[]
    for i,f in enumerate(ids):
        m=screen.measure(obs[i],proj[i],byid[f]['person_box_height_px'],weights)
        records.append(screen.apply(byid[f],m))
        if f in ['frame_000030','frame_000046','frame_000049','frame_000058']:
            for im in evidence[f]['images'].values():assert a.sha(im['path'].replace('/mnt/d/','D:/'))==im['sha256']
    selected=select_budget(records,decisions)
    assert selected['selected_frames']==['frame_000030','frame_000046','frame_000049','frame_000058']
    assert not selected['training_admission']
    chosen=[r['yaw_proxy_deg'] for r in records if r['frame'] in selected['selected_frames']]
    selected.update(updated=stamp,previous_selection=old['selected_frames'],screen_contract=plan['causal_change'],
        numeric_threshold_unchanged=True,ranking_metrics_unchanged=True,selected_yaw_proxy_span_deg=max(chosen)-min(chosen),
        no_framewise_fit_mixing=True,actual_visual_review=True)
    assert time.monotonic()-started<120 and all(a.sha(p)==h for p,h in hashes.items())
    conclusion=dict(updated=stamp,state='completed_original_body_support_screen',seconds=time.monotonic()-started,
        source_fit_sha256=a.sha(FIT/'selected-fit.npz'),selected_frames=selected['selected_frames'],
        previous_selected_frames=old['selected_frames'],selected_yaw_proxy_span_deg=selected['selected_yaw_proxy_span_deg'],
        remaining_supported_failures={r['frame']:r['screen_flags'] for r in records if r['screen_flags']},
        checks=checks,actual_native_body_side_review_added='frame_000058',
        pose_candidates_are_coarse=True,source_mapping_error_found=False,
        change='Screen measurement follows the verified original zero-joint-weight contract; all ignored residuals preserved. Old result and all targets,fit,K,parameters remain byte-identical.',
        quality_accepted=False,gaussian_admitted=False,new_fit_calls=0,new_gaussian_steps=0,
        next='Keep4coarse pose candidates from the same archive. Source-supported shoulder/elbow problems remain in4/12/70, not repaired by a screen change. Apply the same measurement contract to another person only after verifying its executed source; select approximately8good views rather than demanding all sequence frames. No new solver/threshold sweep.')
    a.save(OUT/'selection-reviewed.json',selected);a.save(OUT/'conclusion.json',conclusion)
    a.save(OUT/'checks.json',checks)
    print({k:conclusion[k] for k in ['state','selected_frames','selected_yaw_proxy_span_deg','remaining_supported_failures','new_fit_calls']})


if __name__=='__main__':main()
