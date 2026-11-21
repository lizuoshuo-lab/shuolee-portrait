"""Apply the verified body support contract to one frozen partial sofa run."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import time
import numpy as np
import audit_gaussianavatars_params123 as a
import screen_official_body_contract124 as screen
from select_quality_frames124 import select
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster
from screen_wrist_consistency124 import apply_to_candidates

ROOT=a.REC/'cross-subject-sh0-124'
RUN=ROOT/'sofa-temporal-warmstart-v2'
SRC=ROOT/'sofa-input'
OLD=ROOT/'quality-first-eight124/sofa'
OUT=ROOT/'official-body-screen124/sofa'


def host_path(p):
    p=str(p)
    return Path(p[5].upper()+':/'+p[7:]) if p.startswith('/mnt/') else Path(p)


def main():
    assert not OUT.exists()
    started=time.monotonic();stamp=datetime.now(timezone.utc).isoformat()
    original=a.read(RUN/'plan.json')
    checks={p:a.sha(host_path(p))==h for p,h in original['sources'].items()}
    assert all(checks.values()),[p for p,v in checks.items() if not v]
    old=a.read(OLD/'shortlist.json');manifest=a.read(OLD/'source-manifest.json')
    assert len(old['records'])==len(manifest['records'])==47
    a.save(OUT/'plan.json',dict(updated=stamp,state='registered',source_run=str(RUN),
        verified_original_source_hashes=original['sources'],source_manifest_sha256=a.sha(OLD/'source-manifest.json'),
        old_shortlist_sha256=a.sha(OLD/'shortlist.json'),selector_contract_sha256=a.sha(screen.__file__),
        only_change='Use the same source-verified original nonzero joint weights for the body veto; retain ignored hip residuals, original ranking, feet/temporal/heldout and input-conflict guards.',
        all_sequence_frames_required=False,source_is_one_coherent47frame_partial_run=True,
        new_fit_calls=0,new_detector_calls=0,source_solver_resume=False,budget_CPU_seconds=120))
    rows=a.read(SRC/'data/inputs.json')['records'];rowmap={r['frame']:r for r in rows}
    with np.load(SRC/'data/observations.npz') as z:obs=dict(z)
    assert obs['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]
    conf,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    target=np.zeros((len(rows),118,3),np.float32)
    dst=np.r_[screen.BODY,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    src=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    target[:,dst,:2]=obs['keypoints'][:,src];target[:,dst,2]=conf[:,src]
    target=np.concatenate([mask_outside_raster(target[i:i+1],r['wh'])[0] for i,r in enumerate(rows)])
    targets=dict(zip([r['frame'] for r in rows],target));oldrows={r['frame']:r for r in old['records']}
    weights=np.ones(118);weights[list(screen.IGNORED)]=0
    records=[]
    for m in manifest['records']:
        assert a.sha(m['path'])==m['sha256']
        n=m['frame'];record=oldrows[n]
        with np.load(a.lp(m['path'])) as z:
            proj=z['joints_projected_native'][0].copy()
            assert str(z['frame_id'])==n and str(z['image_sha256'])==rowmap[n]['source_sha256']
        height=rowmap[n]['initial_box_xyxy'][3]-rowmap[n]['initial_box_xyxy'][1]
        before=np.linalg.norm(proj-targets[n][:,:2],axis=1)
        valid=targets[n][screen.BODY,2]**2>0
        assert abs(float(before[screen.BODY][valid].max()/height)-record['body_supported_max_over_height'])<1e-6
        records.append(screen.apply(record,screen.measure(targets[n],proj,height,weights)))
    wrist=a.read(ROOT/'selected-observation-audit124/wrist-preselection/sofa.json')
    records=apply_to_candidates(records,wrist['records'])
    shortlist=select(records,admitted_only=False)
    shortlist.update(updated=stamp,source_run=str(RUN),source_fit_sha256=manifest['source_set_sha256'],
        new_fit_calls=0,all56_not_required=True,thresholds_unchanged=True,
        input_conflict_screen_applied=True,actual_pose_review=False,original_targets_unchanged=True)
    assert not shortlist['training_admission'] and time.monotonic()-started<120
    assert all(a.sha(host_path(p))==h for p,h in original['sources'].items())
    a.save(OUT/'shortlist.json',shortlist)
    result=dict(updated=stamp,state='source_verified_sofa_shortlist_ready',seconds=time.monotonic()-started,
        available_frames=47,old_candidates=old['selected_frames'],new_candidates=shortlist['selected_frames'],
        numeric_survivors=[r['frame'] for r in shortlist['records'] if r['eligible']],
        remaining_flags=dict(Counter(f for r in records for f in r['screen_flags'])),
        all_executed_source_hashes_unchanged=True,all47_fit_hashes_unchanged=True,
        actual_visual_review=False,quality_accepted=False,gaussian_admitted=False,new_fit_calls=0,
        next='Review only these same-run shortlisted native poses and sides if any. Never resume the stopped47/56solver or treat a numeric screen as pose acceptance.')
    a.save(OUT/'result.json',result)
    print(result)


if __name__=='__main__':main()
