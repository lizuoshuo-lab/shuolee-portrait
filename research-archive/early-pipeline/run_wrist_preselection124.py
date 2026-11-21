"""One CPU pass over cached four-person input; no detector, fitting or training."""
from pathlib import Path
from datetime import datetime,timezone
import ast,time
import numpy as np
import audit_gaussianavatars_params123 as a
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster
from screen_wrist_consistency124 import screen,apply_to_candidates,verify

ROOT=a.REC/'cross-subject-sh0-124'
OUT=ROOT/'selected-observation-audit124/wrist-preselection'


def main():
    assert not a.lp(OUT).exists(),'Read existing screen; do not rerun.'
    start=time.monotonic();registered=a.read(OUT.parent/'preselection-wrist-consistency-plan.json')
    for p,d in registered['source_hashes'].items():assert a.sha(Path(p))==d
    official=a.REC/'smplifyx-reference-control/official/smplifyx/utils.py'
    fn=next(x for x in ast.parse(a.lp(official).read_text(encoding='utf-8')).body if isinstance(x,ast.FunctionDef) and x.name=='smpl_to_openpose')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(official),'exec'),ns)
    mapper=ns['smpl_to_openpose'](model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    tests=verify(mapper)
    sources=[Path(__file__),Path(__file__).with_name('screen_wrist_consistency124.py'),official,
        OUT.parent/'preselection-wrist-consistency-plan.json',ROOT/'unique-wrist-input124/sofa16-control/conclusion.json']
    for subject in ('grass','winterhat','sofa','indoor'):
        sources += [ROOT/f'{subject}-input/data/inputs.json',ROOT/f'{subject}-input/data/observations.npz',ROOT/f'{subject}-input/selection.json']
    frozen={str(p):a.sha(p) for p in sources};a.lp(OUT).mkdir()
    a.save(OUT/'plan.json',dict(updated=datetime.now(timezone.utc).isoformat(),state='registered',source_hashes=frozen,
        scope='373cached frames for shadow screening; no new selection of heldout frames, no fit.',
        normalization='Reuse the existing all-input relative-score normalization for exact compatibility. This is not independent heldout validation. No new confidence values are written to training data.',
        difference_from_old_unique_wrist_control='Whole-frame conflict flag; never suppress a wrist root or select a detector owner.',
        budget=registered['budget'],controls=tests))
    summaries=[]
    for subject in ('grass','winterhat','sofa','indoor'):
        src=ROOT/f'{subject}-input';rows=a.read(src/'data/inputs.json')['records'];selection=a.read(src/'selection.json')
        with np.load(a.lp(src/'data/observations.npz'),allow_pickle=False) as z:obs=dict(z)
        ids=obs['frame_ids'].astype(str).tolist();assert ids==[r['frame'] for r in rows]
        confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
        x=np.zeros((len(rows),118,3),np.float32)
        dst=np.r_[[0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11],np.arange(19,25),np.arange(25,67),np.arange(67,118)]
        idx=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
        x[:,dst,:2]=obs['keypoints'][:,idx];x[:,dst,2]=confidence[:,idx]
        x=np.concatenate([mask_outside_raster(x[i:i+1],r['wh'])[0] for i,r in enumerate(rows)])
        heights=[r['initial_box_xyxy'][3]-r['initial_box_xyxy'][1] for r in rows]
        screened=screen(x,heights,mapper,ids)
        # Integrate with already recorded candidate scores, without changing old files.
        candidate_file=ROOT/'quality-first-eight124'/('grass-shortlist.json' if subject=='grass' else f'{subject}/shortlist.json')
        combined=None
        if a.lp(candidate_file).exists():
            existing=a.read(candidate_file);combined=apply_to_candidates(existing['records'],screened)
            a.save(OUT/f'{subject}-candidate-screen.json',dict(records=combined,source_candidate_sha256=a.sha(candidate_file),
                old_scores_preserved=True,old_pose_reviews_preserved=True,training_admission=False,new_fit_calls=0))
        conflicts=[r['frame'] for r in screened if r['screen_flags']]
        summary=dict(subject=subject,frames=len(rows),conflict_frames=conflicts,
            old_eight_conflicts=[f for f in selection['training_frames'] if f in conflicts],
            inconsistent_pairs=sum(p['status']=='conflict' for r in screened for p in r['pairs']),
            insufficient_pair_support=sum(p['status']=='insufficient_duplicate_support' for r in screened for p in r['pairs']),
            existing_review_only_frames_unchanged=True,training_admission=False)
        a.save(OUT/f'{subject}.json',dict(summary=summary,records=screened,source_input_sha256=frozen[str(src/'data/observations.npz')]))
        summaries.append(summary);print(summary)
    assert time.monotonic()-start<120 and sum(s['frames'] for s in summaries)==373
    assert all(a.sha(Path(p))==h for p,h in frozen.items())
    a.save(OUT/'result.json',dict(updated=datetime.now(timezone.utc).isoformat(),state='prefit_consistency_screen_verified',
        seconds=time.monotonic()-start,subjects=summaries,controls=tests,source_hashes_unchanged=True,
        original_targets_unchanged=True,old_fit_and_selection_unchanged=True,fit_calls=0,GPU_calls=0,
        original_quality_thresholds_unchanged=True,pose_quality_accepted=False,gaussian_admitted=False))


if __name__=='__main__':main()
