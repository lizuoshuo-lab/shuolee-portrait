"""Existing SH0 route with reviewed continuous poses and frozen8 RGB frames."""
import argparse
from pathlib import Path
import cross_subject_sh0124 as route
from run_official_gaussians123 import read,save,sha,now

RUN=route.ROOT/'grass-temporal-refinement'
SUBJECT='grass-refinement-sh0'
OUT=route.ROOT/SUBJECT

def verify_admission():
    assert read(RUN/'temporal-progress.json')['state']=='completed'
    admission=read(RUN/'gaussian-admission.json')
    assert admission['pose_admitted_for_gaussian_control'] and admission['frames']==94
    fit=RUN/'official-fit'
    assert sha(fit/'fit/fitted.npz')==admission['fit_sha256']
    assert sha(fit/'data/training-inputs.json')==admission['training_inputs_sha256']
    assert sha(RUN/'review-094/result.json')==admission['full_review_sha256']
    assert len(set(admission['reviewed_frames']))==94
    return admission

def frozen_selection(out,n):
    assert n==94
    contract=read(RUN/'gaussian-handoff-plan.json')
    ids=contract['training_frames'];old=read(route.ROOT/'grass/selection.json')
    assert ids==old['selected_frames'] and len(set(ids))==8
    rows=read(out/'data/training-inputs.json')['records']
    assert set(ids)<=set(r['frame'] for r in rows)
    result=dict(updated=now(),selected_frames=ids,frame_reselection=False,
        source_selection=str(route.ROOT/'grass/selection.json'),
        source_selection_sha256=sha(route.ROOT/'grass/selection.json'),
        records=[dict(frame=r['frame'],slot=i,time_seconds=r['time_seconds'],selected=r['frame'] in ids) for i,r in enumerate(rows)],
        reason='Frozen historical8 frame identities. No reuse of old fit quality scores or result-dependent reselection.')
    save(out/'selection.json',result)
    save(out/'selection-rule.json',dict(updated=now(),selection_count=8,frame_reselection=False,
        source_selection=result['source_selection'],source_selection_sha256=result['source_selection_sha256'],
        rule='Use exactly the pre-registered previous8 frame IDs to isolate the input-pose change.',
        no_manual_frame_replacement=True,full_fitted_roster=94))
    return result

def prepare():
    assert not OUT.exists()
    admission=verify_admission()
    source=dict(route.SOURCES['grass']);source['fit_root']=RUN/'official-fit'
    route.SOURCES[SUBJECT]=source;route.select_eight=frozen_selection
    route.prepare(SUBJECT)
    plan=read(OUT/'plan.json')
    plan.update(only_parameter_change='Reviewed continuous poses replace the independent fit; same8 RGB frame IDs, SH0, masks, K/E and all Gaussian settings.',
        source_fit_quality_accepted=True,source_pose_admission_scope='Visual pose admission to bounded GS; not independent physical3D truth.',
        source_fit_quality_record=str(RUN/'gaussian-admission.json'),
        frame_reselection=False,selected_training_frames_changed=False,
        old_pose_archive_reused=False,cached_subject_fitting=False,
        user_authorization='Continue SMPLify-X temporal -> SH0 mesh-bound reconstruction goal.',
        quality_accepted=False,model_promoted=False,gaussian_admission=admission)
    for file in [Path(__file__),RUN/'gaussian-handoff-plan.json',RUN/'gaussian-admission.json',RUN/'review-094/result.json']:
        plan['source_hashes'][str(file)]=sha(file)
    save(OUT/'plan.json',plan)
    print('GRASS_CONTINUOUS_POSE_SH0_INPUT_PREPARED; TRAIN_NOT_STARTED',flush=True)

def train():
    verify_admission()
    assert read(OUT/'progress.json')['step']==0 and not (OUT/'gaussian-run').exists()
    route.train(SUBJECT)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','train']);args=parser.parse_args()
    {'prepare':prepare,'train':train}[args.mode]()
