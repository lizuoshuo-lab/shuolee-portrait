"""Use the same cross-person Gaussian adapter with a preregistered8+4 roster."""
import argparse, json, traceback
from pathlib import Path
import cross_subject_sh0124 as base
from prepare_winterhat_sh0124 import ROOT,VIDEO,EXPECTED

def prepare():
    source=ROOT/'official-fit';sel=base.read(ROOT/'selection.json')
    base.SOURCES['winterhat']=dict(fit_root=source,video=VIDEO,video_sha256=EXPECTED,record_id='001',label='冬帽人物')
    original=base.select_eight
    def registered(out,n):
        assert n==12
        rule=dict(base.read(ROOT/'plan.json'),selection_count=8,selection_is_heuristic=True,
            no_manual_frame_replacement=True,no_fit_or_Gaussian_loss_selection=True)
        base.save(out/'selection-rule.json',rule)
        result=dict(updated=base.now(),selected_frames=sel['training_frames'],review_only_frames=sel['review_only_frames'],
            training_count=8,excluded_from_Gaussian_RGB_count=4,all_frame_evaluation=12,source_video_roster=112,
            full_video_fit_claim=False,selection_rule_sha256=base.sha(out/'selection-rule.json'),quality_accepted=False)
        base.save(out/'selection.json',result);return result
    base.select_eight=registered
    try:base.prepare('winterhat')
    finally:base.select_eight=original
    out=base.ROOT/'winterhat';p=base.read(out/'plan.json');p.update(cached_subject_fitting=False,
        newly_computed_subject_fitting=True,preprocessing_cached=True,source_video_frames_preprocessed=112,
        evaluation_frames=12,training_frames=8,heldout_pose_frames=sel['review_only_frames'],full_video_fit_claim=False,
        hypothesis='Transfer same Gaussian configuration to new official SMPLify-X input on winterhat',
        comparison_limit='Different subject and preregistered prefit roster; not a single-variable comparison of preprocessing or timing.',
        admission='All8 training originals and masks reviewed; fixed4 review-only frames retained. Review new official fit before Gaussian start.',
        source_pose_frames_changed=12,selected_training_frames_changed=True,frame_reselection=False,
        known_limitation='Long skirt, hat and bag may lie far from underlying naked-body mesh;8-view coverage and initialization remain exploratory.',
        smplify_input_elapsed_seconds=base.read(source/'fit/official-fit-result.json')['elapsed_seconds'],
        source_fit_quality_accepted=False,quality_accepted=False)
    for f in [Path(__file__),ROOT/'selection.json',ROOT/'plan.json']:
        p['source_hashes'][str(f)]=base.sha(f)
    base.save(out/'plan.json',p)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','train']);args=ap.parse_args()
    try:prepare() if args.mode=='prepare' else base.train('winterhat')
    except BaseException as e:
        base.save(base.ROOT/'winterhat-wrapper-failure.json',dict(updated=base.now(),error=repr(e),traceback=traceback.format_exc(),retry=False));raise
