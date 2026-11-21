"""Run the existing bound holdout evaluator against the frozen 005/252 benchmark."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType


ARCHIVE = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/'
               'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
EVALUATOR = ARCHIVE/'sofa-placement006-formal296/evaluate.py'
EVALUATOR_SHA = '2983d35dcee61366d8c46aec1c053fc4b3407b90f2f09619338d7eb7e720be6f'
BENCHMARK_FIT = ARCHIVE/'grass-automatic-sh0-252-formal-heldout-fit262'
BENCHMARK_RENDER = ARCHIVE/'grass-automatic-sh0-252-formal-render262-replay/result.json'
METRIC_HELPER = ARCHIVE/'sofa-release-formal295/review.py'
BENCHMARK_PINS = {
    BENCHMARK_FIT/'contacts.json': 'c3bb362e0ccae884aefbac541e5c40182fc2303b97d639174128e93ea395fea5',
    BENCHMARK_FIT/'preregistration.json': '61958ac4a39d7649584eff2159cb43e9cc05f7233fea21f5b26b017e4fd97bc4',
    BENCHMARK_FIT/'result.json': '4819eb78a4862af77e8fb43eb6f04e8d33ef7da83e571a520f129cb647b5ebaa',
    BENCHMARK_RENDER: 'be077ee165875b3bd8ea769fb10135c4b17c86da53b4440b51a1101c11db0582',
    METRIC_HELPER: '72fb6738b6b6807dc0fe6d8124be41791eb31cdd3fc478a4349a139809a29f8a',
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def within_arc(rows, cfg):
    layout = cfg['capture_layout']
    intervals = layout['front_source_ranges'] + (layout['rear_source_ranges'] if layout['arc']=='full' else [])
    return {row['frame'] for row in rows if any(a<=row['source_frame_index']<=b for a,b in intervals)}


def load_evaluator(run, project, launch, protected):
    need(sha(EVALUATOR)==EVALUATOR_SHA, 'The archived evaluator has changed.')
    src = EVALUATOR.read_text(encoding='utf-8-sig')
    # Keep the archived pose solver and renderer intact. Only adapt identity,
    # capture scope, optional evaluation roster and historical receipt labels.
    replacements = (
        ('fitted == set(inputs["ids"])-set(HOLDOUTS)', 'fitted == within_arc(inputs["rows"], cfg)-set(HOLDOUTS)'),
        ('and len(fitted) == 52 and 0 < len(selected) <= 24', 'and 0 < len(selected) <= sum(scope["view_targets"].values())'),
        ('poses["frame_ids"].tolist() == list(HOLDOUTS)', 'poses["frame_ids"].tolist() == [r["frame"] for r in state.subset["rows"]]'),
        ('archived269_metrics=archived["metrics"]', 'archived252_metrics=archived["metrics"]'),
        ('Four frozen Gaussian holdouts;', 'Registered frozen Gaussian holdouts;'),
        ('exact archived 269 masks', 'exact archived 252 masks'),
        ('Original 269 pixels/camera do not match:', 'Original 252 pixels/camera do not match:'),
        ('complete 006 bound delivery', 'complete candidate bound delivery'),
        ('registered 006 training command', 'registered candidate training command'),
    )
    for old,new in replacements:
        need(src.count(old)==1, f'Cannot match pinned evaluation interface: {old}')
        src=src.replace(old,new)
    evaluator=ModuleType('_archived_006_evaluator_for_005')
    evaluator.__file__=str(EVALUATOR)
    old_path=sys.path[:]
    try:
        exec(compile(src,str(EVALUATOR),'exec'),evaluator.__dict__)
    finally:
        sys.path[:]=[str(project/'src'),*old_path]
    evaluator.__dict__.update(RUN=run,PROJECT=project,COMMIT=launch['candidate_commit'],VIDEO_SHA=launch['source_sha256'],
        HOLDOUTS=protected,OLD_FIT=BENCHMARK_FIT,OLD_RENDER=BENCHMARK_RENDER,OLD_HELPER=METRIC_HELPER,
        OLD_PINS=BENCHMARK_PINS,within_arc=within_arc)
    return evaluator


def cohort(run, project):
    launch = read(run.parent/'preregistration.json')
    git = ['git','-C',str(project)]
    marker=project/'.git'
    if marker.is_file():
        git_dir=marker.read_text(encoding='utf-8').strip().removeprefix('gitdir: ').replace('\\','/')
        if len(git_dir)>3 and git_dir[1:3]==':/':
            # A Windows-created worktree records a drive path. Resolve it
            # for Linux Git without rewriting the frozen worktree metadata.
            git=['git','--git-dir',f'/mnt/{git_dir[0].lower()}/{git_dir[3:]}','--work-tree',str(project)]
    commit = subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip()
    need(commit==launch['candidate_commit'], 'The supplied checkout is not the run’s registered commit.')
    need(all(sha(project/name)==digest for name,digest in launch['code_sha256'].items()),
         'Registered candidate source bytes changed.')
    cfg = read(run/'config.json')
    rows = read(run/'data/inputs.json')['records']
    layout = cfg['capture_layout']
    need(layout['video_sha256']==launch['source_sha256'] and sha(Path(launch['video']))==launch['source_sha256'] and
         sha(run.parent/'camera.json')==launch['camera_config_sha256'], 'Registered video or camera bytes changed.')
    indices = set(layout['held_source_indices'])
    protected = {row['frame']:row['source_frame_index'] for row in rows if row['source_frame_index'] in indices}
    need(set(protected.values())==indices and list(protected)==launch['expected_holdouts'], 'Frozen original-video holdouts differ.')
    inside = within_arc(rows,cfg)
    primary=[name for name in protected if name in inside]
    outside=[name for name in protected if name not in inside]
    need(primary, 'No protected frames within the registered capture scope.')
    return launch,cfg,rows,protected,primary,outside


def scoped_state(evaluator,state,primary,outside,include_outside):
    chosen=set(primary) | (set(outside) if include_outside else set())
    ordered=[name for name in evaluator.HOLDOUTS if name in chosen]
    state.subset=state.seq._selection_inputs(state.subset,ordered,ordered[0])
    state.old_rows=[r for r in state.old_rows if r['frame'] in chosen]
    state.old_records=[r for r in state.old_records if r['frame'] in chosen]
    state.pins[str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
    return dict(protected_frames=evaluator.HOLDOUTS,evaluated_frames=ordered,
        primary_within_capture_arc=primary,outside_capture_arc=outside,
        outside_arc_policy='separate_diagnostic' if include_outside else 'protected_without_evaluation',
        score_mask_source='Exact archived 252 scoring mask per source frame',
        comparison_scope='Matched native RGB, K and scoring masks with independently fitted current/252 poses. '
                         'Training roster, shape, pose recipe, color/alpha objective and 10k/18k budget differ.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check-inputs','check','fit','render'))
    for name in ('run','project','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--outside-arc-diagnostic',action='store_true')
    args=parser.parse_args()
    run,project,out=(p.resolve() for p in (args.run,args.project,args.output))
    need(not out.is_relative_to(run.parent),'Write evaluation output outside the reconstruction directory.')
    if args.mode!='render':
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    launch,cfg,rows,protected,primary,outside=cohort(run,project)
    evaluator=load_evaluator(run,project,launch,protected)
    if args.mode=='check-inputs':
        from shuolee_portrait import official_sequence as seq

        _, archive_rows, _, _ = evaluator.old_benchmark()
        live={r['frame']:r for r in rows}
        for old in archive_rows:
            row=live[old['frame']]
            need(all(row[k]==old[k] for k in ('frame','source_frame_index','source_sha256','wh','K_integer_centers','K','w2c')),
                 f'Archive raster or camera differs: {row["frame"]}')
        inputs=seq.load_inputs(run/'data/inputs.json',run/'data/observations.npz',
            camera_length_contract=cfg['camera_length_contract'],reference_frame=rows[0]['frame'],capture_layout=cfg['capture_layout'])
        need(inputs['scope']['retained_holdouts']==list(protected) and
             all(not inputs['targets'][inputs['ids'].index(frame)].any() for frame in protected),
             'Protected observations reached the training targets.')
        print(json.dumps(dict(state='source_inputs_verified_no_terminal_or_fit',protected_frames=protected,
            primary=primary,outside=outside,fit_candidates=len(inputs['scope']['fit_frame_ids']),
            fitting_calls=0,gpu_calls=0),ensure_ascii=False))
        return
    state=evaluator.preflight()
    scope=scoped_state(evaluator,state,primary,outside,args.outside_arc_diagnostic)
    if args.mode=='check':
        print(json.dumps(dict(state='terminal_inputs_verified',**scope,fit_calls=0,gpu_calls=0),ensure_ascii=False))
    elif args.mode=='fit':
        fit=evaluator.fit_holdouts(state,out)
        state.seq.rt.save(out/'evaluation-scope.json',dict(**scope,fitted_sha256=fit['fitted_sha256']))
        print(fit['state'])
    else:
        fit=read(out/'fit-result.json')
        need(read(out/'evaluation-scope.json')==dict(**scope,fitted_sha256=fit['fitted_sha256']),
             'Render request differs from the frozen evaluation fit scope.')
        with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            receipt=evaluator.render_holdouts(state,out)
        for row in receipt['records']:
            row['evaluation_role']='primary_within_capture_arc' if row['frame'] in primary else 'outside_arc_diagnostic'
        # Keep every pose-screen failure visible. The optional out-of-arc
        # render is a separate section, never included in the primary cohort.
        state.seq.rt.save(out/'evaluation-review.json',dict(**scope,optimizer_updates=0,quality_accepted=False,
            render_receipt_sha256=sha(out/'renders/result.json'),
            primary_records=[r for r in receipt['records'] if r['frame'] in primary],
            outside_arc_records=[r for r in receipt['records'] if r['frame'] in outside]))
        print(receipt['state'])


if __name__=='__main__':
    main()
