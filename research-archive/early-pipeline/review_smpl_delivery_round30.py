"""Compare R30 and R29 using the same fixed cameras and visible native face pixels."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_round30_evaluation_visibility as evaluation
import review_smpl_delivery_round29 as prior


PROJECT=Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r30')
DELIVERY=prior.DELIVERY.with_name('smpl-delivery-round30')
SUBJECTS=prior.SUBJECTS
NEW={'r30_10':10000,'r30':18000}
ARMS=('r29',*NEW)
LABELS=('R29 18,000','R30 10,000','R30 18,000')
RECORD_TITLES=('Original native RGB',*LABELS,'Frozen SMPL-X','Native 2D features')
base,qa,legacy,panels=prior.base,prior.qa,prior.legacy,prior.panels
pin,checked,read,require,descriptor,rebound=prior.pin,prior.checked,prior.read,prior.require,prior.descriptor,prior.rebound


def root_for(subject):
    return qa.review.LAB/f'shuolee-smpl-delivery-round30-{subject}'


def history(subject):
    signed,result,previous=evaluation.baseline(subject)
    grid=previous['static_grid']
    static=result['static']['r29']['receipt']
    require(qa.grid_records(previous,'r29',checked(static))==result['static']['r29']['records'],
            'R29 18,000 fixed renders do not match the prior receipt.')
    for item in (*grid['proof'].values(),*previous['frozen']['meshes'].values()):
        checked(item)
    media=prior.native_media(previous['snapshots']['r29'])
    require(set(media)==set(previous['rois'])=={row['frame'] for row in previous['frozen']['rows']},
            'The previous native pose roster differs.')
    visibility_path=evaluation.OUT/f'{subject}.json'
    visibility=read(visibility_path)
    require(visibility['schema']==evaluation.SCHEMA and visibility['usage']=='evaluation_only' and
            visibility['subject']==subject and visibility['previous_review']==signed['review'] and
            visibility['meshes']==previous['frozen']['meshes'] and set(visibility['frames'])==set(media) and
            visibility['unavailable']==previous['unavailable_holdout_frames'],
            'The native face score masks must derive from the same frozen prior pose roster.')
    checked(visibility['arrays'])
    baseline={key:previous[key] for key in ('subject','asset','frozen','rois','legacy_source',
                                          'unavailable_holdout_frames','unavailable_holdout_reasons')}
    return dict(baseline,static_grid=dict(grid,arms={'r29':grid['arms']['r29']}),baseline_media=media,
        baseline_archive=signed['archive'],baseline_review=signed['review'],baseline_static=static,
        face_visibility=descriptor(visibility_path))


def verify_trajectory(snapshots,config):
    from audit_r23_numerical import read_state

    previous=None
    for arm,step in NEW.items():
        state=read_state(checked(snapshots[arm]['checkpoint']))
        require(state['step']==step==len(state['draws']) and state['pinned_inputs']==config['pinned_inputs'] and
                state['recovery_schema']=='shuolee-r30-optimizer-rng-topology/1' and 'rng' in state and
                all(isinstance(slot,int) and 0<=slot<len(config['train_frames']) for slot in state['draws']),
                'R30 changed the registered main optimizer roster.')
        if previous:
            require(state['draws'][:len(previous['draws'])]==previous['draws'] and
                    state['split_events'][:len(previous['split_events'])]==previous['split_events'],
                    'R30 durations do not share one optimization and topology trajectory.')
        previous=state


def plan(subject,commit,digest):
    code=rebound(prior.implementation,PROJECT=PROJECT)(commit)
    starting=history(subject)
    run=root_for(subject)/'run'
    result_path=pin(run/'gaussian-run/result.json',digest); result=read(result_path)
    config_path=run/'gaussian-run/config.json'; config=read(config_path)
    frozen=starting['frozen']; rows=frozen['rows']
    require(result['state']=='checkpoint_verified' and result['steps']==18000 and
            [item['step'] for item in result['endpoints']]==list(NEW.values()) and
            read(run/'progress.json')['state']=='completed' and config['execution']=='main' and
            config['candidate']=='R30_face_surface' and config['pinned_inputs']['fit']==frozen['fit']['sha256'],
            'R30 formal durations or frozen fit differ.')
    for role,key in (('selected_fit','train_frames'),('holdout','holdout_frames'),('auxiliary_face','auxiliary_frames')):
        require(config[key]==[row['frame'] for row in rows if row['role']==role],
                'R30 changed the original native pose or training roster.')
    reserved=starting['unavailable_holdout_frames']
    held=set(config['holdout_frames']+reserved)
    require(not held.intersection(config['train_frames']+config['auxiliary_frames']) and
            set(config.get('reserved_holdouts',config['holdout_frames']))==held,
            'A reserved pose entered training or was substituted.')
    grid=starting['static_grid']; snapshots={}
    for arm,item in zip(NEW,result['endpoints'],strict=True):
        if starting['legacy_source']:
            source=read(checked(starting['legacy_source']))
            require(config['pinned_inputs']['legacy_source']==starting['legacy_source']['sha256'],
                    'R30 changed the exact archived legacy descriptor.')
            renderer,snapshot=legacy.snapshot(run,item,source,frozen,grid)
            renderer['project']=str(PROJECT)
            snapshot['checkpoint']=descriptor(pin(run/'gaussian-run'/item['path']/'main-state.pt',item['state_sha256']))
        else:
            renderer,snapshot=rebound(base.scan_snapshot,PROJECT=PROJECT)(run,item,frozen,grid)
        grid['arms'][arm]=renderer; snapshots[arm]=snapshot
    verify_trajectory(snapshots,config)
    return dict(starting,schema='shuolee-r30-native-face-review/1',commit=commit,result=descriptor(result_path),
        training_config=descriptor(config_path),script=descriptor(Path(__file__)),implementation=code,
        snapshots=snapshots,quality_accepted=False)


def score_support(box,support,face_box):
    left,top,right,bottom=box
    x0,y0,x1,y1=face_box
    l,t,r,b=max(left,x0),max(top,y0),min(right,x1),min(bottom,y1)
    valid=np.zeros((bottom-top,right-left),dtype=np.bool_)
    if l<r and t<b:
        valid[t-top:b-top,l-left:r-left]=support[t-y0:b-y0,l-x0:r-x0]
    return valid


def face_native(directory,row,media,rois,visibility,masks):
    frame=row['frame']; details=visibility['frames'][frame]
    scores={name:{} for name in ARMS}; panel=None
    require((details['source'],details['matte'],details['role'],details['box'])==
            (row['image'],row['mask'],row['role'],rois['face']['box']),
            'A face score no longer uses its exact source pixels.')
    support=None
    if details['state']=='scored':
        support=masks[details['mask_key']]
        require(support.dtype==np.bool_ and evaluation.array_sha(support)==details['mask_sha256'] and
                int(support.sum())==details['pixels'],'The visible face support differs from its receipt.')
    for part,roi in dict(face=rois['face'],**rois['parts']).items():
        if roi['box'] is None:
            for name in ARMS:
                scores[name][part]=dict(roi,score_state='unscored',reason='no_native_detector_crop')
            continue
        box=panels.crop_overlap(roi['box'],*(item['crop'] for item in media.values()))
        original,matte=prior.original_images(row,box)
        valid=score_support(box,support,details['box']) if support is not None else None
        if valid is not None:
            require(np.all(np.asarray(matte)[valid]>=252),'Trusted face mask escaped original foreground.')
        unscored=details.get('reason','no_trusted_face_pixels_in_region')
        files=[]
        for name,item in media.items():
            if part=='face' or (valid is not None and valid.any()):
                black,white,alpha=(panels.crop_file(checked(item['files'][key]),box,item['crop'])
                                   for key in ('black','white','alpha'))
            score=dict(roi,box=box,score_state='unscored',reason=unscored)
            if valid is not None and valid.any():
                pixels=Image.fromarray(valid.astype(np.uint8)*255)
                score=dict(roi,box=box,score_state='scored',**prior.local_scores(original,pixels,black,alpha))
            scores[name][part]=score
            if part=='face':
                proof=qa.static.r10.strip(directory/f'{frame}-{name}.png',box,(original,black,white,alpha))
                files.append(pin(directory/proof['file'],proof['sha256']))
        if part=='face':
            labels=LABELS if support is not None else tuple(label+' | crop not scored' for label in LABELS)
            panel=qa.stack(directory/f'{frame}-face.png',files,labels)
    return scores,panel


def native(out,current):
    visibility=read(checked(current['face_visibility']))
    selected_panel=rebound(prior.selected_panel,RECORD_TITLES=RECORD_TITLES)
    with np.load(checked(visibility['arrays']),allow_pickle=False) as masks:
        def render(directory,row,media,rois):
            return face_native(directory,row,media,rois,visibility,masks)
        # R29's native loop names its control explicitly; use its existing
        # original and overlay checks with the R30 arms here.
        selected_dir=out/'selected-views'; face_dir=out/'native-face'
        for directory in (selected_dir,face_dir):
            directory.mkdir(exist_ok=True)
        archived=read(checked(current['baseline_archive']))['selected']
        new_media={name:prior.native_media(current['snapshots'][name]) for name in NEW}
        selected,images,scores={}, {}, {name:{} for name in ARMS}
        for row in current['frozen']['rows']:
            frame=row['frame']
            media={'r29':current['baseline_media'][frame],**{name:views[frame] for name,views in new_media.items()}}
            prior.verify_media(row,media)
            values,image=render(face_dir,row,media,current['rois'][frame])
            for name in ARMS:
                scores[name][frame]=dict(role=row['role'],regions=values[name])
            if image is not None:
                images[f'native_{frame}_face']=image
            if row['role']=='selected_fit':
                selected[frame]=selected_panel(selected_dir,row,archived[frame],media)
    return selected,images,scores


def assemble(path,current):
    out=path.parent
    paired=out/'paired'; paired.mkdir(exist_ok=True)
    static={}
    for arm in ARMS:
        receipt=checked(current['baseline_static']) if arm=='r29' else out/'static-grid'/arm/'result.json'
        static[arm]=dict(receipt=descriptor(receipt),records=qa.grid_records(current,arm,receipt))
    images={key:qa.stack(paired/f'{key}.png',[checked(static[arm]['records'][key]['panel']) for arm in ARMS],LABELS)
            for key in current['static_grid']['camera_roster']}
    selected,faces,scores=native(out,current)
    images.update(faces)
    admitted={frame:row['role'] for frame,row in scores['r30'].items()
              if row['regions']['face']['score_state']=='scored'}
    result=dict(schema='shuolee-r30-native-face-review-result/1',state='paired_native_face_and_static_views_rendered',
        subject=current['subject'],commit=current['commit'],static_camera_count=50,static=static,
        native_poses=len(current['frozen']['rows']),native_face_comparisons=len(faces),native_face_scored=admitted,
        auxiliary_face_poses=sum(row['role']=='auxiliary_face' for row in current['frozen']['rows']),
        native_scores=scores,native_score_source='Fixed visible face pixels from original native imagery and frozen posed anatomy.',
        face_visibility=current['face_visibility'],panels=images,selected=selected,plan=descriptor(path),
        unavailable_holdout_frames=current['unavailable_holdout_frames'],
        unavailable_holdout_reasons=current['unavailable_holdout_reasons'],quality_accepted=False)
    print(json.dumps(legacy.save_exact(out/'result.json',result)),flush=True)


def main():
    if len(sys.argv)==5 and sys.argv[1]=='_worker':
        path,name,fd=Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]); current=read(path)
        checked(current['script'])
        for item in current['implementation'].values():
            checked(item)
        if not (path.parent/'static-grid'/name/'result.json').is_file():
            rebound(qa.static.render_grid,PORTABLE=ARMS)(path,name,fd)
        return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check-history','check','render','assemble'))
    parser.add_argument('subject',choices=SUBJECTS)
    parser.add_argument('--result-sha')
    parser.add_argument('--commit')
    args=parser.parse_args()
    if args.action=='check-history':
        starting=history(args.subject)
        visible=read(checked(starting['face_visibility']))['frames']
        print(json.dumps(dict(subject=args.subject,cameras=len(starting['static_grid']['camera_roster']),
            native=len(visible),scored=sum(row['state']=='scored' for row in visible.values()),
            unavailable=starting['unavailable_holdout_frames'])),flush=True)
        return
    if not args.result_sha or not args.commit:
        parser.error('Provide the formal R30 result SHA and commit.')
    out=root_for(args.subject)/'view-comparison'; out.mkdir(exist_ok=True)
    current=plan(args.subject,args.commit,args.result_sha)
    path=out/'plan.json'; legacy.save_exact(path,current)
    if args.action=='check':
        print(json.dumps(descriptor(path)),flush=True)
        return
    if args.action=='render':
        rebound(base.render,ARMS=NEW,__file__=__file__)(path)
    assemble(path,current)


if __name__=='__main__':
    main()
