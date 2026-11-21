"""Compare R31 against both prior runs on fixed camera and native-pixel evidence."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round29 as r29
import review_smpl_delivery_round30 as prior


PROJECT=Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r31')
DELIVERY=prior.DELIVERY.with_name('smpl-delivery-round31')
SUBJECTS=prior.SUBJECTS
R30_INDEX_SHA='df81fe3cdb5f52e186cf76724b8b25b672ea502b2a7c30fdcaa6a3fd4b6003f0'
NEW={'r31_10':10000,'r31':18000}
ARMS=('r29','r30',*NEW)
LABELS=('R29 18,000','R30 18,000','R31 10,000','R31 18,000')
RECORD_TITLES=('Original native RGB',*LABELS,'Frozen SMPL-X','Native 2D features')
base,qa,legacy,panels=prior.base,prior.qa,prior.legacy,prior.panels
pin,checked,read,require,descriptor,rebound=prior.pin,prior.checked,prior.read,prior.require,prior.descriptor,prior.rebound


def root_for(subject):
    return qa.review.LAB/f'shuolee-smpl-delivery-round31-{subject}'


def history(subject):
    control=prior.history(subject)
    index=pin(prior.DELIVERY/'review-receipt.json',R30_INDEX_SHA)
    signed=read(index)['subjects'][subject]
    result=read(checked(signed['review']))
    previous=read(checked(result['plan']))
    grid=previous['static_grid']
    require(result['state']=='paired_native_face_and_static_views_rendered' and
            result['subject']==previous['subject']==subject and result['static_camera_count']==50 and
            previous['frozen']==control['frozen'] and previous['rois']==control['rois'] and
            previous['face_visibility']==control['face_visibility'] and
            all(grid[key]==control['static_grid'][key] for key in ('camera_roster','proof','proxies')),
            'R30 controls changed original pose, cameras or scoring regions.')
    native=r29.native_media(previous['snapshots']['r30'])
    scores=result['native_scores']['r30']
    require(set(native)==set(scores)==set(control['baseline_media']),
            'R30 control does not replay the same original frame roster.')
    static=result['static']['r30']['receipt']
    require(qa.grid_records(previous,'r30',checked(static))==result['static']['r30']['records'],
            'R30 18,000 fixed renders do not match the archived receipt.')
    full_boxes={}
    for row in control['frozen']['rows']:
        frame=row['frame']
        face=scores[frame]['regions']['face']
        require(scores[frame]['role']==row['role'],'R30 changed an original frame role.')
        if face['box'] is None:
            full_boxes[frame]=None
            continue
        expected=panels.crop_overlap(control['rois'][frame]['face']['box'],
            control['baseline_media'][frame]['crop'],native[frame]['crop'])
        require(tuple(face['box'])==tuple(expected),'R30 native full foreground crop changed.')
        full_boxes[frame]=face['box']
    new_grid=dict(grid,arms={'r29':control['static_grid']['arms']['r29'],'r30':grid['arms']['r30']})
    return dict(control,static_grid=new_grid,baseline_archive=signed['archive'],baseline_r30_review=signed['review'],
        baseline_r30_index=descriptor(index),baseline_r30_static=static,baseline_r30_snapshot=previous['snapshots']['r30'],
        full_native_boxes=full_boxes)


def verify_trajectory(snapshots,config):
    from audit_r23_numerical import read_state

    previous=None
    for arm,step in NEW.items():
        state=read_state(checked(snapshots[arm]['checkpoint']))
        require(state['step']==step==len(state['draws']) and state['pinned_inputs']==config['pinned_inputs'] and
                state['recovery_schema']=='shuolee-r31-optimizer-rng-topology/1' and 'rng' in state and
                all(isinstance(slot,int) and 0<=slot<len(config['train_frames']) for slot in state['draws']),
                'R31 changed the registered main optimizer roster.')
        if previous is not None:
            require(state['draws'][:len(previous['draws'])]==previous['draws'] and
                    state['split_events'][:len(previous['split_events'])]==previous['split_events'],
                    'R31 durations do not share one optimization and topology trajectory.')
        previous=state


def plan(subject,commit,digest):
    code=rebound(r29.implementation,PROJECT=PROJECT)(commit)
    starting=history(subject)
    run=root_for(subject)/'run'
    result_path=pin(run/'gaussian-run/result.json',digest); result=read(result_path)
    config_path=run/'gaussian-run/config.json'; config=read(config_path)
    frozen=starting['frozen']; rows=frozen['rows']
    require(result['state']=='checkpoint_verified' and result['steps']==18000 and
            [item['step'] for item in result['endpoints']]==list(NEW.values()) and
            read(run/'progress.json')['state']=='completed' and config['execution']=='main' and
            config['candidate']=='R31_face_surface' and config['pinned_inputs']['fit']==frozen['fit']['sha256'],
            'R31 formal durations or frozen fit differ.')
    for role,key in (('selected_fit','train_frames'),('holdout','holdout_frames'),('auxiliary_face','auxiliary_frames')):
        require(config[key]==[row['frame'] for row in rows if row['role']==role],
                'R31 changed the original native pose or training roster.')
    held=set(config['holdout_frames']+starting['unavailable_holdout_frames'])
    require(not held.intersection(config['train_frames']+config['auxiliary_frames']) and
            set(config.get('reserved_holdouts',config['holdout_frames']))==held,
            'A reserved pose entered training or was substituted.')
    grid=starting['static_grid']; snapshots={}
    for arm,item in zip(NEW,result['endpoints'],strict=True):
        if starting['legacy_source']:
            source=read(checked(starting['legacy_source']))
            require(config['pinned_inputs']['legacy_source']==starting['legacy_source']['sha256'],
                    'R31 changed the exact archived legacy descriptor.')
            renderer,snapshot=legacy.snapshot(run,item,source,frozen,grid)
            renderer['project']=str(PROJECT)
            snapshot['checkpoint']=descriptor(pin(run/'gaussian-run'/item['path']/'main-state.pt',item['state_sha256']))
        else:
            renderer,snapshot=rebound(base.scan_snapshot,PROJECT=PROJECT)(run,item,frozen,grid)
        grid['arms'][arm]=renderer; snapshots[arm]=snapshot
    verify_trajectory(snapshots,config)
    return dict(starting,schema='shuolee-r31-native-face-review/1',commit=commit,result=descriptor(result_path),
        training_config=descriptor(config_path),script=descriptor(Path(__file__)),implementation=code,
        snapshots=snapshots,quality_accepted=False)


def face_native(directory,row,media,rois,visibility,masks,full_box):
    if full_box is not None:
        require(tuple(panels.crop_overlap(full_box,*(item['crop'] for item in media.values())))==tuple(full_box),
                'A candidate native replay does not cover the frozen full foreground crop.')
    scores,panel=rebound(prior.face_native,ARMS=ARMS,LABELS=LABELS)(directory,row,media,rois,visibility,masks)
    supported=scores[ARMS[0]]['face']['score_state']=='scored'
    if supported:
        original,matte=r29.original_images(row,full_box)
    for arm,item in media.items():
        inner=scores[arm]['face']
        require((tuple(inner['box']) if inner['box'] is not None else None)==
                (tuple(full_box) if full_box is not None else None),
                'The native comparison altered its original full crop.')
        require((inner['score_state']=='scored')==supported,'Native eligibility differs between controls and candidates.')
        if supported:
            black,alpha=(panels.crop_file(checked(item['files'][key]),full_box,item['crop']) for key in ('black','alpha'))
            metric=r29.local_scores(original,matte,black,alpha)
            require(metric['foreground_pixels']>=inner['foreground_pixels'],'The trusted inner-face region escaped the source matte.')
            scores[arm]['full_foreground']=dict(box=full_box,score_state='scored',**metric)
        else:
            scores[arm]['full_foreground']=dict(box=full_box,score_state='unscored',reason=inner['reason'])
    return scores,panel


def selected_panel(path,row,source,media):
    source_path=checked(source)
    box=source['crop_xyxy']; width=box[2]-box[0]; height=box[3]-box[1]
    with Image.open(source_path) as archived:
        require(archived.size==(6*width,height+26),'The R30 original, Mesh and keypoint columns changed.')
    crop=panels.crop_overlap(box,*(item['crop'] for item in media.values()))
    original,mesh,features=(panels.crop_file(source_path,crop,box,col=i,header=26) for i in (0,4,5))
    raw,_=r29.original_images(row,crop)
    require(raw.tobytes()==original.tobytes(),'Selected original RGB differs from the archived R30 pixels.')
    gaussians=[panels.crop_file(checked(item['files']['black']),crop,item['crop']) for item in media.values()]
    parts=(original,*gaussians,mesh,features); width,height=original.size
    canvas=Image.new('RGB',(len(parts)*width,height+26),'#181818')
    draw=ImageDraw.Draw(canvas)
    for index,(label,part) in enumerate(zip(RECORD_TITLES,parts,strict=True)):
        require(part.size==original.size,'Selected native comparison crops differ.')
        draw.text((index*width+6,6),label,fill='white')
        canvas.paste(part,(index*width,26))
    if path.is_file():
        with Image.open(path) as archived:
            require(archived.size==canvas.size and archived.convert('RGB').tobytes()==canvas.tobytes(),
                    'Preserve the existing R31 selected frame image.')
    else:
        with path.open('xb') as stream:
            canvas.save(stream,format='PNG',compress_level=1)
    return dict(descriptor(path),frame=row['frame'],source_sha256=row['image']['sha256'],crop_xyxy=crop)


def native(out,current):
    selected_dir=out/'selected-views'; face_dir=out/'native-face'
    for directory in (selected_dir,face_dir):
        directory.mkdir(exist_ok=True)
    archived=read(checked(current['baseline_archive']))['selected']
    visibility=read(checked(current['face_visibility']))
    sources={'r29':current['baseline_media'],'r30':r29.native_media(current['baseline_r30_snapshot'])}
    sources.update({arm:r29.native_media(current['snapshots'][arm]) for arm in NEW})
    selected,images,scores={}, {}, {name:{} for name in ARMS}
    with np.load(checked(visibility['arrays']),allow_pickle=False) as masks:
        for row in current['frozen']['rows']:
            frame=row['frame']
            media={arm:source[frame] for arm,source in sources.items()}
            r29.verify_media(row,media)
            values,image=face_native(face_dir,row,media,current['rois'][frame],visibility,masks,
                                     current['full_native_boxes'][frame])
            for arm in ARMS:
                scores[arm][frame]=dict(role=row['role'],regions=values[arm])
            if image is not None:
                images[f'native_{frame}_face']=image
            if row['role']=='selected_fit':
                selected[frame]=selected_panel(selected_dir/f'{frame}.png',row,archived[frame],media)
    return selected,images,scores


def assemble(path,current):
    out=path.parent
    paired=out/'paired'; paired.mkdir(exist_ok=True)
    control={'r29':current['baseline_static'],'r30':current['baseline_r30_static']}
    static={}
    for arm in ARMS:
        receipt=checked(control[arm]) if arm in control else out/'static-grid'/arm/'result.json'
        static[arm]=dict(receipt=descriptor(receipt),records=qa.grid_records(current,arm,receipt))
    images={key:qa.stack(paired/f'{key}.png',[checked(static[arm]['records'][key]['panel']) for arm in ARMS],LABELS)
            for key in current['static_grid']['camera_roster']}
    selected,faces,scores=native(out,current)
    images.update(faces)
    admitted={frame:row['role'] for frame,row in scores['r31'].items()
              if row['regions']['face']['score_state']=='scored'}
    require(set(admitted)=={frame for frame,row in scores['r31'].items()
                            if row['regions']['full_foreground']['score_state']=='scored'},
            'The two native metrics use different visible-face frame rosters.')
    result=dict(schema='shuolee-r31-native-face-review-result/1',state='paired_native_face_and_static_views_rendered',
        subject=current['subject'],commit=current['commit'],static_camera_count=50,static=static,
        native_poses=len(current['frozen']['rows']),native_face_comparisons=len(faces),native_face_scored=admitted,
        native_full_roi_scored=admitted,auxiliary_face_poses=sum(row['role']=='auxiliary_face' for row in current['frozen']['rows']),
        native_scores=scores,native_score_source={
            'face':'Trusted visible inner-face support from the original native image and frozen pose.',
            'full_foreground':'Fixed R30 native face ROI and original foreground matte; can include hair and neck.'},
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
        parser.error('Provide the formal R31 result SHA and commit.')
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
