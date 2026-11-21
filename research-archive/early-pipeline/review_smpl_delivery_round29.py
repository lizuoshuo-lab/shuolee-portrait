"""Compare R29 endpoints with R28 on the same fixed cameras and native face pixels."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
import archive_smpl_delivery_round23 as panels
import review_smpl_delivery_round28 as prior
import review_smpl_delivery_round28_legacy as legacy


base,qa=prior.base,prior.qa
pin,checked,read,sha,require,descriptor,rebound=prior.pin,prior.checked,prior.read,prior.sha,prior.require,prior.descriptor,prior.rebound
PROJECT=Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r29')
DELIVERY=prior.DELIVERY.with_name('smpl-delivery-round29')
SUBJECTS=('001','005','006','008','016')
NEW={'r29_10':10000,'r29':18000}
ARMS=('r28',*NEW)
LABELS=('R28 18,000','R29 10,000','R29 18,000')
RECORD_TITLES=('Original native RGB',*LABELS,'Frozen SMPL-X','Native 2D features')
R28_INDEX={
    'review-receipt.json':'74b28a65ec8ebf810317b3ab3f048307f0be30acbff7eda82a788c83c7556e27',
    'view-index-001-005-008.json':'e06f985b9ab44d63dc31a1673f56c3a7c414e76d1196ce46763ac3c25b9a59d7',
}


def root_for(subject):
    return qa.review.LAB/f'shuolee-smpl-delivery-round29-{subject}'


def native_media(snapshot):
    if 'media' in snapshot:
        return snapshot['media']
    return {frame:dict(crop=item['box_xyxy'],files=item['files']) for frame,item in snapshot['replay'].items()}


def history(subject):
    is_legacy=subject in legacy.SUBJECTS
    name='view-index-001-005-008.json' if is_legacy else 'review-receipt.json'
    index=pin(prior.DELIVERY/name,R28_INDEX[name]); entry=read(index)['subjects'][subject]
    archive=read(checked(entry['archive']))
    previous_result=read(checked(archive['review']))
    previous_plan=read(checked(previous_result['plan']))
    frozen=previous_plan['frozen']
    frozen=dict(frozen,reference_frame=frozen.get('reference_frame',frozen.get('ref')))
    previous_grid=previous_plan['static_grid']
    require(previous_result['subject']==previous_plan['subject']==archive['subject']==subject and
            previous_result['commit']==previous_plan['commit']==archive['commit'] and
            previous_result['state']=='paired_native_face_and_static_views_rendered' and
            previous_result['static_camera_count']==len(previous_grid['camera_roster'])==50,
            'R28 signed starting camera or subject proof differs.')
    static=previous_result['static']['r28']['receipt']
    require(qa.grid_records(previous_plan,'r28',checked(static))==previous_result['static']['r28']['records'],
            'R28 18,000 fixed camera images do not match their archived receipt.')
    for item in previous_grid['proof'].values():
        checked(item)
    grid=dict(previous_grid,arms={'r28':previous_grid['arms']['r28']})
    for item in frozen['meshes'].values():
        checked(item)
    source=None
    if is_legacy:
        source=previous_plan['legacy_source']
        source_spec=read(checked(source))
        rois=base.source_rois(checked(source_spec['native_observations']),frozen['rows'])
    else:
        rois=previous_plan['rois']
    media=native_media(previous_plan['snapshots']['r28'])
    require(set(media)==set(rois)=={row['frame'] for row in frozen['rows']},'R28 did not retain the native pose roster.')
    return dict(subject=subject,asset=previous_plan['asset'],frozen=frozen,rois=rois,static_grid=grid,
                baseline_archive=entry['archive'],baseline_review=archive['review'],baseline_static=static,
                baseline_index=descriptor(index),baseline_media=media,legacy_source=source,
                unavailable_holdout_frames=previous_plan.get('unavailable_holdout_frames',[]),
                unavailable_holdout_reasons=previous_plan.get('unavailable_holdout_reasons',{}))


def implementation(commit):
    root=PROJECT/'src/shuolee_portrait'
    modules=tuple(sorted(path.name for path in root.glob('face_surface*.py')))
    require('face_surface_training.py' in modules,'The R29 training source checkout is incomplete.')
    proof=rebound(prior.prior.checked_implementation,PROJECT=PROJECT,CORE_MODULES=modules)(commit)
    pointer=(PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')
    git_dir=subprocess.check_output(['/usr/bin/wslpath','-u',pointer],text=True).strip()
    for filename in ('training.py','bound_replay.py'):
        file=root/filename
        stored=subprocess.check_output(['/usr/bin/git','--git-dir',git_dir,'show',
                                        f'{commit}:{file.relative_to(PROJECT).as_posix()}'])
        worktree=file.read_bytes()
        require(worktree==stored or worktree.replace(b'\r\n',b'\n')==stored,
                'R29 renderer or replay source differs from its formal commit.')
        proof[filename]=dict(descriptor(file),git_blob_sha256=sha256(stored).hexdigest())
    return proof


def verify_trajectory(snapshots,config):
    from audit_r23_numerical import read_state

    earlier=None
    for name,step in NEW.items():
        state=read_state(checked(snapshots[name]['checkpoint']))
        require(state['step']==step==len(state['draws']) and state['pinned_inputs']==config['pinned_inputs'] and
                state['recovery_schema']=='shuolee-r29-optimizer-rng-topology/1' and 'rng' in state and
                all(isinstance(slot,int) and 0<=slot<len(config['train_frames']) for slot in state['draws']),
                'The R29 optimizer state changed its registered source frames.')
        if earlier is not None:
            require(state['draws'][:len(earlier['draws'])]==earlier['draws'] and
                    state['split_events'][:len(earlier['split_events'])]==earlier['split_events'],
                    'The two R29 durations do not share one optimizer and topology trajectory.')
        earlier=state


def plan(subject,commit,digest):
    source=implementation(commit)
    baseline=history(subject)
    run=root_for(subject)/'run'
    result_path=pin(run/'gaussian-run/result.json',digest); terminal=read(result_path)
    config_path=run/'gaussian-run/config.json'; config=read(config_path)
    rows=baseline['frozen']['rows']
    require(terminal['state']=='checkpoint_verified' and terminal['steps']==18000 and
            [record['step'] for record in terminal['endpoints']]==list(NEW.values()) and
            read(run/'progress.json')['state']=='completed' and config['execution']=='main' and
            config['candidate']=='R29_face_surface' and
            config['pinned_inputs']['fit']==baseline['frozen']['fit']['sha256'],
            'The formal R29 durations or pinned SMPL-X fit are incomplete.')
    for role,key in (('selected_fit','train_frames'),('holdout','holdout_frames'),('auxiliary_face','auxiliary_frames')):
        require(config[key]==[row['frame'] for row in rows if row['role']==role],
                'R29 changed the native pose or training roster.')
    reserved=baseline['unavailable_holdout_frames']
    require(not set(config['train_frames'])&set(config['holdout_frames']+reserved) and
            set(config.get('reserved_holdouts',config['holdout_frames']))==set(config['holdout_frames']+reserved),
            'A reserved native evaluation pose was replaced or promoted into training.')
    grid=baseline['static_grid']; frozen=baseline['frozen']; snapshots={}
    for name,item in zip(NEW,terminal['endpoints'],strict=True):
        if baseline['legacy_source'] is not None:
            spec=read(checked(baseline['legacy_source']))
            require(config['pinned_inputs']['legacy_source']==baseline['legacy_source']['sha256'],
                    'R29 legacy starts from another exact source descriptor.')
            arm,shot=legacy.snapshot(run,item,spec,frozen,grid)
            arm['project']=str(PROJECT)
            shot['checkpoint']=descriptor(pin(run/'gaussian-run'/item['path']/'main-state.pt',item['state_sha256']))
        else:
            arm,shot=rebound(base.scan_snapshot,PROJECT=PROJECT)(run,item,frozen,grid)
        grid['arms'][name]=arm; snapshots[name]=shot
    verify_trajectory(snapshots,config)
    return dict(baseline,schema='shuolee-r29-native-face-review/1',commit=commit,result=descriptor(result_path),
        training_config=descriptor(config_path),script=descriptor(Path(__file__)),implementation=source,
        snapshots=snapshots,quality_accepted=False)


def original_images(row,box):
    with Image.open(checked(row['image'])) as original, Image.open(checked(row['mask'])) as matte:
        return original.convert('RGB').crop(box),matte.convert('L').crop(box)


def verify_media(row,media):
    for name,item in media.items():
        box=item['crop']
        original,matte=original_images(row,box)
        with Image.open(checked(item['files']['source'])) as saved, Image.open(checked(item['files']['mask'])) as alpha:
            require(np.array_equal(np.asarray(saved.convert('RGB')),np.asarray(original)) and
                    np.array_equal(np.asarray(alpha.convert('L')),np.asarray(matte)),
                    f'{name} changed original native RGB/alpha for {row["frame"]}.')
        for key in ('black','white','alpha'):
            with Image.open(checked(item['files'][key])) as image:
                require(image.size==original.size,'A rendered native channel changed crop dimensions.')


def local_scores(original,matte,black,alpha):
    source=np.asarray(original,dtype=np.float32)/255.
    pixels=np.asarray(black,dtype=np.float32)/255.
    coverage=np.asarray(alpha.convert('L'),dtype=np.float32)/255.
    mask=np.asarray(matte)>=128
    values=coverage[mask]
    return dict(foreground_pixels=int(mask.sum()),low_alpha_095_pixels=int((values<.95).sum()),
                foreground_rgb_mae=float(np.abs(source[mask]-pixels[mask]).mean()) if len(values) else None,
                foreground_mean_transmission=float((1.-values).mean(dtype=np.float64)) if len(values) else None)


def face_native(directory,row,media,rois):
    frame=row['frame']; scores={name:{} for name in ARMS}; region_panels={}
    for part,roi in dict(face=rois['face'],**rois['parts']).items():
        if roi['box'] is None:
            for name in ARMS:
                scores[name][part]=roi
            continue
        box=panels.crop_overlap(roi['box'],*(item['crop'] for item in media.values()))
        original,matte=original_images(row,box)
        verified=row.get('verified_face_mask')
        if verified:
            allowed=panels.crop_file(checked(verified),box,row['verified_box']).convert('L')
            matte=Image.fromarray(np.where(np.asarray(allowed)==255,np.asarray(matte),0).astype(np.uint8))
        files=[]
        for name,item in media.items():
            black,white,alpha=(panels.crop_file(checked(item['files'][key]),box,item['crop'])
                               for key in ('black','white','alpha'))
            scores[name][part]=dict(roi,box=box,**local_scores(original,matte,black,alpha))
            if part=='face':
                proof=qa.static.r10.strip(directory/f'{frame}-{name}.png',box,(original,black,white,alpha))
                files.append(pin(directory/proof['file'],proof['sha256']))
        if part=='face':
            region_panels['face']=qa.stack(directory/f'{frame}-face.png',files,LABELS)
    return scores,region_panels.get('face')


def selected_panel(directory,row,source,media):
    before=checked(source)
    box=source['crop_xyxy']; w=box[2]-box[0]; h=box[3]-box[1]
    with Image.open(before) as image:
        header=image.height-h
        require(image.width==6*w and header==26,'The exact original, Mesh and keypoint columns changed.')
    crop=panels.crop_overlap(box,*(item['crop'] for item in media.values()))
    original,mesh,features=(panels.crop_file(before,crop,box,col=i,header=header) for i in (0,4,5))
    raw,_=original_images(row,crop)
    require(np.array_equal(np.asarray(original),np.asarray(raw)),'Selected native originals differ from the R28 archive.')
    gaussians=[panels.crop_file(checked(item['files']['black']),crop,item['crop']) for item in media.values()]
    paste=rebound(panels.paste_panel,TITLE=RECORD_TITLES)
    output=paste(directory/f'{row["frame"]}.png',(original,*gaussians,mesh,features))
    return dict(output,frame=row['frame'],source_sha256=row['image']['sha256'],crop_xyxy=crop)


def native(out,current):
    source=read(checked(current['baseline_archive']))['selected']
    selected_dir=out/'selected-views'; face_dir=out/'native-face'
    for directory in (selected_dir,face_dir):
        directory.mkdir(exist_ok=True)
    new_media={name:native_media(current['snapshots'][name]) for name in NEW}
    selected,images,score={}, {}, {name:{} for name in ARMS}
    for row in current['frozen']['rows']:
        frame=row['frame']
        media={'r28':current['baseline_media'][frame],**{name:views[frame] for name,views in new_media.items()}}
        verify_media(row,media)
        values,image=face_native(face_dir,row,media,current['rois'][frame])
        for name in ARMS:
            score[name][frame]=dict(role=row['role'],regions=values[name])
        if image is not None:
            images[f'native_{frame}_face']=image
        if row['role']=='selected_fit':
            selected[frame]=selected_panel(selected_dir,row,source[frame],media)
    return selected,images,score


def assemble(path,current):
    out=path.parent
    paired=out/'paired'; paired.mkdir(exist_ok=True)
    static={}
    for name in ARMS:
        receipt=checked(current['baseline_static']) if name=='r28' else out/'static-grid'/name/'result.json'
        static[name]=dict(receipt=descriptor(receipt),records=qa.grid_records(current,name,receipt))
    images={}
    for key in current['static_grid']['camera_roster']:
        images[key]=qa.stack(paired/f'{key}.png',[checked(static[name]['records'][key]['panel']) for name in ARMS],LABELS)
    selected,faces,scores=native(out,current); images.update(faces)
    result=dict(schema='shuolee-r29-native-face-review-result/1',state='paired_native_face_and_static_views_rendered',
        subject=current['subject'],commit=current['commit'],static_camera_count=50,static=static,
        native_poses=len(current['frozen']['rows']),native_face_comparisons=len(faces),
        auxiliary_face_poses=sum(row['role']=='auxiliary_face' for row in current['frozen']['rows']),
        native_scores=scores,native_score_source='Same native source crops and archived 8-bit renderer RGB and alpha.',
        panels=images,selected=selected,plan=descriptor(path),
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
        baseline=history(args.subject)
        print(json.dumps(dict(subject=args.subject,cameras=len(baseline['static_grid']['camera_roster']),
            native=len(baseline['frozen']['rows']),face=sum(roi['face']['box'] is not None for roi in baseline['rois'].values()),
            unavailable=baseline['unavailable_holdout_frames'])),flush=True)
        return
    if not args.result_sha or not args.commit:
        parser.error('Provide the formal R29 result SHA and commit.')
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
