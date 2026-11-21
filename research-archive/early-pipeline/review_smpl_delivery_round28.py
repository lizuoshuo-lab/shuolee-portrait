"""Independently replay R28 at the same signed static and original pose cameras."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round27 as prior


base,qa = prior.prior,prior.qa
pin,checked,read,sha,require,descriptor,rebound = prior.pin,prior.checked,prior.read,prior.sha,prior.require,prior.descriptor,prior.rebound
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r28')
DELIVERY = prior.DELIVERY.with_name('smpl-delivery-round28')
PREVIOUS_SHA = 'adcfd7faa75df235e2617b472f98c3a24cc609171ffa72879f9c06c9247adafe'
HISTORY = ('r22','r27')
NEW = {'r28_10':10000,'r28':18000}
ARMS = (*HISTORY,*NEW)
LABELS = ('R22 18,000','R27 18,000','R28 10,000','R28 18,000')
CORE_MODULES = (*prior.CORE_MODULES,'face_surface_association_r28.py','face_surface_geometry_r28.py')


def root_for(subject):
    return qa.review.LAB/f'shuolee-smpl-delivery-round28-{subject}'


def history(subject):
    index = pin(prior.DELIVERY/'review-receipt.json',PREVIOUS_SHA)
    aggregate = read(index)
    archive = aggregate['subjects'][subject]
    result_path = checked(archive['review'])
    result = read(result_path)
    previous = read(checked(result['plan']))
    baseline = base.history(subject)
    require(result['commit']==previous['commit']==aggregate['commit'] and result['subject']==subject
            and result['state']=='paired_native_face_and_static_views_rendered'
            and (result['static_camera_count'],result['native_poses'],result['auxiliary_face_poses'])==(50,28,4)
            and all(previous[key]==baseline[key] for key in ('frozen','rois','asset')),
            'R28 requires the verified original 50 fixed and 32 native R27 camera records.')
    old_grid = previous['static_grid']
    require(all(old_grid[key]==baseline['static_grid'][key] for key in ('camera_roster','proof','proxies')),
            'R27 original camera or crop evidence differs.')
    grid = dict(old_grid,arms={name:old_grid['arms'][name] for name in HISTORY})
    controls = {}
    for name in HISTORY:
        arm = grid['arms'][name]
        pin(arm['ply'],arm['ply_sha256'])
        pin(arm['camera_path'],arm['camera_sha256'])
        checked(arm['local'])
        checked(arm['world'])
        control = result['static'][name]['receipt']
        require(qa.grid_records(previous,name,checked(control))==result['static'][name]['records'],
                'The archived R22/R27 static images differ from the paired review.')
        controls[name] = control
        prior.verify_native_rows(result['native'][name],baseline)
    return dict(baseline,static_grid=grid,static_archive=controls,r27_index=descriptor(index),
                r27_review=descriptor(result_path),r27_plan=result['plan'],r27_archive=archive['archive'],
                r27_training_config=previous['training_config'],r27_material=previous['material'])


def plan(subject,result_hash,commit):
    implementation = rebound(prior.checked_implementation,PROJECT=PROJECT,CORE_MODULES=CORE_MODULES)(commit)
    run = root_for(subject)/'run'
    result_path = pin(run/'gaussian-run/result.json',result_hash)
    terminal = read(result_path)
    progress = run/'progress.json'
    require(terminal['state']=='checkpoint_verified' and not terminal['quality_accepted']
            and terminal['steps']==18000 and read(progress)['state']=='completed'
            and [row['step'] for row in terminal['endpoints']]==list(NEW.values()),
            'Both R28 duration endpoints must be complete.')
    baseline = history(subject)
    frozen,grid = baseline['frozen'],baseline['static_grid']
    config_path = run/'gaussian-run/config.json'
    config = read(config_path)
    start = read(checked(baseline['r27_training_config']))['pinned_inputs']
    require(all(config['pinned_inputs'][key]==digest for key,digest in start.items()),
            'R28 changed the original training, geometry or material start.')
    pin(run/'data/training-inputs.json',frozen['selected']['sha256'])
    pin(run/'fit/fitted.npz',frozen['fit']['sha256'])
    for role,key in (('selected_fit','train_frames'),('holdout','holdout_frames'),('auxiliary_face','auxiliary_frames')):
        require(config[key]==[row['frame'] for row in frozen['rows'] if row['role']==role],
                'R28 changed the registered original pose or camera roster.')
    require(config['execution']=='main','R28 image delivery requires formal duration outputs.')
    snapshots = {}
    for name,item in zip(NEW,terminal['endpoints'],strict=True):
        grid['arms'][name],snapshots[name] = rebound(base.scan_snapshot,PROJECT=PROJECT)(run,item,frozen,grid)
    return dict(baseline,schema='shuolee-r28-native-face-review/1',commit=commit,result=descriptor(result_path),
        snapshots=snapshots,training_config=descriptor(config_path),implementation=implementation,
        script=descriptor(Path(__file__)),quality_accepted=False)


def main():
    if len(sys.argv)==5 and sys.argv[1]=='_worker':
        rebound(base.worker,NEW=NEW,ARMS=ARMS)(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check-history','check','render','assemble'))
    parser.add_argument('subject',choices=qa.prior.SUBJECTS)
    parser.add_argument('--result-sha')
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action=='check-history':
        baseline = history(args.subject)
        print(json.dumps(dict(subject=args.subject,history=list(HISTORY),cameras=len(baseline['static_grid']['camera_roster']),
                              native_cameras=len(baseline['frozen']['rows']))),flush=True)
        return
    if not args.result_sha or not args.commit:
        parser.error('Use the exact R28 production result and source commit.')
    current = plan(args.subject,args.result_sha,args.commit)
    if args.action=='check':
        print(json.dumps(dict(subject=args.subject,cameras=50,native_poses=28,auxiliary_face_poses=4,
                              points={name:current['snapshots'][name]['points'] for name in NEW})),flush=True)
        return
    out = root_for(args.subject)/'view-comparison'
    out.mkdir(exist_ok=True)
    path = out/'plan.json'
    if not path.exists():
        qa.review.save(path,current)
    require(read(path)==current,'The saved R28 independent review plan changed.')
    if args.action=='render':
        rebound(base.render,ARMS=NEW,__file__=__file__)(path)
        print(json.dumps(dict(subject=args.subject,state='cuda_review_released_gpu',plan=descriptor(path))),flush=True)
    comparison = dict(current,r26_review=current['r27_review'])
    rebound(prior.assemble,ARMS=ARMS,HISTORY=HISTORY,LABELS=LABELS)(path,comparison)


if __name__=='__main__':
    main()
