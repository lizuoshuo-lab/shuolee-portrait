"""Verify R27 and compare the delivered renders at the signed R26 and R22 cameras."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round26 as prior


qa = prior.qa
pin,checked,read,sha,require,descriptor,rebound = prior.pin,prior.checked,prior.read,prior.sha,prior.require,prior.descriptor,prior.rebound
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r27')
DELIVERY = prior.DELIVERY.with_name('smpl-delivery-round27')
PREVIOUS_SHA = '438a33dffc8ea91fda4f2ec381d2291904ac52b87fe7eb8a04488f31f8bbc8db'
HISTORY = ('r22','r26')
NEW = {'r27_10':10000,'r27':18000}
ARMS = (*HISTORY,*NEW)
LABELS = ('R22 18,000','R26 18,000','R27 10,000','R27 18,000')
CORE_MODULES = ('face_surface_bundle.py','face_surface_geometry.py','face_surface_geometry_r27.py',
                'face_surface_split.py','face_surface_training.py')


def root_for(subject):
    return qa.review.LAB/f'shuolee-smpl-delivery-round27-{subject}'


def verify_native_rows(native,baseline):
    frozen,rois = baseline['frozen'],baseline['rois']
    require(set(native)=={row['frame'] for row in frozen['rows']},'Missing named native camera or pose.')
    for row in frozen['rows']:
        frame = row['frame']
        item = native[frame]
        require(item['role']==row['role'] and all(item['camera'][key]==row[key] for key in ('K','wh','w2c','frame'))
                and item['source_sha256']==row['image']['sha256'] and item['mask_sha256']==row['mask']['sha256'],
                'A native render uses a different source, camera, mask or pose.')
        crops = dict(face=rois[frame]['face'],**rois[frame]['parts'])
        for part,score in item['regions'].items():
            require(score['box']==crops[part]['box'],'A signed original face region was recropped.')
            if score['box'] is not None:
                checked(score['panel'])
        if row['role']=='selected_fit':
            for proof in item['body']['files'].values():
                checked(proof)


def history(subject):
    signed = pin(prior.DELIVERY/'review-receipt.json',PREVIOUS_SHA)
    index = read(signed)
    receipt = index['subjects'][subject]
    result_path = checked(receipt['review'])
    result = read(result_path)
    previous = read(checked(result['plan']))
    baseline = prior.history(subject)
    require(index['commit']==result['commit']==previous['commit'] and result['subject']==subject
            and result['state']=='paired_native_face_and_static_views_rendered'
            and (result['static_camera_count'],result['native_poses'],result['auxiliary_face_poses'])==(50,28,4)
            and all(previous[key]==baseline[key] for key in ('frozen','rois','asset')),
            'The R26 result must use the exact original 50 cameras and 24+4+4 native pose roster.')
    old_grid = previous['static_grid']
    require(all(old_grid[key]==baseline['static_grid'][key] for key in ('camera_roster','proof','proxies')),
            'R26 changed the registered camera, anatomical proxies or crop.')
    grid = dict(old_grid,arms={name:old_grid['arms'][name] for name in HISTORY})
    cached_static = {}
    for name in HISTORY:
        arm = grid['arms'][name]
        pin(arm['ply'],arm['ply_sha256'])
        pin(arm['camera_path'],arm['camera_sha256'])
        checked(arm['local'])
        checked(arm['world'])
        cached_static[name] = result['static'][name]['receipt']
        require(qa.grid_records(previous,name,checked(cached_static[name]))==result['static'][name]['records'],
                'The signed static comparison pixels differ.')
        verify_native_rows(result['native'][name],baseline)
    return dict(baseline,static_grid=grid,static_archive=cached_static,
                r26_review=descriptor(result_path),r26_archive=receipt['archive'],r26_index=descriptor(signed),
                r26_training_config=previous['training_config'])


def checked_implementation(commit):
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Use the complete R27 source commit.')
    pointer = (PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')
    git_dir = subprocess.check_output(['/usr/bin/wslpath','-u',pointer],text=True).strip()
    git = ['/usr/bin/git','--git-dir',git_dir,'--work-tree',str(PROJECT)]
    require(subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip()==commit,
            'Use the exact R27 source checkout.')
    proof = {}
    for filename in CORE_MODULES:
        path = f'src/shuolee_portrait/{filename}'
        content = subprocess.check_output([*git,'show',f'{commit}:{path}'])
        proof[filename] = descriptor(pin(PROJECT/path,sha256(content).hexdigest()))
    return proof


def material_evidence(subject,baseline,config=None):
    root = DELIVERY/'input-audit'
    receipt = root/f'{subject}-material-visibility.json'
    arrays = root/f'{subject}-material-visibility.npz'
    registered = read(receipt)
    pin(arrays,registered['material_visibility_npz_sha256'])
    older_path = checked(read(checked(baseline['r26_archive']))['material'])
    older = read(older_path)
    previous = pin(older['material_evidence_file'],older['material_evidence_npz_sha256'])
    frozen = baseline['frozen']
    selected,heldout,aux = ([row['frame'] for row in frozen['rows'] if row['role']==role]
                            for role in ('selected_fit','holdout','auxiliary_face'))
    require(registered['schema']=='shuolee-r27-native-face-material-visibility/1'
            and registered['subject']==subject and registered['fit_sha256']==frozen['fit']['sha256']
            and registered['training_inputs_sha256']==frozen['selected']['sha256']
            and all(registered[key]==older[key] for key in ('baseline_bound_ply_sha256','baseline_raw_ply_sha256'))
            and registered['selected_frames']==selected and registered['retained_holdouts']==heldout
            and registered['admitted_auxiliary_frames']==aux and registered['classification_frames']==selected+aux
            and not registered['heldout_source_images_read'],
            'R27 material visibility must come from the frozen original 24+4 fitting views only.')
    if config is not None:
        pin(receipt,config['pinned_inputs']['material_receipt'])
        pin(arrays,config['pinned_inputs']['material'])
        require(config['pinned_inputs']['aux_replay']==registered['provenance']['auxiliary_replay_sha256'],
                'R27 material evidence and training use different auxiliary replays.')
    with np.load(arrays,allow_pickle=False) as candidate,np.load(previous,allow_pickle=False) as original:
        ids = candidate['stable_ids']
        labels = candidate['labels']
        fields = ('head_scope','observed','geometry_eligible','silhouette_spill',
                  'unknown_unobserved_elongated','possible_protected_material')
        require(all(np.array_equal(candidate[key],original[key]) and candidate[key].dtype==original[key].dtype
                    for key in ('stable_ids','owner_faces')) and len(np.unique(ids))==len(ids)
                and all(candidate[key].shape==ids.shape and candidate[key].dtype==np.bool_ for key in fields)
                and labels.shape==ids.shape and np.isin(labels,np.arange(6)).all(),
                'R27 material visibility is not aligned to the original Gaussian row identities and faces.')
        totals = registered['coverage']['main_24_plus_aux_4']
        for key in ('observed','geometry_eligible','silhouette_spill','possible_protected_material'):
            require(int(candidate[key].sum())==totals[key],f'R27 material evidence count changed: {key}.')
        return dict(receipt=descriptor(receipt),arrays=descriptor(arrays),previous=descriptor(older_path),
                    rows=len(ids),material_labels_changed=int(np.count_nonzero(labels!=original['labels'])),
                    original_stable_ids_and_owners_preserved=True)


def gradient_evidence(run,frozen,material,endpoints):
    path = run/'gaussian-run/native-gradient-audit.json'
    receipt = read(path)
    expected = {str(slot):row['frame'] for slot,row in enumerate(frozen['rows']) if row['role']!='holdout'}
    observed = read(checked(material['receipt']))['coverage']['main_24_plus_aux_4']['observed']
    require(receipt['source_slots']==expected and receipt['observed_rows']==observed
            and all(0<receipt[key]['active_rows_after_filter']<=observed for key in ('rgb','alpha')),
            'Native R27 RGB and alpha gradient evidence has a different 24+4 roster or visibility mask.')
    digest = sha(path)
    require(all(item['frozen_and_geometry']['native_gradient_audit_sha256']==digest for item in endpoints),
            'R27 endpoints did not preserve the same pre-training native gradient evidence.')
    return descriptor(path)


def checkpoint_evidence(snapshots,config,material):
    from audit_r23_numerical import read_state

    draws,events = None,None
    result = {}
    with np.load(checked(material['arrays']),allow_pickle=False) as original:
        initial_ids = original['stable_ids']
        order = initial_ids.argsort()
        for name,snapshot in snapshots.items():
            state_path = checked(snapshot['checkpoint'])
            state = read_state(state_path)
            step = snapshot['step']
            require(state['step']==step==len(state['draws']) and state['pinned_inputs']==config['pinned_inputs']
                    and all(isinstance(slot,int) and 0<=slot<24 for slot in state['draws']),
                    'R27 optimizer state has a different number of draws or training inputs.')
            if draws is not None:
                require(state['draws'][:len(draws)]==draws and state['split_events'][:len(events)]==events,
                        'R27 saved endpoints do not share the same optimizer trajectory.')
            draws,events = state['draws'],state['split_events']
            ids,roots = (state[key].numpy() for key in ('stable_ids','original_row_ids'))
            slots = order[np.searchsorted(initial_ids[order],roots).clip(0,len(order)-1)]
            require(len(ids)==len(roots)==snapshot['points'] and len(np.unique(ids))==len(ids)
                    and np.array_equal(initial_ids[slots],roots),'R27 checkpoint lost original Gaussian row lineage.')
            for key,source in (('observed','observed'),('labels','labels'),('skin','geometry_eligible'),
                               ('spill','silhouette_spill'),('geometry_only','unknown_unobserved_elongated'),
                               ('possible_protected_material','possible_protected_material')):
                if key=='possible_protected_material' and config['execution']=='cuda_probe' and key not in state:
                    # The completed preliminary probes predate this formal endpoint field.
                    continue
                values,expected = state[key].numpy(),original[source][slots]
                require(np.array_equal(values,expected),f'R27 split changed material or visibility membership: {key}.')
            motion = checked(snapshot['drive']).parent/'motion.npz'
            with np.load(motion,allow_pickle=False) as archive:
                require(all(np.array_equal(archive[key],state[key].numpy()) for key in ('stable_ids','labels','spill')),
                        'R27 delivery motion rows differ from the exact training checkpoint.')
            ref,geometry = (state[key].numpy() for key in ('geometry_reference_ids','geometry_reference'))
            current = np.searchsorted(ref,ids).clip(0,len(ref)-1)
            inherited = np.searchsorted(ref,roots).clip(0,len(ref)-1)
            require(geometry.shape==(len(ref),5) and np.isfinite(geometry).all() and (geometry[:,1:]>0).all()
                    and (ref[1:]>ref[:-1]).all() and np.array_equal(ref[current],ids)
                    and np.array_equal(ref[inherited],roots) and np.array_equal(geometry[current],geometry[inherited]),
                    'R27 split children did not preserve the original local geometry envelope.')
            report = state['frozen_and_geometry']
            require(report['maximum_skin_center_increase_m']<=1e-7 and report['maximum_skin_thickness_increase_m']<=1e-7,
                    'A saved R27 endpoint exceeds the original trusted normal envelope.')
            result[name] = dict(checkpoint=snapshot['checkpoint'],step=step,rows=len(ids),
                inherited_rows=int(np.count_nonzero(ids!=roots)),original_material_and_geometry_inherited=True)
    return result


def plan(subject,result_hash,commit):
    implementation = checked_implementation(commit)
    run = root_for(subject)/'run'
    result_path = pin(run/'gaussian-run/result.json',result_hash)
    terminal = read(result_path)
    progress = run/'progress.json'
    require(terminal['state']=='checkpoint_verified' and not terminal['quality_accepted']
            and terminal['steps']==18000 and read(progress)['state']=='completed'
            and [row['step'] for row in terminal['endpoints']]==list(NEW.values()),
            'Both R27 duration endpoints must be complete.')
    baseline = history(subject)
    frozen,grid = baseline['frozen'],baseline['static_grid']
    config_path = run/'gaussian-run/config.json'
    config = read(config_path)
    previous = read(checked(baseline['r26_training_config']))
    start = previous['pinned_inputs']
    require(all(config['pinned_inputs'][key]==start[key] for key in ('bound_ply','raw_state','fit','aux_replay')),
            'R27 and R26 do not start at the same R22 raw state, fit and auxiliary replay.')
    material = material_evidence(subject,baseline,config)
    pin(checked(material['previous']),config['pinned_inputs']['marked_artifact_audit'])
    landmarks = descriptor(pin(run/'fit/joint-candidates.npz',config['pinned_inputs']['native_auxiliary_landmarks']))
    pin(run/'data/training-inputs.json',frozen['selected']['sha256'])
    pin(run/'fit/fitted.npz',frozen['fit']['sha256'])
    for role,key in (('selected_fit','train_frames'),('holdout','holdout_frames'),('auxiliary_face','auxiliary_frames')):
        require(config[key]==[row['frame'] for row in frozen['rows'] if row['role']==role],
                'R27 changed the registered original pose or camera roster.')
    frequency = config['auxiliary_frequency']
    require(config['execution']=='main' and isinstance(frequency,int) and frequency>0
            and sum(terminal['auxiliary_updates'].values())==18000//frequency
            and set(terminal['auxiliary_updates'])==set(config['auxiliary_frames']),
            'R27 main or auxiliary draw schedule disagrees with its receipts.')
    snapshots = {}
    for name,item in zip(NEW,terminal['endpoints'],strict=True):
        grid['arms'][name],snapshots[name] = rebound(prior.scan_snapshot,PROJECT=PROJECT)(run,item,frozen,grid)
    checkpoints = checkpoint_evidence(snapshots,config,material)
    gradients = gradient_evidence(run,frozen,material,terminal['endpoints'])
    return dict(baseline,schema='shuolee-r27-native-face-review/1',commit=commit,result=descriptor(result_path),
        snapshots=snapshots,training_config=descriptor(config_path),training_progress=descriptor(progress),
        split_events=descriptor(run/'gaussian-run/split-events.json'),implementation=implementation,material=material,
        checkpoints=checkpoints,gradients=gradients,native_landmarks=landmarks,
        script=descriptor(Path(__file__)),quality_accepted=False)


def assemble(path,current):
    output = path.parent
    paired = output/'paired'
    paired.mkdir(exist_ok=True)
    old = read(checked(current['r26_review']))
    static,native,panels = {},{},{}
    for name in ARMS:
        receipt = checked(current['static_archive'][name]) if name in HISTORY else output/'static-grid'/name/'result.json'
        static[name] = dict(receipt=descriptor(receipt),records=qa.grid_records(current,name,receipt))
        if name in HISTORY:
            native[name] = old['native'][name]
        else:
            current_native = read(output/'native'/name/'result.json')
            require((current_native['native_poses'],current_native['auxiliary_face_poses'])==(28,4),
                    'R27 independent replay lost a native pose.')
            native[name] = current_native['records']
        verify_native_rows(native[name],current)
    for key in current['static_grid']['camera_roster']:
        items = [static[name]['records'][key]['panel'] for name in ARMS]
        require(all((item['crop_xyxy'],item['columns'])==(items[0]['crop_xyxy'],items[0]['columns'])
                    for item in items),'R27 changed a fixed camera crop or image columns.')
        panels[key] = qa.stack(paired/f'{key}.png',[checked(item) for item in items],LABELS)
    for row in current['frozen']['rows']:
        frame = row['frame']
        for part in ('face',*prior.FACE_LANDMARKS):
            items = [native[name][frame]['regions'][part] for name in ARMS]
            require(all((item['box'],item.get('foreground_pixels'))==(items[0]['box'],items[0].get('foreground_pixels'))
                        for item in items),'R27 paired face frames do not use the original crop and mask.')
            if items[0]['box'] is not None:
                key = f'native_{frame}_{part}'
                panels[key] = dict(qa.stack(paired/f'{key}.png',[checked(item['panel']) for item in items],LABELS),
                                   crop_xyxy=items[0]['box'],frame=frame,role=row['role'],region=part)
    result = dict(state='paired_native_face_and_static_views_rendered',subject=current['subject'],commit=current['commit'],
        static=static,static_camera_count=50,native=native,native_poses=28,auxiliary_face_poses=4,panels=panels,
        native_scores={name:prior.score_summary(native[name]) for name in ARMS},plan=descriptor(path),quality_accepted=False)
    target = output/'result.json'
    if target.exists():
        require(read(target)==result,'Preserve the completed R27 review.')
    else:
        qa.review.save(target,result)
    print(json.dumps(descriptor(target)),flush=True)


def main():
    if len(sys.argv)==5 and sys.argv[1]=='_worker':
        rebound(prior.worker,NEW=NEW,ARMS=ARMS)(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check-history','check','render','assemble'))
    parser.add_argument('subject',choices=qa.prior.SUBJECTS)
    parser.add_argument('--result-sha')
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action=='check-history':
        baseline = history(args.subject)
        evidence = material_evidence(args.subject,baseline)
        print(json.dumps(dict(subject=args.subject,history=list(HISTORY),cameras=len(baseline['static_grid']['camera_roster']),
                              native_cameras=len(baseline['frozen']['rows']),material=evidence)),flush=True)
        return
    if not args.result_sha or not args.commit:
        parser.error('Use the exact R27 production result and full commit.')
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
    require(read(path)==current,'The saved R27 independent review plan changed.')
    if args.action=='render':
        rebound(prior.render,ARMS=NEW,__file__=__file__)(path)
        print(json.dumps(dict(subject=args.subject,state='cuda_review_released_gpu',plan=descriptor(path))),flush=True)
    assemble(path,current)


if __name__=='__main__':
    main()
