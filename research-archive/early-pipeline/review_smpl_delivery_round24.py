"""Independently review R24 graph-driven garments and split coverage at fixed cameras."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round23 as prior


qa = prior.qa
pin,checked,read,sha,require,descriptor,rebound = prior.pin,prior.checked,prior.read,prior.sha,prior.require,prior.descriptor,prior.rebound
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-clothing-surface-r24')
PREVIOUS_SHA = '00529274bf06fd801cead690901b20aebcdcdb4f795a29f016b49943df85426e'
DELIVERY = qa.launch.DELIVERY.with_name('smpl-delivery-round24')
BASE = ('r22','r23_10','r23')
NEW = {'r24_10':10000,'r24':18000}
ARMS = (*BASE,*NEW)
LABELS = ('R22 18,000','R23 10,000','R23 18,000','R24 10,000','R24 18,000')
GRAPH_SCHEMA = 'shuolee-drivable-clothing/2'
GRAPH_FILE = 'garment-graph.npz'
GRAPH_FIELDS = {'node_rest_xyz','node_centers','node_owner_faces','node_normal_neighbors',
                'graph_edges','row_node_ids','row_node_weights','face_groups'}


def root_for(subject):
    return qa.review.LAB/f'shuolee-smpl-delivery-round24-{subject}'


def history(subject):
    index = pin(DELIVERY.with_name('smpl-delivery-round23')/'review-receipt.json',PREVIOUS_SHA)
    released = read(index)['subjects'][subject]
    previous = checked(released['review'])
    result = read(previous)
    old = read(checked(result['plan']))
    require(result['state']=='paired_driven_and_static_views_rendered'
            and result['commit']==old['commit']==prior.COMMIT
            and result['static_camera_count']==50 and result['native_poses']==28
            and result['subject']==old['subject']==subject,
            'The signed R23 native pose and fixed-camera baseline is incomplete.')
    older = read(checked(old['r22_review']))
    r22_plan = read(checked(older['plan']))
    require(old['static_grid']['camera_roster']==r22_plan['static_grid']['camera_roster']
            and old['static_grid']['proof']['camera_manifest']==r22_plan['static_grid']['proof']['camera_manifest'],
            'R23 changed the R22 world camera roster.')
    frozen = old['frozen']
    for item in (frozen['fit'],frozen['selected_mesh'],frozen['heldout_mesh']):
        checked(item)
    for row in frozen['rows']:
        pin(row['image'],row['source_sha256'])
        pin(row['mask'],row['mask_sha256'])
    grid = dict(old['static_grid'],arms={name:old['static_grid']['arms'][name] for name in BASE})
    receipts = {}
    for name in BASE:
        arm = grid['arms'][name]
        pin(arm['ply'],arm['ply_sha256'])
        pin(arm['camera_path'],arm['camera_sha256'])
        entry = result['static'][name]['receipt']
        require(qa.grid_records(old,name,checked(entry))==result['static'][name]['records'],
                'An archived R22 or R23 fixed-view renderer result differs.')
        receipts[name] = entry
        for row in frozen['rows']:
            native = result['native'][name][row['frame']]
            require(all(native['camera'][key]==row[key] for key in ('frame','K','wh','w2c'))
                    and native['source_sha256']==row['source_sha256'] and native['mask_sha256']==row['mask_sha256'],
                    'An archived native comparator changed the source identity or projection.')
            if native['roi'].get('panel'):
                checked(native['roi']['panel'])
    return dict(frozen=frozen,rois=old['rois'],asset=old['asset'],static_grid=grid,static_archive=receipts,
                r23_review=descriptor(previous),r23_archive=released['archive'],r22_review=old['r22_review'],
                r23_index=descriptor(index))


def graph_sidecar(drive):
    path = checked(drive['drive'])
    manifest = read(path)
    require(manifest['schema']==GRAPH_SCHEMA and manifest['garment_graph']==GRAPH_FILE
            and manifest['artifact_role']=='inference_replay' and manifest['optimizer_state_included'] is False
            and GRAPH_FILE in manifest['files_sha256'],
            'R24 delivery requires the exact learned cloth graph archive.')
    graph = pin(path.parent/GRAPH_FILE,manifest['files_sha256'][GRAPH_FILE])
    with np.load(graph,allow_pickle=False) as fields:
        require(set(fields.files)==GRAPH_FIELDS,'The learned cloth graph archive has a different field layout.')
        arrays = {}
        for key in fields.files:
            data = fields[key]
            require(data.dtype.kind in 'biufUS' and (data.dtype.kind not in 'f' or np.isfinite(data).all()),
                    f'Invalid graph array {key}.')
            arrays[key] = dict(shape=list(data.shape),dtype=str(data.dtype))
        delta = np.linalg.norm(fields['node_centers']-fields['node_rest_xyz'],axis=1)
        require(fields['node_centers'].shape==fields['node_rest_xyz'].shape and len(delta)>0
                and len(fields['row_node_ids'])==drive['clothing_points']
                and np.allclose(fields['row_node_weights'].sum(axis=1),1,rtol=0,atol=1e-6),
                'The learned garment graph has inconsistent node or Gaussian rows.')
        counts = dict(nodes=len(delta),edges=len(fields['graph_edges']),
                      displaced_nodes=int(np.sum(delta>1e-6)),
                      displacement_m_p50_p95_max=np.quantile(delta,[.5,.95,1.]).tolist())
    return dict(manifest=descriptor(path),parameters=descriptor(graph),arrays=arrays,learned_displacement=counts)


def implementation(config):
    reported = config['implementation_sha256']
    require(set(prior.IMPLEMENTATION)<=set(reported),'Training did not identify its cloth, drive and support implementation.')
    result = {}
    for name,digest in reported.items():
        if '/' in name:
            path = PROJECT/name
        elif name in prior.IMPLEMENTATION:
            path = PROJECT/prior.IMPLEMENTATION[name]
        else:
            path, = PROJECT.joinpath('src').rglob(name)
        require(path.resolve().is_relative_to(PROJECT.resolve()),'R24 implementation path escapes its checkout.')
        result[name] = descriptor(pin(path,digest))
    return result


def split_checks(path,config,frames,scheduled,*,allow_preview=False):
    events = read(path)
    limits = config['split_guard']['limits']
    require([event['step'] for event in events]==list(scheduled)
            and config['split_guard']['roster']=='all selected training views',
            'The split coverage audit uses a different stage schedule or roster.')
    summary = []
    for event in events:
        require(event['train_cameras']==len(frames)==24,'A split event did not use the full selected camera roster.')
        approved = []
        for trial in event['trials']:
            require([view['frame'] for view in trial['views']]==frames,'A split preview changed a native camera or order.')
            for view in trial['views']:
                values = [view[key] for key in limits]
                passed = all(np.isfinite(values)) and all(view[key]<=limit for key,limit in limits.items())
                require(view['passed']==passed,'A split trial disagrees with its numeric gate.')
            require(trial['passed']==all(view['passed'] for view in trial['views']),
                    'A split trial did not apply every selected camera gate.')
            if trial['passed']:
                approved.append(trial)
        committed = bool(event.get('committed'))
        require((committed==bool(approved) or (allow_preview and not committed)) and len(approved)<=1,
                'A split was committed without one complete passing preview.')
        parents = event['parents']
        require(event['children']==2*parents and (parents>0)==bool(approved),'A rejected split changed topology.')
        if approved:
            require(approved[0] is event['trials'][-1] and approved[0]['proposed_parents']==parents
                    and approved[0]['shift_in_sigma']==event['shift_in_sigma'] and len(set(event['parent_ids']))==parents
                    and isinstance(event['first_child_id'],int),'The committed split differs from the native preview.')
        summary.append(dict(step=event['step'],trials=len(event['trials']),approved=bool(approved),committed=committed,
                            proposed_passing_parents=parents,parents=parents if committed else 0,
                            children=event['children'] if committed else 0,native_cameras=event['train_cameras']))
    return summary


def checkpoint_checks(root,item,graph,events,config):
    from audit_r23_numerical import read_state

    path = root/'main-state.pt'
    state = read_state(path)
    step = item['step']
    draws = state['draws']
    learned = state['surface_graph']
    optimizer = state['surface_optimizer']
    require(state['step']==step==len(draws) and state['split_events']==events
            and state['source_sha256']==config['source_bound_sha256']
            and all(isinstance(slot,int) and 0<=slot<24 for slot in draws),
            'A raw checkpoint changed the original training-only draw sequence or split events.')
    require(len(optimizer['state'])==1 and all(int(fields['step'])==step for fields in optimizer['state'].values()),
            'The shared garment graph optimizer did not take the recorded number of updates.')
    require(len(state['stable_ids'])==len(state['support_face_ids'])==item['free_garment_points']
            and len(set(state['stable_ids'].tolist()))==item['free_garment_points'],
            'The garment checkpoint contains invalid stable identities.')
    with np.load(checked(graph['parameters']),allow_pickle=False) as archive:
        require(set(learned)==set(archive.files) and all(np.array_equal(value.numpy(),archive[name])
                and value.numpy().dtype==archive[name].dtype for name,value in learned.items()),
                'The inference graph differs from the raw trained graph tensors.')
    displaced = graph['learned_displacement']
    require(displaced['nodes']==config['surface_nodes']
            and displaced['displacement_m_p50_p95_max'][-1]<=config['surface_max_displacement_m']+1e-6,
            'The learned garment graph changed its node budget or exceeded the configured displacement.')
    return dict(checkpoint=descriptor(path),draws=draws,graph_optimizer_steps=step,
                inference_graph_exact_checkpoint=True,fixed_identity=state['fixed_identity'])


def plan(subject, expected, commit):
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Pass the full frozen R24 commit.')
    run = root_for(subject)/'run'
    terminal_path = pin(run/'result.json',expected)
    terminal = read(terminal_path)
    require(terminal['state']=='completed' and terminal['total_optimizer_updates']==18000
            and terminal['input_pose_updates']==0 and not terminal['quality_accepted']
            and [row['step'] for row in terminal['endpoints']]==list(NEW.values()),
            'R24 did not complete both same-run duration endpoints.')
    baseline = history(subject)
    frozen = baseline['frozen']
    grid = baseline['static_grid']
    config_path = run/'gaussian-run/config.json'
    config = read(config_path)
    require(Path(config['baseline_run'])==qa.launch.root_for(subject)/'run'
            and config['baseline_step']==18000,
            'R24 did not start from the same exact R22 baseline as R23.')
    pointer = (PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')
    git_dir = subprocess.check_output(['/usr/bin/wslpath','-u',pointer],text=True).strip()
    current_commit = subprocess.check_output(
        ['/usr/bin/git','--git-dir',git_dir,'--work-tree',str(PROJECT),'rev-parse','HEAD'],text=True).strip()
    require(current_commit==commit,'Use the actual frozen R24 source checkout.')
    code = implementation(config)
    splits = run/'gaussian-run/garment-splits.json'
    frames = [row['frame'] for row in frozen['rows'][:24]]
    require(config['training_frames']==frames and config['heldout_frames']==[row['frame'] for row in frozen['rows'][24:]],
            'The R24 training config changed its selected and held-out frame identities.')
    split_summary = split_checks(splits,config,frames,config['splits'])
    events = read(splits)
    drives,graphs,states = {},{},{}
    scan = rebound(prior.scan_drive,PROJECT=PROJECT)
    for (name,step),item in zip(NEW.items(),terminal['endpoints'],strict=True):
        require(item['path']==f'raw-step-{step:06d}','An R24 export uses an unexpected raw checkpoint.')
        root = run/'gaussian-run'/item['path']
        grid['arms'][name],drives[name] = scan(root,item,frozen,grid)
        graphs[name] = graph_sidecar(drives[name])
        states[name] = checkpoint_checks(root,item,graphs[name],events,config)
        require(item['free_garment_points']==config['initial_free_garment_points']+
                sum(row['parents'] for row in split_summary),'Only committed two-child splits may change the row count.')
    require(drives['r24_10']['reference_fixed_rows']['sha256']==drives['r24']['reference_fixed_rows']['sha256'],
            'The two R24 durations changed the protected non-clothing PLY parameters.')
    early,late = (states[name] for name in NEW)
    require(early['draws']==late['draws'][:10000] and early['fixed_identity']==late['fixed_identity'],
            'The 10k snapshot is not a prefix of the same 18k run.')
    for state in states.values():
        del state['draws']
    loss = run/'gaussian-run/loss.jsonl'
    rows = [json.loads(line) for line in loss.read_text(encoding='utf-8').splitlines()]
    require([row['step'] for row in rows]==[1,*range(100,18001,100)]
            and all(row['state']=='training' for row in rows),'R24 main training log is incomplete.')
    return dict(baseline,schema='shuolee-r24-independent-review/1',subject=subject,commit=commit,drives=drives,graphs=graphs,
                result=descriptor(terminal_path),script=descriptor(Path(__file__)),training_log=descriptor(loss),
                training_config=descriptor(config_path),implementation=code,split_audit=descriptor(splits),
                split_summary=split_summary,checkpoints=states,quality_accepted=False)


def worker(plan_path,name,fd):
    current = read(plan_path)
    if not (plan_path.parent/'native'/name/'result.json').is_file():
        rebound(prior.worker,PROJECT=PROJECT,NEW=NEW,STATIC_ARMS=ARMS)(plan_path,name,fd)
    verify_graph_runtime(plan_path.parent,current,name,fd)


def graph_effects(cloth,frames,reference):
    import torch
    from shuolee_portrait.free_clothing_torch import normalized_quaternion_matrix

    learned = cloth.node_centers.clone()
    rest = cloth.node_rest_xyz
    delta = learned-rest
    results = []
    for slot,frame in enumerate(frames):
        cloth.select_mesh_by_timestep(slot)
        before = cloth.get_xyz
        cov = cloth.get_covariance()
        scale = cloth.get_scaling
        matrix = normalized_quaternion_matrix(cloth.get_rotation)
        shaped = matrix*scale[:,None,:]
        manual = shaped@shaped.transpose(-1,-2)
        manual = manual[:,(0,0,0,1,1,2),(0,1,2,1,2,2)]
        packed_error = float((manual-cov).abs().max())
        if slot==reference:
            transported = delta
        else:
            slots = cloth._node_face_slots
            node_rotation = normalized_quaternion_matrix(cloth._node_motion_quat[slot,slots])
            node_scale = cloth._node_motion_scale[slot,slots]
            transported = torch.einsum('nij,nj->ni',node_rotation,delta)*node_scale[:,None]
        expected = (transported[cloth.row_node_ids]*cloth.row_node_weights[...,None]).sum(dim=1)
        cloth.node_centers.copy_(rest)
        cloth.select_mesh_by_timestep(slot)
        neutral = cloth.get_xyz
        require(torch.equal(cov,cloth.get_covariance()),'Changing only graph centers unexpectedly changed covariance.')
        cloth.node_centers.copy_(learned)
        cloth.select_mesh_by_timestep(slot)
        require(torch.equal(before,cloth.get_xyz),'The saved graph was not restored after the in-memory diagnostic.')
        difference = before-neutral
        error = float((difference-expected).abs().max())
        require(error<=4e-6 and packed_error<=2e-8,'Runtime graph geometry or packed covariance disagrees with its inputs.')
        lengths = difference.norm(dim=1)
        results.append(dict(frame=frame,pose_index=slot,role='selected_fit' if slot<24 else 'holdout',
            graph_displaced_rows=int((lengths>1e-6).sum()),
            graph_center_effect_m_p50_p95_max=lengths.quantile(lengths.new_tensor([.5,.95,1.])).cpu().tolist(),
            graph_transport_max_abs_m=error,packed_covariance_max_abs_m2=packed_error))
    return results


def verify_graph_runtime(out,current,name,fd):
    """Change only node positions in the loaded in-memory package and measure its exact effect."""
    require(os.path.samestat(os.fstat(fd),qa.review.GPU_LEASE.stat()),'Missing inherited GPU lease.')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(PROJECT/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.drivable_clothing_bundle import load_drive_bundle

    rt.configure(Path(current['asset']).parents[1])
    planned = current['graphs'][name]
    require(graph_sidecar(current['drives'][name])==planned,'Saved garment graph parameters changed.')
    directory = checked(planned['manifest']).parent
    with gpu_lock(),torch.no_grad(),np.load(checked(planned['parameters']),allow_pickle=False) as frozen:
        model,meta = load_drive_bundle(directory)
        cloth = model.cloth
        restored = cloth.graph_arrays()
        require(meta['schema']==GRAPH_SCHEMA and meta['frame_ids']==[row['frame'] for row in current['frozen']['rows']]
                and set(restored)==set(frozen.files) and all(np.array_equal(restored[key],frozen[key])
                    and restored[key].dtype==frozen[key].dtype for key in frozen.files)
                and cloth.optimizer is None and not cloth.node_centers.requires_grad,
                'Independent replay did not restore the exact inference graph and original poses.')
        results = graph_effects(cloth,meta['frame_ids'],meta['reference_index'])
    qa.review.save(out/'native'/name/'graph-replay.json',dict(graph=planned,records=results,
        camera_and_geometry_verified=True,diagnostic_scope='Temporary neutral node positions with the same saved garment graph topology.'))


def render(path,current):
    out = path.parent
    with qa.review.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in NEW:
            receipt = out/'native'/name/'result.json'
            if receipt.exists() and receipt.with_name('graph-replay.json').exists():
                continue
            with (out/f'{name}.log').open('a',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I','-B',__file__,'_worker',str(path),name,str(lease.fileno())],
                    pass_fds=(lease.fileno(),),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)


def metrics(grid,native,roster):
    close = [key for key,row in roster.items() if row['distance_class']=='close' and row['pitch_deg']]
    require(len(close)==20,'The R24 review lost a registered pitch camera.')
    result = {}
    for name in ARMS:
        stats = {}
        for role in ('selected_fit','holdout'):
            rows = [item for item in native[name].values() if item['role']==role]
            values = [item['roi'] for item in rows if item['roi'].get('foreground_rgb_mae') is not None]
            fg = sum(item['foreground_pixels'] for item in values)
            bg = sum(item['exterior_pixels'] for item in values)
            holes = sum(item['foreground_half_alpha_holes'] for item in values)
            spill = sum(item['exterior_half_alpha_spill'] for item in values)
            require(fg>0 and bg>0,'A paired native group has no reliable ROI.')
            stats[role] = dict(frames=len(rows),visible_torso_frames=len(values),
                mean_torso_rgb_mae=float(np.mean([item['foreground_rgb_mae'] for item in values])),
                torso_fg_samples=fg,torso_bg_samples=bg,torso_fg_half_alpha_holes=holes,
                torso_bg_half_alpha_spill=spill,torso_fg_half_alpha_fraction=holes/fg,
                torso_bg_half_alpha_fraction=spill/bg,mean_native_foreground_below_095=float(np.mean(
                    [item['alpha_regression']['foreground_below_095_fraction'] for item in rows])))
        rows = [grid[name]['records'][key] for key in close]
        inside = [row['opacity']['visible_face_proxy']['low_alpha_095_fraction'] for row in rows]
        rings = [row['anatomical_exterior_ring'] for row in rows]
        stats['close_pitch'] = dict(views=20,mean_low_alpha_on_visible_face_proxy=float(np.mean(
            [value for value in inside if value is not None])),
            mean_anatomical_exterior_ring_alpha=float(np.mean([r['alpha_mass']/r['pixels'] for r in rings])))
        result[name] = stats
    return result


def assemble(path,current):
    out = path.parent
    historical = read(checked(current['r23_review']))
    paired = out/'paired'
    paired.mkdir(exist_ok=True)
    static,native,panels = {},{},{}
    for name in ARMS:
        source = checked(current['static_archive'][name]) if name in BASE else out/'static-grid'/name/'result.json'
        static[name] = dict(receipt=descriptor(source),records=qa.grid_records(current,name,source))
        native[name] = historical['native'][name] if name in BASE else read(out/'native'/name/'result.json')['records']
        require(set(native[name])=={row['frame'] for row in current['frozen']['rows']},'A native branch missed a pose.')
    for key in current['static_grid']['camera_roster']:
        items = [static[name]['records'][key]['panel'] for name in ARMS]
        require(all((item['crop_xyxy'],item['columns'])==(items[0]['crop_xyxy'],items[0]['columns'])
                    for item in items),'An R24 fixed-camera image uses a different crop or layout.')
        panels[key] = qa.stack(paired/f'{key}.png',[checked(item) for item in items],LABELS)
    for row in current['frozen']['rows']:
        key = row['frame']
        items = [native[name][key] for name in ARMS]
        require(all(item['camera']==items[0]['camera'] and item['role']==items[0]['role']
                    and item['source_sha256']==row['source_sha256'] and item['mask_sha256']==row['mask_sha256']
                    and item['roi']['box']==current['rois'][key]['box'] for item in items),
                'R24 changed a paired original source, projection or garment ROI.')
        if items[0]['roi']['box'] is not None:
            score_keys = ('foreground_pixels','exterior_pixels')
            require(all(all(item['roi'][field]==items[0]['roi'][field] for field in score_keys)
                        for item in items),'R24 changed the paired native ROI mask.')
            name = f'native_torso_{key}'
            panels[name] = qa.stack(paired/f'{name}.png',[checked(item['roi']['panel']) for item in items],LABELS)
    result = dict(state='paired_driven_and_static_views_rendered',subject=current['subject'],commit=current['commit'],
        plan=descriptor(path),static=static,static_camera_count=50,native=native,native_poses=28,panels=panels,
        paired_native_metrics=metrics(static,native,current['static_grid']['camera_roster']),graphs=current['graphs'],
        graph_replay={name:descriptor(out/'native'/name/'graph-replay.json') for name in NEW},
        quality_accepted=False)
    target = out/'result.json'
    if target.exists():
        require(read(target)==result,'Preserve the completed R24 review.')
    else:
        qa.review.save(target,result)
    print(json.dumps(descriptor(target)),flush=True)


def main():
    if len(sys.argv)==5 and sys.argv[1]=='_worker':
        worker(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check-history','check','render','assemble'))
    parser.add_argument('subject',choices=qa.prior.SUBJECTS)
    parser.add_argument('--result-sha')
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action=='check-history':
        frozen = history(args.subject)
        print(json.dumps(dict(subject=args.subject,static_cameras=len(frozen['static_grid']['camera_roster']),
                            native_poses=len(frozen['frozen']['rows']),retained_paired_models=list(BASE))),flush=True)
        return
    if not args.result_sha or not args.commit:
        parser.error('Use the actual full R24 result hash and commit.')
    current = plan(args.subject,args.result_sha,args.commit)
    if args.action=='check':
        print(json.dumps(dict(subject=args.subject,static_cameras=50,native_poses=28,
                             graphs={name:row['arrays'] for name,row in current['graphs'].items()})),flush=True)
        return
    out = root_for(args.subject)/'view-comparison'
    out.mkdir(exist_ok=True)
    path = out/'plan.json'
    if not path.exists():
        qa.review.save(path,current)
    require(read(path)==current,'The saved R24 independent review plan differs.')
    if args.action=='render':
        render(path,current)
    assemble(path,current)


if __name__=='__main__':
    main()
