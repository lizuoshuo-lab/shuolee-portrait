"""Review actual R25 anchored garments against the fixed R24 cameras and pose inputs."""

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
import review_smpl_delivery_round24 as prior


qa = prior.qa
pin,checked,read,sha,require,descriptor,rebound = prior.pin,prior.checked,prior.read,prior.sha,prior.require,prior.descriptor,prior.rebound
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-clothing-surface-r25')
PREVIOUS_SHA = 'd34eff119bba53fe8ea08d45d5e16cdfa1c4271f5bfb62a4b67cf97f3b85d4e9'
DELIVERY = prior.DELIVERY.with_name('smpl-delivery-round25')
BASE = ('r22','r24_10','r24')
NEW = {'r25_10':10000,'r25':18000}
ARMS = (*BASE,*NEW)
LABELS = ('R22 18,000','R24 10,000','R24 18,000','R25 10,000','R25 18,000')
SCHEMA = 'shuolee-drivable-clothing/3'
ANCHOR_FILE = 'garment-anchors.npz'


def root_for(subject):
    return qa.review.LAB/f'shuolee-smpl-delivery-round25-{subject}'


def history(subject):
    index = pin(prior.DELIVERY/'review-receipt.json',PREVIOUS_SHA)
    released = read(index)['subjects'][subject]
    baseline = checked(released['review'])
    result = read(baseline)
    plan = read(checked(result['plan']))
    require(result['state']=='paired_driven_and_static_views_rendered' and result['commit']==plan['commit']
            and result['subject']==plan['subject']==subject and result['static_camera_count']==50
            and result['native_poses']==28,'The R24 paired comparison is incomplete.')
    grid = dict(plan['static_grid'],arms={name:plan['static_grid']['arms'][name] for name in BASE})
    for entry in grid['proof'].values():
        checked(entry)
    frozen = plan['frozen']
    for item in (frozen['fit'],frozen['selected_mesh'],frozen['heldout_mesh']):
        checked(item)
    for row in frozen['rows']:
        pin(row['image'],row['source_sha256'])
        pin(row['mask'],row['mask_sha256'])
    receipts = {}
    for name in BASE:
        arm = grid['arms'][name]
        pin(arm['ply'],arm['ply_sha256'])
        pin(arm['camera_path'],arm['camera_sha256'])
        entry = result['static'][name]['receipt']
        require(qa.grid_records(plan,name,checked(entry))==result['static'][name]['records'],
                'An archived R24 or R22 render differs from the fixed-camera receipt.')
        receipts[name] = entry
        for row in frozen['rows']:
            native = result['native'][name][row['frame']]
            require(all(native['camera'][key]==row[key] for key in ('frame','K','wh','w2c'))
                    and native['source_sha256']==row['source_sha256'] and native['mask_sha256']==row['mask_sha256'],
                    'An archived native comparator changed its original source, mask or projection.')
            if native['roi'].get('panel'):
                checked(native['roi']['panel'])
    return dict(frozen=frozen,rois=plan['rois'],asset=plan['asset'],static_grid=grid,static_archive=receipts,
                baseline_review=descriptor(baseline),baseline_archive=released['archive'],
                baseline_index=descriptor(index))


def sidecars(drive):
    graph = rebound(prior.graph_sidecar,GRAPH_SCHEMA=SCHEMA)(drive)
    path = checked(drive['drive'])
    manifest = read(path)
    require(manifest['garment_anchors']==ANCHOR_FILE,'The native R25 manifest does not register its anchors.')
    anchors = pin(path.parent/ANCHOR_FILE,manifest['files_sha256'][ANCHOR_FILE])
    rows = drive['clothing_points']
    nodes = graph['learned_displacement']['nodes']
    expected = {'anchor_reference_normal':((rows,3),np.float32),'anchor_trusted':((rows,),np.bool_),
                'anchor_normal_raw':((rows,1),np.float32),'anchor_tangent_raw':((nodes,3),np.float32)}
    with np.load(anchors,allow_pickle=False) as archive:
        require(set(archive.files)==set(expected) and all(archive[key].shape==shape and archive[key].dtype==dtype
                and np.isfinite(archive[key]).all() for key,(shape,dtype) in expected.items()),
                'R25 anchor arrays do not match the native graph, Gaussian rows or dtype.')
        trust = archive['anchor_trusted']
        return dict(graph,anchors=descriptor(anchors),trusted_rows=int(trust.sum()),clothing_rows=rows)


def checkpoint(root,item,graph,events,config):
    from audit_r23_numerical import read_state

    path = root/'main-state.pt'
    state = read_state(path)
    step = item['step']
    draws = state['draws']
    require(step==state['step']==len(draws) and state['split_events']==events
            and state['source_sha256']==config['source_bound_sha256']
            and all(isinstance(slot,int) and 0<=slot<24 for slot in draws),
            'The saved R25 main checkpoint differs from the selected-only draw or split history.')
    learned = state['surface_optimizer']['state']
    require(len(learned)==2 and all(int(fields['step'])==step for fields in learned.values()),
            'The shared node and tangent parameters do not match the recorded number of graph optimizer updates.')
    groups = state['surface_optimizer']['param_groups']
    require([group['name'] for group in groups]==['garment_nodes','bounded_anchor_tangent'],
            'The R25 checkpoint is missing named node or bounded tangent optimizer state.')
    gaussian = state['garment'][12]
    normal = [group for group in gaussian['param_groups'] if group['name']=='anchor_normal']
    require(len(normal)==1 and len(normal[0]['params'])==1,'The per-row normal residual optimizer is missing.')
    normal_moments = gaussian['state'][normal[0]['params'][0]]
    require(int(normal_moments['step'])==step and all(tuple(normal_moments[key].shape)==(graph['clothing_rows'],1)
            for key in ('exp_avg','exp_avg_sq')),'The trained normal moments no longer follow the garment rows.')
    require(len(state['stable_ids'])==len(state['support_face_ids'])==item['free_garment_points']
            and len(set(state['stable_ids'].tolist()))==item['free_garment_points'],
            'R25 stable clothing row identities differ.')
    for state_field,receipt in (('surface_graph',graph['parameters']),('surface_anchors',graph['anchors'])):
        values = state[state_field]
        with np.load(checked(receipt),allow_pickle=False) as archive:
            require(set(values)==set(archive.files) and all(np.array_equal(value.numpy(),archive[key])
                    and value.numpy().dtype==archive[key].dtype for key,value in values.items()),
                    'R25 inference graph or anchors differ from the actual trained checkpoint.')
    learned_graph = graph['learned_displacement']
    require(learned_graph['nodes']==config['surface_nodes']
            and learned_graph['displacement_m_p50_p95_max'][-1]<=config['surface_max_displacement_m']+1e-6,
            'The learned R25 graph changed its original node count or exceeded its displacement cap.')
    return dict(checkpoint=descriptor(path),draws=draws,fixed_identity=state['fixed_identity'],
                graph_updates=step,normal_updates=step,anchors_and_graph_match_checkpoint=True)


def plan(subject,expected,commit):
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Use the full frozen R25 commit.')
    pointer = (PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')
    git_dir = subprocess.check_output(['/usr/bin/wslpath','-u',pointer],text=True).strip()
    current = subprocess.check_output(
        ['/usr/bin/git','--git-dir',git_dir,'--work-tree',str(PROJECT),'rev-parse','HEAD'],text=True).strip()
    require(current==commit,'Use the exact R25 checkout.')
    run = root_for(subject)/'run'
    terminal_path = pin(run/'result.json',expected)
    terminal = read(terminal_path)
    require(terminal['state']=='completed' and terminal['total_optimizer_updates']==18000
            and terminal['input_pose_updates']==0 and not terminal['quality_accepted']
            and [row['step'] for row in terminal['endpoints']]==list(NEW.values()),
            'R25 requires both verified same-run duration endpoints.')
    baseline = history(subject)
    frozen,grid = baseline['frozen'],baseline['static_grid']
    config_path = run/'gaussian-run/config.json'
    config = read(config_path)
    frames = [row['frame'] for row in frozen['rows']]
    require(Path(config['baseline_run'])==qa.launch.root_for(subject)/'run' and config['baseline_step']==18000
            and config['training_frames']==frames[:24] and config['heldout_frames']==frames[24:],
            'R25 changed the R22 initial state, selected frames or held-out source poses.')
    code = rebound(prior.implementation,PROJECT=PROJECT)(config)
    splits = run/'gaussian-run/garment-splits.json'
    events = read(splits)
    split_summary = prior.split_checks(splits,config,frames[:24],config['splits'])
    scan = rebound(prior.prior.scan_drive,PROJECT=PROJECT)
    drives,graphs,states = {},{},{}
    for (name,step),item in zip(NEW.items(),terminal['endpoints'],strict=True):
        require(item['path']==f'raw-step-{step:06d}','Unexpected R25 raw checkpoint path.')
        root = run/'gaussian-run'/item['path']
        grid['arms'][name],drives[name] = scan(root,item,frozen,grid)
        graphs[name] = sidecars(drives[name])
        states[name] = checkpoint(root,item,graphs[name],events,config)
        require(item['free_garment_points']==config['initial_free_garment_points']+
                sum(row['parents'] for row in split_summary),'A rejected split or other operation changed the row count.')
    early,late = (states[name] for name in NEW)
    require(early['draws']==late['draws'][:10000] and early['fixed_identity']==late['fixed_identity']
            and drives['r25_10']['reference_fixed_rows']['sha256']==drives['r25']['reference_fixed_rows']['sha256'],
            'The two R25 exports changed the protected geometry or main training sequence.')
    for state in states.values():
        del state['draws']
    loss = run/'gaussian-run/loss.jsonl'
    rows = [json.loads(line) for line in loss.read_text(encoding='utf-8').splitlines()]
    require([row['step'] for row in rows]==[1,*range(100,18001,100)]
            and all(row['state']=='training' for row in rows),'The R25 main training log is incomplete.')
    return dict(baseline,schema='shuolee-r25-independent-review/1',subject=subject,commit=commit,drives=drives,graphs=graphs,
        result=descriptor(terminal_path),script=descriptor(Path(__file__)),training_log=descriptor(loss),
        training_config=descriptor(config_path),implementation=code,split_audit=descriptor(splits),split_summary=split_summary,
        checkpoints=states,quality_accepted=False)


def anchor_replay(out,current,name,fd):
    """Inspect a temporary zero-residual copy of the loaded V3 inference model."""
    require(os.path.samestat(os.fstat(fd),qa.review.GPU_LEASE.stat()),'Missing inherited GPU lease.')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(PROJECT/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.drivable_clothing_bundle import load_drive_bundle
    from shuolee_portrait.free_clothing_torch import normalized_quaternion_matrix

    rt.configure(Path(current['asset']).parents[1])
    graph = current['graphs'][name]
    require(sidecars(current['drives'][name])==graph,'The exact V3 graph and anchor sidecars changed.')
    with gpu_lock(),torch.no_grad(),np.load(checked(graph['anchors']),allow_pickle=False) as expected:
        model,info = load_drive_bundle(checked(graph['manifest']).parent)
        cloth = model.cloth
        actual = cloth.anchor_arrays()
        require(info['schema']==SCHEMA and info['frame_ids']==[row['frame'] for row in current['frozen']['rows']]
                and all(np.array_equal(actual[key],expected[key]) and actual[key].dtype==expected[key].dtype
                        for key in expected.files)
                and not any(value.requires_grad for value in
                            (cloth.node_centers,cloth.anchor_normal_raw,cloth.anchor_tangent_raw)),
                'R25 runtime did not restore the exact frozen inference anchors and native poses.')
        normal = cloth.anchor_normal_raw.clone()
        tangent = cloth.anchor_tangent_raw.clone()
        trusted = cloth.anchor_trusted
        rows = []
        for slot,frame in enumerate(info['frame_ids']):
            cloth.select_mesh_by_timestep(slot)
            live = cloth.get_xyz
            cov = cloth.get_covariance()
            A = normalized_quaternion_matrix(cloth.get_rotation)*cloth.get_scaling[:,None,:]
            manual = A@A.transpose(-1,-2)
            packed = manual[:,(0,0,0,1,1,2),(0,1,2,1,2,2)]
            covariance_error = float((packed-cov).abs().max())
            cloth.anchor_normal_raw.zero_()
            cloth.select_mesh_by_timestep(slot)
            without_normal = cloth.get_xyz
            cloth.anchor_tangent_raw.zero_()
            cloth.select_mesh_by_timestep(slot)
            neutral = cloth.get_xyz
            cloth.anchor_normal_raw.copy_(normal)
            cloth.anchor_tangent_raw.copy_(tangent)
            cloth.select_mesh_by_timestep(slot)
            require(torch.equal(live,cloth.get_xyz) and torch.equal(cov,cloth.get_covariance())
                    and covariance_error<=2e-8,'R25 anchor ablation did not restore exact runtime geometry.')
            effect = (live-neutral).norm(dim=1)
            normal_effect = (live-without_normal).norm(dim=1)
            require(not bool((effect[~trusted]>1e-6).any()),'An untrusted row received a learned anchor residual.')
            bounds = effect.new_tensor([.5,.95,1.])
            rows.append(dict(frame=frame,pose_index=slot,role='selected_fit' if slot<24 else 'holdout',
                anchor_effect_m_p50_p95_max=effect.quantile(bounds).cpu().tolist(),
                normal_effect_m_p50_p95_max=normal_effect.quantile(bounds).cpu().tolist(),
                affected_trusted_rows=int(((effect>1e-6)&trusted).sum()),
                affected_untrusted_rows=int(((effect>1e-6)&~trusted).sum()),
                packed_covariance_max_abs_m2=covariance_error))
    qa.review.save(out/'native'/name/'graph-replay.json',dict(graph=graph,records=rows,
        camera_and_geometry_verified=True,
        diagnostic_scope='Temporary zero learned normal/tangent residuals with saved graph, trust and native poses.'))


def worker(path,name,fd):
    current = read(path)
    if not (path.parent/'native'/name/'result.json').is_file():
        rebound(prior.prior.worker,PROJECT=PROJECT,NEW=NEW,STATIC_ARMS=ARMS)(path,name,fd)
    anchor_replay(path.parent,current,name,fd)


def assemble(path,current):
    out = path.parent
    historical = read(checked(current['baseline_review']))
    paired = out/'paired'
    paired.mkdir(exist_ok=True)
    static,native,panels = {},{},{}
    for name in ARMS:
        source = checked(current['static_archive'][name]) if name in BASE else out/'static-grid'/name/'result.json'
        static[name] = dict(receipt=descriptor(source),records=qa.grid_records(current,name,source))
        native[name] = historical['native'][name] if name in BASE else read(out/'native'/name/'result.json')['records']
        require(set(native[name])=={row['frame'] for row in current['frozen']['rows']},'An R25 branch missed a native pose.')
    for key in current['static_grid']['camera_roster']:
        items = [static[name]['records'][key]['panel'] for name in ARMS]
        require(all((item['crop_xyxy'],item['columns'])==(items[0]['crop_xyxy'],items[0]['columns'])
                    for item in items),'The R25 static crop or render columns differ.')
        panels[key] = qa.stack(paired/f'{key}.png',[checked(item) for item in items],LABELS)
    for row in current['frozen']['rows']:
        frame = row['frame']
        items = [native[name][frame] for name in ARMS]
        require(all(item['camera']==items[0]['camera'] and item['role']==items[0]['role']
                    and item['source_sha256']==row['source_sha256'] and item['mask_sha256']==row['mask_sha256']
                    and item['roi']['box']==current['rois'][frame]['box'] for item in items),
                'R25 changed a paired native camera, original frame, mask or ROI.')
        if items[0]['roi']['box'] is not None:
            require(all(all(item['roi'][field]==items[0]['roi'][field] for field in ('foreground_pixels','exterior_pixels'))
                        for item in items),'R25 changed its fixed native ROI mask.')
            key = f'native_torso_{frame}'
            panels[key] = qa.stack(paired/f'{key}.png',[checked(item['roi']['panel']) for item in items],LABELS)
    metrics = rebound(prior.metrics,ARMS=ARMS)(static,native,current['static_grid']['camera_roster'])
    result = dict(state='paired_driven_and_static_views_rendered',subject=current['subject'],commit=current['commit'],
        plan=descriptor(path),static=static,static_camera_count=50,native=native,native_poses=28,panels=panels,
        paired_native_metrics=metrics,graphs=current['graphs'],
        graph_replay={name:descriptor(out/'native'/name/'graph-replay.json') for name in NEW},quality_accepted=False)
    target = out/'result.json'
    if target.exists():
        require(read(target)==result,'Preserve the completed R25 review.')
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
        baseline = history(args.subject)
        print(json.dumps(dict(subject=args.subject,static_cameras=len(baseline['static_grid']['camera_roster']),
            native_poses=len(baseline['frozen']['rows']),history=list(BASE))),flush=True)
        return
    if not args.result_sha or not args.commit:
        parser.error('Use the actual full R25 result hash and commit.')
    current = plan(args.subject,args.result_sha,args.commit)
    if args.action=='check':
        print(json.dumps(dict(subject=args.subject,static_cameras=50,native_poses=28,
            trusted_rows={name:graph['trusted_rows'] for name,graph in current['graphs'].items()})),flush=True)
        return
    out = root_for(args.subject)/'view-comparison'
    out.mkdir(exist_ok=True)
    path = out/'plan.json'
    if not path.exists():
        qa.review.save(path,current)
    require(read(path)==current,'The saved R25 independent review plan changed.')
    if args.action=='render':
        rebound(prior.render,NEW=NEW,__file__=__file__)(path,current)
        print(json.dumps(dict(subject=args.subject,state='cuda_review_released_gpu',plan=descriptor(path))),flush=True)
    assemble(path,current)


if __name__=='__main__':
    main()
