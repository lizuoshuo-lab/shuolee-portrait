"""Independent CPU audit of actual R24 graph checkpoints and split receipts."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath
import subprocess

import numpy as np
import torch

from audit_r23_numerical import (LAB, PARAMS, garment_seed, motion_scope, native_alpha, movement,
    ply_parameters, read_state, tensor_hash, _pin, _read, _require, _sha)


OUTPUT = Path('/mnt/c/Users/Administrator/Documents/New project/outputs/smpl-delivery-round24/input-audit')
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-clothing-surface-r24')
COMMIT = '752a0f65062ff9395ee53f7bdbceb99842bdbee6'
STATIC = ('node_rest_xyz', 'node_owner_faces', 'node_normal_neighbors', 'graph_edges', 'face_groups')


def thresholds(event, config):
    limits = config['split_guard']['limits']
    frames = set(config['training_frames'])
    trials = []
    accepted = None
    for candidate in event['trials']:
        rows = candidate['views']
        _require(len(rows) == len(frames) == event['train_cameras']
                 and {row['frame'] for row in rows} == frames, 'Split guard did not render the complete roster.')
        failures = {}
        for row in rows:
            excess = [key for key, limit in limits.items() if not np.isfinite(row[key]) or row[key] > limit]
            _require(row['passed'] == (not excess), 'A recorded split decision differs from the metric thresholds.')
            if excess:
                failures[row['frame']] = excess
        _require(candidate['passed'] == (not failures) and accepted is None,
                 'Split trial state differs or rendering continued after an accepted trial.')
        if candidate['passed']:
            accepted = candidate
        trials.append(dict(parents=candidate['proposed_parents'], shift=candidate['shift_in_sigma'],
            passed=candidate['passed'], failing_frames=failures,
            metric_maxima={key:max(row[key] for row in rows) for key in limits},
            views=rows))
    if accepted:
        _require(event['parents'] == accepted['proposed_parents'] and event['shift_in_sigma'] == accepted['shift_in_sigma']
                 and event['children'] == 2*event['parents'], 'Accepted split count or displacement differs.')
    else:
        _require(event['parents'] == event['children'] == 0, 'Failed or missing trial changed the split layout.')
    return dict(step=event['step'], committed=event.get('committed', False), trials=trials,
                parents=event['parents'], children=event['children'], train_cameras=event['train_cameras'])


def alpha_rule(logit, shift):
    a = torch.tensor(logit, dtype=torch.float32).sigmoid()
    peak = -torch.expm1(.5*torch.log1p(-a))
    return float(torch.logit((peak/np.exp(-.5*shift**2/(1-shift**2))).clamp(max=.985), eps=1e-7))


def inherit(seed, state):
    live = {int(i):(int(owner), float(alpha), k) for k,(i,owner,alpha) in
            enumerate(zip(seed['ids'],seed['owner'],seed['opacity'],strict=True))}
    next_id, children_of = seed['next'], {}
    for event in state['split_events']:
        if not event.get('committed', False):
            continue
        parents = event['parent_ids']
        count = len(parents)
        _require(count == len(set(parents)) == event['parents'] and count > 0
                 and event['first_child_id'] == next_id and event['children'] == 2*count,
                 'An actual split reused IDs or has inconsistent ancestry.')
        for offset,parent in enumerate(parents):
            owner,alpha,original = live.pop(parent)
            child_opacity = alpha_rule(alpha,event['shift_in_sigma'])
            first,second = next_id+offset,next_id+offset+count
            live[first] = live[second] = owner,child_opacity,original
            children_of[parent] = first,second
        next_id += 2*count
    ids = state['stable_ids'].numpy()
    _require(ids.tolist() == sorted(live) and state['next_stable_id'] == next_id,
             'The actual stable IDs differ from successful split events.')
    owners, alpha, source = [np.asarray(value) for value in zip(*(live[int(i)] for i in ids),strict=True)]
    _require(np.array_equal(owners,state['support_face_ids'].numpy()), 'A garment child changed its frozen support face.')
    return ids, owners, alpha, source, children_of


def anatomy_graph(graph, owners, vertices, faces, assets):
    with np.load(assets/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=False) as data:
        weights, topology = data['weights'], data['f']
    _require(np.array_equal(topology,faces), 'The anatomy map uses different topology.')
    labels = np.zeros(weights.shape[1],dtype=np.int64)
    labels[[13,16,18,20,*range(25,40)]] = 1
    labels[[14,17,19,21,*range(40,55)]] = 2
    labels[[1,4,7,10]] = 3
    labels[[2,5,8,11]] = 4
    votes = np.stack([weights[:,labels==i].sum(axis=1) for i in range(5)],axis=1)
    groups = votes[faces].mean(1).argmax(1)
    _require(np.array_equal(groups,graph['face_groups']), 'The recorded anatomy labels differ from frozen SMPL-X.')
    node_owner = graph['node_owner_faces']
    active = set(owners.tolist())|set(node_owner.tolist())
    by_edge, adjacent = defaultdict(list), {i:set() for i in active}
    for face in active:
        a,b,c = sorted(faces[face])
        for u,v in ((a,b),(a,c),(b,c)):
            by_edge[(groups[face],u,v)].append(face)
    for incident in by_edge.values():
        for face in incident:
            adjacent[face].update(other for other in incident if other != face)
    reachable = {face:{face}|near|{v for u in near for v in adjacent[u]} for face,near in adjacent.items()}
    row_ids, weight = graph['row_node_ids'],graph['row_node_weights']
    _require(np.isfinite(weight).all() and np.all(weight>=0) and np.allclose(weight.sum(1),1.,rtol=0,atol=1e-6),
             'Invalid row weights.')
    row_links = np.column_stack((np.broadcast_to(owners[:,None],row_ids.shape)[weight>0],node_owner[row_ids][weight>0]))
    normal = graph['node_normal_neighbors']
    normal_links = np.column_stack((np.repeat(np.arange(len(node_owner)),2),normal.ravel()))
    normal_links = normal_links[normal_links[:,1]>=0]
    graph_pairs = [row_links,node_owner[graph['graph_edges']],node_owner[normal_links]]
    for pairs in graph_pairs:
        _require(all(int(b) in reachable[int(a)] for a,b in np.unique(pairs,axis=0)),
                 'The actual graph crosses a frozen same-anatomy two-edge support boundary.')
    use = (weight>0).sum(axis=1)
    return dict(active_owner_faces=len(active),nodes_by_anatomy=np.bincount(groups[node_owner],minlength=5).tolist(),
        nonzero_row_links=len(row_links),graph_edges=len(graph_pairs[1]),normal_links=len(graph_pairs[2]),
        multi_node_rows=int((use>1).sum()),garment_normal_nodes=int((normal[:,0]>=0).sum()),
        cross_anatomy_or_beyond_two_hops=0,actual_cross_material_identity_verified=False)


def endpoint(run, old, old_snapshot, seed, step, config, initial=None):
    path = run/'gaussian-run'/f'raw-step-{step:06d}'
    state = read_state(path/'main-state.pt')
    _require(state['step'] == step == len(state['draws']) and state['options'] == config['optimization']
             and state['source_sha256'] == _sha(seed['paths']['bound_ply']), 'Incorrect raw graph checkpoint.')
    manifest = _read(path/'drive/drive.json')
    frames,rows,held = motion_scope(run,old_snapshot,path,seed,manifest,state)
    train_count = len(frames)-len(held)
    _require(all(0<=i<train_count for i in state['draws']), 'A training draw used a held-out view.')
    drive = path/'drive'
    for name,digest in manifest['files_sha256'].items():
        _pin(drive/name,digest)
    fixed = ply_parameters(drive/'fixed-bound.ply')
    _require(all(np.array_equal(fixed[key],value) for key,value in seed['fixed'].items()) and
             state['fixed_identity'] == {key:tensor_hash(value) for key,value in fixed.items()},
             'A protected non-garment row changed.')
    ids,owner,want_alpha,source,children = inherit(seed,state)
    saved = ply_parameters(drive/'clothing-canonical.ply')
    actual = {key:value.detach().numpy() for key,value in zip(PARAMS,state['garment'][1:7],strict=True)}
    _require(all(np.array_equal(saved[key],value) for key,value in actual.items()),
             'Clothing PLY differs from exact checkpoint parameters.')
    original = ids < seed['next']
    _require(np.array_equal(actual['_xyz'][original],seed['xyz'][source[original]]) and
             np.array_equal(actual['_opacity'][original,0],want_alpha[original].astype(np.float32)),
             'An original per-splat centre or opacity changed.')
    logit_error = float(np.max(np.abs(actual['_opacity'][:,0]-want_alpha),initial=0.))
    _require(logit_error < 3e-5,'Child logits differ from the recorded compensated peak rule.')
    reconstructed = {int(i):v for i,v in zip(ids,actual['_xyz'],strict=True)}
    for parent,(a,b) in reversed(children.items()):
        reconstructed[parent] = .5*(reconstructed[a]+reconstructed[b])
    seed_error = float(np.max(np.abs(np.stack([reconstructed[i] for i in seed['ids']])-seed['xyz'])))
    _require(seed_error < 3e-6,'Committed split children do not reconstruct the frozen raw ancestor midpoint.')
    caps = np.full(len(seed['faces']),-np.inf,dtype=np.float32)
    np.maximum.at(caps,seed['owner'],seed['log_sigma'].max(axis=1))
    scale_excess = float(np.max(actual['_scaling'].max(1)-caps[owner]))
    _require(scale_excess <= 2e-6, 'An actual Gaussian escaped the inherited face cap.')
    adam = state['garment'][12]
    for group in adam['param_groups']:
        if group['name'] in ('xyz','opacity'):
            _require(all(key not in adam['state'] for key in group['params']), 'Frozen row XYZ/logit has Adam moments.')
    graph = {key:value.numpy() for key,value in state['surface_graph'].items()}
    with np.load(drive/manifest['garment_graph'],allow_pickle=False) as z:
        _require(set(z.files)==set(graph) and all(np.array_equal(value,z[key]) for key,value in graph.items()),
                 'Executable garment graph differs from saved trainable state.')
    with np.load(drive/'motion.npz',allow_pickle=False) as z:
        assets = Path(_read(old/'run.json')['assets'])
        graph_review = anatomy_graph(graph,owner,z['vertices_world'],z['faces'],assets)
    original_stencils = {}
    if initial is not None:
        _require(all(np.array_equal(graph[key],initial[0][key]) for key in STATIC),
                 'Formal shared graph differs from the pretraining probe graph.')
        original_stencils = initial[1]
        for key,values in (('row_node_ids',graph['row_node_ids']),('row_node_weights',graph['row_node_weights'])):
            expected = np.stack([original_stencils[int(seed['ids'][k])][key] for k in source])
            _require(np.array_equal(values,expected), 'An actual descendant changed its immutable inherited stencil.')
    else:
        for row,k in enumerate(source):
            old_id = int(seed['ids'][k])
            values = {key:graph[key][row] for key in ('row_node_ids','row_node_weights')}
            if old_id in original_stencils:
                _require(all(np.array_equal(value,original_stencils[old_id][key]) for key,value in values.items()),
                         'The two actual probe children did not inherit the same parent stencil.')
            original_stencils[old_id] = values
        _require(len(original_stencils)==len(seed['ids']),'Probe graph cannot establish all seed stencils.')
    g_adam = state['surface_optimizer']
    [slot] = g_adam['param_groups'][0]['params']
    moments = g_adam['state'][slot]
    _require(len(g_adam['state']) == len(g_adam['param_groups']) == 1 and int(moments['step'])==step and
             all(tuple(moments[k].shape)==graph['node_centers'].shape and torch.isfinite(moments[k]).all()
                 for k in ('exp_avg','exp_avg_sq')) and moments['exp_avg'].abs().max()>0,
             'The actual shared node Adam history is incomplete.')
    center_delta = graph['node_centers']-graph['node_rest_xyz']
    ref_center = actual['_xyz']+(center_delta[graph['row_node_ids']]*graph['row_node_weights'][...,None]).sum(axis=1)
    node_stats = movement(graph['node_rest_xyz'],graph['node_centers'])
    _require(0.<node_stats['max_mm']<=config['surface_max_displacement_m']*1000.+.001,
             'Actual learned node movement is zero or outside the declared cap.')
    native = _read(path/'native-ply-parity.json')
    _pin(path/'world.ply',native['sha256'])
    world = ply_parameters(path/'world.ply')
    center_error = float(np.max(np.abs(world['_xyz'][len(seed['fixed_ids']):]-ref_center)))
    _require(center_error < 2e-6,'The learned graph does not reproduce the exported reference-world centres.')
    alpha = native_alpha(old_snapshot,path,rows,held)
    for row in alpha['records']:
        row['r24'] = row.pop('r23')
    report = dict(step=step,checkpoint_sha256=_sha(path/'main-state.pt'),exact_frozen_pose_cameras=len(frames),
        exact_noncloth_rows=len(seed['fixed_ids']),raw_original_xyz_bit_exact=True,original_rows=int(original.sum()),
        live_split_descendants=int((~original).sum()),logit_max_error=logit_error,
        raw_seed_midpoint_max_abs_m=seed_error,max_excess_over_owner_cap_log_sigma=scale_excess,
        independent_reference_world_center_max_abs_m=center_error,
        node_adam_updates=int(moments['step']),node_nonzero_first_moment_rows=int(moments['exp_avg'].ne(0).any(dim=1).sum()),
        node_displacement=node_stats,graph=graph_review,
        induced_row_movement=movement(actual['_xyz'],ref_center),
        split_events=[thresholds(event,config) for event in state['split_events']],
        native_alpha_r22=alpha)
    return report,dict(state=state,graph=graph,centers=ref_center,ids=ids,original_stencils=original_stencils)


def audit(subject, steps):
    seed, old, old_snapshot = garment_seed(subject)
    base = LAB/f'shuolee-smpl-delivery-round24-{subject}'
    probe = base/'probe-01'
    probe_cfg = _read(probe/'gaussian-run/config.json')
    probe_report, probe_state = endpoint(probe,old,old_snapshot,seed,1,probe_cfg)
    _require(probe_report['live_split_descendants']==(0 if subject=='016' else 32), 'Probe changed the wrong rows.')
    run = base/'run'
    formal, loaded = [], []
    gradient = _read(probe/'gaussian-run/cuda-gradient-proof.json')
    paired = None
    if steps:
        config = _read(run/'gaussian-run/config.json')
        # This Git worktree was created on Windows. Use its common object
        # directory rather than parsing the Windows worktree link in Linux Git.
        win_git = PureWindowsPath((PROJECT/'.git').read_text().strip().removeprefix('gitdir: '))
        git = (Path('/mnt')/win_git.drive[0].lower()/Path(*win_git.parts[1:])).parent.parent
        for name,digest in config['implementation_sha256'].items():
            file = 'src/avatar_core/scene/'+name if name=='clothing_gaussian_model.py' else 'src/shuolee_portrait/'+name
            exact = subprocess.check_output(['git','--git-dir',str(git),'show',f'{COMMIT}:{file}'])
            _require(sha256(exact).hexdigest()==digest,'Actual formal implementation differs from the frozen commit.')
        gradient = _read(run/'gaussian-run/cuda-gradient-proof.json')
        _require(set(gradient['node_image_gradients'])=={'rgb','mask'} and
                 all(np.isfinite(v) and v>0 for v in gradient['node_image_gradients'].values()),
                 'Formal actual native RGB or mask produced no node gradient.')
        for step in steps:
            report,state = endpoint(run,old,old_snapshot,seed,step,config,
                                   (probe_state['graph'],probe_state['original_stencils']))
            formal.append(report)
            loaded.append(state)
    if len(loaded)==2:
        a,b = loaded
        _require(a['state']['draws']==b['state']['draws'][:steps[0]] and a['state']['split_events']==b['state']['split_events'] and
                 np.array_equal(a['ids'],b['ids']), 'The same-run 10k draw or guarded topology prefix differs.')
        paired = dict(exact_draw_and_topology_prefix=True,
            nodes_10k_to_18k=movement(a['graph']['node_centers'],b['graph']['node_centers']),
            live_rows_10k_to_18k=movement(a['centers'],b['centers']))
    payload = dict(schema='shuolee-r24-surface-audit/1',subject=subject,probe=probe_report,
        first_step_gradient=gradient,formal=formal,paired=paired,formal_commit=COMMIT if steps else None,
        split_metric_scope='Verify all native CUDA per-view decisions from recorded metrics. Before/after trial rasters were not persisted independently.')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    tag = 'paired' if paired else (str(steps[0]) if steps else 'probe')
    target = OUTPUT/f'{subject}-surface-{tag}.json'
    with target.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(f'{target} sha256={_sha(target)}',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--step',type=int,choices=(10000,18000))
    args=parser.parse_args()
    torch.set_num_threads(4)
    audit(args.subject,() if args.probe else ((args.step,) if args.step else (10000,18000)))
