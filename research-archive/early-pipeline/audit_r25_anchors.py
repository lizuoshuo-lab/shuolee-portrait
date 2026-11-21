"""Compare R24 and R25 stable garment rows and verify actual anchor optimizer state."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath
import subprocess

import numpy as np
from PIL import Image
from scipy.special import erf
from scipy.spatial.transform import Rotation
import torch

from audit_r23_numerical import (LAB, PARAMS, alpha_image, garment_seed, motion_scope,
    movement, ply_parameters, read_state, tensor_hash, _pin, _read, _require, _sha)
from audit_r24_surface import (STATIC, inherit, thresholds)


OUTPUT = Path('/mnt/c/Users/Administrator/Documents/New project/outputs/smpl-delivery-round25/input-audit')
COMMIT = {24:'752a0f65062ff9395ee53f7bdbceb99842bdbee6',25:'43e2df49a5eb039fa14933de822821f0b3473849'}
ENDPOINTS = (10000,18000)
ANCHOR_FIELDS = ('anchor_reference_normal','anchor_trusted','anchor_normal_raw','anchor_tangent_raw')


def project_commit(round_):
    project = Path(f'/mnt/d/Projects/Shuolee-smpl-clothing-surface-r{round_}')
    win = PureWindowsPath((project/'.git').read_text().strip().removeprefix('gitdir: '))
    return (Path('/mnt')/win.drive[0].lower()/Path(*win.parts[1:])).parent.parent


def verify_config(run,round_):
    config = _read(run/'gaussian-run/config.json')
    git = project_commit(round_)
    for name,digest in config['implementation_sha256'].items():
        path = 'src/avatar_core/scene/'+name if name=='clothing_gaussian_model.py' else 'src/shuolee_portrait/'+name
        data = subprocess.check_output(['git','--git-dir',str(git),'show',f'{COMMIT[round_]}:{path}'])
        _require(sha256(data).hexdigest()==digest,'Formal implementation differs from frozen Git blob.')
    return config


def verify_paired_config(before,after):
    fixed = ('optimization','seed','source_bound_sha256','source_guide_sha256','training_frames','heldout_frames',
             'splits','split_guard','surface_nodes','surface_node_lr','surface_max_displacement_m','max_parents_per_split')
    _require(all(before[k]==after[k] for k in fixed),'R24/R25 froze different base conditions.')
    r24,r25 = [cfg['surface_regularization'] for cfg in (before,after)]
    _require(all(r25[key]==value for key,value in r24.items()) and
             set(r25)-set(r24)=={'surface_center','surface_distribution'}, 'An unexpected baseline geometry loss changed.')
    keys = ('surface_regularization','surface_coverage_weight','surface_anchor_normal_lr','surface_anchor_tangent_lr',
            'foreground_alpha_target','exterior_coverage_weight','certain_background_margin_px')
    return {key:dict(r24=before.get(key),r25=after.get(key)) for key in keys}


def quantile(values):
    values = np.asarray(values)
    _require(np.isfinite(values).all(),'Nonfinite anchor measurement.')
    if not len(values):
        return None
    return dict(zip(('p05','p50','p95','max'),(*np.quantile(values,(.05,.50,.95)).tolist(),float(values.max())),strict=True))


def normalize(values):
    return values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-12)


def prepared_graph(graph, params, selected, reference_normal, normal_raw=None, tangent_raw=None):
    # Independent NumPy implementation of the saved local surfel planes,
    # with the exact same trust/reference-normal rows in all compared arms.
    nodes = graph['node_centers'].astype(np.float64)
    rest = graph['node_rest_xyz'].astype(np.float64)
    ids,weight = graph['row_node_ids'][selected],graph['row_node_weights'][selected].astype(np.float64)
    def interp(values):
        return np.sum(values[ids]*weight[...,None],axis=1)
    neighbor = graph['node_normal_neighbors']
    delta = nodes[np.maximum(neighbor,0)]-nodes[:,None]
    cross = np.cross(delta[:,0],delta[:,1])
    supported = (neighbor>=0).all(axis=1) & (np.linalg.norm(cross,axis=1)>1e-8)
    resultant = interp(normalize(cross)*supported[:,None])
    mass = interp(supported[:,None])[:,0]
    coherence = np.linalg.norm(resultant,axis=1)/np.maximum(mass,1e-8)
    measured = normalize(resultant)
    ref = reference_normal.astype(np.float64)
    align = np.sum(measured*ref,axis=1)
    blend = np.clip((mass-.6)/.15,0.,1.)*np.clip((coherence-.5)/.25,0.,1.)*np.clip((align-.2)/.4,0.,1.)
    normal = np.where((blend>=1.-1e-6)[:,None],measured,normalize(ref+(measured-ref)*blend[:,None]))
    base = params['_xyz'][selected].astype(np.float64)-interp(rest)
    correction = np.zeros_like(base)
    signed_residual = np.zeros(len(base))
    tangent_correction = np.zeros_like(base)
    bound = np.clip(np.abs(np.sum(base*ref,axis=1))+.002,.006,.032)
    if normal_raw is not None:
        seed_offset = np.sum(base*ref,axis=1)
        signed_residual = bound*np.tanh(normal_raw[:,0])
        tangent = interp(tangent_raw.astype(np.float64))
        tangent -= np.sum(tangent*normal,axis=1,keepdims=True)*normal
        norm = np.linalg.norm(tangent,axis=1,keepdims=True)
        tangent_correction = .002*tangent*np.where(norm<1e-4,1.,np.tanh(norm)/np.maximum(norm,1e-4))
        correction = (normal-ref)*seed_offset[:,None]+normal*signed_residual[:,None]+tangent_correction
    quat = params['_rotation'][selected][:,[1,2,3,0]].astype(np.float64)
    rotation = Rotation.from_quat(quat).as_matrix()
    scales = np.exp(params['_scaling'][selected].astype(np.float64))
    axis = np.argsort(scales,axis=1)
    sorted_scales = np.take_along_axis(scales,axis,axis=1)
    axis_direction = rotation[np.arange(len(base)),:,axis[:,0]]
    plane_origin = interp(nodes)
    return dict(normal=normal,plane=plane_origin,world=plane_origin+base+correction,rotation=rotation,
                scales=scales,axis=axis_direction,sheet=sorted_scales[:,0]<.8*sorted_scales[:,1],
                normal_residual_m=signed_residual,tangent_residual_m=np.linalg.norm(tangent_correction,axis=1),
                tangent_residual_vector_m=tangent_correction,
                normal_residual_bound_m=bound,normal_support_mass=mass,normal_coherence=coherence,
                actual_local_plane=(mass>=.75)&(coherence>=.75)&(align>=.6))


def measure(model,plane=None):
    plane = model if plane is None else plane
    n = plane['normal']
    center = np.sum((model['world']-plane['plane'])*n,axis=1)
    projections = np.einsum('ni,nij->nj',n,model['rotation'])
    sigma = np.linalg.norm(projections*model['scales'],axis=1)
    ratio = np.minimum(np.abs(center),16*np.maximum(sigma,1e-12))/np.maximum(sigma,1e-12)
    folded = np.abs(center)*erf(ratio/np.sqrt(2.))+sigma*np.sqrt(2./np.pi)*np.exp(-.5*ratio**2)
    cosine = np.clip(np.abs(np.sum(n*model['axis'],axis=1)),0.,1.)
    return dict(signed_center_mm=center*1000.,abs_center_mm=np.abs(center)*1000.,normal_sigma_mm=sigma*1000.,
                folded_expected_mm=folded*1000.,thin_axis_cosine=cosine,thin_axis_angle_deg=np.rad2deg(np.arccos(cosine)))


def paired_metrics(old,new,models,opacity):
    r24,r25 = models
    live = r24['actual_local_plane']&r25['actual_local_plane']
    scopes = {'trusted':np.ones(len(opacity),dtype=bool),'trusted_peak_alpha_ge_point1':opacity>=.1,
              'trusted_live_plane_both':live,'trusted_live_plane_both_peak_alpha_ge_point1':live&(opacity>=.1)}
    measured = {}
    for scope,mask in scopes.items():
        both_sheet = mask&r24['sheet']&r25['sheet']
        metrics = {}
        for key in old:
            active = both_sheet if key in ('thin_axis_cosine','thin_axis_angle_deg') else mask
            metrics[key] = dict(r24=quantile(old[key][active]),r25=quantile(new[key][active]),
                                same_row_delta=quantile((new[key]-old[key])[active]))
        axis_change = np.rad2deg(np.arccos(np.clip(np.abs(np.sum(r24['axis']*r25['axis'],axis=1)),0.,1.)))
        normal_change = np.rad2deg(np.arccos(np.clip(np.sum(r24['normal']*r25['normal'],axis=1),-1.,1.)))
        measured[scope] = dict(rows=int(mask.sum()),same_rows_sheet_both=int(both_sheet.sum()),
                               direction_rotation_deg=quantile(axis_change[both_sheet]),
                               learned_plane_normal_change_deg=quantile(normal_change[mask]),metrics=metrics)
    return measured


def validate_optimizer(optimizer,name,shape,step,lr):
    [group] = [group for group in optimizer['param_groups'] if group['name']==name]
    [id_] = group['params']
    state = optimizer['state'][id_]
    _require(int(state['step'])==step and group['lr']==lr and all(tuple(state[key].shape)==shape and
             torch.isfinite(state[key]).all() for key in ('exp_avg','exp_avg_sq')), 'Anchor Adam state differs from actual schedule.')
    m = state['exp_avg'].detach().numpy()
    return dict(updates=int(state['step']),lr=group['lr'],nonzero_moment_rows=int(np.any(m!=0,axis=1).sum()),
                max_abs_first_moment=float(np.abs(m).max()))


def state_at(run,step,seed,old,config,anchors):
    path = run/'gaussian-run'/f'raw-step-{step:06d}'
    checkpoint = path/'main-state.pt'
    state = read_state(checkpoint)
    _require(state['step']==step==len(state['draws']) and state['options']==config['optimization'] and
             state['source_sha256']==_sha(seed['paths']['bound_ply']), 'Wrong Gaussian checkpoint or immutable seed.')
    manifest = _read(path/'drive/drive.json')
    _require(manifest['schema']==('shuolee-drivable-clothing/3' if anchors else 'shuolee-drivable-clothing/2')
             and manifest['pixel_convention']=='pixel_corners',
             'Executable garment schema differs.')
    frames,rows,held = motion_scope(run,old,path,seed,manifest,state)
    for row in rows:
        corner = np.asarray(row['K_integer_centers']).copy()
        corner[:2,2] += .5
        _require(np.array_equal(corner,row['K']), 'A native camera uses the wrong half-pixel convention.')
    _require(all(0<=i<len(frames)-len(held) for i in state['draws']), 'The draw sequence includes a held-out view.')
    for file,digest in manifest['files_sha256'].items():
        _pin(path/'drive'/file,digest)
    fixed = ply_parameters(path/'drive/fixed-bound.ply')
    _require(all(np.array_equal(value,fixed[key]) for key,value in seed['fixed'].items()) and
             state['fixed_identity']=={key:tensor_hash(value) for key,value in fixed.items()}, 'A frozen non-garment row changed.')
    raw = {key:value.detach().numpy() for key,value in zip(PARAMS,state['garment'][1:7],strict=True)}
    saved = ply_parameters(path/'drive/clothing-canonical.ply')
    _require(all(np.array_equal(value,saved[key]) for key,value in raw.items()), 'Canonical PLY changes a raw Gaussian row.')
    ids,owners,opacity,source,children = inherit(seed,state)
    original = ids<seed['next']
    _require(np.array_equal(raw['_xyz'][original],seed['xyz'][source[original]]) and
             np.array_equal(raw['_opacity'][original,0],opacity[original].astype(np.float32)), 'An original raw centre or opacity changed.')
    alpha_error = float(np.max(np.abs(raw['_opacity'][:,0]-opacity),initial=0.))
    _require(alpha_error<3e-5, 'Compensated split child logit differs from its ancestry.')
    reconstructed = {int(i):v for i,v in zip(ids,raw['_xyz'],strict=True)}
    for parent,(a,b) in reversed(children.items()):
        reconstructed[parent] = .5*(reconstructed[a]+reconstructed[b])
    seed_error = float(np.max(np.abs(np.stack([reconstructed[i] for i in seed['ids']])-seed['xyz'])))
    _require(seed_error<3e-6, 'Split rows no longer preserve the raw parent midpoint.')
    caps = np.full(len(seed['faces']),-np.inf,np.float32)
    np.maximum.at(caps,seed['owner'],seed['log_sigma'].max(axis=1))
    cap_error = float(np.max(raw['_scaling'].max(1)-caps[owners]))
    _require(cap_error<=2e-6, 'A trained canonical Gaussian exceeds its owner sigma cap.')
    graph = {name:value.numpy() for name,value in state['surface_graph'].items()}
    with np.load(path/'drive'/manifest['garment_graph'],allow_pickle=False) as data:
        _require(set(data.files)==set(graph) and all(np.array_equal(value,data[key]) for key,value in graph.items()),
                 'Executable graph differs from optimizer state.')
    body = state['garment'][12]
    for group in body['param_groups']:
        if group['name'] in ('xyz','opacity'):
            _require(not any(id_ in body['state'] for id_ in group['params']), 'Frozen row field has optimizer moments.')
    report = dict(step=step,checkpoint_sha256=_sha(checkpoint),rows=len(ids),protected_rows=len(seed['fixed_ids']),
                  exact_frozen_pose_cameras=len(frames),pixel_convention='pixel_corners (original integer K principal point +0.5)',
                  frozen_original_rows=int(original.sum()),
                  inherited_split_rows=int((~original).sum()),raw_ancestor_midpoint_error_m=seed_error,
                  max_logit_error=alpha_error,max_excess_over_owner_cap_log_sigma=cap_error,
                  events=[thresholds(event,config) for event in state['split_events']])
    data = dict(path=path,ids=ids,source=source,owners=owners,raw=raw,graph=graph,draws=state['draws'],
                children=children,frames=frames,held=held,report=report)
    if anchors:
        saved_anchors = {name:value.numpy() for name,value in state['surface_anchors'].items()}
        with np.load(path/'drive'/manifest['garment_anchors'],allow_pickle=False) as z:
            _require(set(z.files)==set(saved_anchors)==set(ANCHOR_FIELDS) and
                     all(np.array_equal(value,z[key]) for key,value in saved_anchors.items()),
                     'Executable anchors differ from exact optimizer tensors.')
        mask = saved_anchors['anchor_trusted']
        refs = saved_anchors['anchor_reference_normal']
        normal,tangent = [saved_anchors[key] for key in ('anchor_normal_raw','anchor_tangent_raw')]
        _require(mask.dtype==np.bool_ and mask.shape==ids.shape and
                 np.isfinite(refs).all() and np.allclose(np.linalg.norm(refs,axis=1),1.,atol=2e-5) and
                 np.all(normal[~mask]==0) and np.isfinite(normal).all() and np.isfinite(tangent).all(),
                 'Invalid or accidentally updated untrusted anchor state.')
        measured = prepared_graph(graph,raw,np.where(mask)[0],refs[mask],normal[mask],tangent)
        for value,bound in (('normal_residual_m',measured['normal_residual_bound_m']),('tangent_residual_m',.002)):
            _require(np.all(np.abs(measured[value])<=bound+1e-7), 'Learned anchor residual escapes its bound.')
        native = _read(path/'native-ply-parity.json')
        _pin(path/'world.ply',native['sha256'])
        world = ply_parameters(path/'world.ply')['_xyz'][len(seed['fixed_ids']):][mask]
        world_error = float(np.abs(world-measured['world']).max())
        _require(world_error<3e-6, 'Independently reconstructed trusted graph centres differ from the live reference PLY.')
        report.update(trusted_rows=int(mask.sum()),trusted_world_center_max_error_m=world_error,
            normal_adam=validate_optimizer(body,'anchor_normal',normal.shape,step,config['surface_anchor_normal_lr']),
            tangent_adam=validate_optimizer(state['surface_optimizer'],'bounded_anchor_tangent',tangent.shape,step,config['surface_anchor_tangent_lr']),
            nodes_adam=validate_optimizer(state['surface_optimizer'],'garment_nodes',graph['node_centers'].shape,step,config['surface_node_lr']),
            normal_raw_nonzero_rows=int(np.count_nonzero(normal[:,0])),tangent_raw_nonzero_nodes=int(np.any(tangent!=0,axis=1).sum()),
            signed_normal_residual_mm=quantile(measured['normal_residual_m']*1000.),
            abs_normal_residual_mm=quantile(np.abs(measured['normal_residual_m'])*1000.),
            tangent_residual_mm=quantile(measured['tangent_residual_m']*1000.))
        data['anchors'] = saved_anchors
    else:
        g = state['surface_optimizer']
        [group] = g['param_groups']
        [slot] = group['params']
        moments = g['state'][slot]
        _require(int(moments['step'])==step and group['lr']==config['surface_node_lr'] and
                 tuple(moments['exp_avg'].shape)==graph['node_centers'].shape and
                 torch.isfinite(moments['exp_avg']).all(), 'R24 surface optimizer count or stored update differs.')
        report['node_adam_updates'] = int(moments['step'])
    return data


def initial_rows(data,seed):
    result = {}
    for row,source in enumerate(data['source']):
        original = int(seed['ids'][source])
        fields = {key:data['anchors'][key][row] for key in ('anchor_reference_normal','anchor_trusted')}
        fields.update({key:data['graph'][key][row] for key in ('row_node_ids','row_node_weights')})
        if original in result:
            _require(all(np.array_equal(value,result[original][key]) for key,value in fields.items()),
                     'Probe siblings inherited different trust or node stencils.')
        result[original] = fields
    _require(len(result)==len(seed['ids']), 'The probe cannot reconstruct original trust and stencil rows.')
    return result


def verify_initial(data,probe,initial,seed):
    _require(all(np.array_equal(data['graph'][key],probe['graph'][key]) for key in STATIC), 'Frozen graph changed from probe.')
    for key in ('anchor_reference_normal','anchor_trusted','row_node_ids','row_node_weights'):
        source = data['anchors'] if key.startswith('anchor') else data['graph']
        expected = np.asarray([initial[int(seed['ids'][index])][key] for index in data['source']])
        _require(np.array_equal(source[key],expected), 'A current row changed its inherited trust or neighborhood.')


def same_native_alpha(r24,r25,old):
    files = [data['path']/'drive-replay' for data in (r24,r25)]
    rows = [{row['frame']:row for row in _read(directory/'result.json')['records']} for directory in files]
    source = {row['frame']:row for row in _read(old.parent.parent/'data/inputs.json')['records']}
    result = []
    for frozen in r25['held']:
        frame = frozen['frame']
        a,b = [arm[frame] for arm in rows]
        _require(all(a[k]==b[k] for k in ('frame','source_sha256','mask_sha256','pose_index','box_xyxy')) and
                 all(a[k]==frozen[k]==source[frame][k] for k in ('source_sha256','mask_sha256','pose_index') if k!='pose_index') and
                 a['pose_index']==frozen['pose_index'], 'Paired held-out crop or source changed.')
        x0,y0,x1,y1 = a['box_xyxy']
        mask_path = Path(source[frame]['mask'])
        _pin(mask_path,a['mask_sha256'])
        with Image.open(mask_path) as image:
            _require(image.size==tuple(frozen['wh']), 'Original mask has changed dimensions.')
            full_mask = np.asarray(image.convert('L'))
            mask = full_mask[y0:y1,x0:x1]
        fg = mask>=253
        _require(int(fg.sum())==int((full_mask>=253).sum())>0, 'The common crop discards confident foreground.')
        def metrics(values):
            a = values.astype(np.float64)/255.
            return dict(foreground_mean=float(a[fg].mean()),foreground_below_point9=float((a[fg]<.9).mean()),
                        crop_alpha_l1_per_foreground=float(np.abs(a-mask/255.).sum()/fg.sum()),
                        local_mask_zero_exterior_mean=float(a[mask==0].mean()))
        alphas = [alpha_image(directory/row['files']['alpha'],(x1-x0,y1-y0)) for directory,row in zip(files,(a,b),strict=True)]
        result.append(dict(frame=frame,foreground_pixels=int(fg.sum()),crop=a['box_xyxy'],r24=metrics(alphas[0]),r25=metrics(alphas[1])))
    return dict(records=result,scope='Same original native mask, fitted pose and crop. 8-bit alpha; full-image exterior beyond crop is not measured.')


def compare(r24,r25):
    common,a,b = np.intersect1d(r24['ids'],r25['ids'],assume_unique=True,return_indices=True)
    original = (r24['source'][a]==r25['source'][b]) & r25['anchors']['anchor_trusted'][b]
    # R24 and R25 allocate new stable IDs independently. Shared numeric child
    # IDs are not matched even if they happen to share the same ancestor.
    original &= np.isin(common,r25['seed_ids'],assume_unique=True)
    a,b = a[original],b[original]
    _require(np.array_equal(r24['owners'][a],r25['owners'][b]) and
             np.array_equal(r24['raw']['_xyz'][a],r25['raw']['_xyz'][b]) and
             np.array_equal(r24['raw']['_opacity'][a],r25['raw']['_opacity'][b]) and
             all(np.array_equal(r24['graph'][key][a],r25['graph'][key][b]) for key in ('row_node_ids','row_node_weights')),
             'Shared original trusted rows have different owners, seed centre or row stencils.')
    anchors = r25['anchors']
    normals = anchors['anchor_reference_normal'][b]
    a25 = prepared_graph(r25['graph'],r25['raw'],b,normals,anchors['anchor_normal_raw'][b],anchors['anchor_tangent_raw'])
    a24 = prepared_graph(r24['graph'],r24['raw'],a,normals)
    opacity = 1/(1+np.exp(-r25['raw']['_opacity'][b,0]))
    out = dict(planes='Same trusted stable original rows. Measure each arm relative to its own learned local garment plane; anchor trust is geometric.',
               comparisons=paired_metrics(measure(a24),measure(a25),(a24,a25),opacity),
               r25_vs_fixed_r24_plane_sensitivity=paired_metrics(measure(a24),measure(a25,a24),(a24,a25),opacity),
               canonical_center_change=movement(a24['world'],a25['world']))
    return out


def audit(subject,steps):
    seed,old_run,old = garment_seed(subject)
    roots = {n:LAB/f'shuolee-smpl-delivery-round{n}-{subject}' for n in (24,25)}
    probe = roots[25]/'probe-01'
    probe_data = state_at(probe,1,seed,old,_read(probe/'gaussian-run/config.json'),True)
    initial = initial_rows(probe_data,seed)
    _require(probe_data['report']['inherited_split_rows']==32, 'Native probe did not commit 16 split parents.')
    candidate = probe/'gaussian-run'
    difference,formal = None,[]
    changes = None
    if steps:
        runs = [roots[n]/'run' for n in (24,25)]
        baseline_config,config = [verify_config(run,n) for run,n in zip(runs,(24,25),strict=True)]
        difference = verify_paired_config(baseline_config,config)
        candidate = runs[1]/'gaussian-run'
        paired_data = []
        for step in steps:
            r24,r25 = [state_at(run,step,seed,old,cfg,anchor) for run,cfg,anchor in
                      zip(runs,(baseline_config,config),(False,True),strict=True)]
            verify_initial(r25,probe_data,initial,seed)
            _require(all(np.array_equal(r24['graph'][key],r25['graph'][key]) for key in STATIC) and
                     r24['draws']==r25['draws'], 'R24/R25 initial graph or actual draw roster differs.')
            r25['seed_ids'] = seed['ids']
            formal.append(dict(step=step,r24=r24['report'],r25=r25['report'],
                               shared_original_trusted=compare(r24,r25),native_alpha=same_native_alpha(r24,r25,old)))
            paired_data.append(r25)
        if len(paired_data)==2:
            a,b = paired_data
            _require(a['draws']==b['draws'][:steps[0]] and a['children']==b['children'] and
                     np.array_equal(a['ids'],b['ids']), 'R25 same-run raw prefix or immutable topology changed.')
            arrays = [row['anchors'] for row in (a,b)]
            normal,tangent = [np.linalg.norm(arrays[1][key]-arrays[0][key],axis=1)
                              for key in ('anchor_normal_raw','anchor_tangent_raw')]
            trusted = np.flatnonzero(arrays[0]['anchor_trusted'])
            _require(np.array_equal(arrays[0]['anchor_trusted'],arrays[1]['anchor_trusted']),
                     'R25 retained row trust changed between endpoints.')
            arms = [prepared_graph(row['graph'],row['raw'],trusted,anchors['anchor_reference_normal'][trusted],
                        anchors['anchor_normal_raw'][trusted],anchors['anchor_tangent_raw'])
                    for row,anchors in zip((a,b),arrays,strict=True)]
            changes = dict(exact_draw_and_topology_prefix=True,rows=len(a['ids']),
                           normal_raw_same_row_absolute_change=quantile(normal),tangent_raw_same_node_l2_change=quantile(tangent),
                           normal_residual_same_trusted_row_change_mm=quantile(np.abs(arms[1]['normal_residual_m']-arms[0]['normal_residual_m'])*1000.),
                           tangent_residual_vector_same_trusted_row_change_mm=quantile(np.linalg.norm(
                               arms[1]['tangent_residual_vector_m']-arms[0]['tangent_residual_vector_m'],axis=1)*1000.),
                           normal_rows_changed=int(np.count_nonzero(normal)),tangent_nodes_changed=int(np.count_nonzero(tangent)))
    gradient = _read(candidate/'cuda-gradient-proof.json')
    for name in ('node_image_gradients','normal_residual_image_gradients'):
        _require(set(gradient[name])=={'rgb','mask'} and all(np.isfinite(v) and v>0 for v in gradient[name].values()),
                 'Native RGB or mask gradient into actual learned garment offsets is missing.')
    payload = dict(schema='shuolee-r25-independent-anchors/1',subject=subject,probe=probe_data['report'],
        formal_commits=COMMIT if steps else None,formal_config_differences=difference,
        formal=formal,same_run=changes,first_step_gradient=gradient,
        scope='Geometric local planes estimated from executable surfels; independent physical cloth depth and actual rendered visibility are not established by the distance metrics.')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    tag = 'paired' if len(steps)==2 else (str(steps[0]) if steps else 'probe')
    target = OUTPUT/f'{subject}-anchors-{tag}.json'
    with target.open('x',encoding='utf-8') as f:
        json.dump(payload,f,ensure_ascii=False,allow_nan=False,indent=2)
        f.write('\n')
    print(f'{target} sha256={_sha(target)}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    parser.add_argument('--step',type=int,choices=ENDPOINTS)
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    audit(args.subject,() if args.probe else ((args.step,) if args.step else ENDPOINTS))
