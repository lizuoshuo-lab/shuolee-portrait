"""Independent CPU audit of R28 proxy geometry, native color protection and recovery state."""

from argparse import ArgumentParser
from itertools import count, islice
import json
from pathlib import Path
import random

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from audit_r23_numerical import LAB, PARAMS, ply_parameters, _pin, _read, _require, _sha
from audit_r27_face_invariants import arr, geometry_measures, lineage, npz, poses_and_cameras, quantiles, same, save, select


WORKSPACE = Path('/mnt/c/Users/Administrator/Documents/New project')
OUT = WORKSPACE/'outputs/smpl-delivery-round28/input-audit'
OLD = WORKSPACE/'outputs/smpl-delivery-round27/input-audit'
SOURCE_SLOTS = list(range(24))+list(range(28,32))
HOLDOUT_SLOTS = list(range(24,28))
SCALE_KEYS = ('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity')
GROUP_NAMES = ('xyz','f_dc','f_rest','scaling','rotation','opacity')
C0 = .28209479177387814


def _float_match(x,y,label,atol=2e-7,rtol=2e-6):
    _require(x.shape==y.shape and np.allclose(x,y,rtol=rtol,atol=atol),
             f'Independent {label} differs: {np.max(np.abs(x-y))}.')


def frames(triangles):
    a,b,c = np.moveaxis(triangles,1,0)
    ab,ac = b-a,c-a
    length = np.linalg.norm(ab,axis=1)
    x = ab/length[:,None]
    y = np.cross(x,ac)
    height = np.linalg.norm(y,axis=1)
    y /= height[:,None]
    return np.stack((x,y,np.cross(x,y)),axis=2),(length+height)*.5


def closest(point,triangle):
    a,b,c = np.moveaxis(triangle,1,0)
    ab,ac,dp = b-a,c-a,point-a
    ab2,ac2,abac = (ab*ab).sum(1),(ac*ac).sum(1),(ab*ac).sum(1)
    apb,apc = (ab*dp).sum(1),(ac*dp).sum(1)
    det = ab2*ac2-abac**2
    u,v = (ac2*apb-abac*apc)/det,(ab2*apc-abac*apb)/det
    projection = np.linalg.norm(dp-u[:,None]*ab-v[:,None]*ac,axis=1)
    distances = [np.where((u>=0)&(v>=0)&(u+v<=1),projection,np.inf)]
    for p,q in ((a,b),(b,c),(c,a)):
        edge = q-p
        fraction = ((point-p)*edge).sum(1)/(edge*edge).sum(1)
        distances.append(np.linalg.norm(point-p-fraction.clip(0,1)[:,None]*edge,axis=1))
    return np.stack(distances).min(0)


def covariance_params(raw, rows):
    values = {k:raw[k][rows].astype(np.float64) for k in ('_xyz','_scaling','_rotation')}
    values['rotation'] = Rotation.from_quat(values['_rotation'][:,[1,2,3,0]]).as_matrix()
    values['width'] = np.exp(values['_scaling'])
    return values


def measure(source,target,anchor,s_basis,s_scale,t_basis,gaussian):
    x = source.mean(1)+(s_basis @ gaussian['_xyz'][...,None]).squeeze(-1)*s_scale[:,None]
    pose_rotation = s_basis @ gaussian['rotation']
    sigma = gaussian['width']*s_scale[:,None]
    world_axes = pose_rotation*sigma[:,None]
    covariance = world_axes@world_axes.transpose(0,2,1)
    projected = t_basis.transpose(0,2,1)@covariance@t_basis
    target_plane = ((x-anchor)*t_basis[:,:,1]).sum(1)
    owner_plane = gaussian['_xyz'][:,1]*s_scale
    tangent = projected[:,(0,2),:][:,:,(0,2)]
    return dict(target_plane=target_plane,owner_plane=owner_plane,
                target_closest=closest(x,target),owner_closest=closest(x,source),
                fixed_anchor=np.linalg.norm(x-anchor,axis=1),
                target_sigma=np.sqrt(projected[:,1,1]),target_tangent=np.sqrt(np.linalg.eigvalsh(tangent)[:,1]))


def proxy_geometry(old, live, seed_rows, endpoint_rows, assoc, saved, motion, cfg):
    faces,verts = motion['faces'],motion['vertices_world']
    owner,target,bary = (assoc[k] for k in ('owner_faces','target_faces','barycentric'))
    seed,now = covariance_params(old,seed_rows),covariance_params(live,endpoint_rows)
    series = {'r22':{},'r28':{}}
    references = {}
    for slot, pose in enumerate(verts.astype(np.float64)):
        src,dst = pose[faces[owner]],pose[faces[target]]
        a,sc = frames(src)
        b,tc = frames(dst)
        anchor = np.einsum('ni,nij->nj',bary,dst)
        for name,value in (('owner_to_target',b.transpose(0,2,1)@a),
                           ('target_origin',((src.mean(1)-anchor)[:,None,:]@b).squeeze(1)),
                           ('owner_scale',sc),('target_scale',tc)):
            references.setdefault(name,[]).append(value)
        for group,param in (('r22',seed),('r28',now)):
            values = measure(src,dst,anchor,a,sc,b,param)
            for metric,value in values.items():
                series[group].setdefault(metric,[]).append(value)
    series = {group:{k:np.stack(v) for k,v in measures.items()} for group,measures in series.items()}
    references = {k:np.stack(v) for k,v in references.items()}
    references.update(initial_center=series['r22']['target_plane'],initial_normal_sigma=series['r22']['target_sigma'],
                      initial_tangent_sigma=series['r22']['target_tangent'],initial_local_xyz=seed['_xyz'])
    _require(set(saved)==set(references),'Unexpected R28 proxy recovery arrays.')
    for key, value in references.items():
        v=arr(saved[key])
        _require(v.dtype==np.float32 and np.isfinite(v).all(),f'Bad frozen R28 tensor {key}.')
        _float_match(v,value,key,atol=4e-7 if key!='owner_to_target' else 4e-6)
    support = np.zeros(32,bool)
    support[SOURCE_SLOTS] = True
    checks={}
    for key,threshold in (('target_plane','center_goal_m'),('target_sigma','normal_sigma_goal_m')):
        before,after = (np.abs(series[group][key]) for group in ('r22','r28'))
        good = before <= cfg['surface_schedule'][threshold]
        delta = np.maximum(after-before,0)
        native, held = delta[good&support[:,None]],delta[good&~support[:,None]]
        checks[key] = dict(initial_good_registered_pairs=len(native),initial_good_heldout_pairs=len(held),
            registered_max_increase_m=float(native.max(initial=0)),heldout_max_increase_m=float(held.max(initial=0)),
            registered_violations=int((native>1e-7).sum()),heldout_increase_over_tolerance=int((held>1e-7).sum()))
        _require(checks[key]['registered_violations']==0,'Initially-good registered proxy geometry exceeds hard envelope.')
    scopes={'all_32':np.ones((32,len(owner)),bool),'registered_28':np.broadcast_to(support[:,None],(32,len(owner))),
            'heldout_4':np.broadcast_to(~support[:,None],(32,len(owner))),
            'evidence_supported_only':np.zeros((32,len(owner)),bool)}
    scopes['evidence_supported_only'][SOURCE_SLOTS] = assoc['support_views'].T
    summary={}
    for scope,mask in scopes.items():
        summary[scope] = dict(row_pose_pairs=int(mask.sum()))
        for key in series['r22']:
            before,after=(np.abs(series[stage][key][mask])*1000. for stage in ('r22','r28'))
            summary[scope][key+'_mm']=dict(r22=quantiles(before),r28=quantiles(after),
                                          signed_change=quantiles(after-before))
    return dict(rows=len(owner),owner_rows=int((owner==target).sum()),neighbor_rows=int((owner!=target).sum()),
                initially_good_envelope=checks,measures=summary,
                scope='Independent double precision. Distances refer to frozen fitted mesh proxies. '
                      'Held-out poses are measurement-only; no claim of material ground truth or render quality.')


def color_contract(old, live, ids0, row0, seen, state, cfg, data):
    _require(same(data['stable_ids'],ids0) and same(data['owner_faces'],old['binding']),'Color source row IDs or owner changed.')
    guarded = data['source_color_guard']
    _require(guarded.dtype==np.bool_ and guarded.shape==ids0.shape and
             np.all(data['source_color_bins'][guarded]>=2),'Native color lacks repeated evidence.')
    _require(same(guarded[row0],arr(state['source_color_guarded'])) and np.all(seen[guarded[row0]]),
             'Color support changed during topology updates.')
    dc = old['_features_dc'][:,0]
    seed = dc*np.float32(C0)+np.float32(.5)
    low,high = data['source_rgb_low'],data['source_rgb_high']
    options=cfg['color_bounds']
    fallback = options['without_repeated_support']
    max_change = options['supported_max_change']
    lower,upper = np.clip(seed-fallback,0,1),np.clip(seed+fallback,0,1)
    lower[guarded] = np.maximum(np.minimum(low[guarded],seed[guarded]),seed[guarded]-max_change)
    upper[guarded] = np.minimum(np.maximum(high[guarded],seed[guarded]),seed[guarded]+max_change)
    lower,upper = ((a-.5)/np.float32(C0) for a in (lower,upper))
    current_lower,current_upper = arr(state['dc_lower'])[:,0],arr(state['dc_upper'])[:,0]
    _float_match(current_lower,lower[row0],'SH0 lower color cap',atol=5e-7)
    _float_match(current_upper,upper[row0],'SH0 upper color cap',atol=5e-7)
    current=live['_features_dc'][:,0]
    overflow=np.maximum(current-current_upper,current_lower-current).clip(0)
    _require(not np.any(overflow[seen]>1e-7),'Observed SH0 color exceeded its saved source bounds.')
    delta=np.max(np.abs((current-old['_features_dc'][row0,0])*C0),axis=1)
    return dict(repeated_source_rows=int(guarded[row0].sum()),fallback_observed_rows=int((seen&~guarded[row0]).sum()),
                color_bound_max_overflow=float(overflow[seen].max(initial=0)),
                repeated_source_rgb_max_abs_change=quantiles(delta[guarded[row0]]),
                fallback_rgb_max_abs_change=quantiles(delta[seen&~guarded[row0]]))


def recovery(base, state, cfg, n, observed, step):
    model = state['model']
    saved = model[12]
    _require(len(model)==14 and state['recovery_schema']=='shuolee-r28-optimizer-rng-topology/1' and
             np.isfinite(float(model[13])), 'Unexpected recovery state schema.')
    for index,shape in ((9,(n,)),(10,(n,1)),(11,(n,1))):
        v=arr(model[index])
        _require(v.shape==shape and np.isfinite(v).all() and np.all(v>=0),'Bad recovery raster statistics.')
    camera_bits,support=arr(state['seen_cameras']),arr(state['face_surface_view_support'])
    _require(camera_bits.shape==support.shape==(n,) and camera_bits.dtype==np.int64 and support.dtype==np.int32
             and np.all(camera_bits>=0) and np.all(camera_bits<(1<<24))
             and np.all(camera_bits[~observed]==0),'Recovery raster masks are misaligned.')
    bit_count=sum(((camera_bits>>i)&1) for i in range(24))
    _require(same(bit_count,support) and np.all((arr(model[11])[:,0]>0)==(support>0)),
             'Distinct recovery camera counts disagree with raster accumulators.')
    _require(set(saved)=={'state','param_groups'} and len(saved['param_groups'])==len(GROUP_NAMES),
             'Missing saved Adam groups.')
    params = {name:torch.nn.Parameter(model[i].detach().clone()) for i,name in enumerate(GROUP_NAMES,1)}
    opt = torch.optim.Adam([dict(name=grp['name'],params=[params[grp['name']]]) for grp in saved['param_groups']])
    opt.load_state_dict(saved)
    moments={}
    for group in saved['param_groups']:
        name=group['name']
        _require(len(group['params'])==1 and name in GROUP_NAMES and np.isfinite(group['lr']),
                 'Unexpected saved Adam parameter group.')
        original=saved['state'].get(group['params'][0])
        restored=opt.state.get(params[name])
        if name in ('opacity','f_rest'):
            _require(original is None and restored is None,'Frozen SH/opacity has live Adam moments.')
            moments[name]=dict(frozen=True)
            continue
        _require(original is not None and restored is not None and set(original)=={'step','exp_avg','exp_avg_sq'}
                 and int(original['step'])==int(restored['step'])==step,'Incorrect Adam update clock.')
        for k in ('exp_avg','exp_avg_sq'):
            v=arr(original[k])
            _require(v.shape==tuple(params[name].shape) and np.isfinite(v).all() and
                     torch.equal(original[k],restored[k]),f'Restored {name} Adam slot differs.')
            if k=='exp_avg_sq':
                _require(np.all(v>=0),'Negative Adam variance.')
        moments[name]=dict(step=step,shape=list(params[name].shape),nonzero_rows=int(np.any(
             arr(original['exp_avg']).reshape(n,-1)!=0,axis=1).sum()))
    rng = state['rng']
    _require(set(rng)=={'python','numpy','torch_cpu','torch_cuda'},'Missing saved RNG component.')
    r=random.Random(); r.setstate(rng['python'])
    a,keys,i,h,c=rng['numpy']
    rng_np=np.random.RandomState(); rng_np.set_state((a,arr(keys).astype(np.uint32),i,h,c))
    cpu=torch.Generator(device='cpu'); cpu.set_state(rng['torch_cpu'])
    _require(len(rng['torch_cuda'])>=1 and all(x.ndim==1 and x.dtype==torch.uint8 and x.numel()>0
             for x in rng['torch_cuda']),'CUDA RNG bytes missing.')
    rows=_read(base/'data/training-inputs.json')['records']
    buckets={b:[] for b in sorted(set(cfg['training_source_bins']))}
    for i in sorted(range(len(rows)),key=lambda index:rows[index]['source_frame_index']):
        buckets[cfg['training_source_bins'][i]].append(i)
    sampler=np.random.default_rng(cfg['seed'])

    def draws():
        for cycle in count():
            for b in sampler.permutation(list(buckets)):
                group=buckets[b]
                yield group[cycle%len(group)]

    replay=list(islice(draws(),step+8))
    _require(replay[:step]==state['draws'],'Saved sampler prefix fails deterministic reconstruction.')
    expected_aux = {f:sum(1 for j in range(step//8) if j%4==i)
                    for i,f in enumerate(cfg['auxiliary_frames'])}
    _require(state['auxiliary_updates']=={f:k for f,k in expected_aux.items() if k},'Auxiliary replay counters changed.')
    return dict(schema=state['recovery_schema'],adam_slots_reloaded_on_cpu=moments,
                distinct_camera_counts_verified=True,rows_with_source_gradient=int(np.sum(support>0)),
                rng=dict(python_loaded=True,numpy_loaded=True,torch_cpu_loaded=True,
                         torch_cuda_state_bytes=[x.numel() for x in rng['torch_cuda']]),
                exact_sampler_prefix=True,next_draw_slots=replay[step:],auxiliary_update_counts=expected_aux,
                scope='CPU snapshot and state reconstruction. CUDA RNG bytes were checked structurally; '
                      'an interrupted GPU job was not resumed.')


def audit(subject,mode,step,commit):
    base=LAB/f'shuolee-smpl-delivery-round22-{subject}'/'run'
    run=LAB/f'shuolee-smpl-delivery-round28-{subject}'/mode
    cfg_path=run/'gaussian-run/config.json'
    cfg=_read(cfg_path)
    snap=run/'gaussian-run'/f'raw-step-{step:06d}'
    raw=base/'gaussian-run/raw-step-018000'
    before_receipt=_read(raw/'state.json')
    receipt=_read(snap/'result.json')
    state_path=snap/'main-state.pt'
    _pin(state_path,receipt['state_sha256'])
    _pin(raw/'state.pt',before_receipt['state_sha256'])
    baseline=torch.load(raw/'state.pt',map_location='cpu',weights_only=True)['state']
    with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,np.dtypes.Float64DType]):
        state=torch.load(state_path,map_location='cpu',weights_only=True)
    _require(state['step']==receipt['step']==step and state['optimization']==cfg['optimization'] and
             state['pinned_inputs']==cfg['pinned_inputs'] and cfg['candidate']=='R28_face_surface',
             'R28 recovery bound to a different step or config.')
    material_path=OLD/f'{subject}-material-visibility.npz'
    surface_path=OUT/f'{subject}-surface-association.npz'
    color_path=OUT/f'{subject}-native-color-guard.npz'
    sources=dict(bound_ply=raw/'bound-local.ply',raw_state=raw/'state.pt',fit=base/'fit/fitted.npz',
        material=material_path,material_receipt=material_path.with_suffix('.json'),
        aux_replay=base/'gaussian-run/face-supplement/replay.json',
        marked_artifact_audit=WORKSPACE/f'outputs/smpl-delivery-round26/input-audit/{subject}-material-evidence.json',
        native_auxiliary_landmarks=base/'fit/joint-candidates.npz',surface_association=surface_path,
        surface_association_receipt=surface_path.with_suffix('.json'),native_color_guard=color_path,
        native_color_guard_receipt=color_path.with_suffix('.json'),
        color_geometry_diagnosis=OUT/'016-color-geometry-swap/result.json')
    _require(set(sources)==set(cfg['pinned_inputs']), 'Unexpected production source pins.')
    for key,path in sources.items():
        _pin(path,cfg['pinned_inputs'][key])
    material=npz(material_path)
    assoc=npz(surface_path)
    assoc_receipt=_read(surface_path.with_suffix('.json'))
    color=npz(color_path)
    color_receipt=_read(color_path.with_suffix('.json'))
    _require(assoc_receipt['holdout_source_images_read']==color_receipt['heldout_source_images_read']==0 and
             assoc['source_slots'].tolist()==SOURCE_SLOTS and
             set(color_receipt['source_frames'])==set(cfg['train_frames']+cfg['auxiliary_frames']),
             'Geometry/color evidence used held-out frames.')
    ids0=arr(baseline['model']['densification_ids']['tensor'])
    ids=arr(state['stable_ids'])
    old=ply_parameters(raw/'bound-local.ply')
    _require(all(same(old[key],arr(baseline['parameters'][key]['value']['tensor'])) for key in PARAMS)
             and same(old['binding'],arr(baseline['model']['binding']['tensor']))
             and same(old['binding'],material['owner_faces']) and same(ids0,material['stable_ids']),
             'R22 seed parameters or material identity differ.')
    model=state['model']
    live={k:arr(x) for k,x in zip(SCALE_KEYS,model[1:7],strict=True)}
    live['binding']=arr(model[7])
    exported=ply_parameters(snap/'drive/bound-local.ply')
    _require(all(same(exported[k],live[k]) for k in (*PARAMS,'binding')) and model[0]==0
             and len(ids)==receipt['points']==len(np.unique(ids)), 'Endpoint PLY and state do not match.')
    mapping,next_id,events,_=lineage(state['split_events'],ids0,baseline['model']['next_densification_id']['value'],material,cfg)
    _require(set(mapping)==set(ids.tolist()) and next_id==state['next_stable_id'],'Invalid endpoint stable ID clock.')
    origin=np.asarray([mapping[int(s)] for s in ids])
    originals=np.isin(ids,ids0)
    _require(same(arr(state['original_row_ids']),ids0[origin]) and same(live['binding'],old['binding'][origin])
             and same(arr(model[8]),np.bincount(live['binding'],minlength=len(model[8]))),'Binding ownership changed.')
    aliases=dict(labels='labels',spill='silhouette_spill',observed='observed',skin='geometry_eligible',
                 geometry_only='unknown_unobserved_elongated',possible_protected_material='possible_protected_material')
    for name,key in aliases.items():
        _require(same(arr(state[name]),material[key][origin]),f'Changed R28 material {name}.')
    labels,seen,skin,unseen,protected,associated=(arr(state[k]) for k in
        ('labels','observed','skin','geometry_only','possible_protected_material','associated'))
    fixed=~seen&~unseen
    _require(np.all(originals[~seen|protected|(labels>=2)]) and np.all(labels[unseen]==0) and
             not np.any(seen&unseen),'Protected material topology was modified.')
    for name in PARAMS:
        mask=originals if name in ('_opacity','_features_rest') else fixed if name=='_scaling' else ~seen
        _require(same(live[name][mask],old[name][origin[mask]]),f'Frozen parameter changed: {name}.')
    saved_assoc={key:arr(v) for key,v in state['surface_association'].items()}
    _require(set(saved_assoc)==set(assoc) and all(same(saved_assoc[k],assoc[k]) for k in assoc) and
             same(associated,np.isin(ids,assoc['stable_ids'])) and
             np.all(originals[associated]&skin[associated]&~protected[associated]) and
             not set(assoc['stable_ids'].tolist()) & {item['original_id'] for item in events},
             'Fixed skin association was promoted, split or remapped.')
    i=select(ids,assoc['stable_ids']); i0=select(ids0,assoc['stable_ids'])
    support=assoc['support_views']
    groups=np.stack([support[:,assoc['source_bins']==b].any(1) for b in np.unique(assoc['source_bins'])]).sum(0)
    _require(same(old['binding'][i0],assoc['owner_faces']) and np.all(groups>=2) and
             np.all(assoc['barycentric']>=0) and np.allclose(assoc['barycentric'].sum(1),1),'Unsigned mesh proxy association.')
    delta=live['_scaling'][unseen].astype(np.float64)-old['_scaling'][origin[unseen]].astype(np.float64)
    major=old['_scaling'][origin[unseen]].argmax(1)
    ratios=np.exp(delta[np.arange(len(major)),major])
    _require(np.all(delta[np.arange(3)[None]!=major[:,None]]==0) and
             np.all(ratios>=cfg['unseen_major_scale_floor']-4e-7) and np.all(ratios<=1.+2e-7),
             'Geometry-only row exceeded one original axis envelope.')
    motion,meta,pose=poses_and_cameras(base,snap,cfg,receipt)
    _require(same(motion['stable_ids'],ids) and same(motion['labels'],labels) and
             same(motion['spill'],arr(state['spill'])) and state['reference_pose_index']==meta['reference_pose_index'],
             'Exported drive rows or recovery reference pose mismatch.')
    for view,slot in zip(assoc_receipt['view_receipts'],SOURCE_SLOTS,strict=True):
        expected=meta['cameras'][slot]
        _require(view['pose_index']==slot and all(same(view[k],expected[k]) for k in ('frame','K','w2c'))
                 and len(view['mesh_proxy_depth_sha256'])==64 and len(view['material_grid_sha256'])==64,
                 'Association signed crop does not match exact native camera.')
    proxy=proxy_geometry(old,live,i0,i,assoc,state['surface_geometry_reference'],motion,cfg)
    other=np.flatnonzero(skin&~associated)
    _,owner_cap,_=geometry_measures(old,live,live['binding'][other],motion,other,origin[other])
    colors=color_contract(old,live,ids0,origin,seen,state,cfg,color)
    recover=recovery(base,state,cfg,len(ids),seen,step)
    gradients=run/'gaussian-run/native-gradient-audit.json'
    _pin(gradients,state['frozen_and_geometry']['native_gradient_audit_sha256'])
    return dict(schema='shuolee-r28-independent-face-invariants/1',subject=subject,step=step,mode=mode,
        formal_commit=commit,script_sha256=_sha(Path(__file__)),checkpoint_sha256=receipt['state_sha256'],
        endpoint_result_sha256=_sha(snap/'result.json'),config_sha256=_sha(cfg_path),pass_=True,
        source_pins=cfg['pinned_inputs'],raw_points=len(ids),baseline_points=len(ids0),frozen_irrelevant_rows=int(fixed.sum()),
        geometry_only=dict(rows=int(unseen.sum()),modified_rows=int((delta!=0).any(1).sum()),
            original_major_scale_ratio=quantiles(ratios),max_reduction_fraction=float((1-ratios).max(initial=0)),
            scope='Only the original major width, without native photometric support.'),
        material_split=dict(protected_ambiguous_rows=int(protected.sum()),known_non_skin_rows=int((labels>=2).sum()),
            committed_parent_counts_by_label={str(k):sum(x['label']==k for x in events) for k in (0,1)},events=events),
        unchanged_ordinary_skin=dict(rows=len(other),original_owner_hard_cap=owner_cap),
        associated_skin=proxy,native_source_color=colors,recovery=recover,frozen_pose_and_camera=pose,
        native_gradient_audit_sha256=_sha(gradients),quality_accepted=False),state['draws'],state['split_events']


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    parser.add_argument('--steps',type=int,nargs='+',default=(10000,18000))
    parser.add_argument('--mode',default='run')
    parser.add_argument('--commit',default=None)
    parser.add_argument('--no-write',action='store_true')
    args=parser.parse_args()
    _require(args.no_write or (args.commit is not None and len(args.commit)==40),'Pin formal source commit.')
    reports=[]
    for step in args.steps:
        r,draws,events=audit(args.subject,args.mode,step,args.commit)
        path=OUT/f'{args.subject}-face-invariants-{step:06d}.json'
        if not args.no_write:
            save(path,r)
        reports.append((r,draws,events,path))
        print(json.dumps(dict(subject=args.subject,step=step,rows=r['raw_points'],frozen=r['frozen_irrelevant_rows'],
            associated=r['associated_skin']['rows'],caps=r['associated_skin']['initially_good_envelope'],
            color=r['native_source_color'],commits=r['material_split']['committed_parent_counts_by_label'])),flush=True)
    if len(reports)==2 and not args.no_write:
        first,second=reports
        _require(second[1][:len(first[1])]==first[1] and second[2][:len(first[2])]==first[2],
                 'R28 endpoints do not share exact draw and split prefixes.')
        save(OUT/f'{args.subject}-face-invariants-paired.json',dict(schema='shuolee-r28-independent-face-invariants-paired/1',
            subject=args.subject,formal_commit=args.commit,matching_draw_prefix=True,matching_split_prefix=True,
            prefix_steps=len(first[1]),formal=[dict(step=r['step'],checkpoint_sha256=r['checkpoint_sha256'],
                independent_receipt_sha256=_sha(path)) for r,_,_,path in reports]))


if __name__=='__main__':
    main()

