"""Build native training-only skin associations and periocular surface seed evidence."""

import argparse
from dataclasses import fields as dataclass_fields
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import torch

from build_round26_native_face_material import MATERIAL_REVIEW, array_sha, local_hull, native_weights
from build_round27_material_visibility import OUT as R27, decode
from build_round28_legacy_native_evidence import detector_geometry, original_model, pose_arrays
from build_round28_native_color_guard import intervals, source_inputs
from prepare_round28_legacy_source import OUT as R28, descriptor, write_json
from trace_round22_face_pixels import (DELIVERY, REVIEW_SHA, checked, code, load_subject,
                                      material_mask, need, pin, posed_model, preprocess, read, sha)


OUT = DELIVERY.with_name('smpl-delivery-round29')/'input-audit'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r29')
SUBJECTS = ('001', '005', '006', '008', '016')
POLICY = dict(min_skin_mass=.35, min_skin_fraction=.30, min_skin_bins=2,
              min_pooled_skin_over_known=.82, material_veto_mass=.55, material_veto_fraction=.16,
              deficit_alpha=.94, max_seed_rgb_spread=.19,
              max_seeds=48, seed_spacing_m=.0015, seed_source_limit=8,
              min_seed_sigma_m=.00065, max_seed_sigma_m=.0021)


def old_inputs(subject):
    legacy_path = R28/f'{subject}-legacy-source.json'
    material_path = (R28 if legacy_path.is_file() else R27)/f'{subject}-material-visibility.json'
    material = read(material_path)
    data_path = pin(material_path.with_suffix('.npz'), material['material_visibility_npz_sha256'])
    with np.load(data_path, allow_pickle=False) as data:
        arrays = dict(data)
    assoc_path = R28/f'{subject}-surface-association.npz'
    assoc_receipt = read(assoc_path.with_suffix('.json'))
    pin(assoc_path, assoc_receipt['association_npz_sha256'])
    need(assoc_receipt['material_receipt_sha256']==sha(material_path)
         and material['heldout_source_images_read']==0, 'R28 signed skin and association inputs differ.')
    with np.load(assoc_path, allow_pickle=False) as data:
        association = dict(data)
    return legacy_path, material_path, material, arrays, assoc_path, association


def legacy_inputs(path, material, association):
    source = read(path)
    model, _, owner, appearance = original_model(source)
    names = [row['frame'] for row in source['selected_records']]
    reserved = source.get('reserved_holdouts',[row['frame'] for row in source['heldout_records']])
    need(names==material['classification_frames'] and not source['auxiliary_records']
         and set(names).isdisjoint(reserved), 'Changed legacy source roles.')
    need(association['source_slots'].tolist()==list(range(len(names))), 'Changed legacy pose slots.')

    def records():
        with np.load(checked(material['native_skin_grid']), allow_pickle=False) as grids:
            need(grids['frame_ids'].tolist()==names, 'Signed legacy skin grid is out of order.')
            for slot, ((row, obs, proj, _), signed) in enumerate(zip(detector_geometry(source),material['views'],strict=True)):
                box = tuple(map(int,grids['boxes'][slot]))
                grid = grids[f'grid_{slot:03d}']
                need(row['frame']==signed['frame'] and array_sha(grid)==signed['label_mask_sha256']
                     and list(box)==signed['box'], 'Signed legacy grid differs from original observations.')
                yield SimpleNamespace(slot=slot,row=row,signed=signed,box=box,grid=grid,
                    obs=obs,proj=proj,annotation={},allowed=None,
                    rgb=decode(pin(row['image'],row['source_sha256']),box).astype(np.float32)/255.,
                    truth=decode(checked(row['baseline_rgb']),box).astype(np.int16),alpha=None,
                    pre=preprocess(pose_arrays(model,appearance,slot),row),
                    vertices=model.vertices_sequence[slot].numpy())

    return model, owner, checked(source['asset']), source['reference_pose_index'], records()


def r22_inputs(subject, material, association):
    context = load_subject(subject,read(pin(DELIVERY/'review-receipt.json',REVIEW_SHA)))
    model, slots = posed_model(context)
    root = Path(context['arm']['run'])
    pose = context['plan']['poses']['r22']
    proof = pose['proof']
    admission = read(checked(proof['supplement_admission']))
    main = read(pin(root/'data/training-inputs.json',material['training_inputs_sha256']))['records']
    aux = pose['supplement']['frames']
    aux_names = [row['frame'] for row in aux]
    aux_inputs = {row['frame']:row for row in read(pin(root/'data/inputs.json',
                 admission['input_sha256']['native_inputs']))['records']}
    annotations = read(pin(DELIVERY/'input-audit/face-pixel-attribution.json',MATERIAL_REVIEW))['subjects'][subject]['native_annotation']
    held_root = root/'gaussian-run/step-018000/heldout-replay'
    held_receipt = read(held_root/'result.json')
    with np.load(pin(held_root/held_receipt['mesh_file'],held_receipt['mesh_sha256']),allow_pickle=False) as data:
        need(data['frame_ids'].tolist()==material['retained_holdouts'], 'Changed reserved R22 pose order.')
        held_vertices = torch.from_numpy(data['vertices_world'])
    with np.load(checked(proof['supplement_poses']),allow_pickle=False) as data:
        need(data['frame_ids'].tolist()==aux_names and np.array_equal(data['faces'],model.faces.numpy()),
             'Changed R22 auxiliary pose order.')
        aux_vertices = torch.from_numpy(data['vertices_world'])
    model.vertices_sequence = torch.cat((model.vertices_sequence,held_vertices,aux_vertices))
    with np.load(pin(root/'fit/fitted.npz',material['fit_sha256']),allow_pickle=False) as data:
        reference = slots[data['reference_frame_id'].item()]
        observations = {name:(data['observed_native_keypoints'][i],data['joints_projected_native'][i])
                        for i,name in enumerate(slots)}
    with np.load(pin(root/'fit/joint-candidates.npz',admission['input_sha256']['joint_candidates']),allow_pickle=False) as data:
        names = data['frame_ids'].tolist()
        observations.update({name:(data['observed_native_keypoints'][names.index(name)],
                                   data['joints_projected_native'][names.index(name)]) for name in aux_names})
    expected_slots = list(range(len(main)))+list(range(len(main)+len(held_vertices),len(model.vertices_sequence)))
    need(material['classification_frames']==list(slots)+aux_names
         and association['source_slots'].tolist()==expected_slots
         and set(material['classification_frames']).isdisjoint(material['retained_holdouts']), 'Changed R22 source roles.')
    aux_by_frame = {row['frame']:row for row in aux}
    source_by_frame = {row['frame']:row for row in main}|{name:aux_inputs[name] for name in aux_names}

    def records():
        for slot, (signed, pre, grid, _, rgb, truth, alpha) in zip(expected_slots,
                    source_inputs(context,subject,main,pose,material),strict=True):
            name = signed['frame']
            row = source_by_frame[name]
            saved_aux = aux_by_frame.get(name)
            allowed = None if saved_aux is None else decode(pin(saved_aux['valid_path'],saved_aux['valid_mask_sha256']),mode='L')>0
            obs,proj = observations[name]
            if saved_aux is not None:
                row = {**row,'K':saved_aux['camera']['K'],'w2c':saved_aux['camera']['w2c']}
            yield SimpleNamespace(slot=slot,row=row,signed=signed,box=tuple(signed['box']),grid=grid,
                obs=obs,proj=proj,annotation=annotations.get(name,{}),allowed=allowed,rgb=rgb,truth=truth,
                alpha=alpha,pre=pre,vertices=model.vertices_sequence[slot].numpy())

    return model, context['owner'], Path(context['plan']['asset']), reference, records()


def regions(record):
    from shuolee_portrait.face_supplement import REGIONS

    grid, obs, proj, box = record.grid,record.obs,record.proj,record.box
    skin = grid==1
    excluded = np.isin(grid,[2,3,4,5])
    eye_skin = np.zeros_like(skin)
    rgb = record.rgb
    reliable = np.isfinite(obs).all(1)&np.isfinite(proj).all(1)&(obs[:,2]>=.50)
    face = obs[67:118,:2][reliable[67:118]]
    if len(face)<14 or skin.sum()<12 or (span:=np.ptp(face,axis=0).max())<24.:
        return skin,eye_skin,excluded,rgb
    paired = reliable&(np.linalg.norm(obs[:,:2]-proj,axis=1)<=min(6.,.08*span))
    core = np.flatnonzero(paired[67:118])+67
    nose = REGIONS['nose'][paired[REGIONS['nose']]]
    mouth = REGIONS['mouth'][reliable[REGIONS['mouth']]]
    if len(core)<14 or len(nose)<4 or len(mouth)<8:
        return skin,eye_skin,excluded,rgb
    x0,y0,x1,y1 = box
    matte = decode(pin(record.row['mask'],record.row['mask_sha256']),box,'L')
    clear = binary_erosion(matte>=252,iterations=1)
    if record.allowed is not None:
        clear &= record.allowed
    for l,t,r,b in record.row.get('other_person_boxes_xyxy',()):
        left,top,right,bottom = max(x0,int(l)),max(y0,int(t)),min(x1,int(r)),min(y1,int(b))
        if left<right and top<bottom:
            clear[top-y0:bottom-y0,left-x0:right-x0] = False
    for name in ('hair','glasses_rim'):
        signed = record.annotation.get('masks',{}).get(name,{})
        if signed.get('validated'):
            full = material_mask(record.row['wh'],signed['polygons'])
            need(array_sha(full)==signed['full_roi_sha256'], 'Changed signed protected material polygon.')
            excluded |= full[y0:y1,x0:x1]
    margin = max(1,round(.025*span))
    eye_band = np.zeros_like(skin)
    eye_aperture = np.zeros_like(skin)
    for key in ('right_eye','left_eye'):
        ids = REGIONS[key][paired[REGIONS[key]]]
        if len(ids)<5:
            continue
        observed_eye = local_hull(obs[ids,:2],box)
        fitted_eye = local_hull(proj[ids],box)
        if not (observed_eye&fitted_eye).any():
            continue
        aperture = observed_eye|fitted_eye
        eye_aperture |= binary_dilation(aperture,iterations=margin)
        width = np.ptp(obs[ids,:2],axis=0).max()
        outer = binary_dilation(aperture,iterations=max(margin+2,min(18,round(.65*width))))
        eye_band |= outer
    lip = local_hull(obs[mouth,:2],box)|local_hull(proj[mouth],box)
    excluded |= eye_aperture|binary_dilation(lip,iterations=margin)
    skin &= ~excluded
    if skin.sum()<12:
        return skin,eye_skin,excluded,rgb
    paired_face = binary_erosion(local_hull(obs[core,:2],box)&local_hull(proj[core],box),iterations=1)
    luma = rgb@np.asarray([.2126,.7152,.0722],np.float32)
    chroma = rgb[...,[0,2]]/rgb.sum(-1,keepdims=True).clip(1e-5)
    yy,xx = np.mgrid[y0:y1,x0:x1]
    center = ((obs[nose,:2]+proj[nose])*.5).mean(0)
    palette = skin&((xx-center[0])**2+(yy-center[1])**2<=max(6.,.24*span)**2)
    if palette.sum()<12:
        palette = skin
    representative = np.median(chroma[palette],axis=0)
    difference = np.linalg.norm(chroma-representative,axis=-1)
    luma_anchor = np.median(luma[palette])
    chroma_limit = np.clip(3.*np.median(difference[palette]),.024,.052)
    color_ok = (difference<=chroma_limit)&(luma>=max(.15,.62*luma_anchor))&(luma<=min(.97,1.36*luma_anchor))
    supported = binary_erosion(paired_face&clear&color_ok&~excluded,iterations=1)
    eye_skin = supported&eye_band
    skin |= supported
    return skin,eye_skin,excluded,rgb


def composite(record, skin, excluded, rgb, lookup):
    """Accumulate real alpha contributions and source RGB only where native replay matches."""
    x0,y0,_,_ = record.box
    alpha_map = np.full(skin.shape,np.nan,dtype=np.float32)
    count = int(lookup.max())+1
    total, mass, protected = np.zeros((3,count),np.float64)
    color_mass = np.zeros(count,np.float64)
    color = np.zeros((count,3),np.float64)
    square = np.zeros_like(color)
    color_count = np.zeros(count,np.int64)
    sampled = skin|excluded
    yy,xx = np.nonzero(sampled)
    if not len(xx):
        return alpha_map,(total,mass,protected,color_mass,color,square,color_count),0
    xx += x0
    yy += y0
    tiles = np.unique(np.column_stack((xx//16,yy//16)),axis=0)
    failed = 0
    for tx,ty in tiles:
        keep = (xx//16==tx)&(yy//16==ty)
        x,y = xx[keep],yy[keep]
        local_x,local_y = x-x0,y-y0
        rows,weights = native_weights(record.pre,(tx,ty),np.column_stack((x,y)))
        opacity = weights.sum(1)
        error = np.abs(code(weights@record.pre['rgb'][rows])-record.truth[local_y,local_x]).max(1)
        if record.alpha is not None:
            (left,top,right,bottom),reference = record.alpha
            valid = (x>=left)&(x<right)&(y>=top)&(y<bottom)
            delta = np.abs(code(opacity[valid])-reference[y[valid]-top,x[valid]-left])
            error[valid] = np.maximum(error[valid],delta)
        ok = error<=1
        failed += int((~ok).sum())
        alpha_map[local_y[ok],local_x[ok]] = opacity[ok]
        head = lookup[rows]>=0
        ids = lookup[rows[head]]
        weights = weights[:,head]
        on_skin = ok&skin[local_y,local_x]
        on_excluded = ok&excluded[local_y,local_x]
        total[ids] += weights[ok].sum(0,dtype=np.float64)
        mass[ids] += weights[on_skin].sum(0,dtype=np.float64)
        protected[ids] += weights[on_excluded].sum(0,dtype=np.float64)
        reliable_color = on_skin[:,None]&(weights>=.04)&(weights>=.18*opacity[:,None])
        weighted = np.where(reliable_color,weights,0.).astype(np.float64)
        source = rgb[local_y,local_x]
        color_mass[ids] += weighted.sum(0)
        color[ids] += weighted.T@source
        square[ids] += weighted.T@(source*source)
        color_count[ids] += reliable_color.sum(0)
    need(failed/len(xx)<.01, 'Native alpha/RGB parity failed on signed R29 source pixels.')
    return alpha_map,(total,mass,protected,color_mass,color,square,color_count),failed


def raster_surface(vertices, faces, row, box):
    """Nearest projected triangle with perspective-correct depth and barycentrics."""
    x0,y0,x1,y1 = box
    camera = np.asarray(row['w2c'])
    K = np.asarray(row['K'])
    view_vertices = vertices@camera[:3,:3].T+camera[:3,3]
    need(np.all(view_vertices[:,2]>.2), 'R29 proxy crosses the near camera plane.')
    homogeneous = view_vertices@K.T
    xy = homogeneous[:,:2]/homogeneous[:,2:]-0.5
    triangles = xy[faces]
    low = np.maximum(np.ceil(triangles.min(1)-1e-5),[x0,y0]).astype(np.int64)
    high = np.minimum(np.floor(triangles.max(1)+1e-5),[x1-1,y1-1]).astype(np.int64)
    depth = np.full((y1-y0,x1-x0),np.inf,dtype=np.float32)
    owner = np.full(depth.shape,-1,dtype=np.int32)
    barycentric = np.zeros((*depth.shape,3),dtype=np.float32)
    inverse_depth = 1./view_vertices[faces,2]
    for face in np.flatnonzero((low<=high).all(1)):
        tri = triangles[face]
        edge_a,edge_b = tri[1:]-tri[0]
        det = edge_a[0]*edge_b[1]-edge_a[1]*edge_b[0]
        if abs(det)<1e-9:
            continue
        left,top = low[face]
        right,bottom = high[face]+1
        yy,xx = np.mgrid[top:bottom,left:right]
        dx,dy = xx-tri[0,0],yy-tri[0,1]
        b = (dx*edge_b[1]-dy*edge_b[0])/det
        c = (dy*edge_a[0]-dx*edge_a[1])/det
        screen = np.stack((1-b-c,b,c),axis=-1)
        weights = np.maximum(screen,0.)*inverse_depth[face]
        perspective = weights.sum(-1)
        y,x = yy-y0,xx-x0
        nearest = (screen>=-1e-5).all(-1)&(1./perspective<depth[y,x])
        y,x = y[nearest],x[nearest]
        depth[y,x] = 1./perspective[nearest]
        owner[y,x] = face
        barycentric[y,x] = weights[nearest]/perspective[nearest,None]
    return depth,owner,barycentric


def seed_candidates(alpha, eye_skin, depth, owner, barycentric, skin_faces):
    candidates = eye_skin&(owner>=0)&skin_faces[owner.clip(0)]&np.isfinite(depth)&(alpha<POLICY['deficit_alpha'])
    y,x = np.nonzero(candidates)
    order = np.argsort(alpha[y,x],kind='stable')
    x,y = x[order],y[order]
    _,first = np.unique(np.column_stack((x//3,y//3)),axis=0,return_index=True)
    x,y = x[first],y[first]
    return owner[y,x].astype(np.int64),barycentric[y,x]


def seed_evidence(packets, views, vertices, faces, reference, proposed_faces, proposed_bary):
    from shuolee_portrait.face_surface_association_r28 import _normals

    count,view_count = len(proposed_faces),len(views)
    support = np.zeros((count,view_count),bool)
    conflicts = np.zeros(count,bool)
    opacity = np.ones((count,view_count),np.float32)
    pixels = np.full((count,view_count,2),-1,dtype=np.int32)
    colors = np.zeros((count,view_count,3),np.float32)
    pixel_size = np.zeros_like(opacity)
    for column, (packet,view) in enumerate(zip(packets,views,strict=True)):
        triangles = vertices[view.slot,faces[proposed_faces]]
        world = np.einsum('ni,nij->nj',proposed_bary,triangles)
        camera = world@view.w2c[:3,:3].T+view.w2c[:3,3]
        homogeneous = camera@view.K.T
        xy = np.floor(homogeneous[:,:2]/homogeneous[:,2:]).astype(np.int64)
        x0,y0,x1,y1 = view.box
        x,y = (xy[:,0]-x0).clip(0,x1-x0-1),(xy[:,1]-y0).clip(0,y1-y0-1)
        in_view = ((xy>=[x0,y0])&(xy<[x1,y1])).all(1)&(camera[:,2]>.2)
        direction = -view.w2c[:3,:3].T@view.w2c[:3,3]-world
        facing = (_normals(triangles)*direction).sum(1)>=.32*np.linalg.norm(direction,axis=1)
        visible = in_view&facing&(np.abs(camera[:,2]-view.mesh_depth[y,x])<=.004)
        conflicts |= visible&view.excluded[y,x]
        supported = visible&view.skin[y,x]&np.isfinite(packet.alpha[y,x])
        support[:,column] = supported
        pixels[supported,column] = xy[supported]
        opacity[supported,column] = packet.alpha[y[supported],x[supported]]
        colors[supported,column] = packet.rgb[y[supported],x[supported]]
        pixel_size[supported,column] = camera[supported,2]/min(view.K[0,0],view.K[1,1])
    bins = np.asarray([view.source_bin for view in views],np.int64)
    slots = np.asarray([view.slot for view in views],np.int64)
    unique_bins = np.unique(bins)
    independent = np.stack([support[:,bins==source].any(1) for source in unique_bins]).sum(0)
    deficit_views = support&(opacity<POLICY['deficit_alpha'])
    admitted = (independent>=2)&~conflicts&deficit_views.any(1)
    reference_xyz = np.einsum('ni,nij->nj',proposed_bary,vertices[reference,faces[proposed_faces]])
    score = np.where(support,1-opacity,0.).sum(1)/np.maximum(1,support.sum(1))
    ranked = np.flatnonzero(admitted)
    ranked = ranked[np.argsort(-score[ranked],kind='stable')]
    picked, picked_rgb, picked_sigma, picked_columns = [],[],[],[]
    for index in ranked:
        if picked and np.linalg.norm(reference_xyz[picked]-reference_xyz[index],axis=1).min()<POLICY['seed_spacing_m']:
            continue
        columns = np.flatnonzero(support[index])
        medians = np.stack([np.median(colors[index,columns[bins[columns]==source]],axis=0)
                            for source in np.unique(bins[columns])])
        if np.ptp(medians,axis=0).max()>POLICY['max_seed_rgb_spread']:
            continue
        candidates = [min(columns[bins[columns]==source],key=lambda col:opacity[index,col])
                      for source in np.unique(bins[columns])]
        retained = sorted(candidates,key=lambda col:opacity[index,col])[:POLICY['seed_source_limit']]
        picked.append(index)
        picked_rgb.append(np.median(medians,axis=0))
        picked_sigma.append(np.clip(1.3*np.median(pixel_size[index,columns]),
                                    POLICY['min_seed_sigma_m'],POLICY['max_seed_sigma_m']))
        picked_columns.append(retained)
        if len(picked)>=POLICY['max_seeds']:
            break
    picked = np.asarray(picked,np.int64)
    seed_slots = np.full((len(picked),POLICY['seed_source_limit']),-1,np.int64)
    seed_bins = np.full_like(seed_slots,-1)
    for index, columns in enumerate(picked_columns):
        seed_slots[index,:len(columns)] = slots[columns]
        seed_bins[index,:len(columns)] = bins[columns]
    return dict(seed_faces=proposed_faces[picked],seed_barycentric=proposed_bary[picked],
                seed_rgb=np.asarray(picked_rgb,np.float32).reshape(-1,3),seed_sigma_m=np.asarray(picked_sigma,np.float32),
                seed_support_views=support[picked],seed_source_slots=seed_slots,seed_source_bins=seed_bins,
                seed_pixels_xy=pixels[picked],
                seed_deficit=score[picked].astype(np.float32))


def run(subject):
    sys.path[:0] = [str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    from shuolee_portrait.face_surface_association_r28 import SurfaceAssociation, SurfaceView, build_surface_association

    legacy_path, material_path, material, original, prior_path, prior = old_inputs(subject)
    loader = legacy_inputs if legacy_path.is_file() else r22_inputs
    model,owner,asset,reference,records = loader(legacy_path if legacy_path.is_file() else subject,material,prior)
    faces,vertices = model.faces.numpy(),model.vertices_sequence.numpy()
    stable = original['stable_ids']
    need(np.array_equal(owner,original['owner_faces']), 'Changed baseline Gaussian binding order.')
    with np.load(asset,allow_pickle=False) as data:
        need(np.array_equal(data['f'],faces), 'Changed frozen face topology.')
        dominant = data['weights'].argmax(-1)
    # Eye globes are controlled by the eye joints, outside the facial skin patch.
    skin_faces = ~np.isin(dominant[faces],[23,24]).any(1)
    head = np.flatnonzero(original['head_scope'])
    lookup = np.full(len(stable),-1,np.int64)
    lookup[head] = np.arange(len(head))
    bins = prior['source_bins']
    source_bins = np.unique(bins)
    shape = (len(source_bins),len(head))
    color_mass = np.zeros(shape,np.float64)
    color = np.zeros((*shape,3),np.float64)
    square = np.zeros_like(color)
    counts = np.zeros(shape,np.int64)
    packets, views, pins, contributions, candidate_faces, candidate_bary = [],[],[],[],[],[]
    for column,record in enumerate(records):
        skin,eye_skin,excluded,rgb = regions(record)
        alpha,summary,failed = composite(record,skin,excluded,rgb,lookup)
        depth,hit,bary = raster_surface(record.vertices,faces,record.row,record.box)
        found_faces,found_bary = seed_candidates(alpha,eye_skin,depth,hit,bary,skin_faces)
        candidate_faces.append(found_faces)
        candidate_bary.append(found_bary)
        total,mass,protected,c_mass,c_sum,c_sq,c_count = summary
        contributions.append(np.stack((total,mass,protected),axis=1))
        source_bin = record.signed['source_bin']
        need(record.slot==prior['source_slots'][column] and source_bin==bins[column],
             'Expanded native source order differs from the R28 signed roster.')
        angular = int(np.searchsorted(source_bins,source_bin))
        color_mass[angular] += c_mass
        color[angular] += c_sum
        square[angular] += c_sq
        counts[angular] += c_count
        row = record.row
        views.append(SurfaceView(record.slot,source_bin,np.asarray(row['K']),np.asarray(row['w2c']),
                                 record.box,skin,depth,excluded,row['frame']))
        packets.append(SimpleNamespace(alpha=alpha,rgb=rgb))
        pins.append(dict(frame=row['frame'],source_slot=record.slot,source_bin=source_bin,
                         source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],
                         expanded_skin_sha256=array_sha(skin),protected_sha256=array_sha(excluded),
                         native_render_sha256=record.signed['native_render_sha256'],
                         skin_pixels=int(skin.sum()),periocular_pixels=int(eye_skin.sum()),
                         candidate_seed_pixels=len(found_faces),excluded_parity_pixels=failed))
    need([p['frame'] for p in pins]==material['classification_frames'], 'Changed material source roles.')
    total,mass,protected = np.moveaxis(np.stack(contributions),2,0)
    visible = (mass>=POLICY['min_skin_mass'])&(mass>=POLICY['min_skin_fraction']*total)
    votes = np.stack([visible[bins==source].any(0) for source in source_bins]).sum(0)
    protected_veto = ((protected>=POLICY['material_veto_mass'])&
                      (protected>=POLICY['material_veto_fraction']*total)).any(0)
    known_skin = mass.sum(0)>=POLICY['min_pooled_skin_over_known']*total.sum(0)
    eligible = (votes>=POLICY['min_skin_bins'])&known_skin&~protected_veto&skin_faces[owner[head]]
    eligible &= ((original['labels'][head]<=1)&~original['possible_protected_material'][head]
                 &~original['silhouette_spill'][head])
    eligible &= ~np.isin(stable[head],prior['stable_ids'])
    selected = head[eligible]
    centers = []
    with torch.no_grad():
        for slot in range(len(vertices)):
            model.select_mesh_by_timestep(slot)
            centers.append(model.get_xyz[selected].numpy().copy())
    if len(selected):
        fresh,_ = build_surface_association(vertices,faces,stable[selected],owner[selected],np.stack(centers),
                                           np.ones(len(selected),bool),views,reference)
        fresh = fresh.arrays()
        source_rows = lookup[np.searchsorted(stable,fresh['stable_ids'])]
        support = fresh['support_views']&visible[:,source_rows].T
        certified = np.stack([support[:,bins==source].any(1) for source in source_bins]).sum(0)>=2
        certified &= skin_faces[fresh['target_faces']]
        fresh['support_views'] = support
        fresh = {key:value[certified] if key not in ('source_slots','source_bins') else value
                 for key,value in fresh.items()}
    else:
        fresh = {key:value[:0] if key not in ('source_slots','source_bins') else value for key,value in prior.items()}
    order = np.argsort(np.r_[prior['stable_ids'],fresh['stable_ids']])
    combined = {key:np.concatenate((prior[key],fresh[key]))[order] if key not in ('source_slots','source_bins') else prior[key]
                for key in prior}
    SurfaceAssociation.from_arrays({field.name:combined[field.name] for field in dataclass_fields(SurfaceAssociation)})
    expanded_rows = lookup[np.searchsorted(stable,fresh['stable_ids'])]
    low,high,_,_,valid,_ = intervals(color_mass[:,expanded_rows],color[:,expanded_rows],
                                    square[:,expanded_rows],counts[:,expanded_rows])
    seeds = seed_evidence(packets,views,vertices,faces,reference,np.concatenate(candidate_faces),np.concatenate(candidate_bary))
    OUT.mkdir(parents=True,exist_ok=True)
    target = OUT/f'{subject}-native-coverage.npz'
    receipt_path = target.with_suffix('.json')
    need(not target.exists() and not receipt_path.exists(), 'Preserve the existing signed R29 coverage evidence.')
    with target.open('xb') as stream:
        np.savez_compressed(stream,**combined,expanded_ids=fresh['stable_ids'],
                            expanded_rgb_lower=low,expanded_rgb_upper=high,expanded_rgb_valid=valid,**seeds)
    previous_rows = lookup[np.searchsorted(stable,prior['stable_ids'])]
    summary = dict(prior_associated_rows=len(previous_rows),repeated_native_skin_candidates=len(selected),
        new_associated_rows=len(expanded_rows),new_rows_with_rgb_bounds=int(valid.sum()),eye_seeds=len(seeds['seed_faces']),
        signed_skin_alpha_mass=float(mass.sum()),prior_associated_skin_alpha_mass=float(mass[:,previous_rows].sum()),
        new_associated_skin_alpha_mass=float(mass[:,expanded_rows].sum()))
    receipt = dict(schema='shuolee-r29-native-coverage/1',subject=subject,archive_sha256=sha(target),
        material_receipt_sha256=sha(material_path),prior_association_sha256=sha(prior_path),
        baseline_bound_ply_sha256=material['baseline_bound_ply_sha256'],
        baseline_raw_ply_sha256=material['baseline_raw_ply_sha256'],
        training_frames=material['selected_frames'],auxiliary_frames=material['admitted_auxiliary_frames'],
        retained_holdouts=material['retained_holdouts'],reserved_holdouts=material.get('reserved_holdouts',material['retained_holdouts']),
        expanded_rows=len(expanded_rows),seed_count=len(seeds['seed_faces']),
        heldout_source_images_read=0,policy=POLICY,views=pins,summary=summary,source_sha256=sha(__file__))
    write_json(receipt_path,receipt)
    print(dict(subject=subject,archive=descriptor(target),receipt=descriptor(receipt_path),**summary),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=SUBJECTS)
    run(parser.parse_args().subject)
