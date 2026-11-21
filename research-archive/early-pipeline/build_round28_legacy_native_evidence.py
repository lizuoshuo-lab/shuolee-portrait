"""Build the normal R27 material and R28 color inputs for a verified legacy source."""

import argparse
import sys

import numpy as np
import torch

from build_round26_native_face_material import LABELS, array_sha, head_crop, protected_faces
from build_round27_material_visibility import POLICY as MATERIAL_POLICY, classify, compositor, decode, elongated, source_grid
from build_round28_native_color_guard import ARRAYS as COLOR_ARRAYS, POLICY as COLOR_POLICY, intervals, measure
from prepare_round28_legacy_source import OUT, PROJECT, descriptor, write_json
from trace_round22_face_pixels import checked, fields, geometry, need, numbered, pin, ply, preprocess, read, sha


def original_model(source):
    from scene.smplx_gaussian_model import SMPLXGaussianModel

    local=ply(checked(source['local_ply']))
    world=ply(checked(source['world_ply']))
    model=SMPLXGaussianModel(0)
    model._xyz=torch.from_numpy(fields(local,'xyz'))
    model._scaling=torch.from_numpy(numbered(local,'scale',3))
    model._rotation=torch.from_numpy(numbered(local,'rot',4))
    owners=local['binding_0'].astype(np.int64)
    model.binding=torch.from_numpy(owners)
    with np.load(checked(source['pose_mesh']),allow_pickle=False) as mesh:
        model.faces=torch.from_numpy(mesh['faces'].astype(np.int64))
        model.vertices_sequence=torch.from_numpy(mesh['vertices_world'])
        model.num_timesteps=len(mesh['frame_ids'])
        need(mesh['frame_ids'].tolist()==[row['frame'] for row in source['selected_records']],
             'The original legacy pose order differs from its registered rows.')
    need(len(local)==len(world)==source['point_count'] and np.array_equal(local['binding_0'],owners),
         'The saved PLY rows have changed.')
    return model,local,owners,geometry(world)[3:]


def detector_geometry(source):
    with np.load(checked(source['fit']),allow_pickle=False) as fit:
        frames=fit['frame_ids'].tolist()
        observed,projected=(fit[name] for name in ('observed_native_keypoints','joints_projected_native'))
    rows=source['selected_records']
    need(frames==[row['frame'] for row in rows],'Official pose measurements are in a different order.')
    with np.load(checked(source['native_observations']),allow_pickle=False) as data:
        names=data['frame_ids'].tolist()
        contour=[data['joint_names'].tolist().index(f'face-{i}') for i in range(17)]
        detector=[tuple(data[key][names.index(frame),contour] for key in ('keypoints','scores','flip_disagreement'))
                  for frame in frames]
    return zip(rows,observed,projected,detector,strict=True)


def pose_arrays(model,appearance,slot):
    model.select_mesh_by_timestep(slot)
    return (model.get_xyz.numpy(),model.get_scaling.numpy(),model.get_rotation.numpy(),*appearance)


def first_pass(source,model,appearance,head_faces,head_rows):
    lookup=np.full(source['point_count'],-1,np.int64)
    lookup[head_rows]=np.arange(len(head_rows))
    reports=[]; masses=[]; grids=[]; interiors=[]
    faces=model.faces.numpy()
    for slot,(row,obs,proj,detector) in enumerate(detector_geometry(source)):
        box=head_crop(model.vertices_sequence[slot].numpy(),faces,head_faces,row)
        grid,interior,scope,source_stats=source_grid(row,obs,proj,detector,box,{})
        original=decode(checked(row['baseline_rgb']),box).astype(np.int16)
        projected=preprocess(pose_arrays(model,appearance,slot),row)
        mass,parity=compositor(projected,grid,interior,scope,box,original,lookup)
        need(parity['pixels']>0 and parity['excluded_pixels']/parity['pixels']<.01,
             'The saved baseline posed render failed native RGB parity.')
        reports.append(dict(frame=row['frame'],role='selected_fit',source_bin=int(row['source_bin']),
            source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],
            native_render_sha256=row['baseline_rgb']['sha256'],box=list(box),label_mask_sha256=array_sha(grid),
            interior_sha256=array_sha(interior),pixel_scope_sha256=array_sha(scope),parity=parity,source_evidence=source_stats))
        masses.append(mass); grids.append(grid); interiors.append(interior)
        print(dict(subject=source['subject'],frame=row['frame'],skin_pixels=source_stats.get('skin_pixels',0),
                   parity_excluded=parity['excluded_pixels']),flush=True)
    return reports,np.stack(masses),grids,interiors


def full_material(source,local,head,masses,reports):
    result=classify(masses,[view['source_bin'] for view in reports])
    arrays={name:np.zeros(source['point_count'],dtype=values.dtype) for name,values in result.items()}
    for name,values in result.items():
        arrays[name][head]=values
    scope=np.zeros(source['point_count'],bool); scope[head]=True
    arrays['head_scope']=scope
    arrays['unknown_unobserved_elongated']=(scope&(arrays['labels']==0)&~arrays['observed']&
                                            ~arrays['possible_protected_material']&elongated(local))
    counts=dict(label_counts=dict(zip(LABELS,np.bincount(result['labels'],minlength=len(LABELS)).tolist())),
        **{name:int(result[name].sum()) for name in ('observed','geometry_eligible','silhouette_spill','possible_protected_material')})
    return arrays,dict(head_rows=len(head),main=counts,main_plus_auxiliary=counts,
        unknown_unobserved_elongated=int(arrays['unknown_unobserved_elongated'].sum()),
        distinct_source_bins=sorted({view['source_bin'] for view in reports}),added_observed_from_aux=0,added_skin_from_aux=0)


def color_pass(source,model,appearance,arrays,views,grids,interiors):
    labels=arrays['labels']; observed=arrays['observed']; protect=arrays['possible_protected_material']
    eligible=np.flatnonzero(observed&((labels!=0)|~protect))
    lookup=np.full(source['point_count'],-1,np.int64); lookup[eligible]=np.arange(len(eligible))
    angular=sorted({item['source_bin'] for item in views})
    shape=(len(angular),len(eligible))
    mass=np.zeros(shape,np.float64); counts=np.zeros(shape,np.int64)
    color=np.zeros((*shape,3),np.float64); square=np.zeros_like(color)
    reports=[]
    for slot,(row,item,grid,interior) in enumerate(zip(source['selected_records'],views,grids,interiors,strict=True)):
        box=item['box']
        native=decode(pin(row['image'],row['source_sha256']),box).astype(np.float32)/255.
        archived=decode(checked(row['baseline_rgb']),box).astype(np.int16)
        projected=preprocess(pose_arrays(model,appearance,slot),row)
        a,b,c,d,proof=measure(projected,grid,interior,native,archived,None,box,lookup,labels[eligible])
        index=angular.index(item['source_bin'])
        mass[index]+=a; color[index]+=b; square[index]+=c; counts[index]+=d
        if proof['interior_pixels']:
            need(proof['parity_excluded']/proof['interior_pixels']<.01,'The legacy color observation failed native RGB parity.')
        reports.append({key:item[key] for key in ('frame','role','source_bin','source_sha256','mask_sha256','native_render_sha256',
                                                'label_mask_sha256','interior_sha256')}|
                        dict(eligible_source_rows=int((a>0).sum()),parity=proof))
    values=intervals(mass,color,square,counts)
    output={name:np.zeros((source['point_count'],*value.shape[1:]),dtype=value.dtype)
            for name,value in zip(COLOR_ARRAYS,values,strict=True)}
    output['source_rgb_high'].fill(1.)
    for name,value in zip(COLOR_ARRAYS,values,strict=True):
        output[name][eligible]=value
    guard=output['source_color_guard']
    need(np.isfinite(output['source_rgb_low']).all() and np.isfinite(output['source_rgb_high']).all() and
         np.all(output['source_rgb_low']<=output['source_rgb_high']) and np.all(observed[guard]),
         'The R28 legacy color bounds are invalid.')
    metrics=dict(eligible_rows=len(eligible),guarded_rows=int(guard.sum()),observed_rows=int(observed.sum()),
        guarded_by_label=np.bincount(labels[guard],minlength=6).tolist(),
        guarded_bin_histogram=np.bincount(output['source_color_bins'][guard]).tolist())
    return output,reports,metrics


def write_results(source,source_path,owners,material,views,grids,coverage,color,color_views,color_coverage):
    subject=source['subject']; selected=[row['frame'] for row in source['selected_records']]
    heldout=[row['frame'] for row in source['heldout_records']]
    reserved=source.get('reserved_holdouts',heldout)
    stable=np.arange(len(owners),dtype=np.int64)
    grid=OUT/f'{subject}-native-skin-grid.npz'
    material_path=OUT/f'{subject}-material-visibility.npz'
    color_path=OUT/f'{subject}-native-color-guard.npz'
    paths=(grid,material_path,material_path.with_suffix('.json'),color_path,color_path.with_suffix('.json'))
    need(not any(path.exists() for path in paths),'Preserve existing signed legacy evidence outputs.')
    with grid.open('xb') as stream:
        np.savez_compressed(stream,frame_ids=np.asarray(selected),boxes=np.asarray([view['box'] for view in views]),
                            **{f'grid_{i:03d}':value for i,value in enumerate(grids)})
    with material_path.open('xb') as stream:
        np.savez_compressed(stream,stable_ids=stable,owner_faces=owners,**material)
    provenance=dict(source=descriptor(source_path),fit_sha256=source['fit_sha256'],
                    training_inputs_sha256=source['training_inputs_sha256'],source_sha256=sha(__file__))
    common=dict(subject=subject,baseline_bound_ply_sha256=source['local_ply']['sha256'],
        baseline_raw_ply_sha256=source['world_ply']['sha256'],selected_frames=selected,admitted_auxiliary_frames=[],
        retained_holdouts=heldout,reserved_holdouts=reserved,heldout_source_images_read=0)
    material_receipt=dict(common,schema='shuolee-r27-native-face-material-visibility/1',
        material_visibility_npz_sha256=sha(material_path),fit_sha256=source['fit_sha256'],
        training_inputs_sha256=source['training_inputs_sha256'],classification_frames=selected,
        label_codes=dict(enumerate(LABELS)),policy=MATERIAL_POLICY,coverage=coverage,views=views,
        provenance=provenance,native_skin_grid=descriptor(grid),
        scope='Observed original fit frames and source foreground, signed by the legacy posed RGB. '
              'The legacy replay does not contain rendered alpha. No new hair or eyewear material is inferred from identity.')
    write_json(material_path.with_suffix('.json'),material_receipt)
    with color_path.open('xb') as stream:
        np.savez_compressed(stream,stable_ids=stable,owner_faces=owners,**color)
    color_receipt=dict(common,schema='shuolee-r28-native-sh0-color-guard/1',
        source_color_guard_npz_sha256=sha(color_path),material_visibility_npz_sha256=sha(material_path),
        material_visibility_receipt_sha256=sha(material_path.with_suffix('.json')),
        source_frames=selected,policy=COLOR_POLICY,coverage=color_coverage,views=color_views,
        provenance=provenance,source_sha256=sha(__file__),
        rgb_convention='Native PNG RGB codes divided by 255.',
        scope='Per-row RGB measured where the legacy splat contributes materially to original fit views with verified replay parity.')
    write_json(color_path.with_suffix('.json'),color_receipt)
    print(dict(subject=subject,material=descriptor(material_path.with_suffix('.json')),material_npz=descriptor(material_path),
        color=descriptor(color_path.with_suffix('.json')),color_npz=descriptor(color_path),grid=descriptor(grid),
        coverage=coverage,color_coverage=color_coverage),flush=True)


def run(subject):
    sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    path=OUT/f'{subject}-legacy-source.json'
    source=read(path)
    need(source['schema']=='shuolee-r28-legacy-bound-source/1' and source['subject']==subject and
         source['heldout_source_images_read']==0 and not source['auxiliary_records'], 'The original legacy source roster differs.')
    model,local,owners,appearance=original_model(source)
    faces_scope=protected_faces(checked(source['asset']),model.faces.numpy())
    head=np.flatnonzero(faces_scope[owners])
    views,masses,grids,interiors=first_pass(source,model,appearance,faces_scope,head)
    material,coverage=full_material(source,local,head,masses,views)
    color,color_views,color_coverage=color_pass(source,model,appearance,material,views,grids,interiors)
    write_results(source,path,owners,material,views,grids,coverage,color,color_views,color_coverage)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('001','005','008'))
    run(parser.parse_args().subject)
