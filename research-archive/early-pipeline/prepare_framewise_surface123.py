"""Prepare a fixed-ID free clothing initializer from a reviewed final Bundle.

CPU only. `inspect` estimates the final reference grid without extracting any
surface. `run` requires a separately registered plan, explicit observed-valid
masks and a final-fit visual-review receipt. No pose fitting, renderer, GPU,
old Gaussian checkpoint, default all-valid mask or implicit threshold search.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import signal
import sys
import tempfile
import time
import traceback

import numpy as np
from PIL import Image

SCHEMA = 'framewise-real-surface-entry/1'
CONFIG = dict(initial_points=200000, spacing_m=.012, padding_m=.40,
              max_voxels=5000000, min_valid_views=3, vote_fraction=.97, seed=6060)
CONFIG_ORIGIN = dict(file='prepare_native_person122_v3.py', symbol='SURFACE_CONFIG',
    source_fields=dict(min_valid_views='min_observed_views'),
    inherited=['initial_points','spacing_m','padding_m','max_voxels','min_observed_views','vote_fraction','seed'],
    explicit_difference='Current framewise native-pixel surface uses the supplied binary mask unchanged; legacy mask_tolerance_px=2 dilation is not silently restored.')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_frozen_plan(path):
    digest=sha(path);plan=read(path)
    verify_frozen_plan(path,digest)
    return plan,digest


def verify_frozen_plan(path,digest):
    need(sha(path)==digest,'Registered plan changed after it was read; refuse commit.')


def require_posix_budget():
    need(os.name=='posix' and sys.platform.startswith('linux') and
         hasattr(signal,'SIGALRM') and hasattr(signal,'setitimer'),
         'Run initialization in the registered WSL/Linux POSIX environment with SIGALRM/setitimer; Windows inspect remains supported.')


def descriptor(path, root):
    return dict(path=Path(path).relative_to(root).as_posix(),sha256=sha(path))


def checked(base, item):
    need(isinstance(item,dict) and set(item)=={'path','sha256'}, 'Expected explicit path and SHA256.')
    path=Path(item['path']);path=path if path.is_absolute() else base/path
    path=path.resolve(strict=True)
    need(path.is_file() and sha(path)==item['sha256'], 'Missing or changed input: '+str(path))
    return path


def atomic_file(path, writer):
    """No-replace file publication; incomplete temporary bytes never look ready."""
    path=Path(path);need(path.parent.is_dir() and not path.exists(),'Fresh output file required: '+str(path))
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.partial',delete=False) as stream:
            temporary=Path(stream.name);writer(stream);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


def write_json(path, value):
    raw=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
    atomic_file(path,lambda stream:stream.write(raw))


def write_npz(path, **arrays):
    atomic_file(path,lambda stream:np.savez_compressed(stream,**arrays))


def record_run_exception(output,exc,started):
    # This private fresh output directory has one publisher. Its manifest is
    # created only by the final atomic commit, never as a placeholder. A stdout
    # failure or interrupt after that point cannot relabel committed data.
    committed=(Path(output)/'manifest.json').is_file()
    name='post-commit-notification-error.json' if committed else 'failure.json'
    write_json(Path(output)/name,dict(error=repr(exc),traceback=traceback.format_exc(),
        elapsed_seconds=time.monotonic()-started,partial_outputs_preserved=not committed,
        artifact_commit_completed=committed,training_started=False))
    return committed


def estimate(bundle):
    from shuolee_portrait.framewise_clothed_surface import ReferenceGrid
    vertices=bundle.reference.vertices_world.astype(np.float64)
    lower=vertices.min(0)-CONFIG['padding_m'];upper=vertices.max(0)+CONFIG['padding_m']
    shape=np.ceil((upper-lower)/CONFIG['spacing_m']).astype(np.int64)+1
    count=int(np.prod(shape,dtype=object));frames=bundle.report()['frame_ids']
    pixels=sum(bundle.frame(f).wh[0]*bundle.frame(f).wh[1] for f in frames)
    triangles=len(bundle.faces);nframes=len(frames);nvertices=len(vertices)
    # Conservative transparent components, not measured process peak RSS.
    memory=dict(valid_input_plus_copied_valid_and_foreground_masks=3*pixels,
        two_motion_table_sets=2*nframes*triangles*14*8,
        bundle_and_motion_vertices=3*nframes*nvertices*3*8,
        grid_and_volume_scratch=128*count,
        boundary_chunk_and_candidate_allowance=768*1024**2,
        output_and_serialization_allowance=512*1024**2)
    report=dict(schema=SCHEMA,mode='geometry_estimate_only',reference_frame_id=bundle.reference_frame_id,
        frame_ids=frames,frames=nframes,reference_aabb_model_m=[vertices.min(0).tolist(),vertices.max(0).tolist()],
        proposal_bounds_model_m=[lower.tolist(),upper.tolist()],grid_shape=shape.tolist(),grid_voxels=count,
        config=CONFIG,config_origin=CONFIG_ORIGIN,within_fixed_voxel_budget=count<=CONFIG['max_voxels'],
        native_rasters=[dict(frame=f,wh=list(bundle.frame(f).wh)) for f in frames],
        minimum_grid_frame_projection_evaluations=2*count*nframes,
        additional_sample_projection_evaluations=CONFIG['initial_points']*nframes,
        extra_edge_refinement_cost='Approximately (17 scan + up to 16 midpoint + 2 final endpoint) queries per edge vertex per frame; depends on actual extracted surface.',
        estimated_memory_bytes=memory,estimated_working_set_bytes=sum(memory.values()),
        measured_runtime_available=False,runtime_expectation='Potentially minutes to tens of minutes on CPU for millions of voxels and 94 native views; exact KD-boundary, silhouette and edge counts are not benchmarked.',
        physical_metric_scale_measured=False,model_forward_or_surface_extraction_executed=False,gaussian_training_started=False)
    if count<=CONFIG['max_voxels']:
        grid=ReferenceGrid.from_bounds(lower,upper,spacing_m=CONFIG['spacing_m'],
            reference_frame_id=bundle.reference_frame_id,max_voxels=CONFIG['max_voxels'])
        report['grid_axes_sha256']=[hashlib.sha256(a.tobytes()).hexdigest() for a in grid.axes]
    else:
        report['grid_axes_sha256']=None
    return report


def load_validity(path,bundle,bundle_sha):
    """Return frozen explicit native observation support; never infer from alpha."""
    doc=read(path);base=Path(path).parent;names=bundle.report()['frame_ids']
    need(doc.get('schema')=='framewise-observed-valid/1','Unsupported observation-valid manifest.')
    need(doc.get('bundle_manifest_sha256')==bundle_sha and doc.get('reference_frame_id')==bundle.reference_frame_id,
         'Validity evidence is not bound to this final Bundle/reference.')
    records=doc.get('records')
    need(isinstance(records,list) and len(records)==len(names),'Exactly one validity record per frame required.')
    mapping={r['frame_id']:r for r in records}
    need(len(mapping)==len(records) and set(mapping)==set(names),'Validity identities duplicate or differ from Bundle.')
    masks,notes,sources={},{},{}
    for name in names:
        row=mapping[name];frame=bundle.frame(name)
        need(row.get('source_frame')==frame.source_frame,'Validity source-frame identity differs: '+name)
        need(row.get('image_sha256')==sha(frame.image),'Validity source-image hash differs: '+name)
        need(isinstance(row.get('note'),str) and row['note'].strip(),'Explicit observed/occluded support explanation required: '+name)
        need(row.get('external_occlusion_review') in ('reviewed_exclusions','reviewed_no_external_occluder','all_unknown'),
             'Missing external occlusion review; foreground alpha alone is insufficient: '+name)
        source=checked(base,row['valid_mask'])
        with Image.open(source) as im:
            im.load();need(im.mode=='L' and im.size==frame.wh,'Validity mask must be native 8-bit grayscale: '+name)
            values=np.array(im)
        need(np.isin(values,[0,255]).all(),'Validity mask must be exact 0/255: '+name)
        if row['external_occlusion_review']=='all_unknown':need(not values.any(),'All-unknown review conflicts with positive valid pixels.')
        masks[name]=values==255;notes[name]=row['note']
        sources[name]=dict(path=str(source),sha256=sha(source),review=row['external_occlusion_review'],
                           note=row['note'],observed_pixels=int(masks[name].sum()),unknown_pixels=int((~masks[name]).sum()))
    return masks,notes,sources


def color_from_observed_frames(bundle, initialized, valid_masks, deadline):
    """Approximate same-pixel point z-buffer, followed by native RGB averaging.

    This does not certify visibility of a continuous clothing mesh. Unobserved
    points remain in the layout with explicit neutral .5 RGB and zero count.
    """
    from shuolee_portrait.framewise_clothed_surface import _move_queries
    xyz=initialized.xyz_ref;count=len(xyz);transport=initialized.transport
    sums=np.zeros((count,3),np.float64);samples=np.zeros(count,np.uint16);details=[]
    support=transport.motion_support_triangle_ids
    for name in transport.frame_ids:
        need(time.monotonic()<deadline,'Registered CPU color budget expired.')
        frame=bundle.frame(name);w,h=frame.wh
        posed=_move_queries(transport,name,xyz,support)
        camera=posed@frame.w2c[:3,:3].T+frame.w2c[:3,3];q=camera@frame.K_integer_centers.T
        with np.errstate(divide='ignore',invalid='ignore'):uv=q[:,:2]/q[:,2:]
        inside=(camera[:,2]>0)&np.isfinite(uv).all(1)&(uv[:,0]>=-.5)&(uv[:,0]<w-.5)&(uv[:,1]>=-.5)&(uv[:,1]<h-.5)
        ids=np.flatnonzero(inside);pixel=np.floor(uv[ids]+.5).astype(np.int64)
        with Image.open(frame.mask) as im:foreground=np.array(im.convert('L'))==255
        seen=valid_masks[name][pixel[:,1],pixel[:,0]]&foreground[pixel[:,1],pixel[:,0]]
        ids=ids[seen];pixel=pixel[seen];linear=pixel[:,1]*w+pixel[:,0]
        z=np.full(w*h,np.inf);np.minimum.at(z,linear,camera[ids,2])
        visible=camera[ids,2]==z[linear];ids=ids[visible];pixel=pixel[visible]
        with Image.open(frame.image) as im:rgb=np.asarray(im.convert('RGB'),dtype=np.float32)/255.
        sums[ids]+=rgb[pixel[:,1],pixel[:,0]];samples[ids]+=1
        details.append(dict(frame=name,used_points=len(ids),visibility='nearest free-point depth at the same native pixel; approximation only'))
    colors=np.full((count,3),.5,np.float32)
    known=samples>0;colors[known]=(sums[known]/samples[known,None]).astype(np.float32)
    return colors,samples,dict(method='native_nearest_pixel_point_zbuffer_equal_view_mean',frames=details,
        points_with_observed_color=int(known.sum()),neutral_unobserved_points=int((~known).sum()),
        neutral_rgb=.5,old_checkpoint_used=False,continuous_surface_visibility_certified=False,
        color_counts_are_not_independent_material_correspondence=True)


def inspect(args):
    from shuolee_portrait.frame_pose_contract import load_frame_pose_bundle
    bundle=load_frame_pose_bundle(args.bundle);report=estimate(bundle)
    report['bundle_manifest_sha256']=sha(args.bundle)
    report['unresolved_inputs']=[]
    if args.validity_manifest is None:
        report['unresolved_inputs'].append('Explicit per-frame observed-valid masks and external-occlusion explanations have not been supplied; do not derive all-valid masks from foreground alpha.')
    else:
        try:
            _,_,sources=load_validity(args.validity_manifest,bundle,sha(args.bundle));report['validity_sources']=sources
        except Exception as exc:report['unresolved_inputs'].append(str(exc))
    report['fit_visual_review_required_before_run']=True
    write_json(args.report,report);print(json.dumps({k:report[k] for k in ('grid_shape','grid_voxels','estimated_working_set_bytes','unresolved_inputs')}))


def run(args):
    require_posix_budget()
    from shuolee_portrait import (framewise_clothed_surface,free_clothing_transport,frame_pose_contract,
                                  camera_pixel_contract,camera_world_units,alignment_inputs)
    from shuolee_portrait.frame_pose_contract import load_frame_pose_bundle
    from shuolee_portrait.framewise_clothed_surface import ReferenceGrid,initialize_framewise_surface
    from shuolee_portrait.free_clothing_transport import FreeClothingTransport
    plan_path=args.plan.resolve(strict=True);plan,plan_sha=read_frozen_plan(plan_path);base=plan_path.parent
    need(plan.get('schema')==SCHEMA and plan.get('config')==CONFIG,'Registered fixed surface configuration required; no threshold sweep.')
    need(plan.get('config_origin')==CONFIG_ORIGIN,'Explicit inherited parameter and mask-tolerance difference record required.')
    need(plan.get('color_initialization')=='native_nearest_pixel_point_zbuffer_equal_view_mean','Explicit color initialization policy required.')
    need(type(plan.get('cpu_wall_seconds')) is int and plan['cpu_wall_seconds']>0,'Explicit CPU wall budget required.')
    need(type(plan.get('max_working_set_bytes')) is int and plan['max_working_set_bytes']>0,'Explicit CPU memory budget required.')
    actual_sources=dict(runner=Path(__file__))
    for module in (framewise_clothed_surface,free_clothing_transport,frame_pose_contract,
                   camera_pixel_contract,camera_world_units,alignment_inputs):
        actual_sources[module.__name__.split('.')[-1]]=Path(module.__file__)
    need(set(actual_sources)<=set(plan['frozen_sources']),'Complete implementation source bindings required.')
    for name,item in plan['frozen_sources'].items():
        found=checked(base,item)
        if name in actual_sources:
            need(found==actual_sources[name].resolve(),'Imported source differs from frozen source: '+name)
    bundle_path=checked(base,plan['bundle']);bundle_sha=sha(bundle_path)
    bundle=load_frame_pose_bundle(bundle_path)
    need(bundle.reference_frame_id==plan['reference_frame_id'],'Explicit named reference mismatch.')
    need(bundle.report()['frame_ids']==plan['frame_ids'],'Complete registered frame roster mismatch.')
    visual=read(checked(base,plan['final_fit_visual_review']))
    need(visual.get('bundle_manifest_sha256')==bundle_sha and visual.get('reference_frame_id')==bundle.reference_frame_id
         and visual.get('fit_artifact_stage')=='final' and visual.get('exploratory_initialization_allowed') is True,
         'Final-fit visual review for this exact Bundle is not ready; no initializer started.')
    current=estimate(bundle);registered=read(checked(base,plan['geometry_estimate']))
    for key in ('grid_shape','grid_voxels','grid_axes_sha256','reference_frame_id','frame_ids','bundle_manifest_sha256'):
        value=bundle_sha if key=='bundle_manifest_sha256' else current[key]
        need(registered.get(key)==value,'Final reference geometry/roster differs from registered estimate: '+key)
    need(current['within_fixed_voxel_budget'],'Final reference AABB exceeds fixed 5M grid budget; no automatic spacing change.')
    need(current['estimated_working_set_bytes']<=plan['max_working_set_bytes'],'Estimated CPU memory exceeds registered budget.')
    validity_path=checked(base,plan['validity_manifest']);valid,notes,validity_sources=load_validity(validity_path,bundle,bundle_sha)
    length=plan['camera_length_contract']
    need(length.get('translation_unit')=='meters' and length.get('meters_per_camera_unit')==1.,'Bundle E is already normalized; no second scale conversion.')
    output=args.output_dir.resolve();output.mkdir(parents=False,exist_ok=False)
    start=time.monotonic();deadline=start+plan['cpu_wall_seconds']
    write_json(output/'started.json',dict(pid=os.getpid(),plan_sha256=plan_sha,CPU_only=True,bundle_sha256=bundle_sha,
        budget_environment='WSL/Linux POSIX SIGALRM/setitimer'))
    prior_handler=None
    if hasattr(signal,'SIGALRM'):
        def expired(signum,frame):raise TimeoutError('Registered CPU surface wall budget expired.')
        prior_handler=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,plan['cpu_wall_seconds'])
    try:
        lo,hi=current['proposal_bounds_model_m']
        grid=ReferenceGrid.from_bounds(lo,hi,spacing_m=CONFIG['spacing_m'],reference_frame_id=bundle.reference_frame_id,max_voxels=CONFIG['max_voxels'])
        initialized=initialize_framewise_surface(bundle,grid,reference_frame_id=bundle.reference_frame_id,
            valid_masks=valid,validity_notes=notes,camera_length_contract=length,min_valid_views=CONFIG['min_valid_views'],
            vote_fraction=CONFIG['vote_fraction'],seed=CONFIG['seed'],initial_points=CONFIG['initial_points'])
        colors,color_counts,color_report=color_from_observed_frames(bundle,initialized,valid,deadline)
        ids=np.array(initialized.gaussian_ids);support=initialized.transport.motion_support_triangle_ids
        write_npz(output/'initialization.npz',xyz_ref=initialized.xyz_ref,gaussian_ids=ids,
            motion_support_triangle_ids=support,reference_frame_id=np.array(bundle.reference_frame_id),
            translation_unit=np.array('meters'),initial_rgb=colors,color_observation_counts=color_counts,
            positive_counts=initialized.sample_positive_counts,valid_counts=initialized.sample_valid_counts,
            bundle_manifest_sha256=np.array(bundle_sha))
        write_npz(output/'surface.npz',vertices_ref=initialized.surface_vertices_ref,faces=initialized.surface_faces,
                  sampled_surface_face_ids=initialized.surface_sample_face_ids)
        # Transport has its own hash-bound JSON/NPZ. The outer manifest is the
        # atomic commit marker: absence means partial output, never a ready run.
        initialized.transport.save(output/'transport.json')
        restored=FreeClothingTransport.load(output/'transport.json')
        with np.load(output/'initialization.npz',allow_pickle=False) as archive:
            need(tuple(archive['gaussian_ids'].tolist())==restored.gaussian_ids,'Readback Gaussian identities differ.')
            need(np.array_equal(archive['motion_support_triangle_ids'],restored.motion_support_triangle_ids),'Readback support differs.')
            need(np.array_equal(archive['xyz_ref'],initialized.xyz_ref),'Readback XYZ differs.')
            need(archive['reference_frame_id'].item()==restored.reference_frame_id==bundle.reference_frame_id,'Readback reference differs.')
        need(restored.frame_ids==tuple(bundle.report()['frame_ids']),'Readback complete pose roster differs.')
        need(restored.frame_source_sha256==initialized.transport.frame_source_sha256,'Readback frame-source bindings differ.')
        need(np.array_equal(restored.faces,bundle.faces),'Readback motion topology differs.')
        bundle.assert_unchanged();checked(base,plan['validity_manifest'])
        checked(base,plan['final_fit_visual_review']);checked(base,plan['geometry_estimate'])
        for row in validity_sources.values():need(sha(row['path'])==row['sha256'],'Validity mask changed during initialization.')
        for item in plan['frozen_sources'].values():checked(base,item)
        verify_frozen_plan(plan_path,plan_sha)
        need(time.monotonic()<deadline,'Registered CPU budget expired before commit.')
        # Own the final commit after bounded computation. No asynchronous alarm
        # may label an already committed manifest as a failed initialization.
        if prior_handler is not None:signal.setitimer(signal.ITIMER_REAL,0)
        write_json(output/'report.json',dict(schema=SCHEMA,initializer=initialized.report(),color=color_report,
            estimate=current,validity=validity_sources,elapsed_seconds=time.monotonic()-start,
            arrays_roundtrip=True,physical_camera_accepted=False,quality_accepted=False,training_admitted=False,
            all_points_pass_vote_is_not_training_gate=True,no_optimizer=True,no_renderer=True,GPU_calls=0))
        files=['initialization.npz','surface.npz','transport.json','transport.npz','report.json']
        write_json(output/'manifest.json',dict(schema=SCHEMA,state='initialized_for_review',
            reference_frame_id=bundle.reference_frame_id,frame_ids=list(restored.frame_ids),points=len(ids),
            bundle_manifest_sha256=bundle_sha,registered_plan_sha256=plan_sha,
            artifacts={name:descriptor(output/name,output) for name in files},
            training_started=False,quality_accepted=False,world_unit_basis=length,
            commit_contract='Readers require this manifest and every artifact hash; partial files alone are not a ready initializer.'))
        print(json.dumps(dict(state='initialized_for_review',points=len(ids),elapsed_seconds=time.monotonic()-start)))
    except BaseException as exc:
        record_run_exception(output,exc,start)
        raise
    finally:
        if prior_handler is not None:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior_handler)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project-src',type=Path,required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    inspection=sub.add_parser('inspect');inspection.add_argument('--bundle',type=Path,required=True)
    inspection.add_argument('--validity-manifest',type=Path);inspection.add_argument('--report',type=Path,required=True)
    execution=sub.add_parser('run');execution.add_argument('--plan',type=Path,required=True);execution.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.project_src.resolve(strict=True)))
    if args.command=='inspect':inspect(args)
    else:run(args)


if __name__=='__main__':main()
