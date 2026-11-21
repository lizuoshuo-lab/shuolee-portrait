"""Independent evaluation through the pinned official continuous renderer.

The baseline bridge has no skybox. The two four-chunk merged workflows append
one shared coarse skybox, after verifying that Merger omitted all chunk tails.
Only explicit parent scheduling authorizes GPU mode. Protocol/CPU modes never
call CUDA, import a trainer, modify a source model, or create fake hier_opt files.
"""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image

import evaluate_baseline as baseline
from evaluate_hierarchy import Topology, array_digest, author_cpu_load, save_selection, FIXED
from read_hierarchy import read_hierarchy, tau_to_threshold

ROOT=baseline.ROOT;TRACK=baseline.TRACK;OUTROOT=baseline.OUTPUT
REPO=ROOT/'repos/hierarchical-3d-gaussians'
BUILD=TRACK/'hierarchy-build';EXT=BUILD/'python-extensions/cu128'
COMMIT='596dc7081ee74e57a66502f4746328911f763b34'
SETTINGS=(('leaves',None),('tau1',1.),('tau3',3.),('tau6',6.),('tau15',15.))
POST_RANGE=(.005,.1)
save,read_json,digest,require,summarize=baseline.save,baseline.read_json,baseline.digest,baseline.require,baseline.summarize

def source_audit(output):
    require(subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT,'Official main HEAD changed')
    snapshot=output/'source-snapshots';snapshot.mkdir()
    paths=['gaussian_renderer/__init__.py','scene/gaussian_model.py','scene/cameras.py',
           'utils/graphics_utils.py','scene/dataset_readers.py','render_hierarchy.py','train_post.py']
    records=[]
    for relative in paths:
        path=REPO/relative;committed=subprocess.check_output(['git','-C',str(REPO),'show',f'{COMMIT}:{relative}'])
        require(path.read_bytes().replace(b'\r\n',b'\n')==committed.replace(b'\r\n',b'\n'),'Official source changed: '+relative)
        copy=snapshot/'official'/relative;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(path.read_bytes())
        records.append(dict(path=str(path),sha256=digest(path),snapshot=str(copy)))
    post_tree=ast.parse((REPO/'train_post.py').read_text(encoding='utf-8'))
    ranges={n.targets[0].id:ast.literal_eval(n.value) for n in ast.walk(post_tree)
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'limmin','limmax'}}
    require((ranges['limmin'],ranges['limmax'])==POST_RANGE,'Unchanged official post threshold range differs')
    for relative in ('evaluate_official_hierarchy.py','evaluate_baseline.py','evaluate_hierarchy.py','read_hierarchy.py'):
        path=ROOT/'scripts/large_scene_gs'/relative;copy=snapshot/'evaluation'/relative;copy.parent.mkdir(exist_ok=True)
        copy.write_bytes(path.read_bytes());records.append(dict(path=str(path),sha256=digest(path),snapshot=str(copy)))
    extensions=read_json(BUILD/'python-extensions-manifest.json')
    expected={str(Path(row['path']).resolve()).casefold():row['sha256']
              for build in extensions['build_records'].values() for row in build['artifacts_present']}
    modules=[]
    for name in ('gaussian_hierarchy','diff_gaussian_rasterization'):
        found=list((EXT/name).glob('_C*.pyd'));require(len(found)==1,'Ambiguous official extension')
        path=found[0];require(digest(path)==expected[str(path.resolve()).casefold()],'Extension binary identity mismatch')
        modules.append(dict(name=name,path=str(path),sha256=digest(path),bytes=path.stat().st_size))
    native_sources=[]
    for relative in ('cuda_rasterizer/forward.cu','cuda_rasterizer/auxiliary.h','cuda_rasterizer/rasterizer_impl.cu','diff_gaussian_rasterization/__init__.py'):
        path=ROOT/'repos/hierarchy-rasterizer-large-gs'/relative
        copy=snapshot/'rasterizer'/relative;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(path.read_bytes())
        native_sources.append(dict(path=str(path),sha256=digest(path),snapshot=str(copy)))
    result=dict(status='passed_pinned_official_source_and_binary_audit',main_commit=COMMIT,source_files=records,
        rasterizer_sources=native_sources,modules=modules,extension_manifest_sha256=digest(BUILD/'python-extensions-manifest.json'),
        extension_source_contract=extensions['source_audit'],compatibility_patch=extensions['compatibility_patch'],
        continuous_contract='Unchanged render_post(interp_python=True), official wxyz normalization/parameter interpolation and pinned pixel sibling-alpha kernel; immutable properties cached once outside per-frame timing',
        unchanged_post_training_dimensionless_range=list(POST_RANGE),
        cuda_called=False)
    save(output/'source-audit.json',result)
    return result

def threshold_scope(tau,fx):
    """Official post samples dimensionless thresholds, not pixel tau."""
    threshold=None if tau is None else tau_to_threshold(tau,fx)
    inside=None if tau is None else POST_RANGE[0]<=threshold<=POST_RANGE[1]
    return dict(fx_px=float(fx),threshold=threshold,post_training_dimensionless_range=list(POST_RANGE),
        corresponding_tau_range_px=[POST_RANGE[0]*fx/2-.5,POST_RANGE[1]*fx/2-.5],
        in_training_range=inside,diagnostic_scope='leaf_control' if tau is None else ('supported_training_threshold_range' if inside else 'extrapolation_outside_post_training_threshold_range'))

def memory_plan(total_G,node_count,leaf_count,max_selected,sky_count,camera_count=32):
    """Portable preflight bound for our fixed buffers and Python tensor work.

    The rasterizer's binning duplicates/sort scratch depend on projection and
    overdraw; this is deliberately NOT a bound on total native CUDA memory.
    """
    tree_G=total_G-sky_count
    require(0<max_selected<=tree_G,'Invalid planned selection upper bound')
    parts=dict(raw_model_and_tree=236*total_G+60*node_count,
        immutable_property_cache=32*total_G,cut_and_weight_buffers=20*total_G,
        fixed_leaf_indices_parents_nodeids=12*leaf_count,camera_matrices=camera_count*(3*16+3)*4,
        screen_full_model_temporary_upper=24*total_G,
        # Four complete parameter tuples safely overcount simultaneously live
        # gathered/blended/output tensors across sequential official expressions.
        interpolation_parameter_temporary_upper=4*236*(max_selected+sky_count),
        interpolation_index_and_weight_temporary_upper=80*max_selected+4*total_G)
    resident=sum(parts[k] for k in ('raw_model_and_tree','immutable_property_cache','cut_and_weight_buffers','fixed_leaf_indices_parents_nodeids','camera_matrices'))
    return dict(total_G=total_G,tree_G=tree_G,node_count=node_count,leaf_count=leaf_count,
        max_selected_tree_G=max_selected,skybox_G=sky_count,parts_bytes=parts,known_fixed_resident_bytes=resident,
        conservative_Python_tensor_peak_upper_bytes=sum(parts.values()),native_workspace_upper_bound=None,
        native_workspace_limit='Projection-dependent tile duplicate arrays, radix-sort scratch, rasterizer geometry/image work and allocator overhead are additional; no total-memory fit guarantee.',
        feature_cache='Already loaded coefficient-major SH tensor reused; no full-model get_features concatenation per frame',
        memory_scope='Full resident model and cached parameters; no streaming or eviction')

def skybox_cpu(workflow,output):
    import torch
    from plyfile import PlyData
    root=TRACK/'official-training'/workflow['scaffold_run'];receipt=read_json(root/'outcome.json');config=read_json(root/'configuration.json')
    require(receipt['status']=='completed' and config['stage']=='coarse','Coarse dependency incomplete')
    require(config['input_manifest_sha256']==workflow['input_manifest_sha256'],'Coarse global input identity differs')
    directory=Path(receipt['saved_model_directory']);path=directory/'point_cloud.ply'
    require(digest(path)==receipt['point_cloud_sha256'],'Coarse PLY hash mismatch')
    count=int((directory/'pc_info.txt').read_text().strip());require(count==100000,'Expected recorded shared 100k skybox')
    data=PlyData.read(path)['vertex'].data
    require(len(data)>count and len([n for n in data.dtype.names if n.startswith('f_rest_')])==9,'Expected coarse SH1 PLY')
    xyz=np.stack([data[n][:count] for n in ('x','y','z')],axis=1).astype(np.float32)
    sh=np.zeros((count,16,3),dtype=np.float32)
    sh[:,0]=np.stack([data[f'f_dc_{c}'][:count] for c in range(3)],axis=1)
    rest=np.stack([data[f'f_rest_{i}'][:count] for i in range(9)],axis=1).reshape(count,3,3).transpose(0,2,1)
    sh[:,1:4]=rest
    logits=torch.from_numpy(np.array(data['opacity'][:count],dtype=np.float32)).reshape(count,1)
    alpha=torch.sigmoid(logits)
    logs=np.stack([data[f'scale_{c}'][:count] for c in range(3)],axis=1).astype(np.float32)
    quat=np.stack([data[f'rot_{c}'][:count] for c in range(4)],axis=1).astype(np.float32)
    tensors=(torch.from_numpy(xyz),torch.from_numpy(sh),alpha,torch.from_numpy(logs),torch.from_numpy(quat))
    for t in tensors:require(torch.isfinite(t).all().item(),'Nonfinite canonical skybox')
    require((torch.linalg.vector_norm(tensors[4],dim=1)>0).all().item(),'Zero skybox quaternion')
    result=dict(status='passed_single_canonical_coarse_skybox',count=count,coarse_run=workflow['scaffold_run'],
        ply_path=str(path),ply_sha256=digest(path),pc_info_sha256=digest(directory/'pc_info.txt'),
        tensors=[dict(shape=list(t.shape),dtype=str(t.dtype),sha256=array_digest(t.numpy())) for t in tensors],
        representation='First pc_info-count vertices; SH1 channel-major -> coefficient-major and SH3 zero padding; sigmoid source PLY logits once; raw log scales and wxyz',
        append_rule='Exactly one tail for each merged whole-scene model; baseline bridge skybox count remains zero')
    save(output/'shared-skybox-audit.json',result)
    return tensors,result

def camera_cpu(parsed,split,original,output):
    import torch
    source=(REPO/'utils/graphics_utils.py').read_text(encoding='utf-8');tree=ast.parse(source)
    funcs=[n for n in tree.body if getattr(n,'name',None) in {'getWorld2View2','getProjectionMatrix','focal2fov'}]
    require(len(funcs)==3,'Official camera math functions unavailable')
    scope={'np':np,'torch':torch,'math':math}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(REPO/'utils/graphics_utils.py'),'exec'),scope)
    cameras=[];rows=[]
    original_pid={int(p):i for i,p in enumerate(original['point_ids'])}
    for j,index in enumerate(parsed['test_indices'].astype(int)):
        K=parsed['K'][index];c2w=parsed['camtoworlds'][index];w2c=np.linalg.inv(c2w)
        FoVx=scope['focal2fov'](float(K[0,0]),245);FoVy=scope['focal2fov'](float(K[1,1]),136)
        view=torch.from_numpy(scope['getWorld2View2'](w2c[:3,:3].T,w2c[:3,3])).transpose(0,1)
        projection=scope['getProjectionMatrix'](.01,100.,FoVx,FoVy,float(K[0,2]/245),float(K[1,2]/136)).transpose(0,1)
        full=view[None].bmm(projection[None])[0];center=view.inverse()[3,:3]
        cam=SimpleNamespace(FoVx=FoVx,FoVy=FoVy,image_width=245,image_height=136,image_name=split['test_names'][j],
            world_view_transform=view,projection_matrix=projection,full_proj_transform=full,camera_center=center)
        # Independent projection check at registered observations, respecting
        # the official -.5 projected-mean vs integer pixel sampling convention.
        lo,hi=original['registered_point_observation_offsets'][index:index+2]
        pids=original['registered_point_observation_ids'][lo:hi]
        require(len(pids)>0,'Holdout camera has no registered point context')
        sampled=pids[np.linspace(0,len(pids)-1,min(1024,len(pids)),dtype=int)]
        point_rows=np.array([original_pid[int(p)] for p in sampled])
        xyz=parsed['points'][point_rows].astype(np.float64);xyzh=np.column_stack([xyz,np.ones(len(xyz))])
        clip=xyzh@full.numpy();ndc=clip[:,:2]/clip[:,3:4]
        actual=(ndc+1)*np.array([245/2,136/2])
        camera_xyz=xyzh@w2c.T;expected=camera_xyz[:,:2]/camera_xyz[:,2:3]*np.array([K[0,0],K[1,1]])+K[:2,2]
        error=float(np.max(np.abs(actual-expected)))
        require(error<.005,'Official camera matrix projection disagrees with saved K/frame')
        require(np.max(np.abs(center.numpy()-c2w[:3,3]))<1e-6,'Official camera center frame mismatch')
        cameras.append(cam);rows.append(dict(name=cam.image_name,test_index=int(index),source_K=K.tolist(),
            setting_threshold_scope={n:threshold_scope(t,float(K[0,0])) for n,t in SETTINGS},
            official_matrix_projection_max_error_px=error,camera_center=center.numpy().tolist(),
            saved_center_max_abs_error=float(np.max(np.abs(center.numpy()-c2w[:3,3]))),znear=.01,zfar=100.,sample_pairs=len(xyz)))
    require(len(cameras)==32,'Expected all32 frozen holdouts')
    save(output/'official-camera-audit.json',dict(status='passed_CPU_official_matrices_same_saved_intrinsics_global_frame',per_image=rows,
        matrix_dtype='torch.float32',camera_source='Saved baseline K/C2W; unchanged official getWorld2View2/getProjectionMatrix AST, no scene normalization',
        pixel_convention='GraphDeco mean=(K projection-.5) evaluated at integer indices; compare against gsplat K with index+.5 sampling',
        native_clipping='Official Camera defaults znear=.01/zfar=100 and rasterizer p_view.z>.2; different from gsplat baseline far1e10/near.01',
        cuda_called=False))
    return cameras

def context(args,output):
    source=source_audit(output)
    workflow_root=TRACK/'official-hierarchy'/args.workflow_run;workflow=read_json(workflow_root/'configuration.json')
    require(workflow['main_commit']==COMMIT,'Workflow official source differs')
    manifest=read_json(TRACK/'official-inputs'/workflow['input']/'manifest.json')
    require(digest(TRACK/'official-inputs'/workflow['input']/'manifest.json')==workflow['input_manifest_sha256'],'Workflow input manifest hash mismatch')
    training=TRACK/'training/truck-baseline-001';configuration=read_json(training/'configuration.json')
    inputs,split,original,image_rows,report=baseline.input_audit();save(output/'input-audit.json',report)
    parsed,geometry=baseline.geometry_audit(training,configuration['config'],original,split);save(output/'geometry-audit.json',geometry)
    require(digest(training/'parser_geometry.npz')==workflow['baseline_geometry_sha256'],'Workflow/global camera gauge differs')
    mapped=[image_rows[n]['resized_name'] for n in split['test_names']]
    require(mapped==manifest['test_names'],'Official PNG holdout file differs from frozen identities')
    sky,sky_receipt=skybox_cpu(workflow,output);cameras=camera_cpu(parsed,split,original,output)
    baseline_dir=OUTROOT/'truck-baseline-heldout-001';previous=read_json(baseline_dir/'metrics.json')
    require(previous['status']=='completed_independent_post_update_heldout_evaluation','Exact baseline comparison incomplete')
    require([x['name'] for x in previous['per_image']]==split['test_names'],'Baseline camera order differs')
    targets=[];target_paths=[]
    for name in split['test_names']:
        path=Path(inputs['training_images_path'])/image_rows[name]['resized_name']
        with Image.open(path) as im:im.load();targets.append(np.asarray(im).copy())
        target_paths.append(path)
    desc=[]
    for model in args.models:
        if model=='baseline-bridge':
            root=BUILD/'real/truck-baseline-final-hierarchy-001';path=root/'official-creator/hierarchy.hier';receipt=root/'hierarchy-decoding.json'
            kind='baseline_unoptimized';sky_count=0
        elif model=='merged-unoptimized':
            path=workflow_root/'merged-unoptimized.hier';receipt=workflow_root/'merge-unoptimized-outcome.json';kind='unoptimized';sky_count=sky_receipt['count']
        else:
            path=workflow_root/'merged.hier';receipt=workflow_root/'merge-outcome.json';kind='post_optimized';sky_count=sky_receipt['count']
        desc.append(dict(name=model,path=str(path),receipt_path=str(receipt),expected_source_kind=kind,append_skybox=sky_count,
                         available=path.is_file() and receipt.is_file()))
    plan=dict(status='CPU_protocol_prepared',models=desc,settings=[dict(name=n,tau=t) for n,t in SETTINGS],
        continuous_renderer='Pinned gaussian_renderer.render_post(interp_python=True,use_trained_exp=False); pipe SH/cov Python conversions False',
        compare_families='A baseline Creator bridge has no skybox. B unoptimized vs post-optimized merged trees share one canonical100k skybox; leaves/tau1/3/6 controls plus tau15 extrapolation diagnostic. No per-chunk metrics pooled as whole-scene metrics.',
        exposures=False,depth=False,source_audit=source,skybox=sky_receipt,
        holdout_scope='GS photometric optimization only; full-scene published SfM initialization/calibration. No half-image test exposure training.',
        units='Shared normalized SfM gauge, not metres',native_train_budget='coarse29999+4*chunk29999+4*post14999 optimizer calls; differs from baseline30000',
        stage_timing='One-time immutable property decoding separately recorded. Per-frame synchronized GPU cut, interpolation-weight generation and unchanged official render_post (cached properties+parameter interpolation+native raster+clamp/visibility). No timing describes streaming.')
    save(output/'protocol.json',plan)
    return workflow,desc,parsed,split,sky,cameras,targets,target_paths,previous

def model_cpu(desc,workflow,sky,output):
    import torch
    root=output/desc['name'];root.mkdir()
    require(desc['available'],'Actual model/receipt not yet available: '+desc['name'])
    path=Path(desc['path']);receipt_path=Path(desc['receipt_path']);receipt=read_json(receipt_path)
    if desc['name']=='baseline-bridge':
        require(digest(path)==receipt['sha256'],'Baseline Creator hierarchy identity differs')
        preparation=read_json(BUILD/'real/truck-baseline-final-hierarchy-001/preparation.json')
        original_state=read_json(TRACK/'training/truck-baseline-001/outcome.json')
        require(preparation['final_state_sha256']==original_state['final_state_sha256'],'Baseline bridge source stage differs')
    else:
        require(receipt['status']=='completed','Merge process not completed')
        require(receipt.get('source_kind')==desc['expected_source_kind'],'Merge receipt has wrong explicit source kind')
        require(digest(path)==receipt['hierarchy_sha256'] and receipt.get('skybox_in_tree')==0,'Merged tree changed or includes unexpected skybox')
        require(receipt.get('shared_background_skybox_count')==desc['append_skybox'],'Merge background count differs')
    h=read_hierarchy(path)
    require(h.unowned_tail==0,'Merged/baseline tree must have no unowned tail before canonical background')
    topology=Topology(h);gh,tensors,leaves=author_cpu_load(h,root)
    check=topology.audit(leaves,topology.owner[leaves]);require(check['every_leaf_exactly_once'],'Leaf traversal coverage failed')
    require(np.array_equal(np.sort(leaves),topology.leaf_storage),'Author leaf traversal differs from stored tree leaves')
    for name,t in zip(('xyz','SH','activated_alpha','log_scales','wxyz'),tensors[:5]):
        require(t.dtype==torch.float32 and torch.isfinite(t).all().item(),'Nonfinite model '+name)
    require((torch.linalg.vector_norm(tensors[4],dim=1)>0).all().item(),'Zero model quaternion')
    require(torch.isfinite(torch.exp(tensors[3])).all().item() and (torch.exp(tensors[3])>0).all().item(),'Invalid decoded scale')
    require((tensors[2]>=0).all().item(),'Stored hierarchy alpha must have been saved via abs')
    n=len(tensors[0]);append=int(desc['append_skybox'])
    if append:
        require(append==len(sky[0])==100000,'Expected exactly one canonical skybox')
        tensors=tuple(torch.cat((t,background),dim=0) for t,background in zip(tensors[:5],sky))+tuple(tensors[5:])
    require(len(tensors[0])==n+append and int(tensors[5][:,3:5].sum())==n,'Skybox counted in tree or appended more than once')
    result=dict(status='passed_actual_model_identity_and_single_background',model=desc['name'],source_kind=desc['expected_source_kind'],
        source_path=str(path),source_sha256=digest(path),source_receipt_path=str(receipt_path),source_receipt_sha256=digest(receipt_path),
        tree_summary=h.summary(),tree_G=n,background_G=append,total_G=len(tensors[0]),skybox_copy_count=int(append>0),
        leaf_count=len(leaves),node_storage_sha256=array_digest(h.nodes),box_storage_sha256=array_digest(h.boxes,np.float32),
        xyz_storage_sha256=array_digest(h.positions),activated_alpha_gt_one=int((tensors[2][:n]>1).sum()),activated_alpha_max=float(tensors[2][:n].max()),
        official_properties='exp logs; normalize raw wxyz; abs already-activated hierarchy alpha; coefficient-major SH3',
        leaf_control=check,source_receipt=receipt,
        merge_comparison_limit='Merger ownership/falloff remains common, but post changes geometry and may change retained leaves/tree structure; compare measured counts/hashes without assuming equal topology')
    save(root/'model-audit.json',result)
    property_cache_cpu(tensors,root)
    plan=memory_plan(len(tensors[0]),len(h.nodes),len(leaves),len(leaves),append)
    save(root/'memory-preflight-CPU.json',plan)
    return root,h,topology,gh,tensors,leaves,result

def property_cache_cpu(tensors,output):
    """Evaluate actual author property AST, without importing scene or CUDA."""
    import torch
    tree=ast.parse((REPO/'scene/gaussian_model.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GaussianModel')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'get_xyz','get_features','get_scaling','get_rotation','get_opacity'}]
    require(len(methods)==5,'Missing official property definitions')
    scope={'torch':torch};exec(compile(ast.Module(body=methods,type_ignores=[]),'official-property-AST','exec'),scope)
    rows=[]
    for begin in range(0,len(tensors[0]),65536):
        a=[t[begin:begin+65536] for t in tensors[:5]]
        pc=SimpleNamespace(_xyz=a[0],_features_dc=a[1][:,:1],_features_rest=a[1][:,1:16],_opacity=a[2],_scaling=a[3],_rotation=a[4],
            scaling_activation=torch.exp,rotation_activation=torch.nn.functional.normalize,opacity_activation=torch.abs)
        props={n:scope[n].fget(pc) for n in scope if n.startswith('get_')}
        require(torch.equal(props['get_features'],a[1]),'Existing SH array differs from official DC/rest concatenation')
        require(torch.equal(props['get_xyz'],a[0]),'Official XYZ property differs')
        for n in ('get_scaling','get_rotation','get_opacity'):require(torch.isfinite(props[n]).all().item(),'Nonfinite cached property '+n)
        rows.append(dict(start=begin,count=len(a[0]),properties={n:array_digest(t.numpy()) for n,t in props.items()}))
    result=dict(status='passed_all_rows_actual_author_property_AST_CPU',rows_checked=len(tensors[0]),chunk_rows=65536,per_chunk=rows,
        equivalence='GPU cache calls actual GaussianModel property fget once for exp/logscale, normalize/wxyz and abs/activated-alpha. SH reuses existing array; all rows equal original torch.cat(DC,rest). Static source tensors never updated; renderer parent-sign changes target advanced-index copies only.',
        timing='One-time decoding excluded from per-frame rendering, separately recorded; no claim to repeat official per-frame get_features allocation cost',cuda_called=False)
    save(output/'property-cache-CPU-audit.json',result)
    return result

def weight_reference(h,nodeids,camera,theta):
    from read_hierarchy import official_size_proxy_cpu
    sizes=official_size_proxy_cpu(h,camera);parent=h.nodes[nodeids,1];valid=parent>=0
    t=np.ones(len(nodeids),dtype=np.float32);kids=np.ones(len(nodeids),dtype=np.int32)
    p=parent[valid];kids[valid]=h.nodes[p,6]
    par=sizes[p];child=sizes[nodeids[valid]]
    start=np.maximum(np.float32(.5)*par,child);diff=par-start
    calc=(par<=np.float32(2)*np.float32(theta))&(diff>0)
    values=np.ones(len(par),dtype=np.float32)
    values[calc]=np.maximum(np.float32(1)-np.maximum(np.float32(0),np.float32(theta)-start[calc])/diff[calc],np.float32(0))
    t[valid]=values
    return t,kids

def gpu_model(desc,workflow,sky,cameras,targets,split,output):
    import torch
    root,h,topology,gh,tensors,leaves,model_audit=model_cpu(desc,workflow,sky,output)
    sys.path.insert(0,str(EXT));sys.path.insert(0,str(REPO))
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render_post
    import gaussian_renderer
    import diff_gaussian_rasterization as native
    require(Path(gaussian_renderer.__file__).resolve()==(REPO/'gaussian_renderer/__init__.py').resolve(),'Wrong official renderer import')
    require(Path(native._C.__file__).resolve().is_relative_to(EXT.resolve()),'Wrong native renderer binary import')
    torch.cuda.empty_cache();before=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
    plan=read_json(root/'memory-preflight-CPU.json');free,total_vram=torch.cuda.mem_get_info()
    plan.update(free_device_bytes_before_load=free,total_device_bytes=total_vram,
        decision='proceed_with_native_workspace_headroom' if plan['conservative_Python_tensor_peak_upper_bytes']+1024**3<free else 'reject_before_GPU_model_allocation')
    save(root/'memory-preflight-GPU.json',plan)
    require(plan['decision'].startswith('proceed'),'Known Python tensor upper bound leaves less than1GiB native workspace; do not allocate full GPU model')
    gpu=[t.to('cuda') for t in tensors]
    class CachedModel(GaussianModel):
        @property
        def get_features(self):return self._cached_features
        @property
        def get_scaling(self):return self._cached_scales
        @property
        def get_rotation(self):return self._cached_rotations
        @property
        def get_opacity(self):return self._cached_opacity
    pc=CachedModel(3);pc.active_sh_degree=3
    pc._xyz,features,pc._opacity,pc._scaling,pc._rotation,pc.nodes,pc.boxes=gpu
    pc._features_dc=features[:,:1];pc._features_rest=features[:,1:16]
    pc.opacity_activation=torch.abs;pc.inverse_opacity_activation=torch.abs
    pc.skybox_points=int(desc['append_skybox']);pc.pretrained_exposures=None
    torch.cuda.synchronize();began=time.perf_counter()
    pc._cached_features=features
    pc._cached_scales=GaussianModel.get_scaling.fget(pc)
    pc._cached_rotations=GaussianModel.get_rotation.fget(pc)
    pc._cached_opacity=GaussianModel.get_opacity.fget(pc)
    torch.cuda.synchronize();decode=time.perf_counter()-began
    require(all(not t.requires_grad for t in (features,pc._cached_scales,pc._cached_rotations,pc._cached_opacity)),'Property cache must be immutable inference parameters')
    save(root/'property-cache-GPU.json',dict(status='actual_author_property_fget_decoded_once',decode_seconds=decode,
        cached_features_aliases_existing_loaded_SH=features.data_ptr()==pc.get_features.data_ptr(),
        decoder='Actual GaussianModel.get_scaling/get_rotation/get_opacity fget on unmodified raw tensors; identical source function and tensor dtype/device',
        newly_resident_cache_bytes=sum(t.numel()*t.element_size() for t in (pc._cached_scales,pc._cached_rotations,pc._cached_opacity)),
        per_frame_limit='Official render_post source unchanged, property values cached; timing excludes repeated full-model decoding and feature concatenation'))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');viewdir=torch.zeros(3,dtype=torch.float32)
    views=[]
    for c in cameras:
        fields=vars(c).copy()
        for key in ('world_view_transform','projection_matrix','full_proj_transform','camera_center'):fields[key]=fields[key].to('cuda')
        views.append(SimpleNamespace(**fields))
    total=len(pc._xyz);indices=torch.empty(total,device='cuda',dtype=torch.int32);parents=torch.empty_like(indices);nodeids=torch.empty_like(indices)
    weights=torch.zeros(total,device='cuda');kids=torch.ones(total,device='cuda',dtype=torch.int32)
    leaf_gpu=torch.from_numpy(leaves).to('cuda');leaf_nodes=topology.owner[leaves]
    # Leaf weights are1, so parent parameters contribute0; self parent avoids
    # unnecessary root=-1 tail access in this control only.
    leaf_par=leaves.astype(np.int32).copy()
    leaf_parent_gpu=torch.from_numpy(leaf_par).to('cuda');leaf_node_gpu=torch.from_numpy(leaf_nodes).to('cuda')
    torch.cuda.synchronize()
    resident=dict(model=desc['name'],tree_G=model_audit['tree_G'],skybox_G=pc.skybox_points,total_G=total,
        allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),before_load=before,
        raw_tensor_bytes=sum(t.numel()*t.element_size() for t in gpu),cached_parameter_bytes=32*total,
        scope='Whole model/tree plus one canonical background and runtime buffers resident; no streaming or chunk eviction')
    save(root/'resident-memory.json',resident)

    @torch.no_grad()
    def select(j,tau):
        if tau is None:return leaf_gpu,leaf_parent_gpu,leaf_node_gpu,None
        theta=tau_to_threshold(tau,245/(2*math.tan(views[j].FoVx*.5)))
        count=gh.expand_to_size(pc.nodes,pc.boxes,theta,views[j].camera_center.contiguous(),viewdir,indices,parents,nodeids)
        require(0<=count<=model_audit['tree_G'],'Cut count outside actual tree')
        return indices[:count],parents[:count],nodeids[:count],theta

    @torch.no_grad()
    def getweights(j,tau,ids,nids,theta):
        if tau is None:weights[:len(ids)+pc.skybox_points]=1;kids[:len(ids)+pc.skybox_points]=1
        else:gh.get_interpolation_weights(nids.contiguous(),theta,pc.nodes,pc.boxes,cameras[j].camera_center.contiguous(),viewdir,weights,kids)

    @torch.no_grad()
    def render(j,ids,pids):
        return render_post(views[j],pc,pipe,bg,render_indices=ids,parent_indices=pids,
            interpolation_weights=weights,num_node_kids=kids,interp_python=True,use_trained_exp=False)

    results=[]
    for setting,tau in SETTINGS:
        folder=root/setting;folder.mkdir();selections=folder/'selections';selections.mkdir();rows=[];predictions=[]
        leaf_receipt=None
        if tau is None:leaf_receipt=save_selection(selections/'fixed-author-leaves.npz',leaves,leaf_par,leaf_nodes)
        for j in range(3):
            ids,pids,nids,theta=select(j,tau)
            if len(ids):
                getweights(j,tau,ids,nids,theta);warm=render(j,ids,pids);torch.cuda.synchronize();del warm
        torch.cuda.synchronize()
        for j,name in enumerate(split['test_names']):
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
            ids,pids,nids,theta=select(j,tau);torch.cuda.synchronize();cut=time.perf_counter()-began
            cut_peak=torch.cuda.max_memory_allocated();cut_reserved=torch.cuda.max_memory_reserved()
            selected=ids.cpu().numpy().copy();parent_cpu=pids.cpu().numpy().copy();node_cpu=nids.cpu().numpy().copy()
            checked=topology.audit(selected,node_cpu)
            parent_node=topology.parent[node_cpu];parent_reference=np.where(parent_node>=0,topology.start[np.maximum(parent_node,0)],-1)
            require(np.array_equal(parent_cpu,selected if tau is None else parent_reference),'Author parent-index ownership mismatch')
            counts_match=True;root_fallback=False
            if tau is not None:
                expected,sizes=topology.reference_counts(cameras[j].camera_center.numpy(),theta)
                counts_match=bool(np.array_equal(np.bincount(node_cpu,minlength=len(expected)),expected))
                root_fallback=bool(sizes[0]<np.float32(theta)) and not checked['every_leaf_exactly_once']
            receipt=leaf_receipt if tau is None else save_selection(selections/f'{j:02d}-{Path(name).stem}.npz',selected,parent_cpu,node_cpu)
            row=dict(name=name,model=desc['name'],setting=setting,tau=tau,**threshold_scope(tau,245/(2*math.tan(views[j].FoVx*.5))),selected=receipt,
                coverage=checked,author_cut_vs_CPU_predicate_match=counts_match,root_fallback_failure=root_fallback,
                background_count=pc.skybox_points,skybox_copy_count=int(pc.skybox_points>0),
                cut_seconds=cut if tau is not None else 0.,leaf_fixed_index_reuse_seconds=cut if tau is None else None,
                cut_peak_allocated_bytes=cut_peak,cut_peak_reserved_bytes=cut_reserved)
            if not len(ids) or not checked['every_leaf_exactly_once'] or not counts_match:
                row.update(status='invalid_cut_no_render');rows.append(row);save(folder/'per-frame-audit.json',rows);continue
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
            getweights(j,tau,ids,nids,theta);torch.cuda.synchronize();weight_time=time.perf_counter()-began
            weight_peak=torch.cuda.max_memory_allocated();weight_reserved=torch.cuda.max_memory_reserved()
            ts=weights[:len(ids)].cpu().numpy().copy();siblings=kids[:len(ids)].cpu().numpy().copy()
            require(np.isfinite(ts).all() and np.all((ts>=0)&(ts<=1)) and np.all(siblings>0),'Invalid author interpolation weights/sibling counts')
            if tau is None:tr=np.ones(len(ids),np.float32);kr=np.ones(len(ids),np.int32)
            else:tr,kr=weight_reference(h,node_cpu,cameras[j].camera_center.numpy(),theta)
            matched=bool(np.allclose(ts,tr,rtol=0,atol=2e-6) and np.array_equal(siblings,kr))
            row.update(interpolation_seconds=weight_time,weight_peak_allocated_bytes=weight_peak,weight_peak_reserved_bytes=weight_reserved,
                weight_vs_CPU_reference_match=matched,child_weight_min=float(ts.min()),child_weight_max=float(ts.max()),
                sibling_count_min=int(siblings.min()),sibling_count_max=int(siblings.max()),
                interpolation_weight_bytes_sha256=array_digest(ts),sibling_count_bytes_sha256=array_digest(siblings))
            if not matched:
                row.update(status='invalid_interpolation_no_render');rows.append(row);save(folder/'per-frame-audit.json',rows);continue
            np.savez(selections/f'{j:02d}-{Path(name).stem}-interpolation.npz',child_weights=ts,num_parent_children=siblings)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
            rendered=render(j,ids,pids);torch.cuda.synchronize();render_time=time.perf_counter()-began
            render_peak=torch.cuda.max_memory_allocated();render_reserved=torch.cuda.max_memory_reserved()
            rgb=rendered['render'];finite=bool(torch.isfinite(rgb).all().item())
            require(tuple(rgb.shape)==(3,136,245),'Wrong official RGB shape')
            # Author render_post clones the weight buffer then sets sky weights1;
            # it writes sky kids1 into the caller buffer. The source weight
            # buffer is not evidence of the clone's tail values.
            require(torch.all(kids[len(ids):len(ids)+pc.skybox_points]==1).item(),'Official renderer did not assign background sibling counts')
            if not finite:
                row.update(status='nonfinite_render_no_metrics');rows.append(row);save(folder/'per-frame-audit.json',rows);del rendered,rgb;continue
            predictions.append(rgb.permute(1,2,0).cpu().numpy().copy())
            row.update(status='valid_selection_interpolation_finite_render',official_render_post_seconds=render_time,
                sum_synchronized_GPU_stages_seconds=(cut if tau is not None else 0.)+weight_time+render_time,
                render_peak_allocated_bytes=render_peak,render_peak_reserved_bytes=render_reserved,
                background_weights_rule='Pinned render_post clones weights and sets exactly the appended sky range to1; caller kids range observed1; source weights are not silently treated as the clone',
                RGB_min=float(rgb.min()),RGB_max=float(rgb.max()),visible_radii_count=int(len(rendered['radii'])))
            rows.append(row);del rendered,rgb
        valid=[r for r in rows if r['status']=='valid_selection_interpolation_finite_render'];complete=len(valid)==32
        timing=dict(status='all32_valid' if complete else 'partial_failures_no_pooled_metrics',valid_frames=len(valid),total_frames=len(rows),warmups=3,
            stages={},resident_memory=resident,
            boundary='Separate CUDA-synchronized author expand_to_size; get_interpolation_weights (or leaf unit weights); unchanged render_post with cached immutable official properties, parameter interpolation, sibling-alpha rasterization, internal clamp and visibility work. One-time parameter decoding, transfers, CPU audits, image saves and metrics excluded.',
            total_definition='Sum of separately synchronized GPU stages; CPU gaps and extra synchronization exclude continuous playback/streaming throughput claims')
        if complete:
            for key in ('cut_seconds','interpolation_seconds','official_render_post_seconds','sum_synchronized_GPU_stages_seconds'):
                timing['stages'][key]=summarize([r[key] for r in valid])
            timing['selected_tree_G']=summarize([r['selected']['count'] for r in valid])
            timing['threshold_training_scope']=dict(all_views_in_training_range=None if tau is None else all(r['in_training_range'] for r in valid),
                per_view=[{k:r[k] for k in ('name','fx_px','threshold','in_training_range','diagnostic_scope')} for r in valid])
            timing['max_peak_allocated_bytes']=max(max(r[k] for k in ('cut_peak_allocated_bytes','weight_peak_allocated_bytes','render_peak_allocated_bytes')) for r in valid)
            timing['max_peak_reserved_bytes']=max(max(r[k] for k in ('cut_peak_reserved_bytes','weight_peak_reserved_bytes','render_peak_reserved_bytes')) for r in valid)
            np.savez(folder/'predictions.npz',image_names=np.array(split['test_names']),rgb=np.stack(predictions))
        save(folder/'per-frame-audit.json',rows);save(folder/'timings.json',timing)
        results.append(dict(model=desc['name'],setting=setting,tau=tau,folder=folder,complete=complete,predictions=predictions,timing=timing,model_audit=model_audit))
        print(json.dumps(dict(model=desc['name'],setting=setting,valid_frames=len(valid),selected=timing.get('selected_tree_G'))),flush=True)
    require(digest(h.path)==model_audit['source_sha256'],'Read-only model changed during evaluation')
    del pc,gpu,tensors,features,indices,parents,nodeids,weights,kids,leaf_gpu,leaf_parent_gpu,leaf_node_gpu,views
    torch.cuda.synchronize()
    return results

def metrics(all_results,targets,target_paths,split,previous,output):
    import torch
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    import torchmetrics.functional.image.lpips as lpips_source
    paths=[Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth',Path(lpips_source.__file__).parent/'lpips_models/alex.pth']
    require(all(p.is_file() for p in paths),'LPIPS weights missing; no downloading')
    weights=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in paths]
    require(weights==previous['weight_files'],'Perceptual weights differ from frozen baseline')
    ssim=StructuralSimilarityIndexMeasure(data_range=1.).to('cuda').eval();lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to('cuda').eval()
    summaries=[]
    for result in all_results:
        folder=result['folder'];paired=folder/'paired';predicted=folder/'predicted';paired.mkdir();predicted.mkdir();rows=[]
        if not result['complete']:
            summaries.append(dict(model=result['model'],setting=result['setting'],status='incomplete_no_average',timing=result['timing']));continue
        for j,(name,pred,gt8) in enumerate(zip(split['test_names'],result['predictions'],targets)):
            require(np.isfinite(pred).all() and np.all((pred>=0)&(pred<=1)),'Official clamped RGB invalid')
            mse=float(np.mean((pred.astype(np.float64)-gt8.astype(np.float64)/255.)**2));require(mse>0 and np.isfinite(mse),'Invalid PSNR')
            psnr=float(-10*np.log10(mse));pt=torch.from_numpy(pred).permute(2,0,1)[None].to('cuda');gt=torch.from_numpy(gt8).permute(2,0,1)[None].to(device='cuda',dtype=torch.float32)/255.
            with torch.no_grad():sv=float(ssim(pt,gt).cpu());lv=float(lpips(pt,gt).cpu())
            ssim.reset();lpips.reset();require(np.isfinite(sv+lv),'Nonfinite SSIM/LPIPS')
            png=(pred*255).astype(np.uint8);p=predicted/f'{j:02d}-{Path(name).stem}.png';q=paired/p.name
            Image.fromarray(png,'RGB').save(p);Image.fromarray(np.concatenate([gt8,png],axis=1),'RGB').save(q)
            rows.append(dict(name=name,width=245,height=136,mse_numpy_float64=mse,psnr_numpy_db=psnr,ssim_torchmetrics=sv,lpips_alex_torchmetrics=lv,
                target_path=str(target_paths[j]),target_sha256=digest(target_paths[j]),prediction_path=str(p),prediction_sha256=digest(p),paired_path=str(q),paired_sha256=digest(q)))
            del pt,gt
        m=dict(status='complete32_valid_official_continuous_renderer',model=result['model'],setting=result['setting'],tau=result['tau'],test_images=32,
            psnr_numpy_mean_per_image_db=float(np.mean([r['psnr_numpy_db'] for r in rows])),ssim_mean_per_image=float(np.mean([r['ssim_torchmetrics'] for r in rows])),
            lpips_alex_mean_per_image=float(np.mean([r['lpips_alex_torchmetrics'] for r in rows])),per_image=rows,
            psnr_aggregation=previous['psnr_aggregation'],ssim_protocol=previous['ssim_protocol'],lpips_protocol=previous['lpips_protocol'],weight_files=weights,
            photometric_scope='AllRGB245x136; official render_post internal[0,1] clamp before float metrics/PNGquantization; black background; no exposure correction or mask',
            threshold_training_scope=result['timing']['threshold_training_scope'],timing=result['timing'],model_audit=result['model_audit'])
        save(folder/'metrics.json',m);save(folder/'per-image-metrics.json',rows)
        summaries.append({k:v for k,v in m.items() if k not in {'per_image','weight_files'}})
    pairs=[]
    for setting,_ in SETTINGS:
        a=next((r for r in summaries if r['model']=='merged-unoptimized' and r['setting']==setting),None)
        b=next((r for r in summaries if r['model']=='merged-post' and r['setting']==setting),None)
        if a and b and a['status']==b['status']=='complete32_valid_official_continuous_renderer':
            pairs.append(dict(setting=setting,threshold_training_scope=b['threshold_training_scope'],post_minus_unoptimized={k:b[k]-a[k] for k in ('psnr_numpy_mean_per_image_db','ssim_mean_per_image','lpips_alex_mean_per_image')},
                pre_tree_G=a['model_audit']['tree_G'],post_tree_G=b['model_audit']['tree_G'],pre_leaves=a['model_audit']['leaf_count'],post_leaves=b['model_audit']['leaf_count'],
                pre_nodes=a['model_audit']['tree_summary']['node_count'],post_nodes=b['model_audit']['tree_summary']['node_count'],
                node_bytes_identical=a['model_audit']['node_storage_sha256']==b['model_audit']['node_storage_sha256'],
                boxes_identical=a['model_audit']['box_storage_sha256']==b['model_audit']['box_storage_sha256']))
    result=dict(status='complete_selected_models_all32_views' if all(r['complete'] for r in all_results) else 'incomplete_failures_preserved',settings=summaries,paired_merge_stage_comparison=pairs,
        renderer='Pinned official render_post + native hierarchy-rasterizer; true parameter and per-pixel sibling-alpha interpolation, not gsplat opacity approximation',
        comparison_scope='A single-map baseline Creator bridge separate from B actual four-chunk merged pre/post workflows with common100k background. Cross-family differences are not an isolated partitioning/post effect.',
        memory_scope='Full model resident, PyTorch allocated/reserved; no streaming claim',
        clipping_limit='Official rasterizer positive-depth/frustum convention differs from gsplat; official render includes internal clamp in its timer',
        extrapolation_limit='Truck tau15 threshold is outside official post training [.005,.1]; interpret separately, never generalize its quality/failure to in-range post effectiveness.')
    save(output/'metrics.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-run',required=True);p.add_argument('--workflow-run',default='truck-official-hierarchy-001')
    p.add_argument('--models',nargs='+',choices=['merged-unoptimized','merged-post','baseline-bridge'],default=['merged-unoptimized','merged-post','baseline-bridge'])
    p.add_argument('--protocol-only',action='store_true');p.add_argument('--cpu-only',action='store_true')
    args=p.parse_args();require(args.evaluation_run.replace('-','').isalnum() and args.workflow_run.replace('-','').isalnum(),'Invalid run name')
    require(len(set(args.models))==len(args.models),'Duplicate model family')
    output=OUTROOT/args.evaluation_run;output.mkdir(parents=True,exist_ok=False)
    outcome=dict(status='running',started_utc=baseline.now(),protocol_only=args.protocol_only,cpu_only=args.cpu_only,cuda_used=False)
    save(output/'outcome.json',outcome);started=time.perf_counter()
    try:
        workflow,descs,parsed,split,sky,cameras,targets,target_paths,previous=context(args,output)
        versions={k:importlib.metadata.version(k) for k in FIXED};require(versions==FIXED,'Fixed metric/runtime versions changed')
        if args.protocol_only:outcome.update(status='passed_CPU_protocol_pending_actual_models_no_GPU')
        elif args.cpu_only:
            for d in descs:model_cpu(d,workflow,sky,output)
            outcome.update(status='passed_CPU_selected_model_audits_no_GPU')
        else:
            import torch
            require(torch.cuda.is_available(),'CUDA unavailable')
            results=[]
            for d in descs:results.extend(gpu_model(d,workflow,sky,cameras,targets,split,output))
            result=metrics(results,targets,target_paths,split,previous,output)
            outcome.update(status=result['status'],cuda_used=True)
            lines=['# 官方连续层级渲染独立评测','',
                '冻结 32 个留出相机与 245×136 RGB 输入，调用未修改的官方 render_post(interp_python=True) 和固定作者光栅扩展。包括真实父子参数插值及每像素 sibling alpha 修正，曝光关闭。',
                '', 'A baseline bridge 保持原单地图、无 coarse 天空背景；B 未 post 与已 post 的四块合并树各仅追加同一 coarse 的 100,000 个天空高斯一次。A/B 工作流和迭代预算不同，不能作孤立因果归因。','']
            lines+=['τ1、τ3、τ6 对应阈值处于官方 post 的 [0.005,0.1] 训练范围，τ15 仅为范围外推诊断。每张相机的 fx、阈值与 in_training_range 均单独记录；leaves 是完整叶层控制。','']
            for s in result['settings']:
                if s['status']!='complete32_valid_official_continuous_renderer':lines.append(f"{s['model']}/{s['setting']} 有失败，未汇总平均。");continue
                times=s['timing']['stages']
                lines.append(f"{s['model']}/{s['setting']}：PSNR {s['psnr_numpy_mean_per_image_db']:.6f} dB，SSIM {s['ssim_mean_per_image']:.6f}，Alex LPIPS {s['lpips_alex_mean_per_image']:.6f}；cut／权重／官方 render p50 {times['cut_seconds']['p50']*1000:.4f}／{times['interpolation_seconds']['p50']*1000:.4f}／{times['official_render_post_seconds']['p50']*1000:.4f} ms。")
            lines+=['','同一 B 系列的未 post／post 差、实际 leaf／node 数量及结构哈希比较见 metrics.json，不预设优化后 ownership 或结构完全相同。',
                '', '阶段分别 CUDA 同步，官方 render 计时包含缓存参数的插值、光栅化、内部 clamp 与可见性整理；作者原属性解码只做一次并独立记录，不含 CPU 审计、传输、写图和指标。SH 复用已有数组，避免每帧整模型拼接。完整树、缓存和天空背景常驻 GPU，不能解释为 streaming 显存节省或连续播放吞吐率。',
                '', '每帧选点、parent/node IDs、插值权重、 sibling 数、覆盖与有限性检查、真实配对图及固定网络权重哈希均保留。只在一组32帧均通过后平均。留出仅针对 GS 图像优化，全场景 SfM 仍提供标定与稀疏初始化；归一化长度不是米。']
            (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        outcome.update(ended_utc=baseline.now(),elapsed_seconds=time.perf_counter()-started,evaluation_script_sha256=digest(__file__))
        save(output/'outcome.json',outcome);print(json.dumps(outcome),flush=True)
    except Exception as e:
        outcome.update(status='failed_preserved_independent_evaluation',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),ended_utc=baseline.now())
        save(output/'outcome.json',outcome);raise

if __name__=='__main__':main()
