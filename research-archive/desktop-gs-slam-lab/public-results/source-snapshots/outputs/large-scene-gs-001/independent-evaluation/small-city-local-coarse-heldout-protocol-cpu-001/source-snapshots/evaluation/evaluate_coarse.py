"""Independent SmallCity local coarse flat evaluation. CPU plan is the default.

--bind-model-cpu requires a real completed stage. --gpu additionally requires a
fresh explicit exclusive GPU authorization. No tree, cut, LOD or sky append.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,shutil,sys,time,traceback
from types import SimpleNamespace
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/large_scene_gs'))
import evaluate_official_hierarchy as e
INPUT=e.TRACK/'official-inputs/small-city-training-002'
TRAIN=e.TRACK/'official-training/small-city-coarse-i30000-r4-002'
INPUT_SHA='2dd5c04122e290d27934a224921b2506b7955a34d801a0482e0408af536de3d5'
EXPECTED_G=2490040;SKY_PREFIX=100000
SCOPE='Local coarse background model,117 frozen GS-photometric heldout views; 5705 training views. Published full-scene SfM calibration/XYZ/colors may use all photos. Not dense-training equal-budget causal comparison or author-pretrained reproduction.'

def context(output):
    source=e.source_audit(output)
    (output/'source-snapshots/evaluation/evaluate_coarse.py').write_bytes(Path(__file__).read_bytes())
    manifest_path=INPUT/'manifest.json';manifest=e.read_json(manifest_path)
    e.require(e.digest(manifest_path)==INPUT_SHA and manifest['status']=='ready_for_official_small_city_training_cpu_prepared','Wrong/changed002input')
    split_path=INPUT/'frozen-split.json';e.require(e.digest(split_path)==manifest['frozen_split_sha256'],'Frozen split changed')
    split=e.read_json(split_path);names=split['test_names'];train=split['train_names']
    e.require(names==manifest['test_names'] and train==manifest['train_names'] and len(names)==117 and len(train)==5705 and not(set(names)&set(train)),'Split/order/count mismatch')
    config_path=TRAIN/'configuration.json';config=e.read_json(config_path)
    e.require(config['stage']=='coarse' and config['official_commit']==e.COMMIT and not config['source_changed'],'Wrong trainer identity')
    e.require(config['input_manifest_sha256']==INPUT_SHA and config['training_names']==train and config['stage_test_names']==names and config['frozen_global_test_names']==names,'Training split differs')
    e.require(config['train_test_exp'] is False and config['effective_official_arguments']['eval'] is True and config['requested_iterations']==30000 and config['expected_optimizer_update_calls']==29999,'Expected evalTrue/nohalf exposure/I30000 state')
    e.require(config['resolution_pixels']==[256,172] and config['exposure_parameter_updates_enabled'] is False,'Wrong coarse resolution/exposure')
    test_path=INPUT/'aligned/sparse/0/test.txt';e.require(Path(config['official_test_path']).resolve()==test_path.resolve() and e.digest(test_path)==config['official_test_sha256'],'Actual official testfile identity mismatch')
    e.require(test_path.read_text().splitlines()==[Path(n).stem for n in names],'Real official test lines differ')
    reader_path=INPUT/'official-reader-CPU-audit.json';reader=e.read_json(reader_path)
    e.require(reader['status']=='actual_official_reader_AST_CPU_passed' and e.digest(reader_path)==config['official_reader_CPU_audit_sha256'],'Real Reader audit changed')
    proto=Path(manifest['protocol_source']);e.require(e.digest(proto/'manifest.json')==manifest['protocol_manifest_sha256'],'Original camera protocol changed')
    cameras_path=proto/'global-cameras.npz';global_data=np.load(cameras_path);all_names=global_data['names'].tolist();lookup={name:j for j,name in enumerate(all_names)}
    image_root=Path(manifest['images_absolute']);mask_root=Path(manifest['masks_absolute'])
    chunks={c['name']:set(c['train_names'])|set(c['test_names']) for c in manifest['chunks']}
    inputs=output/'independent-inputs';inputs.mkdir();rgb_dir=inputs/'RGB';mask_dir=inputs/'masks';rgb_dir.mkdir();mask_dir.mkdir()
    cameras=[];targets=[];masks=[];target_paths=[];rows=[];K_values=[];poses=[];indices=[]
    camera_math=e.official_camera_math_CPU()
    for name in names:
        j=lookup[name];photo=image_root/name;mask_path=mask_root/(Path(name).stem+'.png')
        with Image.open(photo) as image:
            image.load();e.require(image.size==(1024,690),'Unexpected original photo dimensions');raw=np.asarray(image.convert('RGB')).copy();rgb=np.asarray(image.convert('RGB').resize((256,172))).copy()
        with Image.open(mask_path) as image:
            image.load();e.require(image.mode=='L' and image.size==(1024,690),'Unexpected originalmask');raw_mask=np.asarray(image).copy();mask8=np.asarray(image.resize((256,172))).copy()
        mask=mask8.astype(np.float32)[...,None]/255.
        K=global_data['K'][j].copy();original_K=K.copy();K[0]*=256/1024;K[1]*=172/690;c2w=global_data['c2w'][j].copy()
        camera=e.make_official_camera_CPU(K,c2w,256,172,name,camera_math)
        e.require(np.isfinite(camera.full_proj_transform.numpy()).all(),'Nonfinite camera matrix')
        rgbpath=rgb_dir/(Path(name).stem+'.png');maskpng=mask_dir/(Path(name).stem+'.png');Image.fromarray(rgb).save(rgbpath);Image.fromarray(mask8).save(maskpng)
        membership=sorted(c for c,values in chunks.items() if name in values);group='inside_any_published_chunk_view_list' if membership else 'outside_all_published_chunk_view_lists'
        rows.append(dict(name=name,source_camera_row=j,source_RGB_path=str(photo),source_RGB_sha256=e.digest(photo),source_RGB_decoded_pixels_sha256=e.array_digest(raw),
            source_mask_path=str(mask_path),source_mask_sha256=e.digest(mask_path),source_mask_decoded_pixels_sha256=e.array_digest(raw_mask),source_mask_mode='L uint8',
            original_dimensions=[1024,690],derived_dimensions=[256,172],original_K=original_K.tolist(),K=K.tolist(),C2W=c2w.tolist(),
            RGB_path=str(rgbpath),RGB_sha256=e.digest(rgbpath),mask_png_path=str(maskpng),mask_png_sha256=e.digest(maskpng),mask_float32_sha256=e.array_digest(mask),
            group=group,published_chunk_views=membership,soft_mask_mean=float(mask.mean()),mask_squared_weight_sum=float(np.sum(mask.astype(np.float64)**2))))
        cameras.append(camera);targets.append(rgb);target_paths.append(rgbpath);masks.append(mask);K_values.append(K);poses.append(c2w);indices.append(j)
    counts={g:sum(r['group']==g for r in rows) for g in set(r['group'] for r in rows)}
    e.require(sorted(counts.values())==[22,95],'Expected117 camera-list membership95/22')
    archive=inputs/'evaluated-inputs.npz';np.savez(archive,names=np.array(names),rgb=np.stack(targets),mask=np.stack(masks),K=np.stack(K_values),c2w=np.stack(poses),source_camera_rows=np.array(indices))
    protocol=dict(status='frozen117_actual_RGB_mask_camera_CPU_protocol',view_scope=SCOPE,train_count=5705,test_count=117,test_names=names,
        manifest_path=str(manifest_path),manifest_sha256=INPUT_SHA,frozen_split_path=str(split_path),frozen_split_sha256=e.digest(split_path),
        actual_official_test_path=str(test_path),actual_official_test_sha256=e.digest(test_path),official_reader_audit_sha256=e.digest(reader_path),
        training_configuration_path=str(config_path),training_configuration_sha256=e.digest(config_path),global_camera_archive_path=str(cameras_path),global_camera_archive_sha256=e.digest(cameras_path),
        evaluated_input_npz_path=str(archive),evaluated_input_npz_sha256=e.digest(archive),source_audit=source,views=rows,groups=counts,
        camera_group_semantics='Published chunk camera-list membership, not spatial inside/outside or sole coarse support; all groups evaluated for this global coarse model',
        resize='Source RGB/L default Pillow resize256x172; x K scale256/1024 andy172/690, originalrectified photos untouched',
        masks='Official Camera theneval GTm2/predm full3WH; auxiliary symmetric single-mask valid weighted PSNR separate',
        exposure='Disabled for evaluation. train_test_expFalse; source coarse has no exposure optimizer calls; completed model will independently verify5705identity/exclude117 before binding',
        model_state='Future required completed actualI30000 snapshot with29999optimizer calls; no model SHA invented or GPU access now',
        renderer='Unmodified gaussian_renderer.render flat/direct, indicesNone/use_trained_expFalse/activeSH1. Original PLY2490040 rows includesfirst100k sky; append0, nodesNone/boxesNone, noLOD.',
        native_limit='Pinned native DGR_FIX_AA andz>0.2 retained. Direct render source do_depthTrue retained and included in future timer.',cuda_called=False)
    e.save(output/'protocol.json',protocol);e.save(inputs/'manifest.json',protocol)
    return config,manifest,cameras,targets,target_paths,np.stack(masks),rows,split

def independent_PLY_CPU(path):
    from plyfile import PlyData
    vertices=PlyData.read(path)['vertex'].data
    e.require(len(vertices)==EXPECTED_G,'Expected2390040SfM plus100000sky')
    fields=vertices.dtype.names;rest=sorted([n for n in fields if n.startswith('f_rest_')],key=lambda n:int(n.split('_')[-1]))
    e.require(rest==[f'f_rest_{i}' for i in range(9)] and all(f'rot_{i}' in fields for i in range(4)),'Expected SH1/wxyz PLYschema')
    xyz=np.stack([vertices[n] for n in ('x','y','z')],axis=1).astype(np.float32)
    sh=np.concatenate([np.stack([vertices[f'f_dc_{i}'] for i in range(3)],axis=1)[:,None,:],np.stack([vertices[n] for n in rest],axis=1).reshape(-1,3,3).transpose(0,2,1)],axis=1).astype(np.float32)
    opacity=np.asarray(vertices['opacity'],np.float32)[:,None];logs=np.stack([vertices[f'scale_{i}'] for i in range(3)],axis=1).astype(np.float32);quats=np.stack([vertices[f'rot_{i}'] for i in range(4)],axis=1).astype(np.float32)
    return xyz,sh,opacity,logs,quats

def model_CPU(config,output):
    import torch
    from plyfile import PlyData
    outcome_path=TRAIN/'outcome.json';outcome=e.read_json(outcome_path)
    e.require(outcome['status']=='completed' and outcome['exit_code']==0 and outcome['requested_iterations']==30000 and outcome['optimizer_update_calls']==29999,'Real completed coarse30000 receipt required')
    saved=TRAIN/'model/point_cloud/iteration_30000';path=saved/'point_cloud.ply';info=saved/'pc_info.txt'
    e.require(Path(outcome['saved_model_directory']).resolve()==saved.resolve() and int(info.read_text().strip())==SKY_PREFIX,'Unexpected savedstate/sky prefix')
    e.require(e.digest(path)==outcome['model_parameter_sha256']==outcome['point_cloud_sha256'],'Actual PLYSHA differs from completedreceipt')
    receipt_path=TRAIN/'model-audit.json';receipt=e.read_json(receipt_path)
    e.require(Path(receipt['parameter_path']).resolve()==path.resolve() and receipt['gaussian_count']==EXPECTED_G and receipt['max_SH_degree']==1 and receipt['skybox_prefix_count']==SKY_PREFIX,'Modelaudit schema differs')
    raw=independent_PLY_CPU(path)
    tree=ast.parse((e.REPO/'scene/gaussian_model.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GaussianModel')
    load=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='load_ply_file');scope={'np':np,'PlyData':PlyData}
    exec(compile(ast.Module(body=[load],type_ignores=[]),'fixed-author-load-ply-CPU','exec'),scope)
    a=scope['load_ply_file'](None,str(path),1);author=(a[0].astype(np.float32),np.concatenate([a[1].transpose(0,2,1),a[2].transpose(0,2,1)],axis=1).astype(np.float32),a[3].astype(np.float32),a[4].astype(np.float32),a[5].astype(np.float32))
    reports=[]
    for name,actual,ref in zip(('XYZ','SH_coefficient_major','opacity_logits','log_scales','wxyz'),raw,author):
        e.require(np.array_equal(actual,ref) and np.isfinite(actual).all(),'Independent PLY vs author AST differs/nonfinite '+name)
        reports.append(dict(name=name,shape=list(actual.shape),dtype=str(actual.dtype),sha256=e.array_digest(actual),min=float(actual.min()),max=float(actual.max())))
    tensors=tuple(torch.from_numpy(a) for a in raw)
    e.require(torch.all(torch.linalg.vector_norm(tensors[4],dim=1)>0).item() and torch.all(torch.isfinite(torch.exp(tensors[3])) & (torch.exp(tensors[3])>0)).item(),'Invalid rawquat/scalelog')
    point_vertices=PlyData.read(INPUT/'aligned/sparse/0/points3D.ply')['vertex'].data
    source_xyz=np.stack([point_vertices[n] for n in ('x','y','z')],axis=1).astype(np.float32)
    e.require(np.array_equal(raw[0][SKY_PREFIX:],source_xyz),'Actual coarse non-sky XYZ differs frominput SfM rows')
    exposure_path=TRAIN/'model/exposure.json';exp=e.read_json(exposure_path);train=config['training_names'];test=config['stage_test_names']
    e.require(e.digest(exposure_path)==receipt['exposure']['sha256'] and set(exp)==set(train) and not(set(exp)&set(test)),'Exposureidentity/names mismatch')
    identity=np.eye(3,4,dtype=np.float32);e.require(all(np.array_equal(np.asarray(v,np.float32),identity) for v in exp.values()),'Sourcecoarse exposures not identity')
    props=CPU_properties(tensors)
    audit=dict(status='bound_actual_completed_coarse_PLY_independently_CPU',source_model_path=str(path),source_model_sha256=e.digest(path),training_outcome_sha256=e.digest(outcome_path),
        training_model_receipt_sha256=e.digest(receipt_path),source_configuration_sha256=e.digest(TRAIN/'configuration.json'),iteration30000_optimizer_calls29999=True,
        snapshots='OfficialI30000save beforelastreturn;29999optimizer calls derived from pinnedsource loop+completedreceipt, not independentlyinstrumented',
        total_G=EXPECTED_G,skybox_prefix_G=SKY_PREFIX,append_skybox_G=0,active_SH_degree=1,tree=False,LOD=False,raw_arrays=reports,
        properties=props,non_sky_XYZ_exact_input_rows=True,original_global_frame=True,source_length_units_not_claimed_meters=True,
        opacity='Raw PLY logits→officialsigmoid; neverhierarchy abs or alreadyactivated decode',rotations='rawwxyz→officialnormalize',scales='rawlogs→exp',
        exposure_path=str(exposure_path),exposure_sha256=e.digest(exposure_path),exposure_names=len(exp),test_overlap=0,all_train_identity=True,evaluation_exposure_enabled=False,cuda_called=False)
    e.save(output/'model-binding-CPU.json',audit);shutil.copyfile(outcome_path,output/'bound-training-outcome.json');shutil.copyfile(receipt_path,output/'bound-training-model-audit.json')
    e.save(output/'memory-plan-CPU.json',dict(total_G=EXPECTED_G,raw_bytes=92*EXPECTED_G,cached_decode_bytes=32*EXPECTED_G,screen_temporary_known_upper_bytes=24*EXPECTED_G,
        known_tensor_upper_bytes=148*EXPECTED_G,native_transient_upper_unknown=True,tree_nodes_bytes=0,cut_buffers_bytes=0,sky_append_bytes=0,
        scope='All2490040flat Gaussians resident, SH1 existingfeatures reused; noLOD/streaming claim. Nativeprojection/tiledup/sort/geometry work additionalunknown'))
    return tensors,audit

def CPU_properties(tensors):
    import torch
    tree=ast.parse((e.REPO/'scene/gaussian_model.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GaussianModel')
    functions=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'get_xyz','get_features','get_scaling','get_rotation','get_opacity'}]
    scope={'torch':torch};exec(compile(ast.Module(body=functions,type_ignores=[]),'fixed-author-properties-CPU','exec'),scope)
    pc=SimpleNamespace(_xyz=tensors[0],_features_dc=tensors[1][:,:1],_features_rest=tensors[1][:,1:],_opacity=tensors[2],_scaling=tensors[3],_rotation=tensors[4],
        scaling_activation=torch.exp,rotation_activation=torch.nn.functional.normalize,opacity_activation=torch.sigmoid)
    report=[]
    for name in ('get_xyz','get_features','get_scaling','get_rotation','get_opacity'):
        val=scope[name].fget(pc);e.require(torch.isfinite(val).all().item(),'Nonfinite decodedproperty '+name)
        if name=='get_features':e.require(torch.equal(val,tensors[1]),'ExistingSH4 notexactauthor DC/restcat')
        report.append(dict(name=name,shape=list(val.shape),sha256=e.array_digest(val.numpy()),min=float(val.min()),max=float(val.max())))
    return report

def GPU_render(tensors,audit,cameras,targets,rows,output):
    import torch
    sys.path.insert(0,str(e.EXT));sys.path.insert(0,str(e.REPO))
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    import gaussian_renderer,diff_gaussian_rasterization as native
    e.require(Path(gaussian_renderer.__file__).resolve()==(e.REPO/'gaussian_renderer/__init__.py').resolve() and Path(native._C.__file__).resolve().is_relative_to(e.EXT),'Wrong authorrenderer/nativebinary')
    torch.cuda.empty_cache();free,total=torch.cuda.mem_get_info();plan=e.read_json(output/'memory-plan-CPU.json')
    e.require(plan['known_tensor_upper_bytes']+1024**3<free,'Insufficient nativeheadroom beforeGPUallocation')
    plan.update(free_device_before=free,total_device=total);e.save(output/'memory-preflight-GPU.json',plan)
    class CachedFlat(GaussianModel):
        @property
        def get_features(self):return self._features_direct
        @property
        def get_scaling(self):return self._cached_scales
        @property
        def get_rotation(self):return self._cached_rotation
        @property
        def get_opacity(self):return self._cached_opacity
    gpu=tuple(t.to('cuda') for t in tensors);pc=CachedFlat(1)
    pc._xyz,pc._features_direct,pc._opacity,pc._scaling,pc._rotation=gpu;pc._features_dc=gpu[1][:,:1];pc._features_rest=gpu[1][:,1:]
    pc.active_sh_degree=1;pc.skybox_points=SKY_PREFIX;pc.nodes=None;pc.boxes=None;pc.pretrained_exposures=None
    e.require(pc.opacity_activation is torch.sigmoid,'Wrongopacityactivation: flatPLYneeds sigmoid')
    torch.cuda.synchronize();start=time.perf_counter()
    pc._cached_scales=GaussianModel.get_scaling.fget(pc);pc._cached_rotation=GaussianModel.get_rotation.fget(pc);pc._cached_opacity=GaussianModel.get_opacity.fget(pc)
    torch.cuda.synchronize();e.save(output/'property-cache-GPU.json',dict(seconds=time.perf_counter()-start,existing_SH4_no_perframe_cat=True,raw_model_and_cache_bytes=124*EXPECTED_G,
        functions='Actualpinned fget+defaultsetup sigmoid/exp/normalize onCUDA, once only; immutablearrays',active_SH1=True,nodes_None=True,boxes_None=True,sky_append=0))
    views=[]
    for c in cameras:
        f=vars(c).copy()
        for key in ('world_view_transform','projection_matrix','full_proj_transform','camera_center'):f[key]=f[key].to('cuda')
        views.append(SimpleNamespace(**f))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
    resident=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),whole_device=e.device_memory_snapshot(torch),
        total_G=EXPECTED_G,sky_prefix_G=SKY_PREFIX,sky_append_G=0,scope='Fullflatmodel+staticcache; noLOD/chunk swap/streaming')
    e.save(output/'resident-memory.json',resident)
    @torch.no_grad()
    def draw(j):return render(views[j],pc,pipe,bg,indices=None,use_trained_exp=False)
    warmups=[]
    for j in range(3):
        result=draw(j);torch.cuda.synchronize();finite=bool(torch.isfinite(result['render']).all().item())
        warmups.append(dict(name=rows[j]['name'],finite=finite));del result
    e.save(output/'warmups.json',warmups)
    e.require(all(x['finite'] for x in warmups),'Nonfinite flatwarmup, no pooledmetrics')
    folder=output/'flat';folder.mkdir()
    for d in ('predicted','paired','float-frames'): (folder/d).mkdir()
    records=[];predictions=[]
    for j,row in enumerate(rows):
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();result=draw(j);torch.cuda.synchronize();elapsed=time.perf_counter()-start
        rgb=result['render'];finite=bool(torch.isfinite(rgb).all().item())
        r=dict(name=row['name'],input_G=EXPECTED_G,all_original_rows_rendered_no_subset=True,sky_prefix_G=SKY_PREFIX,sky_append=0,
            official_flat_render_seconds=elapsed,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            whole_device_after=e.device_memory_snapshot(torch),status='finite_flat_RGB' if finite else 'nonfinite_no_metrics')
        if finite:
            e.require(tuple(rgb.shape)==(3,172,256),'Unexpectedflat image shape');pred=rgb.permute(1,2,0).cpu().numpy().copy();predictions.append(pred)
            png=(pred*255).astype(np.uint8);name=f'{j:03d}-{Path(row["name"]).stem}'
            Image.fromarray(png).save(folder/'predicted'/f'{name}.png');Image.fromarray(np.concatenate([targets[j],png],axis=1)).save(folder/'paired'/f'{name}.png');np.save(folder/'float-frames'/f'{name}.npy',pred)
            r.update(RGBmin=float(rgb.min()),RGBmax=float(rgb.max()),visible_radii_count=len(result['radii']))
        records.append(r);e.save(output/'per-frame-render-audit.json',records);del result,rgb
    complete=len(predictions)==117 and all(r['status']=='finite_flat_RGB' for r in records)
    timing=dict(status='complete117_flat' if complete else 'partial_no_average',total_frames=len(records),valid_frames=len(predictions),warmups=warmups,resident=resident,
        boundary='CUDA synchronized unmodified author render withstaticexactpropertycache, directfullinput, source do_depthTrue(native depth work included), DGR_FIX_AA/z>0.2/clamp/visibility. Excludes decode/transfer/CPU/input I/O/metrics. NoLODselection timer.',stages={})
    if complete:
        timing['stages']['official_flat_render_seconds']=e.summarize([r['official_flat_render_seconds'] for r in records]);np.savez(output/'predictions.npz',names=np.array([r['name'] for r in rows]),rgb=np.stack(predictions))
        timing.update(peak_torch_allocated=max(r['torch_peak_allocated_bytes'] for r in records),peak_torch_reserved=max(r['torch_peak_reserved_bytes'] for r in records))
    e.save(output/'timings.json',timing);e.require(e.digest(audit['source_model_path'])==audit['source_model_sha256'],'Readonlymodel changed')
    del pc,gpu,views;torch.cuda.synchronize();return predictions,timing,complete

def masked_numpy(pred,gt8,mask):
    gt=gt8.astype(np.float32)/255.;pe=pred*mask;ge=(gt*mask)*mask
    error=pe.astype(np.float64)-ge.astype(np.float64);numerator=float(np.sum(error**2));denominator=gt8.size
    weight=mask.astype(np.float64)**2;valid_denominator=3*float(weight.sum());valid_numerator=float(np.sum((pred.astype(np.float64)-gt8.astype(np.float64)/255.)**2*weight))
    return pe,ge,dict(RGB_error_numerator=numerator,RGB_error_denominator=denominator,mse_numpy_float64=numerator/denominator,psnr_numpy_db=float(-10*np.log10(numerator/denominator)),
        symmetric_single_mask_valid_weight_diagnostic=dict(RGB_error_numerator=valid_numerator,RGB_error_denominator=valid_denominator,mse=valid_numerator/valid_denominator,psnr_db=float(-10*np.log10(valid_numerator/valid_denominator))))

def GPU_metrics(predictions,targets,masks,rows,timing,audit,output):
    import torch
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    import torchmetrics.functional.image.lpips as lpips_source
    previous=e.read_json(e.OUTROOT/'truck-baseline-heldout-001/metrics.json')
    paths=[Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth',Path(lpips_source.__file__).parent/'lpips_models/alex.pth'];e.require(all(p.is_file() for p in paths),'No metricweight downloads allowed')
    weights=[dict(path=str(p),bytes=p.stat().st_size,sha256=e.digest(p)) for p in paths];e.require(weights==previous['weight_files'],'Fixedmetric weights changed')
    ssim=StructuralSimilarityIndexMeasure(data_range=1.).to('cuda').eval();lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to('cuda').eval()
    folder=output/'flat';(folder/'masked-paired').mkdir();metrics=[]
    for j,(pred,gt,mask,row) in enumerate(zip(predictions,targets,masks,rows)):
        pe,ge,r=masked_numpy(pred,gt,mask);e.require(np.isfinite(r['psnr_numpy_db']),'InvalidPSNR')
        pt=torch.from_numpy(pe).permute(2,0,1)[None].to('cuda');gtensor=torch.from_numpy(ge).permute(2,0,1)[None].to('cuda')
        with torch.no_grad():sv=float(ssim(pt,gtensor).cpu());lv=float(lpips(pt,gtensor).cpu())
        ssim.reset();lpips.reset();e.require(np.isfinite(sv+lv),'Nonfinite metric')
        name=f'{j:03d}-{Path(row["name"]).stem}';paired=folder/'masked-paired'/f'{name}.png';Image.fromarray(np.concatenate([(ge*255).astype(np.uint8),(pe*255).astype(np.uint8)],axis=1)).save(paired)
        r.update(name=row['name'],group=row['group'],ssim_torchmetrics=sv,lpips_alex_torchmetrics=lv,mask_soft_mean=float(mask.mean()),mask_nonzero=int((mask>0).sum()),mask_zero=int((mask==0).sum()),mask_one=int((mask==1).sum()),
            target_path=row['RGB_path'],target_sha256=row['RGB_sha256'],mask=row,masked_paired_path=str(paired),masked_paired_sha256=e.digest(paired),prediction_path=str(folder/'predicted'/f'{name}.png'),prediction_sha256=e.digest(folder/'predicted'/f'{name}.png'))
        metrics.append(r);del pt,gtensor
    groups=[]
    for group in sorted(set(r['group'] for r in metrics)):
        values=[r for r in metrics if r['group']==group];groups.append(dict(group=group,count=len(values),names=[r['name'] for r in values],psnr_mean_per_image_db=float(np.mean([r['psnr_numpy_db'] for r in values])),ssim_mean=float(np.mean([r['ssim_torchmetrics'] for r in values])),alex_lpips_mean=float(np.mean([r['lpips_alex_torchmetrics'] for r in values]))))
    result=dict(status='complete117_independent_local_coarse_flat_heldout',view_scope=SCOPE,model_binding=audit,renderer='Unmodified author gaussian_renderer.render,activeSH1/dodepthTrue/directall/noLOD/exposureFalse',
        psnr_numpy_mean_per_image_db=float(np.mean([r['psnr_numpy_db'] for r in metrics])),ssim_mean_per_image=float(np.mean([r['ssim_torchmetrics'] for r in metrics])),lpips_alex_mean_per_image=float(np.mean([r['lpips_alex_torchmetrics'] for r in metrics])),
        per_image=metrics,groups=groups,weight_files=weights,ssim_protocol=previous['ssim_protocol'],lpips_protocol=previous['lpips_protocol'],psnr_aggregation=previous['psnr_aggregation'],timing=timing,
        mask_scope='GTm2/predm mainfull3WH; supplementary symmetric singlemask denominator3sum(m^2) distinct',exposure_enabled=False,train_test_exp=False,test_exposure_fit=False,
        caveat='This is globalcoarse background training withfixedSfMpositions andSH1, not same-budget dense/hierarchy optimization ablation; publishedSfMcalibration/colors may involveheldout photos')
    e.save(output/'metrics.json',result);e.save(output/'per-image-metrics.json',metrics);e.save(output/'camera-list-group-metrics.json',groups);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation-run',required=True);mode=p.add_mutually_exclusive_group();mode.add_argument('--bind-model-cpu',action='store_true');mode.add_argument('--gpu',action='store_true');a=p.parse_args()
    e.require(a.evaluation_run.replace('-','').isalnum(),'Invalidrunname');output=e.OUTROOT/a.evaluation_run;output.mkdir(exist_ok=False)
    outcome=dict(status='running',cuda_used=False,started_utc=e.baseline.now());e.save(output/'outcome.json',outcome);start=time.perf_counter()
    try:
        e.require({k:e.importlib.metadata.version(k) for k in e.FIXED}==e.FIXED,'Fixedruntime versions changed')
        config,manifest,cameras,targets,target_paths,masks,rows,split=context(output)
        if a.bind_model_cpu or a.gpu:tensors,audit=model_CPU(config,output)
        if a.gpu:
            outcome['gpu_dispatch_attempted']=True;e.save(output/'outcome.json',outcome);predictions,timing,complete=GPU_render(tensors,audit,cameras,targets,rows,output)
            e.require(complete,'Partialflat renders preserved; no117pool')
            result=GPU_metrics(predictions,targets,masks,rows,timing,audit,output);outcome.update(status=result['status'],cuda_used=True)
        elif a.bind_model_cpu:outcome['status']='passed_completed_model_binding_CPU_no_GPU'
        else:outcome['status']='passed117_CPU_protocol_waiting_completed_model_no_GPU'
        outcome.update(ended_utc=e.baseline.now(),elapsed_seconds=time.perf_counter()-start,driver_sha256=e.digest(__file__),core_helper_sha256=e.digest(e.__file__))
        e.save(output/'outcome.json',outcome);print(json.dumps(outcome),flush=True)
    except Exception as error:
        torch_module=sys.modules.get('torch');initialized=bool(torch_module is not None and torch_module.cuda.is_initialized())
        outcome.update(status='failed_preserved_coarse_evaluation',cuda_used=initialized,cuda_context_initialized=initialized,error=str(error),traceback=traceback.format_exc(),ended_utc=e.baseline.now(),driver_sha256=e.digest(__file__))
        e.save(output/'outcome.json',outcome);raise
if __name__=='__main__':main()
