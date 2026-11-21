"""Independent CPU posterior audit of real local City117 x 2 models x 6settings.

No renderer, training, Torch or CUDA imports. Coarse117 evidence is a separate
mode and can never satisfy the1404 completion gate. Missing actual outputs fail.
"""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import argparse,hashlib,importlib.metadata,importlib.util,json,mmap,os,struct,subprocess,sys,time,traceback
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'outputs/large-scene-gs-001';OUTROOT=BASE/'independent-evaluation'
LOCK=BASE/'official-hierarchy/small-city-hierarchy-005'
COARSE=OUTROOT/'small-city-local-coarse-003-flat-heldout-001'
ADAPTER=OUTROOT/'city-local-hierarchy-evaluation-driver-CPU-001/evaluate_local_city_hierarchy.py'
ADAPTER_SHA='c3d9c5c7547cee632f79ba0e66b6d24bbcd35771295b7f4fd23fc5b93ad40610'
CORE_SHA='672533b6aef0f79d64a77976055ce747d05c3f226646e576cb913b0c405af1ec'
CONFIG_SHA='55c7dd48fb06b07340fd0a49e611fd6d4334dc98253bcc45aa7ac2261b6fdcea'
COARSE_SHA='abd4f6af0463dce35648182a66d2ae3d6893417901f3885c5be1eb98368f2e31'
INPUT_SHA='2dd5c04122e290d27934a224921b2506b7955a34d801a0482e0408af536de3d5'
MODELS=('merged-unoptimized','merged-post');SETTINGS=('leaves','tau1','tau3','tau5','tau6','tau15')
FIXED={'torch':'2.7.1+cu128','torchvision':'0.22.1+cu128','numpy':'1.26.4','gsplat':'1.5.3','torchmetrics':'1.7.1'}

def utc():return datetime.now(timezone.utc).isoformat()
def require(v,m):
    if not v:raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    p=Path(p);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for d in iter(lambda:f.read(8*1024**2),b''):h.update(d)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact_file(p,sha):require(Path(p).is_file() and digest(p)==sha,'Missing/changed actual file: '+str(p))
def same_float(a,b,label):require(float(a)==float(b),label+' numerical mismatch')

def numpy_mask_metrics(pred,gt8,mask):
    gt=gt8.astype(np.float32)/255.;pe=pred*mask;ge=(gt*mask)*mask
    error=pe.astype(np.float64)-ge.astype(np.float64);num=float(np.sum(error**2));den=gt8.size
    require(num>0 and np.isfinite(num),'Invalid actual main MSE')
    w=mask.astype(np.float64)**2;vden=3*float(w.sum());vnum=float(np.sum((pred.astype(np.float64)-gt8.astype(np.float64)/255.)**2*w))
    require(vden>0 and vnum>0,'Invalid actual auxiliary mask context')
    return dict(RGB_error_numerator=num,RGB_error_denominator=den,mse_numpy_float64=num/den,psnr_numpy_db=float(-10*np.log10(num/den)),
        symmetric_single_mask_valid_weight_diagnostic=dict(RGB_error_numerator=vnum,RGB_error_denominator=vden,mse=vnum/vden,psnr_db=float(-10*np.log10(vnum/vden)))),pe,ge

def metric_row(r,computed):
    for k in ('RGB_error_numerator','RGB_error_denominator','mse_numpy_float64','psnr_numpy_db'):same_float(r[k],computed[k],k)
    for k,v in computed['symmetric_single_mask_valid_weight_diagnostic'].items():same_float(r['symmetric_single_mask_valid_weight_diagnostic'][k],v,'auxiliary '+k)
    require(np.isfinite(r['ssim_torchmetrics']) and np.isfinite(r['lpips_alex_torchmetrics']),'Reported perceptual metric nonfinite')

def canonical_inputs(output):
    proto=read(COARSE/'protocol.json');path=Path(proto['evaluated_input_npz_path']);exact_file(path,proto['evaluated_input_npz_sha256'])
    exact_file(BASE/'official-inputs/small-city-training-002/manifest.json',INPUT_SHA)
    exact_file(proto['frozen_split_path'],proto['frozen_split_sha256']);split=read(proto['frozen_split_path'])
    require(split['test_names']==proto['test_names'] and len(split['test_names'])==117 and len(split['train_names'])==5705,'Canonical split differs')
    with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    require(data['names'].tolist()==proto['test_names'] and data['rgb'].shape==(117,172,256,3) and data['mask'].shape==(117,172,256,1),'Canonical117 array schema differs')
    for j,row in enumerate(proto['views']):
        require(np.array_equal(data['K'][j],np.asarray(row['K'])) and np.array_equal(data['c2w'][j],np.asarray(row['C2W'])) and ah(data['mask'][j])==row['mask_float32_sha256'],'Frozen mask/K/pose differs')
        for p,s in [('source_RGB_path','source_RGB_sha256'),('source_mask_path','source_mask_sha256'),('RGB_path','RGB_sha256'),('mask_png_path','mask_png_sha256')]:exact_file(row[p],row[s])
    require(sorted(proto['groups'].values())==[22,95],'Canonical camera-list groups differ')
    save(output/'canonical-input-identity.json',dict(protocol_sha256=digest(COARSE/'protocol.json'),input_npz_sha256=digest(path),split_sha256=proto['frozen_split_sha256'],views=117,groups=proto['groups'],K_C2W_masks_original_files_checked=True))
    return proto,data

def topology_only(path,expected_summary):
    """Actual .hier node/box layout only; never allocate Gaussian tensors."""
    path=Path(path)
    with path.open('rb') as f:signed=struct.unpack('<i',f.read(4))[0]
    g=abs(signed);compressed=signed<0;require(g>0,'Invalid actual headerG')
    off=4+(124 if compressed else 236)*g
    with path.open('rb') as f:f.seek(off);n=struct.unpack('<i',f.read(4))[0]
    require(n>0 and path.stat().st_size==off+4+(36 if compressed else 60)*n,'Actual .hier storage length incompatible')
    if compressed:
        dt=np.dtype([('parent','<i4'),('start','<i4'),('childstart','<i4'),('fields','<i2',(4,))])
        raw=np.memmap(path,dtype=dt,mode='r',offset=off+4,shape=(n,));nodes=np.empty((n,7),np.int32)
        nodes[:,0]=raw['fields'][:,0];nodes[:,1]=raw['parent'];nodes[:,2]=raw['start'];nodes[:,3]=raw['fields'][:,2];nodes[:,4]=raw['fields'][:,3];nodes[:,5]=raw['childstart'];nodes[:,6]=raw['fields'][:,1]
    else:nodes=np.memmap(path,dtype='<i4',mode='r',offset=off+4,shape=(n,7))
    summary=dict(compressed=compressed,gaussian_count=g,node_count=n,leaf_gaussian_count=int(nodes[:,3].sum(dtype=np.int64)),merged_gaussian_count=int(nodes[:,4].sum(dtype=np.int64)),root_depth=int(nodes[0,0]),unowned_tail=0)
    for k,v in summary.items():require(expected_summary[k]==v,'Actual node-only header/receipt count differs: '+k)
    require(int(nodes[:,3:5].sum(dtype=np.int64))==g,'Merged tree owns a tail or misses Gaussian ranges')
    # Reuse the immutable, previously CPU-reviewed coverage algorithm, while
    # providing only the real header length. No invented coordinates or model.
    sys.path.insert(0,str(ROOT/'scripts/large_scene_gs'))
    from evaluate_hierarchy import Topology
    topology=Topology(SimpleNamespace(nodes=nodes,positions=range(g)))
    return topology,nodes,summary

def selection_audit(row,folder,ordinal,name,topology,nodes,setting):
    sel=row['selected'];exact_file(sel['path'],sel['file_sha256'])
    require(Path(sel['path']).resolve().is_relative_to((folder/'selections').resolve()),'Selection is not from actual setting folder')
    with np.load(sel['path'],allow_pickle=False) as z:idx=z['indices'].copy();parents=z['parent_gaussian_indices'].copy();ids=z['node_ids'].copy()
    require(idx.dtype==parents.dtype==ids.dtype==np.int32 and len(idx)==sel['count']>0,'Actual selection dtype/count invalid')
    for a,key in [(idx,'selected_index_bytes_sha256'),(parents,'parent_index_bytes_sha256'),(ids,'node_id_bytes_sha256')]:require(ah(a)==sel[key],'Selection bytes SHA differs')
    actual=topology.audit(idx,ids);require(actual==row['coverage'] and actual['every_leaf_exactly_once'],'Fresh actual node topology coverage differs')
    require(row['author_cut_vs_CPU_predicate_match'] and not row['root_fallback_failure'] and row['weight_vs_CPU_reference_match'],'Actual cut/weight was invalid')
    expected_parent=idx if setting=='leaves' else np.where(topology.parent[ids]>=0,topology.start[np.maximum(topology.parent[ids],0)],-1)
    require(np.array_equal(parents,expected_parent),'Actual parent Gaussian index differs')
    ip=folder/'selections'/f'{ordinal:02d}-{Path(name).stem}-interpolation.npz'
    with np.load(ip,allow_pickle=False) as z:ts=z['child_weights'].copy();kids=z['num_parent_children'].copy()
    require(ts.dtype==np.float32 and kids.dtype==np.int32 and len(ts)==len(kids)==len(idx) and np.isfinite(ts).all() and np.all((ts>=0)&(ts<=1)) and np.all(kids>0),'Actual weights/siblings invalid')
    require(ah(ts)==row['interpolation_weight_bytes_sha256'] and ah(kids)==row['sibling_count_bytes_sha256'],'Actual weight bytes hash differs')
    if setting=='leaves':require(np.all(ts==1) and np.all(kids==1) and actual['selected_leaves']==len(topology.leaf_storage),'Leaf control is not all real unit-weight leaves')
    else:
        gate=row['float32_operation_interval_audit'];require(gate['accepted'] and gate['finite_domain'] and gate['exact_parent_children_match'] and gate['outside_interval_count']==0,'Recorded floating interval gate failed')
        require(gate['norm_arithmetic_range']['normal_norm_arithmetic_verified'] and gate['norm_arithmetic_range']['all_selected_and_parent_size_intervals_finite'],'Recorded interval arithmetic assumptions not verified')
        require(gate['negative_control_status']=='passed_perturbation_rejected' and gate['negative_control'] and gate['negative_control']['rejected'],'Actual negative control unavailable/failed; do not claim it passed')
    return dict(selection_file_sha256=sel['file_sha256'],interpolation_npz_sha256=digest(ip),fresh_topology_coverage=True,actual_selection_count=len(idx))

def timing_audit(rows,timing,setting):
    require(timing['status']=='all117_valid' and timing['valid_frames']==timing['total_frames']==117 and timing['warmups_requested']==timing['warmups_completed']==3,'Incomplete actual setting timing')
    keys=('cut_seconds','interpolation_seconds','official_render_post_seconds','sum_synchronized_GPU_stages_seconds')
    for key in keys:
        a=np.asarray([r[key] for r in rows],np.float64);require(np.isfinite(a).all() and np.all(a>=0),'Invalid raw actual timer')
        expected=dict(count=len(a),min=float(a.min()),mean=float(a.mean()),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),max=float(a.max()))
        require(expected==timing['stages'][key],'Actual timer summary differs')
    for row in rows:
        same_float(row['sum_synchronized_GPU_stages_seconds'],row['cut_seconds']+row['interpolation_seconds']+row['official_render_post_seconds'],'Perframe synchronized stage sum')
        if setting=='leaves':require(row['cut_seconds']==0,'Leaf fixed-index reuse misreported as hierarchy cut')
        for stage in ('cut','weight','render'):
            alloc=row[stage+'_peak_allocated_bytes'];reserved=row[stage+'_peak_reserved_bytes'];device=row[stage+'_after_whole_device']
            require(0<=alloc<=reserved and 0<=device['free_bytes']<=device['total_bytes'] and device['whole_device_used_bytes']==device['total_bytes']-device['free_bytes'],'Actual memory schema invalid')
            require('boundary' in device['scope'].lower() and 'not' in device['scope'].lower(),'Whole-device snapshot is not described as a nonpeak boundary')
    require('Separate CUDA-synchronized' in timing['boundary'] and 'CPU gaps' in timing['total_definition'],'Timing boundary or sum scope changed')
    return dict(stages_independently_recomputed=True,whole_device_is_synchronized_boundary_not_transient_peak=True,Torch_allocator_does_not_capture_native_CUB_or_projection_transients=True)

def terminal_query(output,evaluation_run):
    command="$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine) | ConvertTo-Json -Depth 3 -Compress"
    ps=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    r=subprocess.run([str(ps),'-NoLogo','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30,creationflags=subprocess.CREATE_NO_WINDOW);require(r.returncode==0,'Fresh terminal query failed')
    processes=json.loads(r.stdout or '[]');processes=processes if isinstance(processes,list) else [processes]
    spec=importlib.util.spec_from_file_location('cpu_posterior_exact_windows_argv',ROOT/'scripts/large_scene_gs/observe_small_city_process_roles.py');observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
    conflicting=[]
    for p in processes:
        command=p.get('CommandLine') or '';argv=observer.windows_argv(command);names={Path(s).name.lower() for s in argv[1:]}
        python=p['Name'].lower()=='python.exe'
        native=p['Name'].lower() in {'gaussianhierarchycreator.exe','gaussianhierarchymerger.exe','mergerunoptimized.exe','gaussianhierarchymergerunoptimized.exe'}
        trainer=bool(names&{'train_coarse.py','train_single.py','train_post.py'})
        target_eval='evaluate_local_city_hierarchy.py' in names and evaluation_run in argv and '--gpu' in argv
        if native or python and (trainer or target_eval):conflicting.append(p)
    report=dict(observed_utc=utc(),fresh_Win32_Process_query=True,actual_conflicting_GPU_stage_processes=conflicting,scope='Existence/argv/creationtime at this timestamp; no processstart/stop and no CUDA query')
    save(output/'actual-terminal-process-query.json',report);require(not conflicting,'Actual GPU stage/evaluator still live; no terminal/1404 completion claim')
    return report

def coarse_contract_evidence(output):
    proto,data=canonical_inputs(output);m=read(COARSE/'metrics.json')
    with np.load(COARSE/'predictions.npz',allow_pickle=False) as z:pred=z['rgb'].copy();names=z['names'].tolist()
    require(names==proto['test_names'] and len(m['per_image'])==117,'Actual coarse117 metadata differs')
    rows=[]
    for j,r in enumerate(m['per_image']):
        computed,_,_=numpy_mask_metrics(pred[j],data['rgb'][j],data['mask'][j]);metric_row(r,computed);rows.append(dict(name=r['name'],independent_numpy_main_and_auxiliary_exact=True))
    result=dict(status='actual_coarse117_numeric_contract_evidence_only_not1404_completion',actual_source_metrics_SHA=digest(COARSE/'metrics.json'),actual_views=117,rows=rows,ownCity1404_completed=False,actual_local_pre_post_models_bound=False,perceptual_recomputed=False,Torch_imported=False,CUDA_called=False)
    save(output/'coarse117-numeric-evidence.json',result);return result

def full_audit(source,output,evaluation_run):
    # Genuine source models and actual GPU result files are required first.
    require(source.is_dir() and (source/'outcome.json').is_file() and (source/'metrics.json').is_file(),'Actual ownCity1404 GPU evaluation files unavailable; refuse completion')
    outcome=read(source/'outcome.json');overall=read(source/'metrics.json')
    require(outcome['status']=='complete117_each_requested_actual_local_pre_post_setting' and outcome['cuda_used'] and outcome['driver_sha256']==ADAPTER_SHA and outcome['core_evaluator_sha256']==CORE_SHA,'Actual GPU outcome incomplete or wrongsource')
    require(overall['complete_full_leaves_tau1_3_5_6_15_protocol'] and overall['complete_all_requested_settings'] and overall['canonical117_input_scope'],'A subset/partial run cannot certify1404')
    exact_file(ADAPTER,ADAPTER_SHA);exact_file(ROOT/'scripts/large_scene_gs/evaluate_official_hierarchy.py',CORE_SHA);exact_file(LOCK/'configuration.json',CONFIG_SHA)
    exact_file(source/'adapter-source.py',ADAPTER_SHA)
    config=read(LOCK/'configuration.json');proto,data=canonical_inputs(output);local=read(source/'protocol.json')
    require(local['test_names']==proto['test_names'] and local['groups']==proto['groups'] and local['settings']==list(SETTINGS) and local['view_count']==117 and local['hierarchy_configuration_sha256']==CONFIG_SHA,'Actual evaluation protocol differs')
    ownpath=Path(local['independent_input_npz_path']);exact_file(ownpath,local['independent_input_npz_sha256'])
    with np.load(ownpath,allow_pickle=False) as z:
        for key in data:require(np.array_equal(z[key],data[key]),'Actual local frozen input differs: '+key)
    exact_file(local['actual_owncoarse_background_binding']['source_model_path'],COARSE_SHA)
    require(local['canonical_background_sky']['count']==100000 and local['actual_owncoarse_background_binding']['source_model_sha256']==COARSE_SHA,'Wrong single owncoarse background')
    actual=read(source/'actual-workflow-binding-CPU.json');require(actual['status']=='actual_completed_local_pre_post_workflow_CPU_identity_and_freeze_passed' and actual['configuration_sha256']==CONFIG_SHA,'Actual workflow CPU binding missing/failed')
    for r in actual['lineage']:
        directory=BASE/'official-training'/r['run'];exact_file(directory/'outcome.json',r['outcome_sha256']);exact_file(directory/'configuration.json',r['configuration_sha256']);exact_file(r['model_path'],r['model_sha256']);require(read(directory/'outcome.json')['status']=='completed','Real training incomplete')
    source_audit=read(source/'source-audit.json');require(source_audit['status']=='passed_pinned_official_source_and_binary_audit','Official source audit failed')
    for r in source_audit['source_files']+source_audit['rasterizer_sources']:exact_file(r['path'],r['sha256']);exact_file(r['snapshot'],r['sha256'])
    for r in source_audit['modules']:exact_file(r['path'],r['sha256'])
    require({k:importlib.metadata.version(k) for k in FIXED}==FIXED,'Fixed installedmetric/runtime identities differ')
    descriptions={(r['model'],r['setting']):r for r in overall['settings']};require(set(descriptions)=={(m,s) for m in MODELS for s in SETTINGS} and len(overall['settings'])==12,'Expected exactly12 distinct model/settings')
    names=proto['test_names'];frame_checks=[];summaries=[];pairs=[];leaf_arrays={}
    for model in MODELS:
        folder_model=source/model;ma=read(folder_model/'model-audit.json');kind='unoptimized' if model=='merged-unoptimized' else 'post_optimized';receipt_path=LOCK/('merge-unoptimized-outcome.json' if model=='merged-unoptimized' else 'merge-outcome.json');receipt=read(receipt_path)
        require(receipt['status']=='completed' and receipt['exit_code']==0 and receipt['source_kind']==kind and receipt['configuration_sha256']==CONFIG_SHA,'ActualMerger source family invalid')
        exact_file(ma['source_path'],receipt['hierarchy_sha256']);require(ma['source_sha256']==receipt['hierarchy_sha256'] and ma['source_kind']==kind and ma['skybox_copy_count']==1 and ma['background_G']==100000 and ma['total_G']==ma['tree_G']+100000,'Model identity/single sky violated')
        require(receipt['skybox_in_tree']==0 and receipt['shared_background_skybox_count']==100000 and receipt['chunk_order']==list(config['chunk_runs']) and receipt['ownership']==config['merge_ownership'],'Actualmerged sky/order/ownership invalid')
        for dep in receipt['input_hierarchies']:exact_file(dep['path'],dep['sha256']);require(dep['source_kind']==kind,'Fake or mixed pre/post merge input')
        topology,nodes,header=topology_only(ma['source_path'],receipt['hierarchy']);require(header=={k:ma['tree_summary'][k] for k in header} and len(topology.leaf_storage)==ma['leaf_count'],'Actual model header/leafcount differs')
        for setting in SETTINGS:
            folder=folder_model/setting;m=read(folder/'metrics.json');render=read(folder/'per-frame-audit.json');timing=read(folder/'timings.json');per=read(folder/'per-image-metrics.json')
            require(m['status']=='complete117_valid_official_continuous_renderer' and m['view_images']==117 and m['model']==model and m['setting']==setting and m['per_image']==per,'Actual setting metric protocol incomplete')
            require([r['name'] for r in render]==[r['name'] for r in per]==names and len(per)==117,'Actualordered117 missing/duplicated')
            require('117 frozen image' in m['psnr_aggregation'] and 'GTm2/predm' in m['photometric_scope'],'Legacy32/unmasked description leaked into local setting')
            reference=read(COARSE/'metrics.json')
            for k in ('ssim_protocol','lpips_protocol','weight_files'):require(m[k]==reference[k],'FixedSSIM/Alex protocol differs')
            for w in m['weight_files']:exact_file(w['path'],w['sha256'])
            with np.load(folder/'predictions.npz',allow_pickle=False) as z:predictions=z['rgb'].copy();require(z['image_names'].tolist()==names,'Actualfloat archive names differ')
            require(predictions.shape==(117,172,256,3),'Actualfloat archive dimensions differ')
            psnrs=[]
            for j,(row,r) in enumerate(zip(per,render)):
                require(r['status']=='valid_selection_interpolation_finite_render' and r['name']==names[j] and r['background_count']==100000 and r['skybox_copy_count']==1,'Actual invalid frame/sky duplicate')
                tau=None if setting=='leaves' else float(setting[3:]);fx=float(data['K'][j,0,0]);theta=None if tau is None else float(np.float32(2*(tau+.5)/fx));inside=None if tau is None else .005<=theta<=.1
                require(r['threshold']==theta and r['in_training_range']==inside and r['post_training_dimensionless_range']==[.005,.1],'Perframe savedK threshold/range differs')
                meta=proto['views'][j];require(row['width']==256 and row['height']==172 and row['mask']['source_sha256']==meta['source_mask_sha256'] and row['mask']['derived_float32_sha256']==meta['mask_float32_sha256'],'Actualmask/photo dimensions differ')
                filename=f'{j:02d}-{Path(names[j]).stem}';path=folder/'float-frames'/f'{filename}.npy';pred=np.load(path,allow_pickle=False);require(np.array_equal(pred,predictions[j]) and pred.dtype==np.float32 and np.isfinite(pred).all() and np.all((pred>=0)&(pred<=1)),'Actualprediction float invalid/differs')
                computed,pe,ge=numpy_mask_metrics(pred,data['rgb'][j],data['mask'][j]);metric_row(row,computed);psnrs.append(computed['psnr_numpy_db'])
                png=(pred*255).astype(np.uint8)
                for p,expected in [(folder/'predicted'/f'{filename}.png',png),(folder/'paired'/f'{filename}.png',np.concatenate([data['rgb'][j],png],axis=1)),(folder/'masked-paired'/f'{filename}.png',np.concatenate([(ge*255).astype(np.uint8),(pe*255).astype(np.uint8)],axis=1))]:
                    with Image.open(p) as im:require(np.array_equal(np.asarray(im),expected),'ActualPNGpair differs: '+str(p))
                exact_file(row['prediction_path'],row['prediction_sha256']);exact_file(row['paired_path'],row['paired_sha256']);exact_file(row['target_path'],row['target_sha256'])
                selected=selection_audit(r,folder,j,names[j],topology,nodes,setting)
                frame_checks.append(dict(model=model,setting=setting,name=names[j],float_frame_sha256=digest(path),PNG_sha256=digest(folder/'predicted'/f'{filename}.png'),raw_pair_sha256=digest(folder/'paired'/f'{filename}.png'),masked_pair_sha256=digest(folder/'masked-paired'/f'{filename}.png'),independent_PSNR_exact=True,**selected))
            same_float(np.mean(psnrs),m['psnr_numpy_mean_per_image_db'],'Mean117 PSNR')
            for key,single in [('ssim_mean_per_image','ssim_torchmetrics'),('lpips_alex_mean_per_image','lpips_alex_torchmetrics')]:same_float(np.mean([r[single] for r in per]),m[key],'Reported117 '+key)
            ta=timing_audit(render,timing,setting);require(m['timing']==timing,'Timing embedded/file differ');summary=dict(model=model,setting=setting,actual_views=117,actualmetrics_sha256=digest(folder/'metrics.json'),PSNR_independently_recomputed=float(np.mean(psnrs)),ssim_reported_mean=m['ssim_mean_per_image'],AlexLPIPS_reported_mean=m['lpips_alex_mean_per_image'],perceptual_independently_recomputed=False,timing_audit=ta)
            summaries.append(summary);save(output/'actual-verified-frame-progress.json',dict(status='partial_actual_check_progress_not_completion',verified_frames=len(frame_checks),frames=frame_checks))
            if setting=='leaves':leaf_arrays[model]=predictions.copy()
        del topology,nodes
    groups=read(source/'camera-list-group-metrics.json');require(len(groups)==24,'Expected12x2 actual membership group summaries')
    membership={r['name']:r['group'] for r in proto['views']}
    for group in groups:
        per=read(source/group['model']/group['setting']/'per-image-metrics.json');rows=[r for r in per if membership[r['name']]==group['group']]
        require(len(rows)==group['count'] and [r['name'] for r in rows]==group['names'] and len(rows) in (22,95),'Actualgroup membership/order differs')
        for key,single in [('psnr_mean_per_image_db','psnr_numpy_db'),('ssim_mean','ssim_torchmetrics'),('alex_lpips_mean','lpips_alex_torchmetrics')]:same_float(np.mean([r[single] for r in rows]),group[key],'Actualgroup '+key)
    leaf_equal=bool(np.array_equal(leaf_arrays[MODELS[0]],leaf_arrays[MODELS[1]]));leaf_delta=float(np.max(np.abs(leaf_arrays[MODELS[0]]-leaf_arrays[MODELS[1]])))
    require(len(frame_checks)==1404,'Actual1404 frame check count differs')
    terminal=terminal_query(output,evaluation_run)
    result=dict(status='actual1404_complete_independent_CPU_posterior_audit_passed',evaluation_run=evaluation_run,actual_verified_frames=1404,actual_models=2,actual_settings_per_model=6,actual_frames_per_setting=117,
        source_outcome_sha256=digest(source/'outcome.json'),source_overall_metrics_sha256=digest(source/'metrics.json'),source_adapter_SHA=ADAPTER_SHA,source_core_SHA=CORE_SHA,source_config_SHA=CONFIG_SHA,actual_owncoarse_SHA=COARSE_SHA,
        summaries=summaries,actual_pre_post_leaf_RGB_bitwise_equal=leaf_equal,actual_pre_post_leaf_RGB_max_abs_delta=leaf_delta,leaf_equality_measured_not_assumed=True,
        groups_independently_recomputed=True,reportedSSIM_LPIPS_finite_protocol_and_mean_verified=True,SSIM_LPIPS_independently_recomputed=False,
        coverage_recomputed_from_real_nodes_and_saved_selection=True,weights_source_bytes_and_recorded_interval_gate_checked=True,weights_GPU_algorithm_not_reexecuted=True,terminal=terminal,
        mask_protocol='MainGTm2/predm/full3WH=132096, auxiliarysymmetric denominator3sum(m2), unchanged',scope='117GSphotometricheldout; publishedSfMmayuseallphotos; 95/22camera-listmembership notspatialoutside; tau6/15extrapolationseparate; noequalbudgetcoarsecausalcomparison',
        memory_scope='Whole resident tree+singleownsky; cut/weight/native stage Torch allocator vswhole-device synchronized nonpeak boundaries; no streaming/wholedevicepeak claim',Torch_imported=False,CUDA_called=False)
    save(output/'completion-audit.json',result);save(output/'verified-actual-frames.json',frame_checks);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit-run',required=True);p.add_argument('--evaluation-run');p.add_argument('--coarse-contract-evidence',action='store_true');a=p.parse_args()
    require(a.audit_run.replace('-','').isalnum() and len(a.audit_run)<=100,'Invalidauditr un')
    require(a.coarse_contract_evidence != bool(a.evaluation_run),'Choose actualevaluation-run orcoarse-evidence, exclusively')
    if a.evaluation_run:require(a.evaluation_run.replace('-','').isalnum() and len(a.evaluation_run)<=100,'Invalidevaluationrun')
    output=OUTROOT/a.audit_run;output.mkdir(exist_ok=False);start=time.perf_counter();outcome=dict(status='running_CPU_posterior',started_utc=utc(),CUDA_called=False,Torch_imported=False)
    save(output/'outcome.json',outcome);(output/'auditor-source.py').write_bytes(Path(__file__).read_bytes())
    try:
        result=coarse_contract_evidence(output) if a.coarse_contract_evidence else full_audit(OUTROOT/a.evaluation_run,output,a.evaluation_run)
        require('torch' not in sys.modules,'DefaultCPU posterior unexpectedly imported Torch')
        outcome.update(status=result['status'],ended_utc=utc(),elapsed_seconds=time.perf_counter()-start,auditor_source_SHA=digest(__file__),actual1404_completed=result.get('actual_verified_frames')==1404)
        save(output/'outcome.json',outcome);print(json.dumps(outcome),flush=True)
    except Exception as error:
        outcome.update(status='failed_preserved_CPU_posterior_no_completion',error=str(error),traceback=traceback.format_exc(),ended_utc=utc(),elapsed_seconds=time.perf_counter()-start,auditor_source_SHA=digest(__file__),actual1404_completed=False,Torch_imported='torch' in sys.modules)
        save(output/'outcome.json',outcome);raise
if __name__=='__main__':main()
