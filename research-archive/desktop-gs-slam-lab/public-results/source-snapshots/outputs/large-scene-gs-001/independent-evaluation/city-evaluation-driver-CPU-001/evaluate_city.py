"""Author SmallCity reconstruction-view adapter to the fixed official evaluator.

CPU is the default. --gpu is only for a separately authorized exclusive slot.
No supplied author train/test split or exposure is invented. The 12 saved views
are reconstruction diagnostics, never heldout/local reproduction metrics.
"""
from pathlib import Path
import argparse,importlib.util,json,sys,time,traceback
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/large_scene_gs'))
import evaluate_official_hierarchy as e
READY=e.OUTROOT/'small-city-official-renderer-readiness-001'
READY_SCRIPT_SHA='afd90a0ea675d74d53089b29e6a1605db960b7cbc0b4e058d9c5654bd1b08034'
MODEL_SHA='c3ca5134260212bd9f92772fea430c87360684d6d3948aedfe1ad0ad0cd41f2d'
SCOPE='Author pretrained SmallCity, 12 frozen reconstruction views across4 capture passes; author training/test split unavailable. Not heldout quality or local reproduction.'

def context(output,settings):
    source=e.source_audit(output);protocol=e.read_json(READY/'protocol.json')
    (output/'source-snapshots/evaluation/evaluate_city.py').write_bytes(Path(__file__).read_bytes())
    e.require(protocol['training_split_unknown'] and protocol['model_sha256']==MODEL_SHA,'Readiness scope changed')
    manifest_path=e.TRACK/'research/datasets/official-small-city-pretrained/manifest.json'
    e.require(e.digest(manifest_path)==protocol['pretrained_manifest_sha256'],'Author manifest changed')
    manifest=e.read_json(manifest_path)
    e.require(manifest['archive_test_and_independent_extracted_crc_passed'],'Author archive not verified')
    cameras_path=Path(protocol['global_camera_archive'])
    e.require(e.digest(cameras_path)==protocol['global_camera_archive_sha256'],'Global camera archive changed')
    global_cameras=np.load(cameras_path);inputs=np.load(READY/'pilot-inputs.npz')
    names=inputs['names'].tolist();e.require(names==[x['name'] for x in protocol['views']],'Pilot view order changed')
    groups=e.read_json(READY/'chunk-view-groups.json');group_rows={x['name']:x for x in groups['author_pretrained_pilot_reconstruction_only']}
    e.require(e.digest(groups['source_manifest'])==groups['source_manifest_sha256'],'Chunk membership source changed')
    member_manifest=e.read_json(Path(groups['source_manifest']))
    for name,row in group_rows.items():
        actual=sorted(c['name'] for c in member_manifest['chunks'] if name in c['train_names'] or name in c['test_names'])
        e.require(actual==sorted(row['published_chunk_views']),'Camera membership classification differs from real chunk lists')
        expected_group='inside_any_published_chunk_view_list' if actual else 'outside_all_published_chunk_view_lists'
        e.require(row['group']==expected_group,'Camera membership group mislabeled')
    cameras=[];targets=[];target_paths=[];masks=[];mask_records=[];rows=[]
    math_scope=e.official_camera_math_CPU()
    for j,row in enumerate(protocol['views']):
        name=row['name'];index=row['global_camera_row'];K=inputs['K'][j];c2w=inputs['c2w'][j]
        e.require(global_cameras['names'][index]==name,'Camera identity mismatch')
        e.require(np.array_equal(c2w,global_cameras['c2w'][index]),'Original camera frame differs')
        expected=global_cameras['K'][index].copy();expected[0]*=256/1024;expected[1]*=172/690
        e.require(np.array_equal(K,expected),'Independent x/y intrinsics scaling differs')
        e.require(e.digest(row['source_image'])==row['source_image_sha256'] and e.digest(row['source_mask'])==row['source_mask_sha256'],'Source photo/mask changed')
        with Image.open(row['source_image']) as image:
            image.load();e.require(image.size==(1024,690),'Original image dimensions differ');rgb=np.asarray(image.convert('RGB').resize((256,172))).copy()
        with Image.open(row['source_mask']) as image:
            image.load();e.require(image.mode=='L' and image.size==(1024,690),'Original mask mode/dimensions differ');mask=np.asarray(image.resize((256,172))).astype(np.float32)[...,None]/255.
        e.require(np.array_equal(rgb,inputs['rgb'][j]) and np.array_equal(mask,inputs['mask'][j]),'Derived photo/mask differs from frozen pilot')
        target=READY/'derived-inputs'/f'{Path(name).stem}-RGB.png'
        with Image.open(target) as image:e.require(np.array_equal(np.asarray(image),rgb),'Saved derived photo differs')
        camera=e.make_official_camera_CPU(K,c2w,256,172,name,math_scope);cameras.append(camera);targets.append(rgb);target_paths.append(target);masks.append(mask)
        mask_records.append(dict(name=name,source_path=row['source_mask'],source_sha256=row['source_mask_sha256'],decoded_mode='L uint8',
            resize='Unmodified Pillow default resize256x172 thenfloat32/255, matches frozen pilot',derived_float32_sha256=e.array_digest(mask),group=group_rows[name]['group']))
        rows.append(dict(name=name,global_camera_row=index,K=K.tolist(),C2W=c2w.tolist(),official_camera_center=camera.camera_center.numpy().tolist(),
            saved_center_max_abs_error=float(np.max(np.abs(camera.camera_center.numpy()-c2w[:3,3]))),group=group_rows[name],
            threshold_scope={s:e.threshold_scope(t,float(K[0,0])) for s,t in settings}))
    source_frame=e.read_json(READY/'source-frame-audit.json')
    for prefix in ('scaffold','global_input_ply'):
        key='scaffold_path' if prefix=='scaffold' else 'global_input_ply'
        e.require(e.digest(source_frame[key])==source_frame[prefix+'_sha256'],'Verified source-frame input changed')
    e.require(source_frame['status']=='passed_exact_float32_rowwise_global_XYZ','Source frame not established')
    source_root=e.TRACK/'research/datasets/official-small-city-pretrained/extracted/output'
    info=source_root/'scaffold/point_cloud/iteration_30000/pc_info.txt'
    e.require(int(info.read_text().strip().split()[0])==100000,'Author skybox count differs')
    sky,sky_receipt=e.skybox_PLY_CPU(Path(source_frame['scaffold_path']),100000,output,
        dict(source_kind='author_pretrained_scaffold',scaffold_sha256=source_frame['scaffold_sha256'],source_frame_audit=source_frame,pc_info_path=str(info),pc_info_sha256=e.digest(info)))
    desc=dict(name='small-city-author-pretrained',path=str(source_root/'merged.hier'),receipt_path=str(manifest_path),expected_source_kind='author_pretrained',append_skybox=100000,available=True)
    e.save(output/'protocol.json',dict(status='CPU_explicit_reconstruction_protocol',source_audit=source,view_scope=SCOPE,views=rows,
        readiness_protocol_path=str(READY/'protocol.json'),readiness_protocol_sha256=e.digest(READY/'protocol.json'),pilot_input_npz_sha256=e.digest(READY/'pilot-inputs.npz'),
        global_camera_archive_sha256=e.digest(cameras_path),author_manifest_sha256=e.digest(manifest_path),source_frame_audit=source_frame,skybox=sky_receipt,
        source_pretrained_split=None,proposed_local_heldout_not_author_split=True,masks=mask_records,chunk_camera_membership_protocol=groups,
        mask_chain='Pinned Camera+render_hierarchy evaluation GTm2/predm full3WH, symmetric single-mask valid-weight auxiliary separate',
        exposure_correction=False,no_exposure_files_in_author_package=True,metric_protocol='Fixed sameSSIM/AlexLPIPS asTruck, not authorVGG or paper scores',cuda_called=False))
    previous=e.read_json(e.OUTROOT/'truck-baseline-heldout-001/metrics.json')
    return desc,sky,cameras,targets,target_paths,np.stack(masks),mask_records,dict(test_names=names),previous

def model_cpu(desc,sky,cameras,settings,output):
    import torch
    root=output/desc['name'];root.mkdir()
    path=Path(desc['path']);e.require(e.digest(path)==MODEL_SHA,'Author tree changed')
    helper=READY/'readiness.py';e.require(e.digest(helper)==READY_SCRIPT_SHA,'Frozen fast CPU decoder changed')
    (root/'frozen-CPU-model-decoder.py').write_bytes(helper.read_bytes())
    spec=importlib.util.spec_from_file_location('frozen_city_readiness',helper);reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    reader.OUTPUT=root;h=reader.mapped_tree(path)
    e.require(h.unowned_tail==0 and len(h.positions)==18598279 and len(h.nodes)==18598276,'Author tree payload count/tail changed')
    topology=e.Topology(h);gh,tensors,leaves=e.author_cpu_load(h,root)
    coverage=topology.audit(leaves,topology.owner[leaves]);e.require(coverage['every_leaf_exactly_once'] and np.array_equal(np.sort(leaves),topology.leaf_storage),'Leaf coverage mismatch')
    e.require(len(leaves)==13155453,'Actual leaf count differs')
    n=len(tensors[0]);tensors=tuple(torch.cat((t,bg),dim=0) for t,bg in zip(tensors[:5],sky))+tuple(tensors[5:])
    e.require(len(tensors[0])==n+100000 and int(tensors[5][:,3:5].sum())==n,'Sky appended twice or owned bytree')
    audit=dict(status='passed_author_model_identity_single_background_and_CPU_loader',model=desc['name'],source_kind='author_pretrained',
        source_path=str(path),source_sha256=MODEL_SHA,source_receipt_path=desc['receipt_path'],source_receipt_sha256=e.digest(desc['receipt_path']),
        tree_summary=h.summary(),tree_G=n,background_G=100000,total_G=len(tensors[0]),skybox_copy_count=1,leaf_count=len(leaves),
        node_storage_sha256=e.array_digest(h.nodes),box_storage_sha256=e.array_digest(h.boxes,np.float32),xyz_storage_sha256=e.array_digest(h.positions),
        leaf_control=coverage,view_scope=SCOPE,CPU_decoder_script_path=str(helper),CPU_decoder_script_sha256=READY_SCRIPT_SHA,
        representation='RawXYZ/SH, activatedalphaabs(no sigmoid/clamp), exp logscales, normalized wxyz endpoints, official interpolation withoutpostblend normalization')
    e.save(root/'model-audit.json',audit);e.property_cache_cpu(tensors,root)
    selected=[]
    for camera in cameras:
        for name,tau in settings:
            counts,_=topology.reference_counts(camera.camera_center.numpy(),e.tau_to_threshold(tau,camera.image_width/(2*np.tan(camera.FoVx*.5))))
            selected.append(dict(name=camera.image_name,setting=name,CPU_predicate_selected_G=int(counts.sum())))
    maximum=max(x['CPU_predicate_selected_G'] for x in selected)
    plan=e.memory_plan(len(tensors[0]),len(h.nodes),len(leaves),maximum,100000,len(cameras));plan['CPU_actual_view_counts']=selected
    plan['full_tree_leaf_control_known_upper_bytes']=e.memory_plan(len(tensors[0]),len(h.nodes),len(leaves),len(leaves),100000,len(cameras))['conservative_Python_tensor_peak_upper_bytes']
    plan['full_tree_leaf_control_decision']='Do not allocate full-tree leafcontrol on16GiB; Truck compact pixel equivalence is available but separate City compact reference requires explicit future run'
    e.save(root/'memory-preflight-CPU.json',plan)
    return root,h,topology,gh,tensors,leaves,audit

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation-run',required=True);p.add_argument('--gpu',action='store_true')
    p.add_argument('--settings',nargs='+',choices=['tau1','tau3','tau6','tau15'],default=['tau3']);a=p.parse_args()
    e.require(a.evaluation_run.replace('-','').isalnum() and len(set(a.settings))==len(a.settings),'Invalid run/settings')
    settings=tuple((s,float(s[3:])) for s in a.settings);output=e.OUTROOT/a.evaluation_run;output.mkdir(exist_ok=False)
    started=time.perf_counter();outcome=dict(status='running',started_utc=e.baseline.now(),cuda_used=False,view_scope=SCOPE)
    e.save(output/'outcome.json',outcome)
    try:
        e.require({k:e.importlib.metadata.version(k) for k in e.FIXED}==e.FIXED,'Fixed runtime/metric versions changed')
        desc,sky,cameras,targets,target_paths,masks,mask_records,split,previous=context(output,settings)
        prepared=model_cpu(desc,sky,cameras,settings,output)
        if a.gpu:
            outcome['gpu_dispatch_attempted']=True;e.save(output/'outcome.json',outcome)
            results=e.gpu_model(desc,None,sky,cameras,targets,split,output,prepared=prepared,settings=settings)
            result=e.metrics(results,targets,target_paths,split,previous,output,view_scope=SCOPE,mask_arrays=masks,mask_records=mask_records)
            result['comparison_scope']=SCOPE;result['extrapolation_limit']='At256x172 fx121.7425204, tau1/3 are within default post[.005,.1]; tau6/15 outside. Author package training range/config not separately supplied; this is fixed-source nominal range, not recovered author run config.'
            result['memory_scope']='Full author tree plus canonical100k sky resident, no streaming/chunk eviction. City full-tree leafcontrol skipped beforeallocation byhardware bound.'
            e.save(output/'metrics.json',result)
            groups={r['name']:r['group'] for r in mask_records};group_metrics=[]
            for r in results:
                if not r['complete']:continue
                metric=e.read_json(r['folder']/'per-image-metrics.json')
                for group in sorted(set(groups.values())):
                    rows=[x for x in metric if groups[x['name']]==group]
                    group_metrics.append(dict(setting=r['setting'],group=group,view_count=len(rows),names=[x['name'] for x in rows],
                        psnr_mean_per_image_db=float(np.mean([x['psnr_numpy_db'] for x in rows])),ssim_mean=float(np.mean([x['ssim_torchmetrics'] for x in rows])),alex_lpips_mean=float(np.mean([x['lpips_alex_torchmetrics'] for x in rows])),
                        scope='Published chunk camera-list membership only; reconstruction views, not spatial outside or author heldout split'))
            e.save(output/'chunk-camera-membership-metrics.json',group_metrics);outcome.update(status=result['status'],cuda_used=True)
        else:outcome.update(status='passed_CPU_city_model_camera_mask_memory_loader_audits_no_GPU')
        outcome.update(ended_utc=e.baseline.now(),elapsed_seconds=time.perf_counter()-started,driver_sha256=e.digest(__file__),core_evaluator_sha256=e.digest(e.__file__))
        e.save(output/'outcome.json',outcome);print(json.dumps(outcome),flush=True)
    except Exception as error:
        torch_module=sys.modules.get('torch')
        initialized=bool(torch_module is not None and torch_module.cuda.is_initialized())
        outcome.update(status='failed_preserved_city_evaluation',cuda_used=initialized,cuda_context_initialized=initialized,
            error=str(error),traceback=traceback.format_exc(),ended_utc=e.baseline.now(),
            driver_sha256=e.digest(__file__),core_evaluator_sha256=e.digest(e.__file__))
        e.save(output/'outcome.json',outcome);raise
if __name__=='__main__':main()
