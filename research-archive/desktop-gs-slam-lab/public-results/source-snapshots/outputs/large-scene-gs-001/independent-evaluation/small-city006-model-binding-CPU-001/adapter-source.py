"""Own local SmallCity hierarchy evaluation, frozen117 photometric heldout.

Default: CPU source/schema/input/coarse-sky/metric protocol. --bind-model-cpu
requires real complete four-chunk Creator/post/paired-Merger dependencies.
--gpu additionally needs a future explicit exclusive root authorization; the
adapter never launches training or pretends a CPU plan is a completed model.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gc,importlib.util,json,os,shutil,subprocess,sys,time,traceback
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/large_scene_gs'))
import evaluate_official_hierarchy as e
DRIVER=Path(__file__).resolve().parent
ARCHIVES=e.OUTROOT/'city-local-hierarchy-evaluation-driver-CPU-001/archives'
ARCHIVE_MANIFEST_SHA='89a94b24558a44c4cd0ba8524f55467368cc722a1e1dc68af3efacf4ccd727b3'
LOCK=e.TRACK/'official-hierarchy/small-city-hierarchy-006'
CONFIG_SHA=None
NATIVE_WORKFLOW_SHA=None
NATIVE_LOCK=DRIVER/'native-workflow-lock.json'
BINDING_PATH=LOCK/'binding-audits/lineage006-complete-cpu-001/lineage-binding.json'
BINDING_SHA='1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c'
BINDER_SOURCE_SHA='c72fea79bb3b083ead1dfa58e9cf7e7d0134842d764dd3c5482162448dc4db9e'
WORKFLOW_SHA='0e51b5c51d0742a20315d02c37355aaae7e9f1d905069a1f6594da0ee06d7f84'
COARSE_RUN='small-city-coarse-i30000-r4-003'
COARSE_SHA='abd4f6af0463dce35648182a66d2ae3d6893417901f3885c5be1eb98368f2e31'
COARSE_DRIVER_SHA='348567a816ca30d2c91538b63a9fbc523ac03f695e232f95e0d98dffaefdcdc9'
CORE_SHA='672533b6aef0f79d64a77976055ce747d05c3f226646e576cb913b0c405af1ec'
CANON=e.OUTROOT/'small-city-local-coarse-003-flat-heldout-001'
INPUT_SHA='2dd5c04122e290d27934a224921b2506b7955a34d801a0482e0408af536de3d5'
SETTINGS=(('leaves',None),('tau1',1.),('tau3',3.),('tau5',5.),('tau6',6.),('tau15',15.))
SCOPE='Own local SmallCity four-chunk pre/post merged hierarchies,117 frozen GS-photometric heldout; training5705 globalviews and chunk-specific training lists. Published SfM calibration/XYZ/colors may useallphotos. Mixed origins: three legacy003 chunks plus observed11 with unchanged official algorithm and read-only hooks; observed11 remains not formal and its extra synchronization/tracing changes timing/memory. Notauthorpretrained18M reproduction or equal-budget coarse/dense causal comparison.'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def dependencies(output):
    e.require(e.digest(ARCHIVES/'manifest.json')==ARCHIVE_MANIFEST_SHA,'Archived evaluation dependencies changed')
    rows=e.read_json(ARCHIVES/'manifest.json')
    for row in rows:
        e.require(e.digest(row['archive_path'])==row['sha256']==e.digest(row['original_path']),'Frozen dependency bytes changed: '+row['name'])
    e.require(e.digest(e.__file__)==CORE_SHA,'Shared official evaluator changed')
    e.require(e.digest(ARCHIVES/'coarse_evaluator_v3.py')==COARSE_DRIVER_SHA and e.digest(ARCHIVES/'workflow_driver_v4.py')==WORKFLOW_SHA,'Wrong frozen coarse/workflow driver')
    shutil.copytree(ARCHIVES,output/'adapter-dependency-archives')
    (output/'adapter-source.py').write_bytes(Path(__file__).read_bytes())
    coarse=load('own_city_frozen_coarse_v3',ARCHIVES/'coarse_evaluator_v3.py')
    # Archived source has a different __file__; restore its original globals.
    coarse.ROOT=ROOT;coarse.INPUT=e.TRACK/'official-inputs/small-city-training-002';coarse.TRAIN=e.TRACK/'official-training'/COARSE_RUN
    workflow=load('own_city_frozen_workflow_v4',ARCHIVES/'workflow_driver_v4.py')
    workflow.ROOT=ROOT;workflow.OUTPUT=e.TRACK;workflow.REPO=e.REPO
    workflow.NATIVE_REPO=ROOT/'repos/gaussian-hierarchy-large-gs';workflow.BUILD=e.BUILD
    workflow.EXES=workflow.NATIVE_REPO/'build-local/Release'
    workflow.ENV_LAUNCHER=ROOT/'scripts/large_scene_gs/run_local.py';workflow.TRAINING_PYTHON=ROOT/'.venv-large-gs/Scripts/python.exe'
    e.require(workflow.DRIVER_VERSION==4,'Require locked workflow v4')
    return coarse,workflow,rows

def lineage_binding():
    e.require(BINDING_PATH.is_file() and e.digest(BINDING_PATH)==BINDING_SHA,'Actual completed mixed-lineage binding missing/changed')
    binding=e.read_json(BINDING_PATH)
    e.require(binding['status']=='actual_complete_mixed_local_training_lineage_CPU_bound' and binding['binder_source_sha256']==BINDER_SOURCE_SHA and binding['formal_observation_reclassification'] is False,'Wrong completed lineage origin')
    e.require(binding['Creator_post_Merger_or_evaluation_complete'] is False and [r['chunk'] for r in binding['ordered_chunks']]==['0_0','0_1','1_0','1_1'],'Training-only receipt/order scope differs')
    accounting=binding['observed_terminal']
    e.require(accounting['formal_training_run'] is False and accounting['actual_exposure_Adam_calls']==accounting['actual_opacity_grad_condition_visits']==29999 and accounting['actual_Gaussian_Adam_calls']+accounting['actual_None_gradient_skips']==29999 and 30000 in accounting['successful_original_save_iterations'],'Observed actual budget/call/skip accounting differs')
    return binding

def native_configuration():
    path=LOCK/'configuration.json';config=e.read_json(path);plan=e.read_json(LOCK/'plan.json')
    digest=e.digest(path);driver=e.digest(LOCK/'driver-source.py')
    e.require(config['driver_version']>=5 and config['driver_sha256']==driver and plan['configuration_sha256']==digest,'Wrong/changed actual006 driver/config/plan')
    e.require(config['training_lineage_binding_sha256']==BINDING_SHA and Path(config['training_lineage_binding_path']).resolve()==BINDING_PATH.resolve(),'Native config uses another training family')
    e.require(config['input_manifest_sha256']==INPUT_SHA and config['input']=='small-city-training-002' and config['coarse_run']==COARSE_RUN and not config['author_pretrained_model_used'] and config['resolution']==4 and config['pixels']==[256,172],'Actual006 input/coarse/r4 lineage differs')
    binding=lineage_binding();expected={r['chunk']:Path(r['training_directory']).name for r in binding['ordered_chunks']}
    e.require(config['chunk_runs']==expected and list(config['chunk_runs'])==['0_0','0_1','1_0','1_1'],'006 chunk runs/origin order differs')
    e.require(config['post_iterations']==15000 and config['expected_post_optimizer_calls']==14999 and config['post_full_train_defaults']['dimensionless_log_uniform_limit']==[.005,.1],'Official post budget/default range differs')
    e.require(config['official_main_commit']==e.COMMIT and config['official_hierarchy_commit']=='677c8553dc64dfd62c272eca94a291a277733113','Wrong official commits')
    for row in config['source_records']:e.require(e.digest(row['path'])==row['sha256'],'Locked actual006 source changed')
    for row in config['frozen_helper_records']:e.require(e.digest(row['snapshot_path'])==row['sha256'],'Locked actual006 helper changed')
    return config,digest,driver

def freeze_workflow_cpu(output):
    config,digest,driver=native_configuration()
    e.require(not NATIVE_LOCK.exists(),'006 native evaluation lock already exists; preserve it')
    lock=dict(status='actual006_native_source_config_and_completed_training_lineage_CPU_locked_not_trees',adapter_source_sha256=e.digest(__file__),configuration_sha256=digest,driver_source_sha256=driver,configuration_path=str(LOCK/'configuration.json'),driver_source_path=str(LOCK/'driver-source.py'),lineage_binding_path=str(BINDING_PATH),lineage_binding_sha256=BINDING_SHA,CUDA_called=False)
    e.save(NATIVE_LOCK,lock);e.save(output/'006-native-workflow-lock.json',lock)
    return lock

def lock_schema(workflow,output):
    global CONFIG_SHA,NATIVE_WORKFLOW_SHA
    e.require(NATIVE_LOCK.is_file(),'Actual006 native source/config evaluation lock missing; CPU preparation is not real tree completion')
    seal=e.read_json(NATIVE_LOCK)
    e.require(seal['adapter_source_sha256']==e.digest(__file__) and seal['lineage_binding_sha256']==BINDING_SHA,'Wrong immutable006 evaluation lock')
    config,digest,driver=native_configuration()
    e.require(seal['configuration_sha256']==digest and seal['driver_source_sha256']==driver,'Actual006 configuration/source changed after evaluation lock')
    CONFIG_SHA,NATIVE_WORKFLOW_SHA=digest,driver
    inputs=e.TRACK/'official-inputs'/config['input'];manifest=e.read_json(inputs/'manifest.json')
    e.require(e.digest(inputs/'manifest.json')==INPUT_SHA,'Input manifest changed')
    split_audit=workflow.input_split_audit(inputs,manifest)
    e.require(split_audit==config['input_split_audit'],'Actual sparse/0/test Reader partition changed')
    for filename in ('configuration.json','plan.json','driver-source.py'):
        shutil.copyfile(LOCK/filename,output/('workflow-lock006-'+filename))
    shutil.copyfile(NATIVE_LOCK,output/'006-native-workflow-lock.json')
    shutil.copyfile(BINDING_PATH,output/'006-completed-training-lineage-binding.json')
    return config,manifest

def context(output,settings):
    coarse,workflow,archive=dependencies(output);config,manifest=lock_schema(workflow,output)
    coarse_config,_,cameras,targets,target_paths,masks,rows,split=coarse.context(output)
    # Preserve the coarse context as input provenance, and write a separate
    # local-hierarchy protocol without relabeling its previous quality results.
    (output/'protocol.json').rename(output/'coarse-input-contract.json')
    tensors,background_binding=coarse.model_CPU(coarse_config,output)
    e.require(background_binding['source_model_sha256']==COARSE_SHA,'Wrong actual local coarse background')
    del tensors;gc.collect()
    canonical=e.read_json(CANON/'protocol.json');z=np.load(canonical['evaluated_input_npz_path'])
    current=e.read_json(output/'coarse-input-contract.json');own=np.load(current['evaluated_input_npz_path'])
    e.require(e.digest(canonical['evaluated_input_npz_path'])==canonical['evaluated_input_npz_sha256'],'Canonical117 archive changed')
    for key in ('names','rgb','mask','K','c2w','source_camera_rows'):
        e.require(np.array_equal(z[key],own[key]),'117 canonical input array differs: '+key)
    e.require(canonical['test_names']==split['test_names'] and canonical['groups']==current['groups'],'Canonical view order/groups changed')
    sky,sky_audit=e.skybox_PLY_CPU(Path(background_binding['source_model_path']),100000,output,
        dict(source_kind='own_local_completed_coarse003',source_model_sha256=COARSE_SHA,training_run=COARSE_RUN,source_binding=background_binding))
    descriptions=[]
    for name,filename,receipt,kind in [('merged-unoptimized','merged-unoptimized.hier','merge-unoptimized-outcome.json','unoptimized'),('merged-post','merged.hier','merge-outcome.json','post_optimized')]:
        descriptions.append(dict(name=name,path=str(LOCK/filename),receipt_path=str(LOCK/receipt),expected_source_kind=kind,
            append_skybox=100000,available=(LOCK/filename).is_file() and (LOCK/receipt).is_file(),source_model_sha256=None,actual_local_model_bound=False))
    view_records=[dict(**row,threshold_scope={s:e.threshold_scope(t,float(row['K'][0][0])) for s,t in settings}) for row in rows]
    previous=e.read_json(e.OUTROOT/'truck-baseline-heldout-001/metrics.json')
    # Private metadata copy only: the frozen Truck metric file stays intact.
    previous=dict(previous)
    previous['psnr_aggregation']=f'NumPy float64 pixel RGB MSE and -10log10 per image; arithmetic mean of {len(split["test_names"])} frozen image PSNR values, not PSNR of pooled MSE'
    for weight in previous['weight_files']:e.require(Path(weight['path']).is_file() and e.digest(weight['path'])==weight['sha256'],'Fixed metric weights changed or missing; no downloads')
    metric_control=[];coarse_metrics=e.read_json(CANON/'metrics.json');pred=np.load(CANON/'predictions.npz')
    e.require(pred['names'].tolist()==split['test_names'] and len(coarse_metrics['per_image'])==117,'Canonical actual coarse metric order differs')
    for j,old in enumerate(coarse_metrics['per_image']):
        _,_,r=coarse.masked_numpy(pred['rgb'][j],targets[j],masks[j])
        e.require(r['psnr_numpy_db']==old['psnr_numpy_db'] and r['RGB_error_numerator']==old['RGB_error_numerator'] and r['RGB_error_denominator']==old['RGB_error_denominator'],'Independent reused mask/PSNR formula differs')
        e.require(r['symmetric_single_mask_valid_weight_diagnostic']==old['symmetric_single_mask_valid_weight_diagnostic'],'Reused auxiliary mask denominator differs')
        metric_control.append(dict(name=old['name'],main_PSNR_numpy_exact_against_actual_coarse=True,auxiliary_weighted_error_exact=True,full_RGB_denominator=132096))
    e.save(output/'metric-contract-CPU.json',dict(status='fixed117_actual_inputs_and_main_auxiliary_numpy_protocol_passed',fixed_versions=e.FIXED,
        ssim_protocol=previous['ssim_protocol'],lpips_protocol=previous['lpips_protocol'],weight_files=previous['weight_files'],
        psnr_aggregation=previous['psnr_aggregation'],mask_protocol='GTm2/predm across full3WH; auxiliary symmetric single-mask valid weighted denominator3sum(m2), separate',
        source_actual_coarse_metrics_path=str(CANON/'metrics.json'),source_actual_coarse_metrics_sha256=e.digest(CANON/'metrics.json'),
        actual_coarse_quality_is_not_future_hierarchy_quality=True,controls=metric_control,CUDA_called=False))
    e.save(output/'protocol.json',dict(status='CPU_ready_own_local_hierarchy117_inputs_schema_metrics_future_models_unbound',view_scope=SCOPE,
        hierarchy_source_lock='small-city-hierarchy-006',hierarchy_configuration_sha256=CONFIG_SHA,hierarchy_workflow_driver_sha256=NATIVE_WORKFLOW_SHA,training_lineage_binding_path=str(BINDING_PATH),training_lineage_binding_sha256=BINDING_SHA,native_workflow_lock_sha256=e.digest(NATIVE_LOCK),formal_observation_reclassification=False,
        canonical117_input_archive_sha256=canonical['evaluated_input_npz_sha256'],independent_input_npz_path=current['evaluated_input_npz_path'],
        independent_input_npz_sha256=current['evaluated_input_npz_sha256'],test_names=split['test_names'],train_global_count=5705,view_count=117,
        groups=current['groups'],group_semantics=current['camera_group_semantics'],views=view_records,settings=[s for s,t in settings],
        actual_owncoarse_background_binding=background_binding,canonical_background_sky=sky_audit,models=descriptions,
        tree_model_state='Future real complete local models not inferred from plan or CPU schema. No model/node/leaf hash invented.',
        renderer='Frozen unmodified author render_post(interp_python=True); actualendpointnormalize/signflip then linearquat withoutpostblendnormalize; siblingpixelalpha kernel, activatedabs hierarchyalpha; noopacityhack.',
        masks='Same117 GTm2/predm/full3WH main; separate symmetric weighted diagnostic. Source masks unchanged and may retain dynamic foreground objects.',
        exposure='Evaluation use_trained_expFalse; test exposures never fitted. Actual chunk exposures retained/frozen during post, not reused for heldout rendering.',
        memory='Whole tree + one owncoarse100k sky resident for LOD; Torch allocator and synchronized whole-device boundaries reported separately from unknown native transient workspace. No streaming.',
        compact_leaf='Truck exactpixel equivalence retained as code-path evidence; City compact only after bounded same-actual-model/same-camera pixel gate in safe memory. Current adapter does not auto-enable compact.',
        archive_manifest_sha256=ARCHIVE_MANIFEST_SHA,CUDA_called=False))
    e.save(output/'future-model-binding-schema.json',dict(status='schema_prepared_actual_tree_dependencies_pending',configuration_SHA256=CONFIG_SHA,
        chunk_runs=config['chunk_runs'],source_kinds=['unoptimized','post_optimized'],merged_unowned_tail_required=0,post_chunk_unowned_tail_required=100000,
        sky_append_once=100000,expected_merger_order=list(config['chunk_runs']),ownership=config['merge_ownership'],actual_leaf_and_node_counts=None,
        required_gates=['Actualexit0>=30kchunk/coarse lineage/split/source/modelhash and trainedexposure identities','ActualCreator/post/merge exit0 and lockedconfigurationSHA; stageinputhashes','Independently decode pre/post and verify anchors5fields exact,nodes/boxes unchanged,posttailhalfwriterroundtrip','Actualmergedreadonlyloader vs authorCPU arrayhashes,finite/activatedalpha/scale/quaternion/topology','Neverinfermergedleaf/node equality; compare measured post structure','No CUDA until allactual model gates plus rootexclusiveauthorization and actualno-trainers'],CUDA_called=False))
    return coarse,workflow,config,manifest,descriptions,sky,cameras,targets,target_paths,masks,rows,split,previous

def local_training_gate(workflow,config,manifest,output):
    binding=lineage_binding();training=[]
    for source in [binding['coarse']]+binding['ordered_chunks']:
        stage,chunk=source['stage'],source['chunk'];directory=Path(source['training_directory'])
        receipt=e.read_json(directory/'outcome.json');configuration=e.read_json(directory/'configuration.json')
        observed=chunk=='1_1'
        expected_directory=e.TRACK/('official-training-observation' if observed else 'official-training')/directory.name
        e.require(directory.resolve()==expected_directory.resolve(),'Training source directory kind differs')
        e.require(e.digest(directory/'outcome.json')==source['training_receipt']['sha256'] and e.digest(directory/'configuration.json')==source['configuration']['sha256'] and e.digest(directory/'model-audit.json')==source['model_audit_sha256'],'Actual training receipt/config/audit differs from completed binding')
        e.require(receipt['status']==('completed_observation_only_final_finite_not_formal' if observed else 'completed') and receipt['exit_code']==0 and receipt['requested_iterations']==30000,'Incomplete actual mixed-origin local training')
        e.require(source['training_source_kind']==('observed_local_official_training' if observed else 'legacy_local_official_training'),'Training origin sourcekind differs')
        if observed:e.require(source['formal_training_run'] is False and receipt['formal_completed_model_claimed'] is False,'Observed origin cannot be formal-reclassified')
        else:e.require(source['actual_Gaussian_Adam_calls'] is None,'Legacy derived count relabeled as actual')
        parameter=Path(source['model']['path']);e.require(e.digest(parameter)==source['model']['sha256']==receipt['model_parameter_sha256'],'Actual local modelparameter SHA mismatch')
        expected_names=manifest['train_names'] if stage=='coarse' else next(c['train_names'] for c in manifest['chunks'] if c['name']==chunk)
        e.require(configuration['training_names']==expected_names and not set(expected_names)&set(manifest['test_names']),'Actual mixed training image family changed')
        exp=workflow.exposure_audit(directory/'model/exposure.json',expected_names,manifest['test_names'])
        e.require(exp['sha256']==source['exposure']['sha256'] and exp['test_overlap']==0,'Actual trained exposure differs from completed lineage')
        training.append(dict(stage=stage,chunk=chunk,run=directory.name,training_directory=str(directory),training_source_kind=source['training_source_kind'],formal_training_run=source['formal_training_run'],actual_Gaussian_Adam_calls=source['actual_Gaussian_Adam_calls'],derived_Gaussian_call_upper_bound=source['derived_Gaussian_call_upper_bound'],outcome_sha256=e.digest(directory/'outcome.json'),configuration_sha256=e.digest(directory/'configuration.json'),model_path=str(parameter),model_sha256=e.digest(parameter),saved_model_directory=receipt['saved_model_directory'],exposure=exp,source_receipt=receipt))
    result=dict(status='allfive_actual_completed_mixed_local_source_split_model_exposure_receipts_passed',records=training,training_lineage_binding_path=str(BINDING_PATH),training_lineage_binding_sha256=BINDING_SHA,formal_observation_reclassification=False,observed_timing_memory_limit=binding['observed_terminal']['observation_overhead'],CUDA_called=False)
    e.save(output/'actual-local-training-lineage-CPU.json',result)
    return training

def stage_and_merge_gate(workflow,config,manifest,sky_audit,output):
    lineage=local_training_gate(workflow,config,manifest,output)
    # Load only frozen helper functions. Their globals are restored explicitly;
    # never call a plan/execution function or create files in source lock005.
    codec=load('own_city_locked_hierarchy_codec',ARCHIVES/'read_hierarchy.py')
    previous=sys.modules.get('read_hierarchy');sys.modules['read_hierarchy']=codec
    try:helper=load('own_city_locked_post_freeze',ARCHIVES/'run_official_hierarchy.py')
    finally:
        if previous is None:del sys.modules['read_hierarchy']
        else:sys.modules['read_hierarchy']=previous
    helper.ROOT=ROOT;helper.OUTPUT=e.TRACK;helper.REPO=e.REPO;helper.BUILD=e.BUILD;helper.EXES=ROOT/'repos/gaussian-hierarchy-large-gs/build-local/Release'
    scaffold=Path(sky_audit['ply_path']).parent;stage_records=[]
    chunks={c['name']:c for c in manifest['chunks']}
    for chunk in config['chunk_runs']:
        directory=LOCK/'trained_chunks'/chunk
        creator=e.read_json(directory/'creator-outcome.json');post=e.read_json(directory/'post-outcome.json')
        actual_training=next(r for r in lineage if r['chunk']==chunk)
        for receipt in (creator,post):e.require(receipt['training_lineage_binding_sha256']==BINDING_SHA,'Native stage training lineage changed')
        e.require(creator['source_training_receipt']==actual_training['source_receipt'] and creator['input_model']['sha256']==actual_training['model_sha256'],'Creator not derived from selected actual completed local chunk')
        for receipt,kind in ((creator,'unoptimized'),(post,'post_optimized')):
            e.require(receipt['status']=='completed' and receipt['exit_code']==0 and receipt['source_kind']==kind and receipt['configuration_sha256']==CONFIG_SHA,'Incomplete/wrong actualCreator/post receipt')
        before_path=directory/'hierarchy.hier';after_path=directory/'hierarchy.hier_opt'
        e.require(e.digest(before_path)==creator['hierarchy_sha256'] and e.digest(after_path)==post['hierarchy_sha256'],'Actualstagehierarchy changed')
        e.require(post['requested_iterations']==15000 and post['optimizer_update_calls']==14999 and post['dimensionless_sampling']==[.005,.1] and post['exposure_optimizer_updates']==0 and not post['depth_loss_present_in_unchanged_post'],'Wrong postworkflow budget/source contract')
        exp=workflow.exposure_audit(directory/'exposure.json',chunks[chunk]['train_names'],manifest['test_names'])
        e.require(exp['sha256']==actual_training['exposure']['sha256']==creator['copied_exposure_sha256']==post['frozen_exposure']['sha256'],'Actual frozen trained exposure changed')
        before=e.read_hierarchy(before_path);after=e.read_hierarchy(after_path,expected_unowned_tail=100000)
        anchors=codec.read_anchors(directory/'anchors.bin',len(before.positions))
        e.require(e.digest(directory/'anchors.bin')==creator['anchor_sha256'] and len(anchors)==creator['anchor_count'],'Actualanchors changed')
        freeze=helper.post_freeze_audit(before,after,anchors,scaffold,100000)
        e.require(freeze['status']=='frozen_anchor_and_skybox_fields_verified','Independentactualpostfreeze failed')
        stage_records.append(dict(chunk=chunk,creator_receipt_sha256=e.digest(directory/'creator-outcome.json'),post_receipt_sha256=e.digest(directory/'post-outcome.json'),
            before_sha256=creator['hierarchy_sha256'],after_sha256=post['hierarchy_sha256'],before_summary=before.summary(),after_summary=after.summary(),actual_recomputed_freeze=freeze,exposure=exp))
        del before,after,anchors;gc.collect()
    merges=[]
    for filename,kind in [('merge-unoptimized-outcome.json','unoptimized'),('merge-outcome.json','post_optimized')]:
        receipt=e.read_json(LOCK/filename)
        e.require(receipt['status']=='completed' and receipt['exit_code']==0 and receipt['source_kind']==kind and receipt['configuration_sha256']==CONFIG_SHA and receipt['training_lineage_binding_sha256']==BINDING_SHA,'Merge incomplete/changed scope')
        e.require(receipt['chunk_order']==list(config['chunk_runs']) and receipt['chunk_centers']=={n:chunks[n]['center'] for n in config['chunk_runs']} and receipt['ownership']==config['merge_ownership'],'Merge order/centers/ownership differs')
        e.require(receipt['skybox_in_tree']==0 and receipt['shared_background_skybox_count']==100000 and Path(receipt['shared_background_skybox']).resolve()==scaffold.resolve(),'Mergedtree unexpected sky/coarse identity')
        deps=receipt['input_hierarchies'];e.require([r['chunk'] for r in deps]==list(config['chunk_runs']) and all(r['source_kind']==kind for r in deps),'Mergeinputs source kind/order differs')
        for d in deps:
            expected=LOCK/'trained_chunks'/d['chunk']/('hierarchy.hier' if kind=='unoptimized' else 'hierarchy.hier_opt')
            e.require(Path(d['path']).resolve()==expected.resolve() and e.digest(expected)==d['sha256'],'Actual merge input path/hash changed')
        if kind=='unoptimized':
            variant=e.read_json(e.BUILD/'merger-unoptimized/manifest.json');exe=Path(variant['executable']['path'])
            e.require(e.digest(e.BUILD/'merger-unoptimized/manifest.json')==config['unoptimized_filename_variant_manifest_sha256'],'Merger filenameonly variant changed')
        else:exe=ROOT/'repos/gaussian-hierarchy-large-gs/build-local/Release/GaussianHierarchyMerger.exe'
        e.require(e.digest(exe)==receipt['merger_executable_sha256'],'ActualMerger exe changed')
        merges.append(dict(source_kind=kind,receipt_path=str(LOCK/filename),receipt_sha256=e.digest(LOCK/filename)))
    result=dict(status='actual_completed_local_pre_post_workflow_CPU_identity_and_freeze_passed',configuration_sha256=CONFIG_SHA,training_lineage_binding_sha256=BINDING_SHA,formal_observation_reclassification=False,lineage=lineage,stage_records=stage_records,merge_records=merges,CUDA_called=False)
    e.save(output/'actual-workflow-binding-CPU.json',result);return result

def model_cpu(desc,sky,cameras,settings,output):
    # This shared model_cpu validates the actual merge SHA, dependencies and
    # tail0, matches the author CPU loader, and appends this own sky once.
    prepared=e.model_cpu(desc,None,sky,output);root,h,topology,gh,tensors,leaves,audit=prepared
    receipt=e.read_json(Path(desc['receipt_path']))
    for key in ('compressed','gaussian_count','node_count','leaf_gaussian_count','merged_gaussian_count','root_depth','unowned_tail'):
        e.require(receipt['hierarchy'][key]==audit['tree_summary'][key],'Actual merged storage/count differs from saved receipt: '+key)
    audit.update(view_scope=SCOPE,workflow_lock006_configuration_sha256=CONFIG_SHA,coarse_source_model_sha256=COARSE_SHA,exposure_enabled=False,
        source_length_units_not_claimed_meters=True,compact_leaf_only_planned_until_actual_local_pixel_gate=True)
    e.save(root/'model-audit.json',audit)
    counts=[];maximum=0
    for camera in cameras:
        fx=camera.image_width/(2*np.tan(camera.FoVx*.5))
        for name,tau in settings:
            if tau is None:count=len(leaves)
            else:reference,_=topology.reference_counts(camera.camera_center.numpy(),e.tau_to_threshold(tau,fx));count=int(reference.sum())
            counts.append(dict(name=camera.image_name,setting=name,CPU_predicate_selected_tree_G=count));maximum=max(maximum,count)
    plan=e.memory_plan(audit['total_G'],len(h.nodes),len(leaves),maximum,100000,len(cameras))
    plan.update(CPU_actual_view_counts=counts,actual_full_tree_leaf_upper_bytes=e.memory_plan(audit['total_G'],len(h.nodes),len(leaves),len(leaves),100000,len(cameras))['conservative_Python_tensor_peak_upper_bytes'],
        compact_leaf_policy='Plan only. Exact actual-local samecamera pixel gate required before using compact as reference; neverstreaming or sameGPUstate memorycomparison.')
    e.save(root/'memory-preflight-CPU.json',plan);return prepared

def no_live_trainers(output,model):
    path=DRIVER/'query_windows_gpu_roles_006.py'
    e.require(e.digest(path)=='eb9bab149ebc03f289c8e17bdfcd223b9b5d3168cb1ee3adf9804b496e3b689e','006 private PS7 process query helper changed')
    helper=load('eval006_process_query',path)
    report=helper.query('pre_gpu',evaluator_source=Path(__file__).resolve())
    e.save(output/(model+'-preGPU-actual-training-process-audit.json'),report)
    e.require(not report['actual_conflicting_GPU_stage_processes'],'Actual training/native/other GPU evaluation process is live; GPU evaluation refused')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation-run',required=True)
    p.add_argument('--settings',nargs='+',choices=[s for s,t in SETTINGS],default=[s for s,t in SETTINGS])
    mode=p.add_mutually_exclusive_group();mode.add_argument('--bind-model-cpu',action='store_true');mode.add_argument('--gpu',action='store_true');mode.add_argument('--freeze-workflow-cpu',action='store_true');a=p.parse_args()
    e.require(a.evaluation_run.replace('-','').isalnum() and len(a.evaluation_run)<=100 and len(set(a.settings))==len(a.settings),'Invalidrun/settings')
    settings=tuple((s,t) for s,t in SETTINGS if s in a.settings);output=e.OUTROOT/a.evaluation_run;output.mkdir(exist_ok=False)
    start=time.perf_counter();outcome=dict(status='running',started_utc=e.baseline.now(),cuda_used=False,requested_GPU=a.gpu,gpu_dispatch_attempted=False,view_scope=SCOPE)
    e.save(output/'outcome.json',outcome)
    try:
        if a.freeze_workflow_cpu:
            freeze_workflow_cpu(output);outcome.update(status='actual006_native_source_and_lineage_CPU_locked_not_trees',driver_sha256=e.digest(__file__),core_evaluator_sha256=e.digest(e.__file__),cuda_used=False);e.save(output/'outcome.json',outcome);print(json.dumps(outcome),flush=True);return
        e.require({k:e.importlib.metadata.version(k) for k in e.FIXED}==e.FIXED,'Fixed runtime/metric versions changed')
        if a.bind_model_cpu or a.gpu:
            e.require(all((LOCK/f).is_file() for f in ('merged-unoptimized.hier','merged.hier','merge-unoptimized-outcome.json','merge-outcome.json')),'Actual006 pre/post merged trees and receipts missing; refuse model binding/GPU before coarse or tree loading')
        coarse,workflow,config,manifest,desc,sky,cameras,targets,target_paths,masks,rows,split,previous=context(output,settings)
        if a.bind_model_cpu or a.gpu:
            e.require(all(d['available'] for d in desc),'Actual own local pre/post merged model files/receipts unavailable; CPU plan is not training completion and GPU is refused')
            actual=stage_and_merge_gate(workflow,config,manifest,e.read_json(output/'skybox-audit.json') if (output/'skybox-audit.json').exists() else e.read_json(output/'protocol.json')['canonical_background_sky'],output)
            results=[]
            for d in desc:
                prepared=model_cpu(d,sky,cameras,settings,output)
                actual_audit=prepared[-1]
                d.update(actual_local_model_bound=True,source_model_sha256=actual_audit['source_sha256'],actual_tree_summary=actual_audit['tree_summary'],actual_leaf_count=actual_audit['leaf_count'])
                if a.gpu:
                    no_live_trainers(output,d['name']);outcome['gpu_dispatch_attempted']=True;e.save(output/'outcome.json',outcome)
                    results.extend(e.gpu_model(d,None,sky,cameras,targets,split,output,prepared=prepared,settings=settings))
                del prepared;gc.collect()
            bound_protocol=e.read_json(output/'protocol.json')
            bound_protocol.update(status='actual_completed_own_local_pre_post_model_binding_passed',models=desc,tree_model_state='Actual complete source/receipt/training/anchor/sky/count/finite/topology/authorloader audits bound both real models. Node/leaf topology equality is not assumed.')
            e.save(output/'protocol.json',bound_protocol)
            if a.gpu:
                masks_meta=[dict(name=r['name'],group=r['group'],source_path=r['source_mask_path'],source_sha256=r['source_mask_sha256'],derived_float32_sha256=r['mask_float32_sha256']) for r in rows]
                result=e.metrics(results,targets,target_paths,split,previous,output,view_scope=SCOPE,mask_arrays=masks,mask_records=masks_meta)
                # The immutable Truck helper pairs its original five settings.
                # Build all requested local pairs here, including new tau5.
                local_pairs=[]
                for setting,tau in settings:
                    pre=next(r for r in results if r['model']=='merged-unoptimized' and r['setting']==setting)
                    post=next(r for r in results if r['model']=='merged-post' and r['setting']==setting)
                    if not(pre['complete'] and post['complete']):continue
                    pm=e.read_json(pre['folder']/'metrics.json');qm=e.read_json(post['folder']/'metrics.json')
                    a0,b0=pre['model_audit'],post['model_audit'];rgb0=np.stack(pre['predictions']);rgb1=np.stack(post['predictions'])
                    local_pairs.append(dict(setting=setting,threshold_training_scope=post['timing']['threshold_training_scope'],
                        post_minus_unoptimized={k:qm[k]-pm[k] for k in ('psnr_numpy_mean_per_image_db','ssim_mean_per_image','lpips_alex_mean_per_image')},
                        pre_tree_G=a0['tree_G'],post_tree_G=b0['tree_G'],pre_leaves=a0['leaf_count'],post_leaves=b0['leaf_count'],pre_nodes=a0['tree_summary']['node_count'],post_nodes=b0['tree_summary']['node_count'],
                        node_bytes_identical=a0['node_storage_sha256']==b0['node_storage_sha256'],boxes_identical=a0['box_storage_sha256']==b0['box_storage_sha256'],
                        all117_RGB_bitwise_equal=bool(np.array_equal(rgb0,rgb1)),max_abs_RGB_difference=float(np.max(np.abs(rgb0-rgb1)))))
                result['paired_merge_stage_comparison']=local_pairs
                result['extrapolation_limit']='ActualownCityr4 post range[.005,.1], supported tau1/3/5; tau6/15 are separate extrapolation diagnostics.'
                result.pop('cross_renderer_comparison_limit',None)
                groups={r['name']:r['group'] for r in rows};group_metrics=[]
                for r in results:
                    if not r['complete']:continue
                    metric=e.read_json(r['folder']/'per-image-metrics.json')
                    for group in sorted(set(groups.values())):
                        sub=[x for x in metric if groups[x['name']]==group]
                        group_metrics.append(dict(model=r['model'],setting=r['setting'],group=group,count=len(sub),names=[x['name'] for x in sub],
                            psnr_mean_per_image_db=float(np.mean([x['psnr_numpy_db'] for x in sub])),ssim_mean=float(np.mean([x['ssim_torchmetrics'] for x in sub])),alex_lpips_mean=float(np.mean([x['lpips_alex_torchmetrics'] for x in sub]))))
                e.save(output/'camera-list-group-metrics.json',group_metrics)
                complete=all(r['complete'] for r in results) and len(results)==2*len(settings)
                result.update(comparison_scope=SCOPE,exposure_enabled=False,source_lock006_configuration_SHA=CONFIG_SHA,training_lineage_binding_sha256=BINDING_SHA,native_workflow_lock_sha256=e.digest(NATIVE_LOCK),formal_observation_reclassification=False,actual_owncoarse_SHA=COARSE_SHA,
                    canonical117_input_scope=True,complete_all_requested_settings=complete,complete_full_leaves_tau1_3_5_6_15_protocol=complete and settings==SETTINGS,
                    range_limit='r4fx121.7425204:tau1/3/5withinactualpostdefault[.005,.1],tau6/15outsideandseparateextrapolationdiagnostics; nevercalloutsidefailuresoverallpostineffectiveness',memory_scope='Fullresidentactualowntree+oneowncoarse100ksky; noauthor18Msubstitution/streaming')
                e.save(output/'metrics.json',result);e.require(complete,'Actualpartialframes/settings preserved; no fullprotocolcompletion claim')
                outcome.update(status='complete117_each_requested_actual_local_pre_post_setting',cuda_used=True)
            else:outcome['status']='passed_actual_local_pre_post_model_binding_CPU_no_GPU'
        else:outcome['status']='passed117_local_inputs_schema_coarse_sky_metric_contract_CPU_future_trees_unbound'
        outcome.update(training_lineage_binding_sha256=BINDING_SHA,native_workflow_lock_sha256=e.digest(NATIVE_LOCK),hierarchy_configuration_sha256=CONFIG_SHA,hierarchy_workflow_driver_sha256=NATIVE_WORKFLOW_SHA,formal_observation_reclassification=False,ended_utc=e.baseline.now(),elapsed_seconds=time.perf_counter()-start,driver_sha256=e.digest(__file__),core_evaluator_sha256=e.digest(e.__file__))
        e.save(output/'outcome.json',outcome);print(json.dumps(outcome),flush=True)
    except Exception as error:
        torch_module=sys.modules.get('torch');initialized=bool(torch_module is not None and torch_module.cuda.is_initialized())
        outcome.update(status='failed_preserved_own_local_city_evaluation',cuda_used=initialized,cuda_context_initialized=initialized,error=str(error),traceback=traceback.format_exc(),ended_utc=e.baseline.now(),driver_sha256=e.digest(__file__),core_evaluator_sha256=e.digest(e.__file__))
        e.save(output/'outcome.json',outcome);raise
if __name__=='__main__':main()
