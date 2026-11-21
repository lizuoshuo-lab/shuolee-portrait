"""Independent real006 unoptimized Merger binding and CPU memory accounting.

No native/renderer dispatch, Torch import, GPU query, training or sealed edits.
Prepare has no merged-model completion; bind requires real terminal Merger bytes.
"""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'outputs/large-scene-gs-001';OUT=BASE/'independent-evaluation'
LOCK=BASE/'official-hierarchy/small-city-hierarchy-006'
DRIVER=OUT/'city-local-hierarchy-evaluation-driver-CPU-006'
CORE=ROOT/'scripts/large_scene_gs/evaluate_official_hierarchy.py'
LEDGER=OUT/'city-anchor-lineage-driver-CPU-006/audit_city_hierarchy_lineage_006_posebytes_v2.py'
CODEC=LOCK/'helper-snapshots/read_hierarchy.py'
ALL4=OUT/'small-city006-all4-creator-anchor-source-exposure-CPU-001/report.json'
INPUT=BASE/'official-inputs/small-city-training-002'
CANONICAL=OUT/'small-city-local-coarse-003-flat-heldout-001/independent-inputs/evaluated-inputs.npz'
VARIANT=BASE/'hierarchy-build/merger-unoptimized/manifest.json'
ORDER=['0_0','0_1','1_0','1_1']
BINDING=LOCK/'binding-audits/lineage006-complete-cpu-001/lineage-binding.json'
PIN={BINDING:'1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c',
 LOCK/'configuration.json':'562c24c7bf56d5debfd32b62fc523bce7133dc1d322a29eb76e55508a5440e47',
 LOCK/'driver-source.py':'a44e97310e02b742ed3222d5e9d5e6777526d8485412662c0b46ced0ce1f50a6',
 ROOT/'scripts/large_scene_gs/run_small_city_hierarchy_006.py':'a44e97310e02b742ed3222d5e9d5e6777526d8485412662c0b46ced0ce1f50a6',
 DRIVER/'evaluate_local_city_hierarchy_006.py':'260827a71f429fb49f0454c96054cdc3b3aaa08867225a56dafc525c39a75bb3',
 OUT/'city-hierarchy-posterior-auditor-CPU-006/audit_city_1404_006.py':'5c9c1b7da3d2bbeeb072a977aa69bac09c81dc5511f71c7c51fe0e35139e4d8c',
 DRIVER/'query_windows_gpu_roles_006.py':'eb9bab149ebc03f289c8e17bdfcd223b9b5d3168cb1ee3adf9804b496e3b689e',
 DRIVER/'native-workflow-lock.json':'0fac8a7e861fb05f5b67d3edab7851dacf4d6af99f211d16f1cf78c9b3eebea8',
 CORE:'672533b6aef0f79d64a77976055ce747d05c3f226646e576cb913b0c405af1ec',
 LEDGER:'aa4d620778231f1b9c405d95b4c6a04cf2af734eab5a8941817a0cfa0b408bb0',
 CODEC:'583c96f50b6f486d73bbbbaf9faf053ed15d486a7c7b48e15ef8e3ebc0964538',
 ALL4:'17be8b334c09f3dfbe91ba5e56d0e8107b674b379bc8977dc7447c12d572f774',
 CANONICAL:'60ab56f705c6db347b00f821b08a44e49c0d2dca4f0f06e9a978b9c73c8bce25'}
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def exact_sources():
    for p,d in PIN.items():require(sha(p)==d,'Bound source changed: '+str(p))
    config=read(LOCK/'configuration.json');binding=read(BINDING)
    require(not binding['formal_observation_reclassification'] and list(config['chunk_runs'])==ORDER,'Actual mixed origin/order changed')
    variant=read(VARIANT);require(sha(VARIANT)==config['unoptimized_filename_variant_manifest_sha256'],'Filename variant manifest changed')
    for p,d in [(variant['source_path'],variant['official_source_sha256']),(variant['derived_source_path'],variant['derived_source_sha256']),(variant['patch_path'],variant['patch_sha256']),(variant['executable']['path'],variant['executable']['sha256'])]:require(sha(p)==d,'Filename variant artifact changed: '+str(p))
    official=Path(variant['source_path']).read_bytes();derived=Path(variant['derived_source_path']).read_bytes()
    require(official.count(b'/hierarchy.hier_opt')==1 and derived==official.replace(b'/hierarchy.hier_opt',b'/hierarchy.hier',1),'Variant differs beyond one mode0 input literal')
    library=ROOT/'repos/gaussian-hierarchy-large-gs/build-local/Release/GaussianHierarchy.lib'
    require(sha(library)==variant['native_library_sha256'],'Variant linked native-library artifact changed')
    source_contracts=[]
    for row in config['source_records']:
        if Path(row['path']).name in ['mainHierarchyMerger.cpp','hierarchy_explicit_loader.cpp','hierarchy_loader.cpp','hierarchy_writer.cpp','writer.cpp','common.h']:
            require(sha(row['path'])==row['sha256'],'Official Merger source contract changed');source_contracts.append(row)
    require(len({Path(r['path']).name for r in source_contracts})==6,'Incomplete frozen ownership/storage source records')
    return config,binding,variant,source_contracts
def creator_edges(config,binding,fresh_trees):
    all4=read(ALL4);require(all4['status']=='completed_actual_all_four006_Creator_anchor_source_exposure_CPU_ledger_only','Actual all4 Creator audit missing')
    manifest=read(INPUT/'manifest.json');require(sha(INPUT/'manifest.json')==config['input_manifest_sha256'],'Actual input manifest changed')
    rows=[]
    for index,c in enumerate(ORDER):
        original=binding['ordered_chunks'][index];record=all4['records'][index];folder=LOCK/'trained_chunks'/c
        rp=folder/'creator-outcome.json';receipt=read(rp);tree=folder/'hierarchy.hier'
        require(record['chunk']==original['chunk']==c and sha(rp)==record['protected_sources'][str(rp)],'Actual Creator receipt changed since independently measured arrays')
        require(receipt['status']=='completed' and receipt['exit_code']==receipt['actual_child_poll_exit_code']==0 and receipt['source_kind']=='unoptimized','Creator not actual terminal-completed')
        require(receipt['training_origin']==original and receipt['training_source_kind']==original['training_source_kind'] and receipt['training_lineage_binding_sha256']==PIN[BINDING] and receipt['configuration_sha256']==PIN[LOCK/'configuration.json'],'Actual Creator origin/config differs')
        require(receipt['hierarchy_sha256']==record['actual_tree_sha256'] and receipt['hierarchy']==record['actual_hierarchy'] and receipt['hierarchy']['unowned_tail']==0,'Creator actual tree family/tail differs')
        require(sha(folder/'exposure.json')==original['exposure']['sha256']==record['actual_original_and_Creator_exposure_sha256'],'Creator original frozen exposure bytes changed')
        if fresh_trees:require(sha(tree)==receipt['hierarchy_sha256'],'Actual Merger source tree bytes changed')
        rows.append(dict(chunk=c,path=str(tree),sha256=receipt['hierarchy_sha256'],source_kind='unoptimized',source_stage_receipt_sha256=sha(rp),training_source_kind=original['training_source_kind'],training_lineage_binding_sha256=PIN[BINDING],summary=receipt['hierarchy']))
    return manifest,rows
def memory_recipe():
    # Execute only the pinned pure AST function, never import renderer/core deps.
    tree=ast.parse(CORE.read_text(encoding='utf-8'));f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='memory_plan')
    scope={'require':require};exec(compile(ast.Module(body=[f],type_ignores=[]),'pinned-core-memory-plan-AST','exec'),scope)
    return scope['memory_plan'],hashlib.sha256(ast.dump(f,include_attributes=False).encode()).hexdigest()
def input_and_leaf_gate_plan():
    with np.load(CANONICAL,allow_pickle=False) as data:
        names=data['names'].astype(str).tolist();require(len(names)==len(set(names))==117,'Canonical heldout count/order invalid')
        require('pass1_1100.jpg' in names,'Fixed geometric reference view missing')
        index=names.index('pass1_1100.jpg')
        # This is a plan, not a rendered City equivalence result.
        return dict(canonical_npz_path=str(CANONICAL),canonical_npz_sha256=PIN[CANONICAL],ordered_camera_count=117,
            settings=['leaves','tau1','tau3','tau5','tau6','tau15'],paired_frame_count=1404,
            compact_leaf_pixel_gate=dict(status='planned_not_executed_not_enabled',camera_name=names[index],canonical_camera_index=index,
              contract='Actual same merged City tree; author CPU leaf traversal order, one owncoarse100k sky appended once; full-tree explicit leaves t1/kids1 vs compact directall through unchanged official renderer, same camera/K/dtype/mask/exposureFalse. Float RGB bitwise equality and source hashes required; preserve depth ties/order.',
              safety='Only root explicit GPU scheduling plus fresh safe-memory gates may run. If full-tree explicit-leaf path cannot pass conservative known-tensor/native-headroom gate, stop and retain limitation; do not use a reduced model or Truck proof as City full-model pixel proof.',
              scope='One predetermined actual-model/camera gate is code-path evidence, not 117-image quality completion or streaming proof.'),
            metric_and_exposure_scope='No metrics computed here. Later immutable260 renderer and672 metric core, same117/masks, heldout exposureFalse; mixed observed timing differs from old formal.')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','bind'],required=True);p.add_argument('--run',required=True);a=p.parse_args()
    require(a.run.replace('-','').isalnum() and len(a.run)<=110,'Unsafe output namespace')
    output=OUT/a.run;output.mkdir(exist_ok=False);shutil.copyfile(__file__,output/'source-snapshot.py')
    began=time.perf_counter();outcome=dict(status='running_independent_unoptimized_Merger_CPU_'+a.mode,CUDA_called=False,Torch_imported=False,actual_merged_model_bound=False,actual_1404_complete=False)
    save(output/'outcome.json',outcome)
    try:
        receiptpath=LOCK/'merge-unoptimized-outcome.json';model=LOCK/'merged-unoptimized.hier'
        if a.mode=='bind':
            require(receiptpath.is_file() and model.is_file(),'Actual006 unoptimized Merger output/receipt absent; refuse before any model decoding')
            early=read(receiptpath);require(early['status']=='completed' and early['exit_code']==early['actual_child_poll_exit_code']==0,'Actual unoptimized Merger not terminal-completed; refuse CPU binding')
        config,binding,variant,contracts=exact_sources();manifest,edges=creator_edges(config,binding,a.mode=='bind');memory,ast_sha=memory_recipe()
        protocol=input_and_leaf_gate_plan();coarse=Path(binding['coarse']['model']['path']);sky=100000
        require(int((coarse.parent/'pc_info.txt').read_text())==sky,'Own coarse sky count changed')
        hardware_source=OUT/'small-city-local-coarse-003-flat-heldout-001/resident-memory.json';hardware=read(hardware_source)['whole_device']['total_bytes']
        common=dict(source_sha256=sha(__file__),configuration_sha256=PIN[LOCK/'configuration.json'],training_lineage_binding_path=str(BINDING),training_lineage_binding_sha256=PIN[BINDING],formal_observation_reclassification=False,
            sealed_adapter_source_sha256=PIN[DRIVER/'evaluate_local_city_hierarchy_006.py'],sealed_posterior_source_sha256=PIN[OUT/'city-hierarchy-posterior-auditor-CPU-006/audit_city_1404_006.py'],sealed_native_workflow_lock_sha256=PIN[DRIVER/'native-workflow-lock.json'],
            source_contract_records=contracts,filename_only_variant_manifest_sha256=sha(VARIANT),filename_only_variant_source_exact=True,creator_edges=edges,protocol=protocol,
            shared_sky=dict(owncoarse_path=str(coarse),sha256=binding['coarse']['model']['sha256'],append_during_render_exactly_once=sky,appended_in_this_CPU_audit=0,coarse_field_bit_evidence_report_path=str(ALL4),coarse_field_bit_evidence_sha256=PIN[ALL4]),
            implementation='Mode0 getWeight uses full3D Euclidean distance to own/nearest other center,1 below0.95d,0 above1.05d,linear within5% band. Retained Gaussian activated opacity multiplied by weight. Zero-valid local nodes return children; root mean reset to own center and four child summary endpoints duplicated at new root. Not alpha-conserving deduplication; no independent full float/FMA ownership replay claimed.',
            memory_formula_core_sha256=PIN[CORE],memory_formula_AST_sha256=ast_sha,prior_actual_hardware_total_bytes=hardware,hardware_reference_path=str(hardware_source),hardware_reference_sha256=sha(hardware_source),
            hardware_limit='Prior synchronized total-capacity observation only, not current free memory, measured OOM or whole-device/native peak/fit guarantee.',CPU_time_seconds=time.perf_counter()-began,CUDA_called=False,Torch_imported=False,formal_posts_or_paired_Merger_or1404_complete=False,units='Published SfM aligned units, not GT or metres')
        if a.mode=='prepare':
            # Output root duplicates one endpoint per chunk, with one new root node.
            # Counts here are source-derived upper bounds, not actual merged counts.
            g=sum(e['summary']['gaussian_count'] for e in edges)+4;n=sum(e['summary']['node_count'] for e in edges)+1;l=sum(e['summary']['leaf_gaussian_count'] for e in edges)
            upper=memory(g+sky,n,l,g,sky,117)
            report=dict(common,status='CPU_prepared_actual_four_Creator_sources_no_merged_model_binding',actual_merged_model_bound=False,actual_merged_counts=None,
                source_derived_output_upper_counts=dict(G=g,nodes=n,leaves=l,scope='All source storedGaussians+4 duplicate roots and source nodes+1 root; ownership pruning may reduce. No actual merged count assertion.'),source_derived_memory_upper_accounting=upper)
        else:
            receipt=read(receiptpath);expected=[str(Path(variant['executable']['path'])),str(LOCK/'trained_chunks'),'0',str(INPUT/'chunks'),str(model)]+ORDER
            require(receipt['command_argv']==expected and receipt['source_kind']=='unoptimized' and receipt['configuration_sha256']==common['configuration_sha256'] and receipt['training_lineage_binding_sha256']==PIN[BINDING],'Actual merger argv/source kind/config differs')
            require(receipt['chunk_order']==ORDER and receipt['chunk_centers']=={c['name']:c['center'] for c in manifest['chunks']} and receipt['ownership']==config['merge_ownership'],'Actual Merger order/centers/ownership differs')
            require(receipt['input_filename_only_variant']==variant and receipt['merger_executable_sha256']==receipt['executable_sha256']==receipt['executable_after_sha256']==variant['executable']['sha256'],'Actual Merger executable/variant identity differs')
            require(receipt['input_hierarchies']==[{k:v for k,v in r.items() if k!='summary'} for r in edges],'Actual original-Creator dependencies differ; no post/pilot substitution')
            require(receipt['skybox_in_tree']==0 and receipt['shared_background_skybox_count']==sky and Path(receipt['shared_background_skybox']).resolve()==coarse.parent.resolve(),'Actual merged tail/coarse source differs')
            require(receipt.get('actual_child_pid',0)>0 and receipt['actual_child_birth']['creation_FILETIME_100ns']>0,'Actual Merger PID/birth evidence absent')
            protected={str(p):sha(p) for p in [model,receiptpath,coarse]};require(protected[str(model)]==receipt['hierarchy_sha256'] and protected[str(coarse)]==common['shared_sky']['sha256'],'Actual merged/owncoarse bytes changed')
            codec=load('merge006_bound_codec',CODEC);ledger=load('merge006_vector_readonly',LEDGER);h,numeric=ledger.mapped(model,codec,0)
            require(h.summary()==receipt['hierarchy'],'Actual independently decoded merged summary differs')
            require(h.nodes[0,3]==0 and h.nodes[0,4]==4 and h.nodes[0,6]==4,'Actual merged root local/child layout differs from mode0 source')
            centers=np.asarray([c['center'] for c in manifest['chunks']],np.float32)
            require(np.array_equal(h.positions[h.nodes[0,2]:h.nodes[0,2]+4],centers),'Actual root endpoint positions differ from ordered input centers')
            count=h.nodes[:,3].astype(np.int64);positive=np.flatnonzero(count);order=positive[np.argsort(h.nodes[positive,2],kind='stable')];count=count[order];starts=h.nodes[order,2].astype(np.int64)
            l=int(count.sum());offset=np.cumsum(count)-count;leafids=(np.repeat(starts-offset,count)+np.arange(l,dtype=np.int64)).astype('<i4')
            require(np.all(h.nodes[order,0]==0) and np.all(h.nodes[order,6]==0) and (len(leafids)<2 or np.all(leafids[1:]>leafids[:-1])),'Actual source leaf ranges/depth/order invalid')
            np.savez_compressed(output/'actual-leaf-storage-ids.npz',leaf_storage_ids=leafids)
            g=len(h.positions);n=len(h.nodes);resident_leaf=memory(g+sky,n,l,l,sky,117);worst_cut=memory(g+sky,n,l,g,sky,117)
            compact=dict(render_input_G=l+sky,known_fixed_resident_bytes=268*(l+sky)+117*(3*16+3)*4,conservative_known_tensor_peak_upper_bytes=292*(l+sky)+117*(3*16+3)*4,no_GPU_tree_or_cut_buffers=True,native_workspace_upper_bound=None,scope='Planned distinct compact leaf-only input path, not enabled, not same LOD state and not streaming')
            require(all(sha(p)==d for p,d in protected.items()),'Read-only merged/coarse/receipt source bytes changed while audited')
            report=dict(common,status='completed_actual_unoptimized006_Merger_CPU_binding_and_memory_accounting_only',actual_merged_model_bound=True,actual_merged_model_path=str(model),actual_merged_model_sha256=sha(model),actual_merge_receipt_path=str(receiptpath),actual_merge_receipt_sha256=sha(receiptpath),actual_merged_counts=h.summary(),numeric_storage=numeric,
                actual_leaf_ids_path=str(output/'actual-leaf-storage-ids.npz'),actual_leaf_ids_sha256=sha(output/'actual-leaf-storage-ids.npz'),actual_leaf_ID_int32_le_sha256=hashlib.sha256(leafids.tobytes()).hexdigest(),actual_leaf_ID_scope='Sorted actual storage IDs; no author traversal or GPU cut/weight/pixel-equivalence claim',
                full_tree_explicit_leaf_memory_accounting=resident_leaf,full_tree_all_storage_cut_upper_accounting=worst_cut,compact_leaf_reference_accounting=compact,all_actual_bound_bytes_unchanged=True)
        require('torch' not in sys.modules,'CPU-only binder imported Torch')
        require(all(sha(p)==d for p,d in PIN.items()),'A sealed source changed during CPU audit')
        report['CPU_time_seconds']=time.perf_counter()-began;save(output/'report.json',report)
        outcome.update(status=report['status'],report_sha256=sha(output/'report.json'),actual_merged_model_bound=report['actual_merged_model_bound'],elapsed_seconds=time.perf_counter()-began);save(output/'outcome.json',outcome)
        print(json.dumps(dict(status=report['status'],report_path=str(output/'report.json'),report_sha256=sha(output/'report.json'),actual_merged_model_bound=report['actual_merged_model_bound'],CUDA_called=False,Torch_imported=False)),flush=True)
    except Exception as exc:
        outcome.update(status='failed_preserved_unoptimized006_Merger_CPU_binding',error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-began,Torch_imported='torch' in sys.modules);save(output/'outcome.json',outcome);raise
if __name__=='__main__':main()
