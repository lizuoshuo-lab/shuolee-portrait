"""Independent Remaining chunk10/chunk11 I15000 selected physical audit. --prepare reads no model arrays.

Actual audit refuses running/partial receipts before imports/model reads. It never
starts/stops a GPU process, releases ownership, repeats all5 numerical audits,
or substitutes pilot64. A later parent-owned Merger may be live at audit time.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,re,sys,subprocess
from datetime import datetime,timezone
sys.dont_write_bytecode=True
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab');BASE=ROOT/'outputs/large-scene-gs-001'
LOCK=BASE/'official-hierarchy/small-city-hierarchy-006'
SPECS={
    '1_0':dict(index=2,short='10',creator_SHA='77e681acdcaa72fdf87fb7b286289176c451b32748af8f05153a82ab162e84d8',
               before_SHA='abd60ea621272d7a880c5ab38b752be8c973eac65def953c03e2dd844ef9142f',
               npz_SHA='13129a78e47be0cfdeb0fc1fad44c0a36d3f1ac2a767f7a3a3e9eeb4e8c7ba18',
               G=4194329,A=2850201,L=2850099,A_minus_L=102,origin_kind='legacy_local_official_training'),
    '1_1':dict(index=3,short='11',creator_SHA='f86cad9937dd451ea2c48cedfbb1b1f2032606b527ad72d98fdc2276a6af661c',
               before_SHA='3c75cc1645f2082bf6d79628e0ebed5cc3324c162d67638b560c8c1b61bdc297',
               npz_SHA='6d7a281bf2ce6cb65eb1a6842c33661435878318e147961744ba5bb503f970bb',
               G=4268945,A=2939552,L=2939146,A_minus_L=406,origin_kind='observed_local_official_training')
}
CHUNK=DIR=RUN=SUP=LAUNCH=EXTRA=SPEC=None
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(v,m):
    if not v:raise RuntimeError(m)
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def stored_node_box_bytes_equal(before,after):
    # The compressed writer stores the node-count header after124 bytes/G,
    # then20-byte nodes and16-byte boxes. Compare exactly those36*N bytes.
    require(before.compressed and after.compressed and len(before.nodes)==len(after.nodes),'Compressed node/box layout differs')
    remaining=36*len(before.nodes)
    with before.path.open('rb') as x,after.path.open('rb') as y:
        x.seek(4+124*len(before.positions)+4);y.seek(4+124*len(after.positions)+4)
        while remaining:
            size=min(8*1024*1024,remaining);a=x.read(size);b=y.read(size)
            require(len(a)==len(b)==size and a==b,'Original compressed node/box raw bytes changed')
            remaining-=size
    return True
def gate():
    r=read(DIR/'post-outcome.json');s=read(SUP/'supervisor-outcome.json')
    require(r.get('status')=='completed' and r.get('source_kind')=='post_optimized' and r.get('exit_code')==r.get('actual_child_poll_exit_code')==0,
            'Selected formal post for '+CHUNK+' not actual completed/exit0; no partial model audit')
    require(r.get('requested_iterations')==15000 and r.get('optimizer_update_calls')==14999 and r.get('optimizer_call_count_source_derived') is True,
            'Formal15000/source-derived14999 required; pilot64 is ineligible')
    require(s.get('status')=='completed_one_actual_stage_only' and s.get('child_exit_code')==0 and s.get('child_handle_terminal') and s.get('GPU_stage_processes_released'),
            'Selectedpost supervision actual owned child/descendants not terminal')
    return r,s
def main():
    global CHUNK,DIR,RUN,SUP,LAUNCH,EXTRA,SPEC
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--chunk',choices=tuple(SPECS),required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--audit-run');a=p.parse_args()
    CHUNK=a.chunk;SPEC=SPECS[CHUNK];DIR=LOCK/'trained_chunks'/CHUNK
    RUN='city006-post'+SPEC['short']+'-i15000-r2-stage-001'
    SUP=BASE/'official-hierarchy-stage-supervision-006-revision-002'/RUN
    LAUNCH=BASE/'official-hierarchy-stage-supervision-launch-006-revision-002'/RUN/'launch-receipt.json'
    EXTRA=SUP.parent/'runtime-source-supplements'/RUN
    if a.prepare:
        current=read(DIR/'post-outcome.json') if (DIR/'post-outcome.json').is_file() else {'status':'not_yet_dispatched'}
        print(json.dumps({'status':'CPU_physical_audit_source_prepared_not_executed','audit_source':str(Path(__file__)),'audit_source_SHA':sha(__file__),
          'actual_current_stage_status':current['status'],'required_before_execution':'OriginalI15000 finalsave+actualexit0+nativecompleted+supervisorcompleted',
          'GPU_started':False,'arrays_read':False,'chunk':CHUNK,'original_Creator_tree_path':str(DIR/'hierarchy.hier'),'original_Creator_expected_SHA':SPEC['before_SHA'],
          'expected_training_origin_kind':SPEC['origin_kind'],'actual_expected_post_child_birth':current.get('actual_child_birth'),
          'expected_stage_run':RUN,'only_after_root_explicit_selected_CPU_authorization_and_true_same_stage_terminal':True}));return
    require(a.audit_run is not None and a.audit_run.replace('-','').isalnum() and len(a.audit_run)<=110,'Unique safe independent CPU audit namespace required')
    stage,sup=gate()  # Mandatory terminal gate before any large array read/output completion artifact.
    out=BASE/'recovery-audit'/a.audit_run;out.mkdir(parents=True,exist_ok=False)
    (out/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    native=ROOT/'scripts/large_scene_gs/run_small_city_hierarchy_006.py'
    require(sha(native)=='a44e97310e02b742ed3222d5e9d5e6777526d8485412662c0b46ced0ce1f50a6','Frozen native006 changed')
    d=load('_remaining_formalpost_CPU_independent_native',native)
    d.execution_source_gate();_,binding=d.fixed_sources()  # Source/receipt bytes only, not an all5 array scan.
    config,plan=read(LOCK/'configuration.json'),read(LOCK/'plan.json')
    manifest=read(BASE/'official-inputs/small-city-training-002/manifest.json')
    creator=read(DIR/'creator-outcome.json');before_input=read(DIR/'post-stage/post-input-audit.json')
    origin=binding['ordered_chunks'][SPEC['index']];coarse=Path(binding['coarse']['model']['path']).parent
    require(origin['chunk']==CHUNK and origin['training_source_kind']==SPEC['origin_kind'] and
            creator['training_origin']==origin and creator['formal_observation_reclassification'] is False,'Selected chunk/origin/observed identity differs')
    require((CHUNK!='1_1') or (origin['actual_Gaussian_Adam_calls']==29951 and origin['derived_Gaussian_call_upper_bound']==29999 and origin['formal_training_run'] is False),'Observed11 actual29951/derived29999/not-formal identity differs')
    ledger_path=BASE/'independent-evaluation/small-city006-all4-creator-anchor-source-exposure-CPU-001/report.json'
    require(sha(ledger_path)=='17be8b334c09f3dfbe91ba5e56d0e8107b674b379bc8977dc7447c12d572f774','Measured Creator A/L ledger changed')
    ledger_record=read(ledger_path)['records'][SPEC['index']]
    require(ledger_record['chunk']==CHUNK,'Measured Creator ledger index/chunk differs')
    classification=ledger_record['actual_anchor_classification']
    sets_path=Path(classification['actual_set_npz_path'])
    require(sha(sets_path)==classification['actual_set_npz_sha256']==SPEC['npz_SHA'],'Actual selected chunk A/L storage sets changed')
    tracked=[Path(__file__),ledger_path,sets_path,native,LOCK/'driver-source.py',LOCK/'configuration.json',LOCK/'plan.json',d.BINDING_PATH,
      DIR/'creator-outcome.json',DIR/'post-outcome.json',DIR/'hierarchy.hier',DIR/'hierarchy.hier_opt',DIR/'anchors.bin',DIR/'exposure.json',
      DIR/'post-stage/post-input-audit.json',DIR/'post-stage/post-freeze-audit.json',SUP/'supervisor-outcome.json',LAUNCH,
      EXTRA/'runtime-control-revision.json',EXTRA/'actual-runtime-entry-source.py',EXTRA/'actual-time-parser-source.py']
    protected_before={str(f):sha(f) for f in tracked}
    require(creator['status']=='completed' and creator['source_kind']=='unoptimized' and creator['exit_code']==0 and creator['actual_child_poll_exit_code']==0,'Actual original Creator incomplete')
    require(sha(DIR/'creator-outcome.json')==SPEC['creator_SHA'],'Selected original Creator receipt changed')
    require(sha(DIR/'hierarchy.hier')==creator['hierarchy_sha256']==before_input['hierarchy_sha256']==SPEC['before_SHA'],'Formalpost did not start from original Creator')
    require(stage['command_argv']==d.post_command(DIR,BASE/'official-inputs/small-city-training-002/chunks'/CHUNK,coarse,manifest,15000),'Actual official I15000 argv changed')
    require(stage['source_training_receipt']==creator['source_training_receipt']==read(origin['training_receipt']['path']) and stage['training_origin']==creator['training_origin']==origin,
            'Formalpost actual selected training receipt/origin differs')
    require(stage['input_model']==creator['input_model'] and stage['input_model']['sha256']==origin['model']['sha256'] and sha(origin['model']['path'])==origin['model']['sha256'],'Selected original model input identity differs')
    require(stage['training_lineage_binding_sha256']==d.BINDING_SHA and stage['configuration_sha256']==sha(LOCK/'configuration.json') and stage['formal_observation_reclassification'] is False,'Mixed origin/config source changed')
    require(stage['exposure_optimizer_updates']==0 and stage['depth_loss_present_in_unchanged_post'] is False,'Official post exposure/depth protocol changed')
    require(sha(DIR/'anchors.bin')==creator['anchor_sha256']==before_input['anchors_sha256'],'Original anchors changed')
    require(sha(coarse/'point_cloud.ply')==binding['coarse']['model']['sha256']==before_input['scaffold_sha256'],'Actual coarse Sky source changed')
    archive=plan['camera_archives'][CHUNK];require(creator['creator_train_camera_archive']==archive and archive['heldout_pose_count']==0,'Original train-only pose archive detached')
    for f in archive['files']:require(sha(f['path'])==f['sha256'],'Actual train-only archive bytes changed')
    text=(DIR/'post-stage/stage.log').read_text(encoding='utf-8',errors='replace')
    require('[ITER 15000] Saving Gaussians' in text and 'Training complete.' in text,'Actual full15000 finalsave/return log missing')
    final=sup['actual_final_stage']
    require(final['receipt_path']==str(DIR/'post-outcome.json') and final['model_path']==str(DIR/'hierarchy.hier_opt') and
            final['receipt_sha256']==sha(DIR/'post-outcome.json') and final['model_sha256']==stage['hierarchy_sha256'] and
            final['source_kind']=='post_optimized' and final['actual_receipt']==stage,'Selected actual native receipt/model/final handle lineage differs')
    require(sup['active_stage_lock']['stage']=='post' and sup['active_stage_lock']['chunk']==CHUNK and
            sup['active_stage_lock']['supervision_run']==RUN and sup['stage']=='post' and sup['chunk']==CHUNK,'Selected supervision run/stage/chunk differs')
    require(stage['hierarchy_sha256']==sha(DIR/'hierarchy.hier_opt'),'Saved formalpost model changed')
    helper,codec,training,_=d.frozen_helpers(LOCK)
    before=codec.read_hierarchy(DIR/'hierarchy.hier');after=codec.read_hierarchy(DIR/'hierarchy.hier_opt',expected_unowned_tail=100000)
    require(before.summary()==creator['hierarchy'] and after.summary()==stage['hierarchy'] and
            len(before.positions)==classification['G']==SPEC['G'] and classification['A_minus_L']['count']==SPEC['A_minus_L'],
            'Actual decoded selected hierarchy counts/layout differs')
    anchors=codec.read_anchors(DIR/'anchors.bin',len(before.positions));require(len(anchors)==creator['anchor_count'],'Actual A count differs')
    fields=('positions','rotations','log_scales','opacities','sh');physical={}
    for label,h in [('original_Creator',before),('formalpost15000',after)]:
        stats={'count':len(h.positions),'all5_fields_finite':True};above=0;lo=float('inf');hi=float('-inf')
        for b in range(0,len(h.positions),65536):
            for f in fields:require(np.isfinite(getattr(h,f)[b:b+65536]).all(),'Nonfinite '+label+'/'+f)
            with np.errstate(over='ignore',under='ignore'):scales=np.exp(np.asarray(h.log_scales[b:b+65536],np.float32))
            require(np.isfinite(scales).all() and (scales>0).all(),'Invalid finite positive expσ '+label)
            norms=np.linalg.norm(np.asarray(h.rotations[b:b+65536],np.float64),axis=1);require(np.isfinite(norms).all() and (norms>0).all(),'Zero/invalid qnorm '+label)
            k=np.asarray(h.opacities[b:b+65536],np.float32);require((k>=0).all(),'Negative stored absκ '+label)
            above+=int(np.count_nonzero(k>1));lo=min(lo,float(k.min()));hi=max(hi,float(k.max()))
        stats.update(exp_f32_scales_finite_positive=True,quaternion_norm_nonzero=True,stored_abs_kappa_finite_nonnegative=True,kappa_above1_count=above,kappa_min=lo,kappa_max=hi,
                     opacity_semantics='Hierarchical absκ activation is unbounded above; do not impose traditional sigmoid probability<=1.')
        physical[label]=stats
    require(len(after.positions)==len(before.positions)+100000 and after.unowned_tail==100000,'Single appended shared100kSky contract failed')
    require(np.array_equal(before.nodes,after.nodes) and np.array_equal(before.boxes.astype(np.float32),after.boxes),'Original stored nodes/boxes changed')
    node_box_raw_bytes_exact=stored_node_box_bytes_equal(before,after)
    anchor_fields={}
    for f in fields:
        x=np.ascontiguousarray(getattr(before,f)[anchors],dtype=np.float32);y=np.ascontiguousarray(getattr(after,f)[anchors],dtype=np.float32)
        require(np.array_equal(x,y),'Frozen A decoded raw field changed: '+f)
        anchor_fields[f]={'decoded_float32_values_exact':True,'float32_bits_equal':bool(np.array_equal(x.view(np.uint32),y.view(np.uint32))),
                          'max_abs_delta':float(np.max(np.abs(x-y))) if x.size else 0.0}
    with np.load(sets_path,allow_pickle=False) as sets:
        leaves=np.asarray(sets['leaves_sorted'],dtype='<i8'); actual_A=np.asarray(sets['anchors_sorted'],dtype='<i8')
    require(len(leaves)==classification['L']['count']==before.summary()['leaf_gaussian_count']==SPEC['L'] and
            hashlib.sha256(leaves.tobytes()).hexdigest()==classification['L']['sorted_int64_le_sha256'] and
            len(actual_A)==classification['A']['count']==len(anchors)==SPEC['A'] and
            hashlib.sha256(actual_A.tobytes()).hexdigest()==classification['A']['sorted_int64_le_sha256'] and
            np.array_equal(np.sort(anchors.astype(np.int64)),actual_A),'Actual selected chunk A/L ID values/counts detached from original anchors/tree')
    require(np.all(leaves[1:]>leaves[:-1]) and 0<=leaves[0] and leaves[-1]<len(before.positions) and
            np.all(np.isin(leaves,actual_A)),'Actual selected chunk leaves are not sorted original-tree IDs inside frozen A')
    leaf_fields={}
    for field in fields:
        equal_bits=True; maximum=0.0
        for b in range(0,len(leaves),65536):
            selected=leaves[b:min(len(leaves),b+65536)]
            x=np.ascontiguousarray(getattr(before,field)[selected],dtype=np.float32)
            y=np.ascontiguousarray(getattr(after,field)[selected],dtype=np.float32)
            require(np.array_equal(x,y),'Actual frozen leaf endpoint field changed: '+field)
            equal_bits=equal_bits and bool(np.array_equal(x.view(np.uint32),y.view(np.uint32)))
            maximum=max(maximum,float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))))
        leaf_fields[field]={'actual_leaf_rows':len(leaves),'decoded_float32_values_exact':True,'float32_bits_equal':equal_bits,'max_abs_delta':maximum}
    freeze=helper.post_freeze_audit(before,after,anchors,coarse,100000)
    require(freeze==stage['frozen_fields']==read(DIR/'post-stage/post-freeze-audit.json'),'Actual anchor/Skyhalf roundtrip receipt differs')
    exposure=d.exposure_audit(DIR/'exposure.json',origin['training_names'],manifest['test_names'])
    require(exposure['sha256']==origin['exposure']['sha256']==creator['copied_exposure_sha256']==stage['frozen_exposure']['sha256']==before_input['exposure']['sha256'],'Original→Creator→post exposure bytes edge changed')
    runtime=read(EXTRA/'runtime-control-revision.json');launch=read(LAUNCH)
    require(runtime==sup['runtime_control_revision']==sup['active_stage_lock']['runtime_control_revision']==sup['active_stage_lock']['supervisor_birth']['runtime_control_revision'],'Composite runtime source supplement differs')
    for key,path,copy in [('actual_runtime_entry_sha256',runtime['actual_runtime_entry_path'],EXTRA/'actual-runtime-entry-source.py'),('actual_parser_sha256',runtime['actual_parser_path'],EXTRA/'actual-time-parser-source.py')]:
        require(sha(path)==sha(copy)==runtime[key],'Runtime actual entry/parser/snapshot changed')
    require(sha(runtime['source_seal_path'])==runtime['source_seal_sha256']=='c3d54d7f447b1ee316ee9d56d3f92b85fdda6303bfcbaa6837f5bd832f43f9cc','Runtime reviewed seal changed')
    require(sup['exact_child_birth_argv_observed'] and not sup['identity_mismatch_observations'] and sup['active_stage_lock_released'] and sup['actual_final_stage']['receipt_sha256']==sha(DIR/'post-outcome.json'),'Actual owned terminal receipt/birth/argv/release unproven')
    require(launch['start_process_pid']==sup['supervisor_parent_pid'] and launch['supervisor_source_sha256']==runtime['actual_runtime_entry_sha256'],'Actual Hidden source/launcher parent differs')
    # Observe selected historical identities only. Another parent-owned Merger/post can legitimately be live.
    parser=load('_remaining_formalpost_CPU_zoned_parser',runtime['actual_parser_path']);baseline=load('_remaining_formalpost_CPU_PS7_runtime',ROOT/'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_006.py')
    identities={(launch['start_process_pid'],parser.creation_ms(launch['start_process_creation_utc'])),(sup['supervisor_pid'],sup['active_stage_lock']['supervisor_birth']['creation_Unix_ms']),
                (sup['child_pid'],sup['actual_child_birth']['creation_Unix_ms']),(stage['actual_child_pid'],stage['actual_child_birth']['creation_Unix_ms'])}
    for line in (SUP/'process-observations.jsonl').read_text(encoding='utf-8').splitlines():
        for child in json.loads(line).get('actual_observed_child_descendants',[]):identities.add((int(child['ProcessId']),parser.creation_ms(child['CreationDate'])))
    query="$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Depth 4 -Compress"
    raw=subprocess.run([str(baseline.PW),'-NoProfile','-NonInteractive','-Command',query],env=baseline.PW_ENV,capture_output=True,text=True,encoding='utf-8',errors='strict',check=True,timeout=30).stdout.strip()
    rows=json.loads(raw) if raw else [];rows=rows if isinstance(rows,list) else [rows]
    for row in rows:
        if int(row['ProcessId']) in {v[0] for v in identities}:require((int(row['ProcessId']),parser.creation_ms(row['CreationDate'])) not in identities,'Selected selectedpost historical process/descendant still live')
    argv=d.load_module('_remaining_formalpost_CPU_exact_windows_argv',d.PROCESS_HELPER).windows_argv
    others=d.gpu_process_roles([r for r in rows if r['Name'].lower().startswith(('python','gaussianhierarchy'))],argv)
    require({f:sha(f) for f in protected_before}==protected_before and 'torch' not in sys.modules,'Immutable source/model/receipt bytes changed during independent CPU audit')
    peaks=[int(x) for x in re.findall(r'peak memory:\s*(\d+)',text,re.I)]
    result={'status':'actual_selected15000_formalpost_physical_source_and_owned_terminal_CPU_audit_passed','UTC':datetime.now(timezone.utc).isoformat(),
      'chunk':CHUNK,'source_kind':'post_optimized','requested_iterations':15000,'optimizer_call_count_source_derived':14999,'optimizer_call_count_instrumented':False,
      'protected_source_and_output_before_after':protected_before,'original_hierarchy':before.summary(),'actual_full15000_hierarchy':after.summary(),
      'independent_physical_numeric':physical,'actual_anchor_count':len(anchors),'actual_A_raw5fields':anchor_fields,'actual_L_raw5fields':leaf_fields,'actual_leaf_count':len(leaves),'actual_A_minus_L_count':classification['A_minus_L']['count'],'anchor_exact_scope':'Decoded float32 numerical values after native storage expansion are required exact; each field bit-equality is separately measured, not assumed. No normalization or unit-quaternion constraint.','nodes_and_stored_boxes_exact':True,
      'node_and_box_compressed_raw_bytes_exact':node_box_raw_bytes_exact,
      'single_shared100kSky_halfwriter_roundtrip':freeze,'frozen_original_train_exposure':exposure,'training_lineage_binding_SHA':d.BINDING_SHA,
      'training_lineage_kind':binding['training_lineage_kind'],'observation_reclassified_formal':False,
      'mixed_origin_scope':'Same frozen151 binding and actual source/config/receipt bytes verified; only original selected chunk '+CHUNK+' /coarse/selectedpost arrays checked. Other training arrays not rescanned.',
      'runtime_source_supplement':runtime,'selected_actual_Popen_and_seen_descendant_births_absent':sorted(identities),
      'other_actual_parent_GPU_stage_owners_at_query':others,'global_GPU_free_claimed':False,
      'official_logged_peak_allocated_bytes':max(peaks) if peaks else None,'memory_scope':'Original official Torch allocated peak only, no reserved/total claim.',
      'pilot64_used_as_input':False,'all4_formalpost_or_Merger_or1404_complete_claimed':False,'GPU_or_process_or_lock_control_performed':False,
      'broad_all5_array_audit_repeated':False,'CUDA_initialized':False,'old64failures_or_archives_or_original_sources_modified':False}
    (out/'completion-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'report':str(out/'completion-audit.json'),'SHA':sha(out/'completion-audit.json'),'G':len(after.positions),'A':len(anchors)}))
if __name__=='__main__':main()
