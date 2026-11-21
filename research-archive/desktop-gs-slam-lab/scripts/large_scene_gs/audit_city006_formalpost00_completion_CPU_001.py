"""Independent FIRST I15000 post00 physical audit. --prepare reads no model arrays.

Actual audit refuses running/partial receipts before imports/model reads. It never
starts/stops a GPU process, releases ownership, repeats all5 numerical audits,
or substitutes pilot64. A later parent-owned Merger may be live at audit time.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,re,sys,subprocess
from datetime import datetime,timezone
sys.dont_write_bytecode=True
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab');BASE=ROOT/'outputs/large-scene-gs-001'
LOCK=BASE/'official-hierarchy/small-city-hierarchy-006';DIR=LOCK/'trained_chunks/0_0'
RUN='city006-post00-i15000-r2-stage-001'
SUP=BASE/'official-hierarchy-stage-supervision-006-revision-002'/RUN
LAUNCH=BASE/'official-hierarchy-stage-supervision-launch-006-revision-002'/RUN/'launch-receipt.json'
EXTRA=SUP.parent/'runtime-source-supplements'/RUN
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
def gate():
    r=read(DIR/'post-outcome.json');s=read(SUP/'supervisor-outcome.json')
    require(r.get('status')=='completed' and r.get('source_kind')=='post_optimized' and r.get('exit_code')==r.get('actual_child_poll_exit_code')==0,
            'First formalpost00 not actual completed/exit0; no partial model audit')
    require(r.get('requested_iterations')==15000 and r.get('optimizer_update_calls')==14999 and r.get('optimizer_call_count_source_derived') is True,
            'Formal15000/source-derived14999 required; pilot64 is ineligible')
    require(s.get('status')=='completed_one_actual_stage_only' and s.get('child_exit_code')==0 and s.get('child_handle_terminal') and s.get('GPU_stage_processes_released'),
            'Firstpost supervision actual owned child/descendants not terminal')
    return r,s
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--audit-run');a=p.parse_args()
    if a.prepare:
        current=read(DIR/'post-outcome.json')
        print(json.dumps({'status':'CPU_physical_audit_source_prepared_not_executed','audit_source':str(Path(__file__)),'audit_source_SHA':sha(__file__),
          'actual_current_stage_status':current['status'],'required_before_execution':'OriginalI15000 finalsave+actualexit0+nativecompleted+supervisorcompleted',
          'GPU_started':False,'arrays_read':False,'actual_expected_post_child_birth':current['actual_child_birth']}));return
    require(a.audit_run is not None and a.audit_run.replace('-','').isalnum() and len(a.audit_run)<=110,'Unique safe independent CPU audit namespace required')
    stage,sup=gate()  # Mandatory terminal gate before any large array read/output completion artifact.
    out=BASE/'recovery-audit'/a.audit_run;out.mkdir(parents=True,exist_ok=False)
    (out/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    native=ROOT/'scripts/large_scene_gs/run_small_city_hierarchy_006.py'
    require(sha(native)=='a44e97310e02b742ed3222d5e9d5e6777526d8485412662c0b46ced0ce1f50a6','Frozen native006 changed')
    d=load('_formalpost00_CPU_independent_native',native)
    d.execution_source_gate();_,binding=d.fixed_sources()  # Source/receipt bytes only, not an all5 array scan.
    config,plan=read(LOCK/'configuration.json'),read(LOCK/'plan.json')
    manifest=read(BASE/'official-inputs/small-city-training-002/manifest.json')
    creator=read(DIR/'creator-outcome.json');before_input=read(DIR/'post-stage/post-input-audit.json')
    origin=binding['ordered_chunks'][0];coarse=Path(binding['coarse']['model']['path']).parent
    tracked=[native,LOCK/'driver-source.py',LOCK/'configuration.json',LOCK/'plan.json',d.BINDING_PATH,
      DIR/'creator-outcome.json',DIR/'post-outcome.json',DIR/'hierarchy.hier',DIR/'hierarchy.hier_opt',DIR/'anchors.bin',DIR/'exposure.json',
      DIR/'post-stage/post-input-audit.json',DIR/'post-stage/post-freeze-audit.json',SUP/'supervisor-outcome.json',LAUNCH,
      EXTRA/'runtime-control-revision.json',EXTRA/'actual-runtime-entry-source.py',EXTRA/'actual-time-parser-source.py']
    protected_before={str(f):sha(f) for f in tracked}
    require(creator['status']=='completed' and creator['source_kind']=='unoptimized' and creator['exit_code']==0 and creator['actual_child_poll_exit_code']==0,'Actual original Creator incomplete')
    require(sha(DIR/'creator-outcome.json')=='7920acf67814ab13c2525344defe8d20357e11538ae3e9900b1d8e9d5d4f9853','First original Creator receipt changed')
    require(sha(DIR/'hierarchy.hier')==creator['hierarchy_sha256']==before_input['hierarchy_sha256']=='9b3e8ad0901601410e7e73d902cdb448051258d582e41574f6b0fd98f7a30cd7','Formalpost did not start from original Creator')
    require(stage['command_argv']==d.post_command(DIR,BASE/'official-inputs/small-city-training-002/chunks/0_0',coarse,manifest,15000),'Actual official I15000 argv changed')
    require(stage['source_training_receipt']==creator['source_training_receipt']==read(origin['training_receipt']['path']) and stage['training_origin']==creator['training_origin']==origin,
            'Formalpost actual selected training receipt/origin differs')
    require(stage['input_model']==creator['input_model'] and stage['input_model']['sha256']==origin['model']['sha256'] and sha(origin['model']['path'])==origin['model']['sha256'],'Selected original model input identity differs')
    require(stage['training_lineage_binding_sha256']==d.BINDING_SHA and stage['configuration_sha256']==sha(LOCK/'configuration.json') and stage['formal_observation_reclassification'] is False,'Mixed origin/config source changed')
    require(stage['exposure_optimizer_updates']==0 and stage['depth_loss_present_in_unchanged_post'] is False,'Official post exposure/depth protocol changed')
    require(sha(DIR/'anchors.bin')==creator['anchor_sha256']==before_input['anchors_sha256'],'Original anchors changed')
    require(sha(coarse/'point_cloud.ply')==binding['coarse']['model']['sha256']==before_input['scaffold_sha256'],'Actual coarse Sky source changed')
    archive=plan['camera_archives']['0_0'];require(creator['creator_train_camera_archive']==archive and archive['heldout_pose_count']==0,'Original train-only pose archive detached')
    for f in archive['files']:require(sha(f['path'])==f['sha256'],'Actual train-only archive bytes changed')
    text=(DIR/'post-stage/stage.log').read_text(encoding='utf-8',errors='replace')
    require('[ITER 15000] Saving Gaussians' in text and 'Training complete.' in text,'Actual full15000 finalsave/return log missing')
    require(stage['hierarchy_sha256']==sha(DIR/'hierarchy.hier_opt'),'Saved formalpost model changed')
    helper,codec,training,_=d.frozen_helpers(LOCK)
    before=codec.read_hierarchy(DIR/'hierarchy.hier');after=codec.read_hierarchy(DIR/'hierarchy.hier_opt',expected_unowned_tail=100000)
    require(before.summary()==creator['hierarchy'] and after.summary()==stage['hierarchy'],'Actual decoded hierarchy counts/layout differs')
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
    anchor_fields={}
    for f in fields:
        x=np.ascontiguousarray(getattr(before,f)[anchors],dtype=np.float32);y=np.ascontiguousarray(getattr(after,f)[anchors],dtype=np.float32)
        require(np.array_equal(x,y),'Frozen A decoded raw field changed: '+f)
        anchor_fields[f]={'decoded_float32_values_exact':True,'float32_bits_equal':bool(np.array_equal(x.view(np.uint32),y.view(np.uint32))),
                          'max_abs_delta':float(np.max(np.abs(x-y))) if x.size else 0.0}
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
    parser=load('_formalpost00_CPU_zoned_parser',runtime['actual_parser_path']);baseline=load('_formalpost00_CPU_PS7_runtime',ROOT/'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_006.py')
    identities={(launch['start_process_pid'],parser.creation_ms(launch['start_process_creation_utc'])),(sup['supervisor_pid'],sup['active_stage_lock']['supervisor_birth']['creation_Unix_ms']),
                (sup['child_pid'],sup['actual_child_birth']['creation_Unix_ms']),(stage['actual_child_pid'],stage['actual_child_birth']['creation_Unix_ms'])}
    for line in (SUP/'process-observations.jsonl').read_text(encoding='utf-8').splitlines():
        for child in json.loads(line).get('actual_observed_child_descendants',[]):identities.add((int(child['ProcessId']),parser.creation_ms(child['CreationDate'])))
    query="$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Depth 4 -Compress"
    raw=subprocess.run([str(baseline.PW),'-NoProfile','-NonInteractive','-Command',query],env=baseline.PW_ENV,capture_output=True,text=True,encoding='utf-8',errors='strict',check=True).stdout.strip()
    rows=json.loads(raw) if raw else [];rows=rows if isinstance(rows,list) else [rows]
    for row in rows:
        if int(row['ProcessId']) in {v[0] for v in identities}:require((int(row['ProcessId']),parser.creation_ms(row['CreationDate'])) not in identities,'Selected firstpost historical process/descendant still live')
    argv=d.load_module('_formalpost00_CPU_exact_windows_argv',d.PROCESS_HELPER).windows_argv
    others=d.gpu_process_roles([r for r in rows if r['Name'].lower().startswith(('python','gaussianhierarchy'))],argv)
    require({f:sha(f) for f in protected_before}==protected_before and 'torch' not in sys.modules,'Immutable source/model/receipt bytes changed during independent CPU audit')
    peaks=[int(x) for x in re.findall(r'peak memory:\s*(\d+)',text,re.I)]
    result={'status':'actual_first15000_formalpost00_physical_source_and_owned_terminal_CPU_audit_passed','UTC':datetime.now(timezone.utc).isoformat(),
      'chunk':'0_0','source_kind':'post_optimized','requested_iterations':15000,'optimizer_call_count_source_derived':14999,'optimizer_call_count_instrumented':False,
      'protected_source_and_output_before_after':protected_before,'original_hierarchy':before.summary(),'actual_full15000_hierarchy':after.summary(),
      'independent_physical_numeric':physical,'actual_anchor_count':len(anchors),'actual_A_raw5fields':anchor_fields,'nodes_and_stored_boxes_exact':True,
      'single_shared100kSky_halfwriter_roundtrip':freeze,'frozen_original_train_exposure':exposure,'training_lineage_binding_SHA':d.BINDING_SHA,
      'training_lineage_kind':binding['training_lineage_kind'],'observation_reclassified_formal':False,
      'mixed_origin_scope':'Same frozen151 binding and actual source/config/receipt bytes verified; only originalchunk00/coarse/firstpost arrays checked. Other4 training arrays not rescanned.',
      'runtime_source_supplement':runtime,'selected_actual_Popen_and_seen_descendant_births_absent':sorted(identities),
      'other_actual_parent_GPU_stage_owners_at_query':others,'global_GPU_free_claimed':False,
      'official_logged_peak_allocated_bytes':max(peaks) if peaks else None,'memory_scope':'Original official Torch allocated peak only, no reserved/total claim.',
      'pilot64_used_as_input':False,'all4_formalpost_or_Merger_or1404_complete_claimed':False,'GPU_or_process_or_lock_control_performed':False,
      'broad_all5_array_audit_repeated':False,'CUDA_initialized':False,'old64failures_or_archives_or_original_sources_modified':False}
    (out/'completion-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'report':str(out/'completion-audit.json'),'SHA':sha(out/'completion-audit.json'),'G':len(after.positions),'A':len(anchors)}))
if __name__=='__main__':main()
