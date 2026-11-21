"""Independent read-only post64 result/physical/source/terminal audit; never release locks."""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
import subprocess
import re
from datetime import datetime,timezone
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab');BASE=ROOT/'outputs/large-scene-gs-001'
LOCK=BASE/'official-hierarchy/small-city-hierarchy-006';PILOT=BASE/'official-post-preflight/small-city-post00-i64-r4-006'
SUP=BASE/'official-hierarchy-stage-supervision-006/city006-post00-i64-stage-001'
OUT=BASE/'recovery-audit/small-city-post006-i64-completion-CPU-002'
OUT.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def require(v,m):
    if not v:raise RuntimeError(m)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
source=ROOT/'scripts/large_scene_gs/run_small_city_post_preflight_006.py'
require(sha(source)=='7b8069644f74db81f61db9d897765db456514730ac772eea5adfa7176d27139f','Frozen post006 source differs')
p=load('_independent_post006_final_audit',source)
d,codec,helper,training,config,manifest=p.modules()
pilot_config,stage=read(PILOT/'configuration.json'),read(PILOT/'outcome.json')
supervisor=read(SUP/'supervisor-outcome.json')
creator=read(LOCK/'trained_chunks/0_0/creator-outcome.json')
binding=read(d.BINDING_PATH)
ref=binding['ordered_chunks'][0];coarse=Path(binding['coarse']['model']['path']).parent
tracked=[source,PILOT/'wrapper-source.py',PILOT/'configuration.json',PILOT/'plan.json',PILOT/'outcome.json',
         PILOT/'source-binding.json',PILOT/'post-freeze-audit.json',SUP/'supervisor-outcome.json',
         LOCK/'configuration.json',LOCK/'plan.json',LOCK/'driver-source.py',d.BINDING_PATH,
         LOCK/'trained_chunks/0_0/creator-outcome.json']
directory=PILOT/'creator-input';original=LOCK/'trained_chunks/0_0'
tracked += [directory/n for n in ('hierarchy.hier','anchors.bin','exposure.json','hierarchy.hier_opt')]
tracked += [original/n for n in ('hierarchy.hier','anchors.bin','exposure.json')]
before_hash={str(f):sha(f) for f in tracked}
p.require_unchanged(read(PILOT/'protected-formal-files.json'))
require(stage['status']=='completed_post64_preflight_only' and stage['exit_code']==stage['actual_child_poll_exit_code']==0,'Actual post64 return not completed')
require(stage['source_kind']=='post64_preflight' and stage['eligible_for_formal_merge'] is False and stage['formal15000_post_complete'] is False,'Pilot reclassified as formal')
require(stage['requested_iterations']==64 and stage['optimizer_update_calls']==63 and 'not optimizer instrumentation' in stage['update_count_scope'],'64/63 count semantics differ')
require(stage['training_lineage_binding_sha256']==d.BINDING_SHA and stage['formal_observation_reclassification'] is False and stage['formal_source_configuration_sha256']==p.CONFIG_SHA,'Mixed origin binding changed')
require(stage['command_argv']==d.post_command(directory,p.INPUT/'chunks/0_0',coarse,manifest,64),'Actual unchanged official post64 argv differs')
require(pilot_config['wrapper_source_sha256']==sha(source)==sha(PILOT/'wrapper-source.py'),'Actual pilot wrapper/snapshot differs')
require(creator['status']=='completed' and creator['exit_code']==0 and creator['source_kind']=='unoptimized' and creator['source_training_receipt']==read(ref['training_receipt']['path']),'Creator source receipt differs')
require(creator['training_lineage_binding_sha256']==d.BINDING_SHA and creator['input_model']['sha256']==ref['model']['sha256'],'Creator actual model binding differs')
require(creator['command_argv']==read(LOCK/'plan.json')['creator_command_previews']['0_0'],'Creator exact train-only argv differs')
archive=read(LOCK/'plan.json')['camera_archives']['0_0']
require(creator['creator_train_camera_archive']==archive and archive['heldout_pose_count']==0,'Creator included heldout poses')
for f in archive['files']:require(sha(f['path'])==f['sha256'],'Train-only camera archive changed')
for n,key in [('hierarchy.hier','hierarchy_sha256'),('anchors.bin','anchor_sha256'),('exposure.json','copied_exposure_sha256')]:
    require(sha(original/n)==sha(directory/n)==creator[key]==stage['copied_input_hashes'][n],'Copied Creator input differs: '+n)
require(sha(directory/'hierarchy.hier_opt')==stage['output_hierarchy_sha256'],'Actual saved post hierarchy hash changed')
require(supervisor['child_exit_code']==supervisor['actual_child_handle_poll_exit_code']==0 and supervisor['child_handle_terminal'] is True,'Actual supervision Popen exit evidence missing')
require(supervisor['error_type']=='ValueError' and 'Invalid isoformat' in supervisor['error'] and supervisor['active_stage_lock_released'] is False,'Original supervisor terminal failure context changed')
text=(directory/'post-stage/stage.log').read_text(encoding='utf-8',errors='replace')
require('[ITER 64] Saving Gaussians' in text and 'Training complete.' in text,'Actual final-save/normal-return log absent')
peak_values=[int(x) for x in re.findall(r'peak memory:\s*(\d+)',text,re.I)]
# This is a fresh independent entry, not an execution/audit-post rewrite of the original receipt.
pre=codec.read_hierarchy(directory/'hierarchy.hier')
post=codec.read_hierarchy(directory/'hierarchy.hier_opt',expected_unowned_tail=100000)
anchors=codec.read_anchors(directory/'anchors.bin',len(pre.positions))
pre_summary=pre.summary(); source_summary=dict(creator['hierarchy'])
require(pre_summary['path']==str(directory/'hierarchy.hier') and source_summary['path']==str(original/'hierarchy.hier'),'Actual copied hierarchy/source path differs')
require({k:v for k,v in pre_summary.items() if k!='path'}=={k:v for k,v in source_summary.items() if k!='path'} and post.summary()==stage['hierarchy'],'Decoded actual hierarchy fields differ beyond the intentionally copied path')
require(len(anchors)==creator['anchor_count']==2978564,'Actual anchor count differs')
fields=('positions','rotations','log_scales','opacities','sh');physical={}
for hname,h in [('original_Creator',pre),('pilot_post',post)]:
    count=len(h.positions);proof={'all_fields_finite':True,'count':count}
    for b in range(0,count,65536):
        for f in fields:require(np.isfinite(getattr(h,f)[b:b+65536]).all(),'Nonfinite '+hname+'/'+f)
        with np.errstate(over='ignore',under='ignore'):activated=np.exp(h.log_scales[b:b+65536].astype(np.float32))
        require(np.isfinite(activated).all() and (activated>0).all(),'Invalid exp(scale) '+hname)
        q=np.linalg.norm(h.rotations[b:b+65536].astype(np.float64),axis=1)
        require(np.isfinite(q).all() and (q>0).all(),'Invalid quaternion norm '+hname)
        alpha=h.opacities[b:b+65536];require(((alpha>=0)&(alpha<=1)).all(),'Activated alpha outside[0,1] '+hname)
    proof.update(exp_f32_scales_finite_positive=True,quaternion_norm_nonzero=True,activated_alpha_in_0_1=True)
    physical[hname]=proof
require(len(post.positions)==len(pre.positions)+100000 and post.unowned_tail==100000,'Expected one appended100k coarse sky')
require(np.array_equal(pre.nodes,post.nodes) and np.array_equal(pre.boxes.astype(np.float32),post.boxes),'Stored node/box topology changed')
anchor_proof={}
for f in fields:
    a=np.ascontiguousarray(getattr(pre,f)[anchors],dtype=np.float32);b=np.ascontiguousarray(getattr(post,f)[anchors],dtype=np.float32)
    require(np.array_equal(a,b),'Frozen anchor values changed: '+f)
    anchor_proof[f]={'all_decoded_values_equal':True,'float32_bits_equal':bool(np.array_equal(a.view(np.uint32),b.view(np.uint32))),
                     'max_abs_delta':float(np.max(np.abs(a-b))) if a.size else 0.0}
freeze=helper.post_freeze_audit(pre,post,anchors,coarse,100000)
require(freeze==stage['frozen_fields']==read(PILOT/'post-freeze-audit.json'),'Saved anchor/Skyhalf physical audit disagrees')
exposure=d.exposure_audit(directory/'exposure.json',ref['training_names'],manifest['test_names'])
require(exposure['sha256']==ref['exposure']['sha256']==stage['frozen_trained_exposure']['sha256'],'Frozen actual trained exposure changed')
require(exposure['matrix_count']==1467 and exposure['test_overlap']==0 and stage['depth_loss_present'] is False and stage['exposure_optimizer_updates']==0,'Post input/test/depth/exposure policy differs')
# Confirm the all5 binding's parameters/sources remain actual, without any dispatcher/lock mutation.
binder,_=d.fixed_sources();all5=[]
for item in [binding['coarse']]+binding['ordered_chunks']:
    actual=binder.bind_one(Path(item['training_directory']),item['stage'],item['chunk'],manifest,d,training,codec,
                           observed=item['training_source_kind']=='observed_local_official_training')
    require(actual==item,'Actual mixed training source/model/finite/Sky differs')
    all5.append({'stage':actual['stage'],'chunk':actual['chunk'],'training_source_kind':actual['training_source_kind'],
                 'model_sha256':actual['model']['sha256'],'raw_finite':True})
# Independent terminal observation; the active lock is recorded, never released here.
query="Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'python|GaussianHierarchy'} | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Compress"
raw=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',query],capture_output=True,text=True,check=True).stdout.strip()
rows=json.loads(raw) if raw else [];rows=rows if isinstance(rows,list) else [rows]
parser=d.load_module('_post006_CPU_terminal_args',d.PROCESS_HELPER).windows_argv
owners=d.gpu_process_roles(rows,parser)
terminal_pids={stage['wrapper_pid'],stage['child_pid'],supervisor['supervisor_pid'],supervisor['child_pid']}
require(not owners and not any(r['ProcessId'] in terminal_pids for r in rows),'Actual training/native/supervised process still live')
lock_state={'path':str(d.ACTIVE_STAGE),'exists_at_audit':d.ACTIVE_STAGE.exists(),
            'sha256_at_audit':sha(d.ACTIVE_STAGE) if d.ACTIVE_STAGE.exists() else None,'released_by_this_audit':False}
failed=BASE/'official-training/small-city-chunk11-i30000-r4-003/model/point_cloud/iteration_30000/point_cloud.ply'
require(sha(failed)=='7f3a8a7fa25032658ebc70f6a8f2cbabe464257316ea7cb9fa041ed98bce48fc','Original failed11 PLY changed')
after_hash={f:sha(f) for f in before_hash};require(after_hash==before_hash,'Frozen/source/output/failed supervisor files changed during CPU audit')
p.require_unchanged(read(PILOT/'protected-formal-files.json'))
require('torch' not in sys.modules,'CPU audit unexpectedly imported Torch')
report={'status':'actual_post64_pilot_physical_completion_CPU_verified_supervisor_failure_preserved',
        'audited_utc':datetime.now(timezone.utc).isoformat(),'pilot_source_kind':'post64_preflight','eligible_for_formal_merge':False,
        'requested_iterations':64,'optimizer_calls_source_derived':63,'optimizer_call_count_measured':False,
        'source_before':before_hash,'source_after':after_hash,'all5_actual_mixed_training_sources_rechecked':all5,
        'original_hierarchy':pre.summary(),'post_hierarchy':post.summary(),'independent_physical_numeric':physical,
        'anchor_count':len(anchors),'independent_anchor5fields':anchor_proof,'single_shared_sky_roundtrip':freeze,
        'frozen_train_exposure':exposure,'post_elapsed_seconds':stage['elapsed_seconds'],
        'official_logged_torch_cuda_peak_allocated_bytes':max(peak_values) if peak_values else None,
        'peak_memory_scope':'Original official logged allocated peak only; not reserved/total and64 does not prove full15000 fits.',
        'fresh_actual_process_snapshot':rows,'actual_gpu_owners':owners,'terminal_original_stage_PIDs_absent':sorted(terminal_pids),
        'preserved_supervisor_failure':{'path':str(SUP/'supervisor-outcome.json'),'sha256':sha(SUP/'supervisor-outcome.json'),
               'status':supervisor['status'],'error_type':supervisor['error_type'],'error':supervisor['error'],
               'actual_child_handle_poll_exit_code':0,'active_stage_lock_released_in_original_receipt':False},
        'active_lock_observation':lock_state,'no_reconciliation_or_lock_release_performed':True,'GPU_started':False,
        'CUDA_initialized':False,'original_failed11_PLY_sha256':sha(failed),'source_kind_observed_training_preserved':True}
write(OUT/'completion-audit.json',report)
write(OUT/'source-manifest.json',{'audit_source_sha256':sha(__file__),'output_sha256':sha(OUT/'completion-audit.json')})
print(json.dumps({'status':report['status'],'report':str(OUT/'completion-audit.json'),'report_sha256':sha(OUT/'completion-audit.json'),
                 'post_gaussians':len(post.positions),'anchors':len(anchors),'peak_allocated_bytes':report['official_logged_torch_cuda_peak_allocated_bytes']}))
