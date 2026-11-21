"""Read-only exact remaining Creator/formalpost commands and current memory observations."""
from pathlib import Path
import importlib.util,hashlib,json,sys,subprocess
from datetime import datetime,timezone
sys.dont_write_bytecode=True
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab');BASE=ROOT/'outputs/large-scene-gs-001'
LOCK=BASE/'official-hierarchy/small-city-hierarchy-006'
OUT=BASE/'recovery-audit/small-city-native006-remaining-stage-plan-CPU-001'
OUT.mkdir(parents=True,exist_ok=False)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
driver_source=ROOT/'scripts/large_scene_gs/run_small_city_hierarchy_006.py'
assert sha(driver_source)=='a44e97310e02b742ed3222d5e9d5e6777526d8485412662c0b46ced0ce1f50a6'
d=load('_remaining_native006_CPU_plan',driver_source)
config,plan=read(LOCK/'configuration.json'),read(LOCK/'plan.json')
manifest=read(BASE/'official-inputs/small-city-training-002/manifest.json')
binder,binding=d.fixed_sources();d.execution_source_gate()
assert config['formal_observation_reclassification'] is False and config['post_iterations']==15000
source_before={str(p):sha(p) for p in [driver_source,LOCK/'driver-source.py',LOCK/'configuration.json',LOCK/'plan.json',d.BINDING_PATH]}
model_refs=[binding['coarse']]+binding['ordered_chunks']
for item in model_refs:
    assert sha(item['model']['path'])==item['model']['sha256']
    assert sha(item['configuration']['path'])==item['configuration']['sha256']
    assert sha(item['training_receipt']['path'])==item['training_receipt']['sha256']
pilot=BASE/'official-post-preflight/small-city-post00-i64-r4-006'
pilot_outcome=read(pilot/'outcome.json')
pilot_audit=BASE/'recovery-audit/small-city-post006-i64-completion-CPU-004/completion-audit.json'
assert sha(pilot_audit)=='f3fa672b435932d32f6a0b144f21c0f639cf062f0d2776aee78140190f905526'
pilot_verified=read(pilot_audit)
common=[str(d.TRAINING_PYTHON),'-u',str(d.ENV_LAUNCHER),str(driver_source)]
fixed=['--run',d.FIXED_RUN,'--input',config['input'],'--coarse-run',config['coarse_run']]
for c in d.ORDER:fixed+=['--chunk-run',c+'='+config['chunk_runs'][c]]
coarse=Path(binding['coarse']['model']['path']).parent
stages=[]
for c,item in zip(d.ORDER,binding['ordered_chunks']):
    directory=LOCK/'trained_chunks'/c
    source_paths=[directory/n for n in ('creator-outcome.json','hierarchy.hier','anchors.bin','exposure.json')]
    creator_available=all(p.is_file() for p in source_paths)
    receipt=read(source_paths[0]) if source_paths[0].is_file() else None
    completed=bool(creator_available and receipt['status']=='completed' and receipt.get('exit_code')==0)
    native_argv=plan['creator_command_previews'][c]
    fullpost_argv=d.post_command(directory,binder.INPUT/'chunks'/c,coarse,manifest,15000)
    wrapper_creator=common+['--stage','creator']+fixed+['--chunk',c,'--execute']
    wrapper_post=common+['--stage','post']+fixed+['--chunk',c,'--execute']
    stages.append({'chunk':c,'training_origin_kind':item['training_source_kind'],
        'observation_reclassified_formal':False,'training_source_model_sha256':item['model']['sha256'],
        'actual_training_gaussian_count':item['model']['gaussian_count'],
        'nonSky_candidate_record_upper_bound':item['model']['gaussian_count']-100000,
        'actual_Creator_available_at_CPU_plan':creator_available,'actual_Creator_complete_at_CPU_plan':completed,
        'actual_Creator_status_seen':receipt.get('status') if receipt else None,
        'Creator_wrapper_argv':wrapper_creator,'unchanged_native_Creator_child_argv':native_argv,
        'formal_post_wrapper_argv':wrapper_post,'unchanged_train_post_child_argv':fullpost_argv,
        'post_requested_iterations':15000,'post_source_derived_optimizer_calls':14999,
        'post_count_scope':'Unchanged unconditional official Adam.step for iterations1..14999; final15000 backward/save then return. No per-Gaussian nonzero-update instrumentation.',
        'post_initialization':'Start from this actual frozen unoptimized Creator hierarchy; never from pilot64 result.',
        'post_required_actual_inputs':[str(p) for p in source_paths],
        'post_output':str(directory/'hierarchy.hier_opt'),'eligible_source_kind':'post_optimized_after_actual_completed_full15000_receipt',
        'original_global_test_pose_archive_overlap':0})
gpu_query=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
gpu_lines=[]
for line in gpu_query.stdout.strip().splitlines():
    fields=[v.strip() for v in line.split(',')]
    gpu_lines.append({'name':fields[0],'memory_total_MiB':int(fields[1]),'memory_used_MiB':int(fields[2]),'memory_free_MiB':int(fields[3])})
ps="[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; $o=Get-CimInstance Win32_OperatingSystem; [pscustomobject]@{total_physical_KiB=$o.TotalVisibleMemorySize; free_physical_KiB=$o.FreePhysicalMemory; disk_free_bytes=(Get-PSDrive -Name C).Free} | ConvertTo-Json -Compress"
mem=json.loads(subprocess.run(['powershell.exe','-NoProfile','-Command',ps],capture_output=True,text=True,check=True).stdout)
query="Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'python|GaussianHierarchy'} | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Compress"
raw=subprocess.run(['powershell.exe','-NoProfile','-Command',query],capture_output=True,text=True,check=True).stdout.strip()
rows=json.loads(raw) if raw else [];rows=rows if isinstance(rows,list) else [rows]
parser=d.load_module('_remaining006_exact_windows_argv',d.PROCESS_HELPER).windows_argv
gpu_owners=d.gpu_process_roles(rows,parser)
assert mem['disk_free_bytes']>=80*1024**3
assert {p:sha(p) for p in source_before}==source_before and 'torch' not in sys.modules
report={'status':'CPU_exact_remaining_native006_stage_plan_no_execution','created_utc':datetime.now(timezone.utc).isoformat(),
        'frozen_source_before_after':source_before,'actual_training_lineage_kind':binding['training_lineage_kind'],
        'training_lineage_binding_sha256':d.BINDING_SHA,'formal_observation_reclassification':False,
        'suggested_serial_stage_order':['Creator0_1','Creator1_0','Creator1_1','fullpost0_0','fullpost0_1','fullpost1_0','fullpost1_1'],
        'stage_commands_and_dependencies':stages,'parent_single_stage_dispatch_required':True,'automatic_dispatch_from_this_plan':False,
        'formal_fullpost_shared_defaults':{'resolution':4,'pixels':[256,172],'budget':15000,'source_derived_steps':14999,
             'feature_lr':.0005,'opacity_lr':.01,'scaling_lr':.001,'dimensionless_LoD_limit_log_uniform':[.005,.1],
             'data_device':'cpu','depth_loss':False,'exposure_optimization':False,'frozen_pretrained_exposure_used':True,
             'anchors_locked':True,'single_shared_sky_appended_once':100000,'original_nodes_boxes_retained':True},
        'pilot64_physical_verified_receipt':{'path':str(pilot_audit),'sha256':sha(pilot_audit),'eligible_for_merge':False,
             'official_logged_allocated_peak_bytes':pilot_verified['official_logged_torch_cuda_peak_allocated_bytes'],
             'GPU_seconds':pilot_outcome['elapsed_seconds']},
        'fresh_hardware_observation':{'NVML_readonly_query':gpu_lines,'CPU_disk':mem,'actual_GPU_owner_processes':gpu_owners,
              'scope':'Point-in-time NVML memory/free and full exact Python/native role observation; no CUDA context or training initialized.'},
        'memory_plan':{'official_post_tensor_count_fixed_after_Creator':True,
             'pilot_peak_GiB':pilot_verified['official_logged_torch_cuda_peak_allocated_bytes']/1024**3,
             'formalpost_budget_changes_iteration_count_not_densification':True,
             'no_fullrun_fit_claim_from64':'Different views/random LoD selections may increase renderer workspace peak; measured64 allocated peak is not total/reserved or worst-case15000.',
             'pre_each_stage':'Fresh no-active-lock/source/data/model/prior-real-receipt/process check; NVML free must be reviewed against latest actual allocated peak; keep80GiB disk reserve.',
             'failure_policy':'Preserve real failure/handle/source/model; do not auto-retry, lower resolution/budget, edit tensors or reuse pilot64 as formal.'},
        'paired_Merger_future_gate':{'unoptimized':'All4 actual Creator receipts source_kindunoptimized with same006 config/lineage/source/currentSHA.',
             'post_optimized':'All4 actual full15000 receipts source_kindpost_optimized with14999 source-derived counter; no64/preflight/author substitute.',
             'order':list(d.ORDER),'ownership':'Official mode0, identical original centers,5% opacity falloff; not alpha-conserving deduplication.',
             'background':'Merged hierarchy has zeroSky; later rendering adds one common100k coarseSky.'},
        'GPU_started':False,'CUDA_context_initialized':False,'Creator_or_post_or_Merger_launched':False,
        'old_sources_configs_failures_or_observed_origin_changed':False}
(OUT/'plan.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'plan':str(OUT/'plan.json'),'plan_sha256':sha(OUT/'plan.json'),
    'current_NVML':gpu_lines,'current_GPU_owner_count':len(gpu_owners),'disk_free_GiB':mem['disk_free_bytes']/1024**3}))
