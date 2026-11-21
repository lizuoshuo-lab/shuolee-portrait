"""Validate independent006 pilot configuration and actual absent Creator rejection; no GPU."""
from pathlib import Path
import hashlib
import importlib.util
import sys
import json
from datetime import datetime,timezone
sys.dont_write_bytecode=True
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab');BASE=ROOT/'outputs/large-scene-gs-001'
PILOT=BASE/'official-post-preflight/small-city-post00-i64-r4-006'
OUT=PILOT/'cpu-validation-001';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
source=ROOT/'scripts/large_scene_gs/run_small_city_post_preflight_006.py'
spec=importlib.util.spec_from_file_location('_post006_CPU001',source);p=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=p;spec.loader.exec_module(p)
driver,codec,helper,training,config,manifest=p.modules()
static=read(PILOT/'configuration.json');plan=read(PILOT/'plan.json')
assert sha(source)==sha(PILOT/'wrapper-source.py')==static['wrapper_source_sha256']
assert static['source_kind']=='post64_preflight' and static['eligible_for_formal_merge'] is False
assert static['requested_iterations']==64 and static['expected_optimizer_update_calls']==63
assert static['training_lineage_binding_sha256']==driver.BINDING_SHA
assert static['formal_observation_reclassification'] is False
assert plan['actual_Creator_available'] is False
assert plan['GPU_started'] is False
assert config['post_iterations']==15000 and config['expected_post_optimizer_calls']==14999
args=plan['effective_official_arguments']
assert args['iterations']==64 and args['resolution']==4 and args['eval'] and not args['train_test_exp']
tracked={str(f):sha(f) for f in [source,PILOT/'wrapper-source.py',PILOT/'configuration.json',PILOT/'plan.json',
          p.LOCK/'configuration.json',p.LOCK/'plan.json',p.LOCK/'driver-source.py',driver.BINDING_PATH]}
protected=read(PILOT/'protected-formal-files.json');p.require_unchanged(protected)
try:p.source_binding(driver,codec,helper,training,config,manifest)
except RuntimeError as e:
    assert 'not available' in str(e)
    rejection={'rejected':True,'reason':str(e),'scope':'actual_missing006_Creator00_tree_and_receipts','no_model_or_tree_substitute':True}
else:raise AssertionError('Absent realCreator accepted')
assert 'torch' not in sys.modules
assert {f:sha(f) for f in tracked}==tracked
p.require_unchanged(protected)
report={'status':'passed_post006_CPU_protocol_source_and_actual_missing_Creator_control',
        'UTC':datetime.now(timezone.utc).isoformat(),'sources_unchanged':tracked,'negative_control':rejection,
        'requested_iterations':64,'derived_optimizer_call_upper_bound':63,'formal15000_model_claimed':False,
        'source_kind':'post64_preflight','eligible_for_formal_merge':False,'lineage_binding_sha256':driver.BINDING_SHA,
        'formal_source_configuration_sha256':p.CONFIG_SHA,'wrapper_sha256':sha(source),
        'official_post_command':plan['official_post_command_preview'],'Torch_imported':False,'CUDA_initialized':False,
        'GPU_started':False,'Creator_executed':False,'post_executed':False,'protected_formal_files_unchanged':True}
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'report':str(OUT/'verification.json'),'report_sha256':sha(OUT/'verification.json')}))
