"""Actual006 CPU controls, no native/CUDA execution or model/receipt changes."""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
from datetime import datetime, timezone
sys.dont_write_bytecode = True
ROOT = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE = ROOT / 'outputs/large-scene-gs-001'
LOCK = BASE / 'official-hierarchy/small-city-hierarchy-006'
OUT = LOCK / 'cpu-validation-001'
OUT.mkdir(exist_ok=False)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
source = ROOT / 'scripts/large_scene_gs/run_small_city_hierarchy_006.py'
spec = importlib.util.spec_from_file_location('_actual006_verification', source)
d = importlib.util.module_from_spec(spec); sys.modules[spec.name] = d; spec.loader.exec_module(d)
config, plan = read(LOCK / 'configuration.json'), read(LOCK / 'plan.json')
manifest = read(BASE / 'official-inputs/small-city-training-002/manifest.json')
helper, codec, training, records = d.frozen_helpers(LOCK)
source_before = {str(p):sha(p) for p in [source, LOCK/'driver-source.py', LOCK/'configuration.json',
                                      LOCK/'plan.json', d.BINDING_PATH, d.BINDER_PATH, d.BINDER_SNAPSHOT]}
assert sha(source) == sha(LOCK/'driver-source.py') == config['driver_sha256']
assert config['training_lineage_binding_sha256'] == d.BINDING_SHA
assert config['formal_observation_reclassification'] is False
assert plan['dependencies']['all_five_actual_bindings']['all_five_saved_model_bindings_rechecked'] is True
d.execution_source_gate()
positive = d.lineage_gate(config, manifest, training, codec)
assert positive['all_five_saved_model_bindings_rechecked']
controls = []
def reject(name, fn):
    try: fn()
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        controls.append({'name':name,'rejected':True,'error':str(exc),'native_or_GPU_invoked':False}); return
    raise AssertionError('Unsafe gate did not reject: '+name)
for chunk in d.ORDER:
    reject('actual_missing_Creator_'+chunk, lambda c=chunk:d.creator_gate(c,config,manifest,training,codec,helper,plan['camera_archives']))
    reject('actual_missing_formalpost_'+chunk, lambda c=chunk:d.post_gate(c,config,manifest,training,codec,helper,plan['camera_archives']))
bad = dict(config); bad['training_lineage_binding_sha256']='0'*64
reject('wrong_actual_lineage_SHA',lambda:d.lineage_gate(bad,manifest,training,codec))
bad = dict(config); bad['training_lineage_kind']='four_legacy_formal_training'
reject('observation_relabeled_formal',lambda:d.lineage_gate(bad,manifest,training,codec))
bad = dict(config); bad['input_manifest_sha256']='0'*64
reject('wrong_input_manifest_SHA',lambda:d.lineage_gate(bad,manifest,training,codec))
assert 'torch' not in sys.modules
source_after={p:sha(p) for p in source_before}
assert source_after==source_before
binder,_=d.fixed_sources()
terminal=binder.fresh_terminal()
assert terminal['actual_gpu_stage_processes']==[]
report={'status':'passed_actual006_CPU_source_model_and_missing_stage_controls',
        'UTC':datetime.now(timezone.utc).isoformat(),'source_before':source_before,'source_after':source_after,
        'all_five_actual_model_bindings':positive,'controls':controls,'negative_control_count':len(controls),
        'creator00_command':plan['creator_command_previews']['0_0'], 'actual_GPU_process_observation':terminal,
        'Torch_imported':False,'CUDA_initialized':False,'GPU_started':False,'Creator_executed':False,
        'post15000_executed':False,'Merge_executed':False,'old_sources_changed':False}
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Compatibility audit index is CPU evidence only; it does not mint an execution receipt.
(LOCK/'cpu-binding-verification.json').write_text(json.dumps({'status':report['status'],
        'verification_path':str(OUT/'verification.json'),'verification_sha256':sha(OUT/'verification.json'),
        'driver_sha256':sha(source),'configuration_sha256':sha(LOCK/'configuration.json'),
        'negative_control_count':len(controls),'GPU_started':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'report':str(OUT/'verification.json'),
                 'report_sha256':sha(OUT/'verification.json'),'negative_controls':len(controls)}))
