"""Bounded CPU functional check; no model completion or CUDA claim."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
OUT=D.parent/'small-city006-posebytes-v2-tail-window-CPU-001'
OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=D/'audit_city_hierarchy_lineage_006_posebytes.py'
new=D/'audit_city_hierarchy_lineage_006_posebytes_v2.py'
assert sha(old)=='0148a71ae3ccf9678a6824e06f83cf1cc73c909db3001695ed09a509736ed1a2'
assert sha(new)=='aa4d620778231f1b9c405d95b4c6a04cf2af734eab5a8941817a0cfa0b408bb0'
before_hash={str(p):sha(p) for p in [old,new,D/'audit_city_hierarchy_lineage_006.py']}
o=load('old_window',old);m=load('fixed_window',new)
g=65536+3;fields={'positions':(3,),'rotations':(4,),'log_scales':(3,),'opacities':(),'sh':(16,3)}
a=types.SimpleNamespace(**{k:np.zeros((g,)+s,np.float32) for k,s in fields.items()})
b=types.SimpleNamespace(**{k:np.concatenate([getattr(a,k),np.full((100000,)+s,7,np.float32)]) for k,s in fields.items()})
b.positions[g-1,2]=.25
leaf=np.ones(g,np.bool_)
try:o.leaf_changes(a,b,leaf)
except IndexError as e:negative={'rejected':True,'exception':type(e).__name__,'message':str(e)}
else:raise AssertionError('Old tail window unexpectedly passed')
full=m.leaf_changes(a,b,leaf)
assert full['fields']['positions']['changed_leaf_rows']==1
assert full['fields']['positions']['max_abs_decoded_difference']==.25
assert all(r['actual_leaf_rows']==g for r in full['fields'].values())
assert all(r['changed_leaf_rows']==0 for k,r in full['fields'].items() if k!='positions')
leaf[:]=False;leaf[-2:]=True
subset=m.leaf_changes(a,b,leaf)
assert all(r['actual_leaf_rows']==2 for r in subset['fields'].values())
assert subset['fields']['positions']['changed_leaf_rows']==1
assert before_hash=={p:sha(p) for p in before_hash}
result={'status':'passed_bounded_CPU_tail_window_functional_control_not_model_completion','fixture_G':g,'appended_sky_rows':100000,'old_version_negative_control':negative,'new_full_leaf_result':full,'new_two_tail_leaf_result':subset,'sources_sha256':before_hash,'CUDA_called':False,'Torch_imported':'torch' in sys.modules,'actual_four_formal_posts_complete':False,'actual_1404_complete':False}
assert not result['Torch_imported']
(OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'],'report_path':str(OUT/'report.json'),'report_sha256':sha(OUT/'report.json')}))
