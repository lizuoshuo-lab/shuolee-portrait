"""Static independent peer review only; never imports audit/main or model arrays."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'outputs/large-scene-gs-001'
OUT=BASE/'independent-evaluation/small-city006-formalpost00-completion-template-peer-CPU-001'
OLD=ROOT/'scripts/large_scene_gs/audit_city006_formalpost00_completion_CPU_001.py'
NEW=ROOT/'scripts/large_scene_gs/audit_city006_formalpost00_completion_CPU_002.py'
BASELINE=ROOT/'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_006.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(OLD)=='1eee64f9a2e28aaba6b4ac6cf7e5d8b9e6476e0342ba4955cfaf1c6923b8c428'
assert sha(NEW)=='1fcd7c22871f92bf49de27ec0c21cfb1d4c32a38dc9518a79310f3e795bcec87'
assert sha(BASELINE)=='3db6b7ee81d0b7df592609509f8dd565e146179d399f4ca8fbee9baf65e85cf7'
protected={str(p):sha(p) for p in [OLD,NEW,BASELINE]}
old=ast.parse(OLD.read_text(encoding='utf-8'));new=ast.parse(NEW.read_text(encoding='utf-8'))
timeout_count=0;scope_count=0
for n in ast.walk(new):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess' and n.func.attr=='run':
        selected=[k for k in n.keywords if k.arg=='timeout']
        if selected:
            assert len(selected)==1 and isinstance(selected[0].value,ast.Constant) and selected[0].value.value==30
            n.keywords=[k for k in n.keywords if k.arg!='timeout'];timeout_count+=1
    if isinstance(n,ast.Dict):
        idx=[i for i,k in enumerate(n.keys) if isinstance(k,ast.Constant) and k.value=='anchor_exact_scope']
        if idx:
            assert len(idx)==1;index=idx[0]
            assert isinstance(n.values[index],ast.Constant) and n.values[index].value=='Decoded float32 numerical values after native storage expansion are required exact; each field bit-equality is separately measured, not assumed. No normalization or unit-quaternion constraint.'
            del n.keys[index];del n.values[index];scope_count+=1
assert timeout_count==scope_count==1
assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
assert protected=={p:sha(p) for p in protected}
OUT.mkdir(exist_ok=False);shutil.copyfile(__file__,OUT/'review-source-snapshot.py')
diff=''.join(difflib.unified_diff(OLD.read_text(encoding='utf-8').splitlines(True),NEW.read_text(encoding='utf-8').splitlines(True),fromfile='immutable001',tofile='immutable002'))
(OUT/'actual-narrow-source.diff').write_text(diff,encoding='utf-8')
report={'status':'passed_static_first_formalpost00_completion_template_002_narrow_CPU_peer_review_only','sources_sha256':protected,
 'actual_narrow_diff_path':str(OUT/'actual-narrow-source.diff'),'actual_narrow_diff_sha256':sha(OUT/'actual-narrow-source.diff'),
 'normalized_full_AST_equals_001':True,'changed_behavior':'Only late privatePS7/CIM snapshot subprocess timeout30; exact same terminal PID+birth matching and other-owner reporting. Timeout rejects actual audit; it never releases ownership or retries GPU.',
 'changed_metadata':'Anchor equality gate remains decodedfloat32 numerical exact; separate measured rawbit bool retained for each field; no quaternion renormalization added.',
 'source_contract_review':['Nativecompleted exit/poll0, requestedI15000, source-derived14999 and supervision owned child/descendant terminal gate before large model arrays; shortpilot64 ineligible.','Exact originalCreator SHA, actual official post argv, immutable006/151 mixed origin/input/coarse/anchors/Sky/exposure edges, original15000 save/normalreturn log.','Finite stored fivegroups, positive exp(logscales), nonzero quaternion norm, unbounded activatedalpha>=0 without silent clamp; actual one100k tail and exactnodes/boxes.','Same actualanchor IDs and frozen helper halfwriter roundtrip; original→Creator→formalpost exposureSHA and no exposure step/depthloss.','Actual reviewed composite runtime entry/parser/seal supplements and Hidden launch/ownedFILETIME identities; query excludes only selected historical identities and does not demand other parent stages/GPU globally empty.'],
 'actual_formalpost00_arrays_or_terminal_audit_executed':False,'template_main_or_prepare_executed':False,'actual_post_or_Merger_or1404_completion_proved_here':False,
 'CUDA_called':False,'Torch_imported':'torch' in sys.modules,'model_arrays_read':False,'process_query_or_process_control_executed':False,'sources_unchanged':True}
assert not report['Torch_imported']
(OUT/'peer-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'report_path':str(OUT/'peer-review.json'),'report_sha256':sha(OUT/'peer-review.json')}))
