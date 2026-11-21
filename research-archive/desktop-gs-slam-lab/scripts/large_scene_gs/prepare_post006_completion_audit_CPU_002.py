"""Preserve001's path-only audit failure and prepare002 with copied-path-aware provenance."""
from pathlib import Path
import hashlib,json
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
original=ROOT/'scripts/large_scene_gs/audit_small_city_post006_completion_CPU_001.py'
text=original.read_text(encoding='utf-8')
failure=ROOT/'outputs/large-scene-gs-001/recovery-audit/small-city-post006-i64-completion-CPU-001/failure.json'
assert not failure.exists()
failure.write_text(json.dumps({'status':'independent_CPU_audit_failed_preserved','actual_exit_code':1,
    'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
    'error':'Decoded actual hierarchy summary differs','boundary':'Source Creator hierarchy was copied to pilot; path field is intentionally different.001 compared full summary including path.',
    'model_numeric_failure_established':False,'original_model_or_receipt_modified':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=text.replace('small-city-post006-i64-completion-CPU-001','small-city-post006-i64-completion-CPU-002')
old="require(pre.summary()==creator['hierarchy'] and post.summary()==stage['hierarchy'],'Decoded actual hierarchy summary differs')"
new="""pre_summary=pre.summary(); source_summary=dict(creator['hierarchy'])
require(pre_summary['path']==str(directory/'hierarchy.hier') and source_summary['path']==str(original/'hierarchy.hier'),'Actual copied hierarchy/source path differs')
require({k:v for k,v in pre_summary.items() if k!='path'}=={k:v for k,v in source_summary.items() if k!='path'} and post.summary()==stage['hierarchy'],'Decoded actual hierarchy fields differ beyond the intentionally copied path')"""
assert old in text;text=text.replace(old,new)
target=ROOT/'scripts/large_scene_gs/audit_small_city_post006_completion_CPU_002.py'
assert not target.exists();compile(text,str(target),'exec');target.write_text(text,encoding='utf-8')
print(target)
