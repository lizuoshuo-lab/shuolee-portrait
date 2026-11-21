"""Preserve003's native-ZIP/non-Git checkout mismatch and use the frozen official archive."""
from pathlib import Path
import hashlib,json
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
source=ROOT/'scripts/large_scene_gs/audit_small_city_post006_completion_CPU_003.py'
s=source.read_text(encoding='utf-8')
failure=ROOT/'outputs/large-scene-gs-001/recovery-audit/small-city-post006-i64-completion-CPU-003/failure.json'
assert not failure.exists()
failure.write_text(json.dumps({'status':'independent_CPU_audit_failed_preserved','actual_exit_code':1,
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'error':'git check_output exit128: native repo is an acquired official ZIP checkout with no .git',
    'physical_post_and_anchor_and_sky_and_terminal_checks_reached_before_source_contract_failure':True,
    'new004_source_contract':'Use frozen official native677c source ZIP SHA739cdaf... and actual ClusterMerger bytes, matching existing immutable build/source acquisition protocol.',
    'model_or_receipt_or_original_finite_gate_changed':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s=s.replace('small-city-post006-i64-completion-CPU-003','small-city-post006-i64-completion-CPU-004')
old="canonical=subprocess.check_output(['git','-C',str(d.NATIVE_REPO),'show',d.NATIVE_COMMIT+':ClusterMerger.cpp'])"
new="""import zipfile
native_archive=read(d.BUILD/'python-extensions-manifest.json')['source_audit']['hierarchy']
require(native_archive['commit']==d.NATIVE_COMMIT and native_archive['changes']==[] and sha(native_archive['archive'])=='739cdaf62c9325294f5c0a8ec495f234138f6d765a023589532edce846c0d241','Official native ZIP acquisition source differs')
with zipfile.ZipFile(native_archive['archive']) as z:
    candidates=[n for n in z.namelist() if n.endswith('/ClusterMerger.cpp')]
    require(len(candidates)==1,'Native ClusterMerger ZIP path ambiguous')
    canonical=z.read(candidates[0])"""
assert old in s;s=s.replace(old,new)
target=ROOT/'scripts/large_scene_gs/audit_small_city_post006_completion_CPU_004.py'
assert not target.exists();compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print(target)
