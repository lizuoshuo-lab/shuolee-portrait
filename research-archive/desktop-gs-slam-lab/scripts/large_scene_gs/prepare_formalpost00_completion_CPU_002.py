"""Preserve prepared001; add bounded30s read-only CIM query and exact-value scope in002."""
from pathlib import Path
import hashlib
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
old=ROOT/'scripts/large_scene_gs/audit_city006_formalpost00_completion_CPU_001.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='1eee64f9a2e28aaba6b4ac6cf7e5d8b9e6476e0342ba4955cfaf1c6923b8c428'
s=old.read_text(encoding='utf-8')
source="env=baseline.PW_ENV,capture_output=True,text=True,encoding='utf-8',errors='strict',check=True)"
assert source in s;s=s.replace(source,"env=baseline.PW_ENV,capture_output=True,text=True,encoding='utf-8',errors='strict',check=True,timeout=30)")
s=s.replace("'actual_A_raw5fields':anchor_fields,", "'actual_A_raw5fields':anchor_fields,'anchor_exact_scope':'Decoded float32 numerical values after native storage expansion are required exact; each field bit-equality is separately measured, not assumed. No normalization or unit-quaternion constraint.',")
target=ROOT/'scripts/large_scene_gs/audit_city006_formalpost00_completion_CPU_002.py'
assert not target.exists();compile(s,str(target),'exec');target.write_text(s,encoding='utf-8')
print(target);print(hashlib.sha256(target.read_bytes()).hexdigest())
