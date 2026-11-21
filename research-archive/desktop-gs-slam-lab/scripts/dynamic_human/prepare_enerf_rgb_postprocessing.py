from pathlib import Path
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human'
for source,dest,replaces in [
 ('compare_enerf_forearm_heldout.py','compare_enerf_rgb_heldout.py',{'dynamic-human-enerf-forearm-matte-repair-072':'dynamic-human-enerf-rgb-pose-correction-074','Local matte repair':'Corrected RGB pose'}),
 ('audit_trained_enerf_forearm.py','audit_trained_enerf_rgb.py',{'dynamic-human-enerf-forearm-matte-repair-072':'dynamic-human-enerf-rgb-pose-correction-074','dynamic-human-enerf-head-motion-066':'dynamic-human-enerf-rgb-pose-correction-074/head'}),
]:
 s=(S/source).read_text()
 for a,b in replaces.items():assert a in s;s=s.replace(a,b)
 assert not (S/dest).exists();(S/dest).write_text(s)
print('Native comparison and independent trained-graph audit prepared.')
