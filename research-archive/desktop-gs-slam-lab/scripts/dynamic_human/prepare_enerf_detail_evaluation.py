"""Prepare frozen evaluation of the detail-weight control; no training edits."""
from pathlib import Path
import shutil

S=Path(__file__).resolve().parent
L=S.parents[1]
old='dynamic-human-enerf-rgb-pose-correction-074'
new='dynamic-human-enerf-detail-weight-076'
R=L/'outputs'/new
(R/'heldout').mkdir(exist_ok=True)
for name in ['capture-affines-estimated.npz','motion-extension-audit.json']:
    src=L/'outputs'/old/'heldout'/name
    dst=R/'heldout'/name
    if dst.exists():
        assert dst.read_bytes()==src.read_bytes()
    else:
        shutil.copy2(src,dst)

def replace_once(text,a,b):
    assert text.count(a)==1,(a,text.count(a))
    return text.replace(a,b,1)

text=(S/'render_enerf_rgb_heldout.py').read_text(encoding='utf8')
text=replace_once(text,old,new)
text=text.replace('Evaluate074on','Evaluate076on')
(S/'render_enerf_detail_heldout.py').write_text(text,encoding='utf8')

text=(S/'compare_enerf_rgb_heldout.py').read_text(encoding='utf8')
text=replace_once(text,old,new)
text=replace_once(text,"assert read(I/'independent-image-audit.json')['images']==54",
    "B=L/'outputs/"+old+"/heldout'\nassert read(B/'native-evaluation/independent-image-audit.json')['images']==18")
text=replace_once(text,"[I/'graph_estimated/native-evaluation',E]","[B/'native-evaluation',E]")
text=replace_once(text,"next(x for x in read(I/'paired-comparison/summary.json')['records'] if x['camera']==c and x['frame']==t)['metrics']['graph_estimated']['foreground']",
    "next(x for x in read(B/'native-evaluation/metrics.json')['records'] if x['camera']==c and x['frame']==t)['foreground_psnr_db']")
text=replace_once(text,"['Original photograph','Residual graph baseline','Corrected RGB pose']",
    "['Original photograph','Corrected RGB pose','Detail-weighted RGB']")
(S/'compare_enerf_detail_heldout.py').write_text(text,encoding='utf8')

text=(S/'audit_trained_enerf_rgb.py').read_text(encoding='utf8')
text=replace_once(text,"R=L/'outputs/"+old+"'","R=L/'outputs/"+new+"'")
(S/'audit_trained_enerf_detail.py').write_text(text,encoding='utf8')

runner='''#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
S=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human
python "$S/render_enerf_detail_heldout.py"
python "$S/audit_trained_enerf_detail.py"
'''
(S/'run_enerf_detail_heldout.sh').write_text(runner,encoding='utf8',newline='\n')
print('076 evaluation prepared: unchanged 074 motion, native inputs, fixed comparison regions.')
