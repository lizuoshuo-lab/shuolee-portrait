"""Prepare all078evaluations without modifying the active training runner."""
from pathlib import Path
import shutil,json
S=Path(__file__).resolve().parent;L=S.parents[1]
old='dynamic-human-enerf-detail-weight-076';new='dynamic-human-enerf-temporal-basis-078';B='dynamic-human-enerf-rgb-pose-correction-074'
R=L/'outputs'/new;(R/'heldout').mkdir(exist_ok=True)
for name in ['capture-affines-estimated.npz','motion-extension-audit.json']:
    source=L/'outputs'/B/'heldout'/name;target=R/'heldout'/name
    if target.exists():assert target.read_bytes()==source.read_bytes()
    else:shutil.copy2(source,target)
def once(t,a,b):
    assert t.count(a)==1,(a,t.count(a));return t.replace(a,b,1)

t=(S/'render_enerf_detail_heldout.py').read_text(encoding='utf8');t=once(t,old,new)
t=once(t,'from enerf_graph_renderer import GraphMotion,ResidualGraphRenderer','from enerf_temporal_graph import GraphMotion,ResidualGraphRenderer')
t=t.replace('Evaluate076on','Evaluate078on');(S/'render_enerf_temporal_heldout.py').write_text(t,encoding='utf8')
t=(S/'compare_enerf_detail_heldout.py').read_text(encoding='utf8');t=once(t,old,new);t=t.replace('Detail-weighted RGB','Local temporal basis')
(S/'compare_enerf_temporal_heldout.py').write_text(t,encoding='utf8')
t=(S/'review_enerf_detail_context.py').read_text(encoding='utf8');t=once(t,old,new);t=t.replace('Detail-weighted RGB','Local temporal basis')
(S/'review_enerf_temporal_context.py').write_text(t,encoding='utf8')
t=(S/'postprocess_enerf_detail.py').read_text(encoding='utf8');t=once(t,old,new)
t=once(t,"['protocol.json','graph-protocol.json','initial-and-sampling-audit.json']","['protocol.json','initial-and-sampling-audit.json']")
t=once(t,"assert sha(R/'graph.npz')==sha(B/'graph.npz')","""assert sha(R/'graph.npz')==sha(B/'graph.npz')
ga=read(R/'residual_graph/graph-protocol.json');gb=read(B/'residual_graph/graph-protocol.json')
assert ga.pop('coefficients')==23040 and gb.pop('coefficients')==6912
assert ga.pop('temporal_basis')=='globalcubic_plus7compact_cubic_Bsplines'
assert ga==gb""")
t=t.replace("base_protocol_initialization_graph_and_sampling_equal_to074","base_protocol_initialization_spatial_graph_sampling_equal_except_new_temporal_basis")
t=t.replace("['protocol','graph-protocol','initial-and-sampling-audit']","['protocol','spatial_graph','initial-and-sampling-audit','graph_protocol_except_explicit_temporal_extension']")
t=t.replace("added_loss=read(R/'protocol.json')['loss']","temporal_change=read(R/'protocol.json')['temporal_basis']")
t=t.replace('Detail-weighted RGB','Local temporal basis').replace('compare_enerf_detail_heldout.py','compare_enerf_temporal_heldout.py').replace('076 numerical','078 numerical')
(S/'postprocess_enerf_temporal.py').write_text(t,encoding='utf8')

t=(S/'audit_trained_enerf_detail.py').read_text(encoding='utf8');t=once(t,old,new)
t=t.replace('from scipy.spatial.transform import Rotation','from scipy.spatial.transform import Rotation\nfrom scipy.interpolate import BSpline')
t=once(t,'==[15000,15000]','==[15000,15000,15000,15000]')
t=once(t,"rot=.1*np.tanh(rawR);trans=.03*cp['scene_scale']*np.tanh(rawd)","""rot=.1*np.tanh(rawR);trans=.03*cp['scene_scale']*np.tanh(rawd)
localR=.03*np.tanh(cp['graph']['local_rotation_raw'].numpy().astype(np.float64))
locald=.01*cp['scene_scale']*np.tanh(cp['graph']['local_translation_raw'].numpy().astype(np.float64))
assert np.isfinite(localR).all() and np.isfinite(locald).all()
centers=cp['graph']['local_centers'].numpy().astype(np.float64)
assert np.array_equal(centers,np.linspace(-1.5,1.5,7))
bspline=BSpline.basis_element([-2,-1,0,1,2],extrapolate=False)
def basis_local(tau):return np.nan_to_num(bspline((tau-centers)/.5))-np.nan_to_num(bspline(-centers/.5))
assert np.count_nonzero(basis_local(0))==0
assert np.linalg.norm(localR)>0 and np.linalg.norm(locald)>0""")
t=once(t,"rv=np.einsum('k,nkd->nd',basis,rot);d=np.einsum('k,nkd->nd',basis,trans)","rv=np.einsum('k,nkd->nd',basis,rot)+np.einsum('k,nkd->nd',basis_local(tau),localR);d=np.einsum('k,nkd->nd',basis,trans)+np.einsum('k,nkd->nd',basis_local(tau),locald)")
t=t.replace('trained_graph_optimizer_steps=[15000,15000]','trained_graph_optimizer_steps=[15000,15000,15000,15000]')
t=t.replace('CPU independent SciPy rotations for all384nodes/all195contexts','CPU independent SciPy B-spline values and rotations for all384nodes/all195contexts')
(S/'audit_trained_enerf_temporal.py').write_text(t,encoding='utf8')
(S/'run_enerf_temporal_heldout.sh').write_text('''#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
S=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human
python "$S/render_enerf_temporal_heldout.py"
python "$S/audit_trained_enerf_temporal.py"
''',encoding='utf8',newline='\n')
print('078heldout motion unchanged; adapted120image comparisons and independent SciPy spline audit.')
