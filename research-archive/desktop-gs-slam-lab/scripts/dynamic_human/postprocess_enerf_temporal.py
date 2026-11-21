"""CPU comparisons against corrected-RGB control, requiring finished rendering."""
from pathlib import Path
import subprocess,sys,json,hashlib
S=Path(__file__).resolve().parent; L=S.parents[1]
R=L/'outputs/dynamic-human-enerf-temporal-basis-078'
B=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert len(read(R/'residual_graph/native-evaluation/metrics.json')['records'])==102
assert len(read(R/'heldout/native-evaluation/metrics.json')['records'])==18
for name in ['protocol.json','initial-and-sampling-audit.json']:
    assert read(R/'residual_graph'/name)==read(B/'residual_graph'/name),name
assert sha(R/'graph.npz')==sha(B/'graph.npz')
ga=read(R/'residual_graph/graph-protocol.json');gb=read(B/'residual_graph/graph-protocol.json')
assert ga.pop('coefficients')==23040 and gb.pop('coefficients')==6912
assert ga.pop('temporal_basis')=='globalcubic_plus7compact_cubic_Bsplines'
assert ga==gb
report=dict(status='base_protocol_initialization_spatial_graph_sampling_equal_except_new_temporal_basis',base=B.name,candidate=R.name,
    graph_sha256=sha(R/'graph.npz'),checks=['protocol','spatial_graph','initial-and-sampling-audit','graph_protocol_except_explicit_temporal_extension'],
    temporal_change=read(R/'protocol.json')['temporal_basis'],goal_complete=False)
(R/'training-protocol-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
def run(script,*args):
    subprocess.run([sys.executable,str(S/script),*map(str,args)],check=True)
run('audit_enerf_articulated_images.py','--branch','residual_graph','--result-root',R)
args=['--before-root',B/'residual_graph','--after-root',R/'residual_graph','--output-root',R,
      '--before-label','Corrected RGB pose','--after-label','Local temporal basis']
run('compare_enerf_articulated_control.py',*args)
run('temporal_enerf_articulated_control.py',*args)
run('compare_enerf_temporal_heldout.py')
run('temporal_enerf_heldout_cameras.py','--before-root',B/'heldout','--after-root',R/'heldout',
    '--output-root',R/'heldout','--before-label','Corrected RGB pose','--after-label','Local temporal basis')
print('078 numerical and native panel comparisons complete. Actual visual decision pending.')
