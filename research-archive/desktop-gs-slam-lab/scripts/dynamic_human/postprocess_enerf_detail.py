"""CPU comparisons against corrected-RGB control, requiring finished rendering."""
from pathlib import Path
import subprocess,sys,json,hashlib
S=Path(__file__).resolve().parent; L=S.parents[1]
R=L/'outputs/dynamic-human-enerf-detail-weight-076'
B=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert len(read(R/'residual_graph/native-evaluation/metrics.json')['records'])==102
assert len(read(R/'heldout/native-evaluation/metrics.json')['records'])==18
for name in ['protocol.json','graph-protocol.json','initial-and-sampling-audit.json']:
    assert read(R/'residual_graph'/name)==read(B/'residual_graph'/name),name
assert sha(R/'graph.npz')==sha(B/'graph.npz')
report=dict(status='base_protocol_initialization_graph_and_sampling_equal_to074',base=B.name,candidate=R.name,
    graph_sha256=sha(R/'graph.npz'),checks=['protocol','graph-protocol','initial-and-sampling-audit'],
    added_loss=read(R/'protocol.json')['loss'],goal_complete=False)
(R/'training-protocol-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
def run(script,*args):
    subprocess.run([sys.executable,str(S/script),*map(str,args)],check=True)
run('audit_enerf_articulated_images.py','--branch','residual_graph','--result-root',R)
args=['--before-root',B/'residual_graph','--after-root',R/'residual_graph','--output-root',R,
      '--before-label','Corrected RGB pose','--after-label','Detail-weighted RGB']
run('compare_enerf_articulated_control.py',*args)
run('temporal_enerf_articulated_control.py',*args)
run('compare_enerf_detail_heldout.py')
run('temporal_enerf_heldout_cameras.py','--before-root',B/'heldout','--after-root',R/'heldout',
    '--output-root',R/'heldout','--before-label','Corrected RGB pose','--after-label','Detail-weighted RGB')
print('076 numerical and native panel comparisons complete. Actual visual decision pending.')
