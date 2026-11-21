from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
HERE=Path(__file__).parent;ROOT=Path(__file__).resolve().parents[4]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
report=json.loads((HERE/'report.json').read_text(encoding='utf-8'));assert report['status']=='completed_actual1470train_initial_gaussian_projection_CPU_proxy_not_cause'
snap=HERE/'source-snapshots';snap.mkdir()
for path,sha in report['source_records'].items():
    p=Path(path)
    if p.suffix in ('.py','.cu','.h'):
        relative=p.relative_to(ROOT);target=snap/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);assert digest(target)==sha
out=ROOT/'outputs/large-scene-gs-001/independent-evaluation'
previous={}
for label in ('001','002'):
    d=out/('small-city-chunk11-initial-projection-CPU-'+label)
    previous[label]=[dict(path=str(p),sha256=digest(p)) for p in sorted(d.iterdir()) if p.is_file()]
value=dict(status='actual_geometry1470_CPU_proxy_complete_not_runtime_cause',created_utc=datetime.now(timezone.utc).isoformat(),
    output_source_records=[dict(path=str(HERE/n),sha256=digest(HERE/n)) for n in ('audit_projection_cpu.py','report.json','per-camera.json','all1470-initial-projection.npz','outcome.json','report.zh-CN.md')],
    previous001_failure_and002_circle_proxy_preserved=previous,
    source_snapshot_scope='Read-only source bytes, model/input files notcopied ormodified',
    actual_native_axis_footprint_branch='rasterize_points.cu rects[P,2]nonnull→forward.cu391axisceil3sqrt(Cdiagplus)→getRectinttruncation',
    cause_established=False,first_NaN_epoch_established=False,CUDA_called=False,Torch_imported=False,models_repaired=False,
    planned_owner_observer_trace_is_separate=True)
write(HERE/'provenance.json',value)
print(json.dumps(dict(provenance_sha256=digest(HERE/'provenance.json'),report_sha256=digest(HERE/'report.json'),zhreport_sha256=digest(HERE/'report.zh-CN.md'))))
