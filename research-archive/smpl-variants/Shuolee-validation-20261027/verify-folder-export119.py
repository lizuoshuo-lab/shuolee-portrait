from pathlib import Path
import hashlib,json
import numpy as np
import pycolmap
from plyfile import PlyData
root=Path('/opt/gs-slam-lab/shuolee-folder119-validation-20261027/runs/sofa119-cb11ad325834b399')
result=json.loads((root/'result.json').read_text())
validation=json.loads((root/'export/validation.json').read_text())
training=json.loads((root/'gaussian-run/result.json').read_text())
run=json.loads((root/'run.json').read_text())
ply=PlyData.read(root/'export/point_cloud.ply',mmap=False)['vertex'].data
for name in ply.dtype.names: assert np.isfinite(ply[name]).all(),name
sha=hashlib.sha256((root/'export/point_cloud.ply').read_bytes()).hexdigest()
assert sha==result['sha256']==validation['sha256']
assert result['state']=='completed' and result['steps']==training['steps']==18000
assert sum(training['per_frame_updates'])==18000
assert not any(run[k] for k in ('old_subject_loaded','old_fitted_person_loaded','old_gaussian_loaded','old_optimizer_loaded'))
recon=pycolmap.Reconstruction(str(root/'export/colmap'))
cameras=json.loads((root/'export/cameras.json').read_text())
assert len(recon.images)==len(cameras)==result['frames']==89
assert validation['format_passed'] is True
assert all(p.suffix in ('.ply','.json','.txt') for p in (root/'export').rglob('*') if p.is_file())
report=dict(state='passed',run=str(root),reader='plyfile and pycolmap 3.10.0',points=len(ply),property_count=len(ply.dtype.names),sha256=sha,finite=True,frames=len(recon.images),steps=18000,native_render_checks=len(validation['native_render_checks']),max_native_export_error=max(x['max_abs'] for x in validation['native_render_checks']),fresh_subject_and_optimizer=True,viewer_required=False)
Path('/mnt/d/Projects/Shuolee-validation-20261027/folder119-independent-export.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
