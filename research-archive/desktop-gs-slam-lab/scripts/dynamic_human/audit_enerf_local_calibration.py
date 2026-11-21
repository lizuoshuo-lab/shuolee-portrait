"""Independent source/epipolar score audit for the frozen local protocol."""
from pathlib import Path
import json
import numpy as np
from audit_enerf_input import digest
from enerf_native_loader import EnerfNativeDataset

LAB=Path(__file__).resolve().parents[2]
out=LAB/'outputs/dynamic-human-enerf-local-time-029'
manifest=out/'calibration-only-manifest.json'
spec=json.loads(manifest.read_text())
ds=EnerfNativeDataset(manifest,'train',allow_calibration_only=True)
available={(r['camera'],r['frame']) for r in ds.rows}
assert set(spec['selected_frames'])==set(range(6,13))|set(range(16,23))|set(range(26,33))
try:
    EnerfNativeDataset(manifest,'train')
except ValueError as e:
    assert 'Calibration-only' in str(e)
else:
    raise AssertionError('Calibration input guard absent')
for r in spec['images']:
    assert digest(ds.root/r['path'])==r['sha256']
    assert digest(ds.root/r['mask_path'])==r['mask_sha256']
graph=out/'graph-neighbors3';counts=[];maximum=0.
for f in sorted(graph.glob('tracks-*.npz')):
    _,c,d,t=f.stem.split('-');frame=int(t[1:]);v=np.load(f)
    wa=np.array(ds.cameras[c]['w2c']);wb=np.array(ds.cameras[d]['w2c'])
    R=wb[:3,:3]@wa[:3,:3].T;T=wb[:3,3]-R@wa[:3,3]
    F=np.linalg.inv(np.array(ds.cameras[d]['K'])).T@np.cross(np.eye(3),T)@R@np.linalg.inv(np.array(ds.cameras[c]['K']))
    x=np.c_[v['source_xy'],np.ones(len(v['source_xy']))];values=[]
    for k,lag in enumerate(v['offsets']):
        assert (c,frame) in available and (d,frame+int(lag)) in available
        y=np.c_[v['target_xy'][:,k],np.ones(len(x))];l2=x@F.T;l1=y@F
        dist=(y*l2).sum(1)/np.sqrt((l1[:,:2]**2).sum(1)+(l2[:,:2]**2).sum(1))
        values.append(dist)
    diff=float(np.abs(np.stack(values,1)-v['signed_sampson']).max())
    assert diff<1e-8;maximum=max(maximum,diff);counts.append(len(x))
for group in ['same-index','offset-index']:
    ap=json.loads((out/group/'appearance-manifest.json').read_text())
    assert not available.intersection((r['camera'],r['frame']) for r in ap['images'])
result=dict(status='local_calibration_source_scores_and_split_verified',source_images=len(spec['images']),source_files_checked=2*len(spec['images']),track_files=len(counts),matches=sum(counts),maximum_score_error=maximum,calibration_appearance_overlap=0,appearance_loader_guard='passed',scope='Empirical local frame correspondence only. All18cameras calibration-visible, three cameras appearance-heldout. This check is not hardware synchronization verification.')
(out/'independent-calibration-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
