"""Independent source, geometry-score, and controlled-initialization checks."""
from pathlib import Path
import json,hashlib
import numpy as np
from enerf_native_loader import EnerfNativeDataset
from audit_enerf_input import digest

LAB=Path(__file__).resolve().parents[2]
out=LAB/'outputs/dynamic-human-enerf-time-control-027'
diag=LAB/'outputs/dynamic-human-enerf-correspondence-026'
graph=diag/'graph-neighbors3';g=json.loads((graph/'global-offset-hypothesis.json').read_text())
manifest=LAB/'outputs/dynamic-human-enerf-full-023/actor1_4-native-input-audit.json'
ds=EnerfNativeDataset(manifest,'train');available={(r['camera'],r['frame']) for r in ds.rows}
# Keep the hypothesis bytes used by preparation, before exact-DP certification
# was added. Recover only when its exact SHA matches the recorded receipt.
prior=dict(g);prior['best_loss']=prior.pop('multi_start_best_loss');prior.pop('exact_graph_solver')
encoded=(json.dumps(prior,indent=2,allow_nan=False)+'\n').encode('utf8')
receipt=json.loads((out/'offset-index/source-observations.json').read_text())
if hashlib.sha256(encoded).hexdigest()!=receipt['offset_hypothesis_sha256']:encoded=encoded.replace(b'\n',b'\r\n')
assert hashlib.sha256(encoded).hexdigest()==receipt['offset_hypothesis_sha256']
(graph/'global-offset-hypothesis-pre-dp.json').write_bytes(encoded)
assert digest(out/'same-index/train-visual-hull-init.npz')==digest(out/'offset-index/train-visual-hull-init.npz')
rows=[]
for file in sorted(graph.glob('tracks-*.npz')):
    _,c,d,t=file.stem.split('-');frame=int(t[1:]);v=np.load(file);ca=ds.cameras[c];cb=ds.cameras[d]
    wa=np.array(ca['w2c']);wb=np.array(cb['w2c']);R=wb[:3,:3]@wa[:3,:3].T;T=wb[:3,3]-R@wa[:3,3]
    E=np.cross(np.eye(3),T)@R
    F=np.linalg.solve(np.array(cb['K']).T,E)@np.linalg.inv(np.array(ca['K']))
    x=np.c_[v['source_xy'],np.ones(len(v['source_xy']))];all_errors=[]
    for k,lag in enumerate(v['offsets']):
        assert (c,frame) in available and (d,frame+int(lag)) in available
        y=np.c_[v['target_xy'][:,k],np.ones(len(x))];Fx=(F@x.T).T;Fty=(F.T@y.T).T
        r=np.einsum('ij,ij->i',y,Fx)/np.sqrt(np.sum(Fx[:,:2]**2,axis=1)+np.sum(Fty[:,:2]**2,axis=1));all_errors.append(r)
    err=np.stack(all_errors,1);difference=float(np.max(np.abs(err-v['signed_sampson'])));assert difference<1e-8
    rows.append(dict(file=file.name,points=len(x),max_error_difference=difference))
source_checks=[]
for group in ['same-index','offset-index']:
    s=json.loads((out/group/'source-observations.json').read_text())
    assert len(s['source_frames'])==15
    for c,t in s['source_frames'].items():
        assert (c,t) in available
        assert t==49+(g['offsets'][c] if group=='offset-index' else 0)
        v=np.load(out/group/f'train-camera-{c}.npz');assert int(v['source_frame'])==t;assert str(v['source_sha256'])==s['source_rgb_sha256'][c]
    source_checks.append(dict(condition=group,source_frames=s['source_frames']))
result=dict(status='source_indices_splits_sampson_and_identical_initialization_verified',track_files=len(rows),track_points=sum(r['points'] for r in rows),score_checks=rows,initialization_sha256=digest(out/'same-index/train-visual-hull-init.npz'),source_checks=source_checks,offset_fit_frames=[9,29],offset_independent_check_frame=49,heldout_rgb_or_masks_used_for_offset_fit=False,manual_review=dict(images=['identity-09-01-t049.png','identity-01-10-t049.png','identity-14-16-t049.png','identity-04-03-t029.png'],selection='Fastest source features with spatial separation, no error-improvement gate',scope='24 native 128px patch triplets visually inspected; most correspond to same hands, toys, clothing points; perspective and exposure blur remain. This does not certify all feature tracks.'),limitations=['Only original training cameras have estimated offsets','Heldout-view image49 remains unchanged; its physical alignment with reference camera08 is unverified','Common 6000-step budget, seed73, source init identical; Gaussian counts may diverge through the same densification rule','Same-index initialization reused by offset condition, no new visual hull advantage','Only one moving instant, no continuous reconstruction quality claim'])
(out/'control-protocol-and-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ['score_checks','source_checks','manual_review']}))
