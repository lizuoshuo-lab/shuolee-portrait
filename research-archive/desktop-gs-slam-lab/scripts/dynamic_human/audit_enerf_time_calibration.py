"""Check calibration provenance, scores and rejection of unobservable offsets."""
from pathlib import Path
import json
import numpy as np
from audit_enerf_input import digest
from enerf_native_loader import EnerfNativeDataset
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-time-calibration-028';reports=[]
for directory in [out,out/'observable-motion']:
    manifest=directory/'calibration-only-manifest.json';spec=json.loads(manifest.read_text());ds=EnerfNativeDataset(manifest,'train',allow_calibration_only=True);available={(r['camera'],r['frame']) for r in ds.rows};assert min(spec['selected_frames'])>59
    try:EnerfNativeDataset(manifest,'train')
    except ValueError as e:assert 'Calibration-only' in str(e)
    else:raise AssertionError('Calibration input guard absent')
    for r in spec['images']:
        assert digest(ds.root/r['path'])==r['sha256'];assert digest(ds.root/r['mask_path'])==r['mask_sha256']
    graph=directory/'graph-neighbors3';raw=json.loads((graph/'timing-diagnostic.json').read_text());counts=[]
    for f in sorted(graph.glob('tracks-*.npz')):
        _,c,d,t=f.stem.split('-');frame=int(t[1:]);v=np.load(f);wa=np.array(ds.cameras[c]['w2c']);wb=np.array(ds.cameras[d]['w2c']);R=wb[:3,:3]@wa[:3,:3].T;T=wb[:3,3]-R@wa[:3,3];F=np.linalg.inv(np.array(ds.cameras[d]['K'])).T@np.cross(np.eye(3),T)@R@np.linalg.inv(np.array(ds.cameras[c]['K']));x=np.c_[v['source_xy'],np.ones(len(v['source_xy']))]
        values=[]
        for k,lag in enumerate(v['offsets']):
            assert (c,frame) in available and (d,frame+int(lag)) in available
            y=np.c_[v['target_xy'][:,k],np.ones(len(x))];l2=x@F.T;l1=y@F;dist=(y*l2).sum(1)/np.sqrt((l1[:,:2]**2).sum(1)+(l2[:,:2]**2).sum(1));values.append(dist)
        diff=float(np.abs(np.stack(values,1)-v['signed_sampson']).max());assert diff<1e-8;counts.append(len(x))
    hypothesis=json.loads((graph/'global-offset-hypothesis.json').read_text());reports.append(dict(directory=directory.relative_to(out).as_posix(),source_manifest_sha256=digest(manifest),source_images=len(spec['images']),source_files_checked=2*len(spec['images']),tracks_files=len(counts),matches=sum(counts),score_recomputation='passed',appearance_loader_guard='passed',hypothesis_status=hypothesis['status']))
result=dict(status='calibration_inputs_and_geometric_scores_verified_no_global_retiming_accepted',reports=reports,conclusions=['First calibration anchors109,129 yielded inconsistent offsets; independent149 has>=12 moving matches in only3/48 pairs, insufficient whole-rig check','Motion-selected anchors460,640 lose enough shared cross-camera moving tracks that the fit graph disconnects; numeric gauges across components are unidentifiable','Original train-only9,29 to49 diagnosis remains a local result, not proof of constant whole-sequence camera offset','No original images, filenames or appearance models changed by calibration runs'],next='Use time-local calibration with explicit separate calibration images and appearance-only heldout targets, or improve moving-part correspondences; do not reuse failed global offsets.',retiming_performed=False)
(out/'independent-audit-and-decision.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
