"""Real-source isolation test and native cache correspondence for local study."""
from pathlib import Path
import json,copy
import numpy as np
from enerf_native_loader import EnerfNativeDataset
from audit_enerf_input import digest
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-local-time-029';cal=json.loads((out/'calibration-only-manifest.json').read_text());cal_ids={(r['camera'],r['frame']) for r in cal['images']};reports=[]
assert digest(out/'same-index/train-visual-hull-init.npz')==digest(out/'offset-index/train-visual-hull-init.npz')
for group in ['same-index','offset-index']:
    p=out/group/'appearance-manifest.json';s=json.loads(p.read_text());ds=EnerfNativeDataset(p,'train');bad=copy.deepcopy(s)
    for r in bad['images']:
        if r['split']=='heldout_view':r.update(path='MUST_NOT_OPEN_HELDOUT_RGB.jpg',sha256='invalid',mask_path='MUST_NOT_OPEN_HELDOUT_MATTE.jpg',mask_sha256='invalid')
    temp=out/group/'heldout-inaccessible-test-manifest.json';temp.write_text(json.dumps(bad,indent=2)+'\n',encoding='utf8');isolated=EnerfNativeDataset(temp,'train');assert len(ds)==len(isolated)==375
    refs={r['camera']:r for r in s['images'] if r['split']=='train' and r['common_frame']==34};cache_count=0
    for i,row in enumerate(ds.rows):
        assert (row['camera'],row['frame']) not in cal_ids;orig=ds[i];other=isolated[i]
        for k in ['rgb','foreground_matte','valid_pixels','K','w2c']:assert np.array_equal(orig[k],other[k])
        assert orig['common_frame']==row['common_frame'] and orig['frame']==row['frame']
        if row['common_frame']==34:
            v=np.load(out/group/f'train-camera-{row["camera"]}.npz')
            for k,alt in [('rgb','rgb'),('matte','foreground_matte'),('valid','valid_pixels'),('K','K'),('w2c','w2c')]:assert np.array_equal(v[k],orig[alt])
            assert int(v['source_frame'])==row['frame'];cache_count+=1
    assert cache_count==15;reports.append(dict(condition=group,source_manifest_sha256=digest(p),training_images=375,all_training_inputs_unchanged_when_test_files_inaccessible=True,first_frame_native_caches=cache_count,calibration_appearance_overlap=0))
    print('VERIFIED',group,flush=True)
result=dict(status='both_real_appearance_input_boundaries_and_native_caches_verified',conditions=reports,shared_initial_gaussians_sha256=digest(out/'same-index/train-visual-hull-init.npz'),scope='75testRGB/matte paths intentionally made unavailable in EACH copied manifest; all375train observations remain identical. Original source files untouched. Camera matrices and earlier temporal calibration are shared by design.',temporal_holdout=False)
(out/'input-isolation-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
