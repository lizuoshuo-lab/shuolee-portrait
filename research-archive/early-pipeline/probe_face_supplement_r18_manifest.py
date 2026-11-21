from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
from PIL import Image

PROJECT=Path('/mnt/d/Projects/Shuolee-smpl-face-supplement-20261111')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime
from shuolee_portrait.face_supplement import _front_depth, prepare_face_supplements

def require(condition,message):
    if not condition:
        raise AssertionError(message)

K=np.asarray([[100.,0.,64.],[0.,110.,48.],[0.,0.,1.]])
front=np.asarray([[-1.,-1.,2.],[1.,-1.,4.],[0.,1.,3.]])
reproject=front @ K.T
screen=reproject[:,:2]/reproject[:,2:3]
weights=np.asarray([.2,.3,.5])
depth=_front_depth(np.stack((weights @ screen,[10000.,10000.])),
                   np.concatenate((front*2,front)),np.arange(6).reshape(2,3),K)
expected=1./np.sum(weights/front[:,2])
require(np.allclose(depth[0],expected,atol=1e-10,rtol=0) and np.isinf(depth[1]),
        'Perspective-correct nearest mesh intersection control failed')
with TemporaryDirectory() as tmp:
    require(prepare_face_supplements(Path(tmp),[{'frame':'test'}])[1]['state']=='skipped_missing_joint_candidate_contract',
            'Legacy input compatibility control failed')

proof = dict(state='passed',scope='CPU native image/independent mask, exact frozen fit and camera admission',
             source_sha256=runtime.sha(PROJECT/'src/shuolee_portrait/face_supplement.py'),
             fixture_sha256=runtime.sha(__file__),
             controls=dict(expected_nearest_depth_m=expected,computed_nearest_depth_m=float(depth[0]),
                           no_hit_is_unknown=True,legacy_without_joint_archive_skips=True),
             subjects={})
for subject in ('016','006'):
    root=Path('/opt/gs-slam-lab')/('shuolee-smpl-delivery-round17-'+subject)/'run'
    rows=runtime.read(root/'data/training-inputs.json')['records']
    selection=runtime.read(root/'fit/selection.json')
    source={row['frame']:row for row in runtime.read(root/'data/inputs.json')['records']}
    held={Path(source[name][field]) for name in selection['retained_holdouts'] for field in ('image','mask')}
    reads=[]
    original_read=Path.read_bytes
    def check_read(path):
        require(path not in held,f'Holdout bytes entered admission: {path}')
        reads.append(path)
        return original_read(path)
    t0=perf_counter()
    with patch.object(Path,'read_bytes',check_read):
        supplements,evidence=prepare_face_supplements(root,rows)
    with np.load(root/'fit/joint-candidates.npz',allow_pickle=False) as fit:
        fit_ids=fit['frame_ids'].tolist()
        details=[]
        for item in supplements:
            name=item.row['frame']
            src=source[name]
            record=next(row for row in evidence['candidates'] if row['frame']==name)
            require(item.row['K']==src['K'] and item.row['K_integer_centers']==src['K_integer_centers']
                    and item.row['source_w2c']==src['w2c'],'Camera was altered or interpolated')
            require(item.row['source_sha256']==src['source_sha256'] and item.row['mask_sha256']==src['mask_sha256'],
                    'Selected source bytes differ')
            with Image.open(src['mask']) as m:
                alpha=np.asarray(m.convert('L').crop(item.box)).astype(np.float32)/255
            require(np.array_equal(alpha,item.alpha) and np.all(alpha[item.valid]>=.99) and np.any(~item.valid),
                    'Auxiliary target changed the independent native alpha or included unknown pixels')
            require(record['native_roi']['valid_sha256']==hashlib.sha256(item.valid.tobytes()).hexdigest()
                    and record['native_roi']['alpha_sha256']==hashlib.sha256(item.alpha.tobytes()).hexdigest(),
                    'In-memory target does not match its receipt')
            E=np.asarray(item.row['w2c'])
            replay=item.pose['vertices_world']@E[:3,:3].T+E[:3,3]
            error=float(np.max(np.abs(replay-fit['vertices_camera'][fit_ids.index(name)])))
            require(error<1e-7,'Supplemental vertices did not replay to the archived camera geometry')
            details.append(dict(frame=name,box=list(item.box),known_pixels=int(item.valid.sum()),
                                unknown_crop_pixels=int((~item.valid).sum()),frozen_camera_mesh_error_m=error,
                                admitted_regions=[key for key,v in record['regions'].items() if v['status']=='admitted']))
    require(len(evidence['fit_candidate_frame_ids'])==52 and len(evidence['full_body_frame_ids'])==24
            and len(evidence['retained_holdouts'])==4
            and not set(evidence['chosen_frame_ids'])&set(evidence['retained_holdouts']+evidence['full_body_frame_ids']),
            'Protected source and candidate roster changed')
    proof['subjects'][subject]=dict(seconds=perf_counter()-t0,selected=details,source_reads=len(reads),
                                   candidate_status=dict(Counter(row['status'] for row in evidence['candidates'])),
                                   source_inputs=evidence['input_sha256'],full_body=evidence['full_body_coverage'],
                                   supplemented=evidence['supplemented_coverage'],retained_holdouts=evidence['retained_holdouts'],
                                   audit=evidence)
path=Path('/mnt/c/Users/Administrator/Documents/New project/outputs/smpl-delivery-round18/input-audit/face-supplement-admission-archive-proof.json')
runtime.save(path,proof)
print(json.dumps(dict(path=str(path),sha256=runtime.sha(path),selected={s:[r['frame'] for r in v['selected']]
       for s,v in proof['subjects'].items()}),indent=2))
