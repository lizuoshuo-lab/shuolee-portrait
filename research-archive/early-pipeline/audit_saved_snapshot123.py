"""Read-only replay of saved arrays; no fitted/model/scaled geometry is written."""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
ROUTE = T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route'
A = ROUTE/'training-interface-audit/saved-snapshot'
EVIDENCE = json.loads((A/'audit-scale-evidence.json').read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):h.update(block)
    return h.hexdigest()
def save(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
fit_path=ROUTE/'shared-three-frame/fit.npz'
before=sha(fit_path)
with np.load(fit_path,allow_pickle=False) as z: fit={k:z[k] for k in z.files}
assert before==EVIDENCE['source_hashes'][str(fit_path)]
video=EVIDENCE['video']; assert sha(video['path'])==video['sha256']
inputs_path=next(p for p in EVIDENCE['source_hashes'] if p.endswith('data\\inputs.json'))
rows=json.loads(Path(inputs_path).read_text(encoding='utf-8-sig'))['records']
assert sha(inputs_path)==EVIDENCE['source_hashes'][inputs_path]
by_id={r['frame']:r for r in rows}; assert len(by_id)==len(rows)
checks=[]
for i, raw_id in enumerate(fit['frame_ids']):
    key=str(raw_id); row=by_id[key]; independent=next(x for x in EVIDENCE['frame_checks'] if x['frame_id']==key)
    assert row['source_frame_index']==int(fit['source_frames'][i])==independent['source_frame']
    assert np.array_equal(row['w2c'],fit['w2c'][i])
    assert np.array_equal(row['K_integer_centers'],fit['K_integer_centers'][i])
    for kind in ['image','mask']:
        p=independent['files'][kind]['path']; assert sha(p)==independent['files'][kind]['sha256']
        with Image.open(p) as im:
            im.load()  # Header size/hash alone do not guarantee complete decoding.
            assert list(im.size)==row['wh']
    rot=fit['w2c'][i,:3,:3]; trans=fit['w2c'][i,:3,3]
    replay=(fit['vertices_camera'][i]-trans)@rot
    assert np.allclose(replay,fit['vertices_world'][i],atol=1e-12,rtol=0)
    projected=(fit['vertices_world'][i]@rot.T+trans)@fit['K_integer_centers'][i].T
    projected=projected[:,:2]/projected[:,2:]
    direct=fit['vertices_camera'][i]@fit['K_integer_centers'][i].T
    direct=direct[:,:2]/direct[:,2:]
    checks.append(dict(frame_id=key,source_frame=int(fit['source_frames'][i]),pose_index=i,
        original_roster_slot=independent['source_roster_slot'],full_image_and_mask_decoded=True,
        K_E_exact=True,world_replay_max=float(np.max(abs(replay-fit['vertices_world'][i]))),
        same_frame_projection_max_px=float(np.max(np.linalg.norm(projected-direct,axis=1)))))
assert np.array_equal(fit['reference_vertices_world'],fit['vertices_world'][1])
assert fit['betas'].shape==(1,10)
assert sha(fit_path)==before
result=dict(updated=datetime.now(timezone.utc).isoformat(),
    state='real_identity_and_numeric_replay_verified_world_scale_not_established',
    source_video=video,source_fit_sha256=before,frames=checks,
    shared_beta_shape=list(fit['betas'].shape),reference_frame_id=str(fit['frame_ids'][1]),
    registered_roster_frames=len(rows),validated_saved_frames=len(checks),
    physical_length_unit_verified=False,real_meter_bundle_created=False,
    reason='The new sequence inverse transform omits an explicit common model-to-SfM scale. Numerical roundtrip cannot validate units.',
    older_scale_estimates=EVIDENCE['existing_scale_evidence'],
    original_inputs_preserved=True,new_geometry_or_camera=False,
    new_model_forwards=0,real_fit=0,gaussian_steps=0,training_admitted=False,quality_accepted=False)
save(A/'saved-input-result.json',result)
print(json.dumps({k:result[k] for k in ['state','frames','real_meter_bundle_created','new_geometry_or_camera']}))
