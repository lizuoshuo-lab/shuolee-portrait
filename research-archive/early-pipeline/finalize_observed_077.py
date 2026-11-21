"""Record the completed experiment without promoting a failed reconstruction."""
from pathlib import Path
import datetime, hashlib, json, shutil
import numpy as np

W = Path(__file__).parent
B = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
V = B.parent / 'viewer'
R = B / 'current/person-static-observed-correspondence-fresh-077'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def save(p, data):
    t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    t.replace(p)

rows=read(R/'data/training-inputs.json')['records']
meta=read(V/'model-whole-body-static-correspondence.json')
export=read(R/'gaussian-run/viewer-export/delivery.json')
T=np.asarray(export['world_to_viewer']['rotation'])
origin=np.asarray(export['world_to_viewer']['origin_world'])
assert len(rows)==len(meta['cameras'])==175
assert sha(V/meta['model'])==export['sha256']==meta['sha256']
rng=np.random.default_rng(77)
maximum=0.
for row,cam in zip(rows,meta['cameras']):
    E,K=np.asarray(row['w2c']),np.asarray(row['K'])
    assert (row['source_frame_index'],row['frame'],row['camera_kind'])==(cam['raw_source_frame'],cam['img_name'],cam['camera_kind'])
    assert row['camera_kind']=='fresh_full_scene_sfm'
    Kv=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1]])
    np.testing.assert_allclose(K,Kv,atol=1e-10,rtol=0)
    xc=rng.uniform(-1,1,(64,3));xc[:,2]=rng.uniform(1,10,64)
    xw=(xc-E[:3,3])@E[:3,:3]
    xv=(xw-origin)@T.T
    nv=(xv-np.asarray(cam['position']))@np.asarray(cam['rotation'])
    a,b=xc@K.T,nv@Kv.T
    maximum=max(maximum,float(abs(a[:,:2]/a[:,2:]-b[:,:2]/b[:,2:]).max()))
assert maximum<1e-8
camera_check=R/'gaussian-run/viewer-camera-roundtrip.json'
save(camera_check,dict(checked_at=now,cameras=175,synthetic_rays_per_camera=64,
    max_projection_difference_px=maximum,passed=True,
    camera_counts=dict(fresh_full_scene_sfm=175,interpolated=0),
    claim_boundary='Coordinate/export consistency only, not estimated camera accuracy'))

screens=[R/'review'/n for n in ['viewer-preset.png','viewer-standalone-gaussians.png','viewer-standalone-points.png']]
assert all(p.is_file() and p.stat().st_size>10000 for p in screens)
manual=read(R/'review/viewer-interaction-review.json')
assert manual['manual_open_file'] and manual['gaussian_mode_seen'] and manual['point_mode_seen']
assert manual['points_displayed']==export['points']==300827
visual=read(R/'review/visual-decision.json')
assert visual['actually_inspected'] and not visual['quality_accepted']
visual.update(local_independent_ply_import='passed',local_viewer_checked_at=now,
    local_viewer_points=export['points'],local_viewer_modes=['gaussian','point_cloud'],
    local_viewer_screenshots=[str(p) for p in screens],pc00_opened_this_round=False)
save(R/'review/visual-decision.json',visual)
paired=read(R/'review/paired-transfer-comparison.json')
assert not paired['all_numeric_gates_passed']
paired.update(visual_inspection_complete=True,quality_accepted=False)
save(R/'review/paired-transfer-comparison.json',paired)
public_path=V/'models/whole-body-static-correspondence/delivery.json'
public=read(public_path)
public.update(local_standalone_open_verified=True,local_preset_open_verified=True,
    local_viewer_proof=[dict(path=str(p),sha256=sha(p)) for p in screens],
    camera_export_check=str(camera_check),quality_accepted=False,pc00_opened_this_round=False)
save(public_path,public)
registry=read(B/'gaussian-deliveries.json')
registry['rounds']['077'].update(local_standalone_open_verified=True,
    local_gaussian_and_centers_verified=True,local_viewer_proof=public['local_viewer_proof'],quality_accepted=False)
registry['updated']=now;save(B/'gaussian-deliveries.json',registry)
status=read(B/'work-status.json')
assert status['latest_completed_static_delivery']['round']=='077'
status['latest_completed_static_delivery'].update(local_viewer_passed=True,camera_export_check=str(camera_check))
status['updated']=now;save(B/'work-status.json',status)
for name in ['visual-decision.json','paired-transfer-comparison.json']:
    shutil.copy2(R/'review'/name,V/'body-observed-correspondence-current'/name)
publication=read(R/'publication.json')
for entry in publication['published_assets']:
    entry['sha256']=sha(V/'body-observed-correspondence-current'/entry['file'])
publication.update(updated=now,local_independent_ply_import=True)
save(R/'publication.json',publication)
page=V/'body-smplx.html'
content=page.read_text(encoding='utf8').replace('当前最新人物高斯（质量状态见075）','当前最新人物高斯（质量状态见页首）')
page.write_text(content,encoding='utf8')
for f in ['compare_transfer_077.py','publish_observed_077.py','update_deck_077.mjs','finalize_observed_077.py','prepare_person_077.sh','review_after_077.sh','review_person_077.sh']:
    p=W/f
    if p.exists():shutil.copy2(p,R/'source-final'/f)
result=read(R/'gaussian-run/result.json')
assert result['steps']==18000 and result['from_scratch']
deck=Path(publication['deck'])
assert sha(deck)==publication['deck_sha256']
save(R/'completion.json',dict(updated=now,round='077',workflow_complete=True,
    training_steps=18000,initial_points=200000,final_points=300827,from_scratch=True,
    whole_scene_camera_frames=175,interpolated_camera_frames=0,representation='shared_static_world',
    quality_accepted=False,root_cause_established=False,all_numeric_gates_passed=False,
    standard_ply=str(V/meta['model']),ply_sha256=export['sha256'],compatibility_passed=True,
    local_standalone_viewer_passed=True,pc00_deployed=False,
    camera_roundtrip_px=maximum,report_url=publication['report_url'],
    deck=str(deck),deck_sha256=sha(deck),continuous_video_full_review=False,
    next_step=visual['next_step']))
print(json.dumps(dict(complete=True,quality_accepted=False,points=300827,
    camera_export_max_px=maximum,ply_sha256=export['sha256'],deck_sha256=sha(deck)),ensure_ascii=False))
