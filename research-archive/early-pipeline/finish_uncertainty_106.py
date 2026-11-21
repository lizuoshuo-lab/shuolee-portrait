from pathlib import Path
import json,hashlib,shutil,datetime
import numpy as np
T=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/shared-uncertainty-bundle-106'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
r=read(T/'human106-deck-receipt.json');source=Path(r['source']);final=Path(r['finalPath'])
assert sha(source)==r['source_sha256'] and sha(final)==r['final_sha256'] and not r['canonical_replaced']
valid=read(T/'human106-validation.json')
assert valid['packageIntegrity']['status']=='pass' and valid['presentationLayout']['exitCode']==0 and valid['firstPartyImport']['passed']
assert r['changed_slides']==[52] and len(r['renders'])==52 and (O/'published-report.png').exists()
staged=source.with_suffix('.staged.pptx');assert not staged.exists()
shutil.copy2(final,staged);assert sha(staged)==r['final_sha256'];staged.replace(source)
r.update(canonical_replaced=True,canonical_path=str(source),canonical_sha256=sha(source));save(T/'human106-deck-receipt.json',r)
shutil.copy2(T/'human106-final-52.png',O/'slide52.png');shutil.copy2(T/'human106-validation.json',O/'presentation-validation.json')
now=datetime.datetime.now().astimezone().isoformat()
save(O/'presentation-review.json',dict(created=now,canonical=str(source),sha256=sha(source),slides=52,changed_slides=[52],first51_render_hashes_unchanged=True,changed_slide_visually_inspected=True,actual_powerpoint_application_opened=False))
v=read(O/'visual-review.json');v['inspected']+=['published-report.png','projection-review-slide.png','slide52.png'];save(O/'visual-review.json',v)
p=read(O/'publication.json');p.update(ui_verified=True,screenshot=str(O/'published-report.png'),presentation=str(source),viewer_actual_loaded_points=310961,viewer_actual_SH=3,viewer_sha_check_passed=True,viewer_latest_diagnostic='106');save(O/'publication.json',p)
h=read(O/'handoff.json');h.update(report=p['report'],presentation=str(source),presentation_sha256=sha(source),viewer_gaussian_round='098',latestDiagnostic='106',no_107_job_started=True,source_camera_and_gaussian_unchanged=True,metadata_updated=True)
save(O/'handoff.json',h)
sourcefiles=[B.parents[2]/'repos/GaussianAvatars_SMPLX-shuolee/fit_uncertainty_shared_bundle_106.py',T/'publish_uncertainty_106.py',T/'update_deck_106.mjs',T/'finish_uncertainty_106.py']
save(O/'source-manifest.json',{str(p):sha(p) for p in sourcefiles})
# The rejected candidate is retained only as small diagnostic numbers, never
# as a new training initializer. Raw inputs, all scene baselines and098 stay.
assert not read(O/'result.json')['numeric_candidate_passed']
with np.load(O/'solution.npz') as z:
    np.savez_compressed(O/'diagnostic-camera-points.npz',w2c=z['w2c'],face_xyz=z['face_xyz'],face_prediction=z['face_prediction'])
paths=[T/'human106-candidate.pptx',final,O/'input.npz',O/'solution.npz']
save(O/'cleanup-candidates.json',dict(created=now,scope='Only this turn generated duplicate PPTs and rejected106full solver arrays. Primary sources and canonical report retained.',
    canonical=str(source),canonical_sha256=sha(source),small_diagnostic=str(O/'diagnostic-camera-points.npz'),
    records=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in paths],executed=False))
print(json.dumps(dict(canonical=str(source),sha256=sha(source),changed_slides=[52],report=p['report'],goal_complete=False)))
