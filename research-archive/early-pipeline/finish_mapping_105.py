from pathlib import Path
import json,hashlib,shutil,datetime
T=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/shared-mapping-contract-105'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
r=read(T/'human105-deck-receipt.json');source=Path(r['source']);final=Path(r['finalPath'])
assert sha(source)==r['source_sha256'] and sha(final)==r['final_sha256'] and not r['canonical_replaced']
valid=read(T/'human105-validation.json')
assert valid['packageIntegrity']['status']=='pass' and valid['presentationLayout']['exitCode']==0 and valid['firstPartyImport']['passed']
assert r['changed_slides']==[52] and len(r['renders'])==52 and (O/'published-report.png').exists()
staged=source.with_suffix('.staged.pptx');assert not staged.exists()
shutil.copy2(final,staged);assert sha(staged)==r['final_sha256'];staged.replace(source)
r.update(canonical_replaced=True,canonical_path=str(source),canonical_sha256=sha(source));save(T/'human105-deck-receipt.json',r)
shutil.copy2(T/'human105-final-52.png',O/'slide52.png');shutil.copy2(T/'human105-validation.json',O/'presentation-validation.json')
now=datetime.datetime.now().astimezone().isoformat()
save(O/'presentation-review.json',dict(created=now,canonical=str(source),sha256=sha(source),slides=52,changed_slides=[52],
    first51_render_hashes_unchanged=True,changed_slide_visually_inspected=True,actual_powerpoint_application_opened=False))
save(O/'visual-review.json',dict(created=now,inspected=['mapping-before-after.jpg','published-report.png','slide52.png'],
    finding='The same profile-view nose and mouth detections disagree with the silhouette in both old and new rectified images. Source images only cropped for display, no local realignment. The mask is not semantic or geometric GT.',
    quality_accepted=False,new_gaussian=False))
p=read(O/'publication.json');p.update(ui_verified=True,screenshot=str(O/'published-report.png'),presentation=str(source));save(O/'publication.json',p)
h=read(O/'handoff.json');h.update(report=p['report'],presentation=str(source),presentation_sha256=sha(source),viewer_gaussian_round='098',latestDiagnostic='105',
    input_convention_checks='Current098all175frame/ray/feature/world/viewermappingspass. The strictSFMvsOpenCVRGBcheck failed as previously known; independentall175geometrycheckpass. Originalstrictfailureandamendmentretained.',
    no_106_job_started=True,source_camera_and_gaussian_unchanged=True,
    metadata_change='Viewer latest/status metadata updated after audit. Source image/camera/model data unmodified.',
    decisive_mask_result='Current291outsideallalreadyoutsideoldmapping;0newoutside,2newinside; does not establish whether maskorlandmarksarewrong.')
save(O/'handoff.json',h)
save(O/'source-manifest.json',{str(p):sha(p) for p in [
    B.parents[2]/'repos/GaussianAvatars_SMPLX-shuolee/audit_shared_mapping_105.py',
    T/'publish_mapping_105.py',T/'update_deck_105.mjs',T/'finish_mapping_105.py']})
print(json.dumps(dict(canonical=str(source),sha256=sha(source),changed_slides=[52],report=p['report'],goal_complete=False)))
