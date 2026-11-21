from pathlib import Path
import json,hashlib,shutil,datetime
W=Path('C:/Users/Administrator/Documents/New project');T=W/'shuolee-slide-update'
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/shared-camera-points-allview-104';A=B/'current/shared-observation-support-audit-103'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
r=read(T/'human104-deck-receipt.json');source=Path(r['source']);final=Path(r['finalPath'])
assert sha(source)==r['source_sha256'] and sha(final)==r['final_sha256']
assert r['changed_slides']==[52] and len(r['renders'])==52
assert (O/'published-report.png').exists()
staged=source.with_suffix('.staged.pptx');assert not staged.exists()
shutil.copy2(final,staged);assert sha(staged)==r['final_sha256'];staged.replace(source)
assert sha(source)==r['final_sha256']
r['canonical_replaced']=True;r['canonical_path']=str(source);r['canonical_sha256']=sha(source);save(T/'human104-deck-receipt.json',r)
shutil.copy2(T/'human104-final-52.png',O/'slide52.png');shutil.copy2(T/'human104-validation.json',O/'presentation-validation.json')
now=datetime.datetime.now().astimezone().isoformat()
save(O/'presentation-review.json',dict(created=now,canonical=str(source),sha256=sha(source),slides=52,
    changed_slides=[52],first51_render_hashes_unchanged=True,changed_slide_visually_inspected=True,
    actual_powerpoint_application_opened=False))
save(O/'visual-review.json',dict(created=now,inspected=['all-view-shared-projection-review.jpg','source-mask-feature-review.jpg','published-report.png','slide52.png'],
    source_pixels_unchanged=True,observations='Projection residuals shorten in the fixed raw1example. Raw124still has visible mismatches near nose and mouth. Mask-contour overlays show several outside detections, especially profileviews. These pictures do not establish a corrected Gaussian model.',
    quality_accepted=False,training_allowed=False))
save(A/'visual-review.json',dict(created=now,inspected=['review/raw000001-support.jpg','review/raw000124-support.jpg'],
    observation='Initial support is sparse near nose/lips; the historical499and18000references have densealpha. Those references are not one training trajectory, nor evidence that coverage alone fixes ghosting.',
    optimizer_updates=0,quality_accepted=False))
p=read(O/'publication.json');p.update(ui_verified=True,screenshot=str(O/'published-report.png'),presentation=str(source));save(O/'publication.json',p)
handoff=read(O/'handoff.json');handoff.update(presentation=str(source),report=p['report'],viewer_gaussian_round='098',latestDiagnostic='104',
    gaussian_training_started=False,live_processes=0,
    key_next_evidence='104bodyuppergate failed only torso: p90delta0.5094587437vs0.5threshold. Source-maskcheck found291/5815belowhalf,232exactzero in95frames. Do not call all invalid or silentlydrop them. Queryappropriate image/visibility/measurementuncertainty and retain all cameras.',
    preparation_repair='Early preparation decompressed the sameNPZarrays pertrack, stoppedPID310 before nativeBA and cachedimmutableinputonce. Subsequentfreshpreparation completed and Ceres converged40iterations. No previousfittedXYZreused.')
save(O/'handoff.json',handoff)
save(O/'source-manifest.json',{str(p):sha(p) for p in [
    B.parents[2]/'repos/GaussianAvatars_SMPLX-shuolee/refit_shared_camera_points_104.py',
    B.parents[2]/'repos/GaussianAvatars_SMPLX-shuolee/audit_shared_visibility_104.py',
    B.parents[2]/'repos/PGSR-shuolee/audit_shared_observation_support_103.py',
    T/'publish_shared_bundle_104.py',T/'inspect_visibility_104.py',T/'update_deck_104.mjs']})
print(json.dumps(dict(canonical=str(source),sha256=sha(source),published=True,quality_accepted=False)))
