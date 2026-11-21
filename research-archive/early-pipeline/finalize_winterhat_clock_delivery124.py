"""Persist completed local UI inspection without claiming Company00 execution."""
from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a
from publish_cache_clock124 import atomic, main as report

root=a.REC/'cross-subject-sh0-124/winterhat-clock18k-cache124'
out=root/'delivery/step-018000';proof=root/'ui-review'
pub=a.read(root/'delivery/publication.json');v=a.read(out/'validation.json')
package=a.read(out/'package-result.json');now=datetime.now(timezone.utc).isoformat()
assert a.read(root/'process-exit.json')['exit_code']==0
assert v['points']==74330 and a.sha(out/'point_cloud.ply')==v['sha256']
assert a.sha(out/'scene.zip')==package['sha256']
paths=[proof/name for name in ('preset.png','openfile-gaussian.png','openfile-points.png','final-viewer.png')]
assert all(a.lp(p).exists() for p in paths)
ui=dict(updated=now,actual_ui_verified=True,preset_loaded=True,open_file_zip=True,
    gaussian_mode=True,point_mode=True,points=74330,camera_count=1,reference_frame='frame_000071',
    imported_camera_selected_in_UI=True,sha256_binding_parser_verified=True,original_K_preserved=True,
    official_SIBR_GUI_verified=False,independent_Spark_GUI_verified=False,company00_verified=False,
    quality_accepted=False,proofs=[str(p) for p in paths],proof_sha256={p.name:a.sha(p) for p in paths})
atomic(out/'viewer-ui-review.json',ui)
pub['ui_checked']=True;pub['ui_review']=str(out/'viewer-ui-review.json');pub['updated']=now
atomic(root/'delivery/publication.json',pub)
mid=pub['modelId']
for folder in (a.V,a.S):
    model=a.read(folder/f'model-{mid}.json');model['localViewerUiVerified']=True
    model['company00Verified']=False;model['qualityAccepted']=False
    atomic(folder/f'model-{mid}.json',model)
    atomic(folder/'models'/mid/'viewer-ui-review.json',ui)
    assert a.sha(folder/'models'/mid/'point_cloud.ply')==v['sha256']
    assert a.sha(folder/'exports/winterhat-clock18k-cache124_SH0_018000_with_camera.zip')==package['sha256']
registry_path=a.T/'shuolee/gaussian-deliveries.json';registry=a.read(registry_path)
registry['rounds'][mid]=dict(updated=now,run='winterhat-clock18k-cache124',steps=18000,
    points=74330,initial_points=20908,sha256=v['sha256'],reference_frame='frame_000071',training_frames=8,
    compatibility_passed=True,quality_accepted=False,pc00_opened_this_round=False,
    local_viewer_validation=ui,native_validation=v,
    download_path=str(a.V/'models'/mid/'point_cloud.ply'),sync_path=str(a.S/'models'/mid/'point_cloud.ply'),
    paired_package=pub['packagePath'],download_url='http://127.0.0.1:8870/exports/winterhat-clock18k-cache124_SH0_018000_with_camera.zip',
    viewer_url=pub['viewerUrl'],quality_review=str(out/'visual-review.json'),
    source_pose_quality_accepted=False,snapshot_not_dynamic=True)
registry['updated']=now;registry['latest_diagnostic_round']=mid
atomic(registry_path,registry)
report()
print('WINTERHAT_CLOCK_DELIVERY_LOCAL_UI_VERIFIED_QUALITY_OPEN')
