"""Commit the already inspected indoor candidate; leave quality and cold timing open."""
from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a
from update_indoor_clock_state124 import atomic
from publish_indoor_clock124 import main as report


def main():
    root=a.REC/'cross-subject-sh0-124/indoor-zoom-clock18k124'
    out=root/'delivery/step-018000';proof=root/'ui-review'
    pub=a.read(root/'delivery/publication.json');v=a.read(out/'validation.json')
    package=a.read(out/'package-result.json');now=datetime.now(timezone.utc).isoformat()
    assert a.read(root/'process-exit.json')['exit_code']==0
    assert a.read(root/'gaussian-run/step-018000/visual-review.json')['actual_visual_review']
    assert a.read(out/'visual-review.json')['actual_visual_review']
    assert v['points']==a.read(root/'gaussian-run/step-018000/result.json')['points']
    assert a.sha(out/'point_cloud.ply')==v['sha256'] and a.sha(out/'scene.zip')==package['sha256']
    assert v['reference_frame']=='frame_000105'
    paths=[proof/name for name in ('preset.png','openfile-gaussian.png','openfile-points.png','final-viewer.png')]
    assert all(a.lp(p).exists() for p in paths)
    ui=dict(updated=now,actual_ui_verified=True,preset_loaded=True,open_file_zip=True,
        gaussian_mode=True,point_mode=True,points=v['points'],camera_count=1,reference_frame=v['reference_frame'],
        imported_camera_selected_in_UI=True,sha256_binding_parser_verified=True,derived_full_K_preserved=True,
        camera_scope='Person ROI enlarged2x; not original whole-frame K. Pixel mapping included in ZIP.',
        official_SIBR_GUI_verified=False,independent_Spark_GUI_verified=False,company00_verified=False,
        quality_accepted=False,proofs=[str(p) for p in paths],proof_sha256={p.name:a.sha(p) for p in paths})
    atomic(out/'viewer-ui-review.json',ui)
    pub.update(ui_checked=True,ui_review=str(out/'viewer-ui-review.json'),updated=now)
    atomic(root/'delivery/publication.json',pub)
    mid=pub['modelId']
    for folder in (a.V,a.S):
        model=a.read(folder/f'model-{mid}.json');model.update(localViewerUiVerified=True,company00Verified=False,qualityAccepted=False)
        atomic(folder/f'model-{mid}.json',model);atomic(folder/'models'/mid/'viewer-ui-review.json',ui)
        assert a.sha(folder/'models'/mid/'point_cloud.ply')==v['sha256']
        assert a.sha(folder/'exports/indoor-zoom-clock18k124_SH0_018000_with_camera.zip')==package['sha256']
    registry_path=a.T/'shuolee/gaussian-deliveries.json';registry=a.read(registry_path)
    registry['rounds'][mid]=dict(updated=now,run=root.name,steps=18000,points=v['points'],initial_points=20908,
        sha256=v['sha256'],reference_frame=v['reference_frame'],training_frames=6,
        compatibility_passed=True,quality_accepted=False,pc00_opened_this_round=False,
        local_viewer_validation=ui,native_validation=v,
        download_path=str(a.V/'models'/mid/'point_cloud.ply'),sync_path=str(a.S/'models'/mid/'point_cloud.ply'),
        paired_package=pub['packagePath'],download_url='http://127.0.0.1:8870/exports/indoor-zoom-clock18k124_SH0_018000_with_camera.zip',
        viewer_url=pub['viewerUrl'],quality_review=str(out/'visual-review.json'),
        source_pose_quality_accepted=False,snapshot_not_dynamic=True)
    registry['updated']=now;registry['latest_diagnostic_round']=mid;atomic(registry_path,registry)
    report()
    print('INDOOR_CLOCK_DELIVERY_LOCAL_UI_VERIFIED_QUALITY_OPEN')


if __name__=='__main__':main()
