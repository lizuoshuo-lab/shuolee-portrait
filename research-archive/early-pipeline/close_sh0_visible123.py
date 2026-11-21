"""Record actual UI acceptance of the colour-only SH0 deliverable."""
from datetime import datetime, timezone
import json
import audit_gaussianavatars_params123 as a
from publish_sh0_visible123 import SOURCE,ID,OUT,SID


def main():
    stamp=datetime.now(timezone.utc).isoformat()
    native=a.read(OUT/'native-result.json');fields=a.read(OUT/'field-validation.json')
    assert native['passed'] and all(x['bitwise_equal'] for x in native['views'])
    meta=a.read(a.V/('model-'+ID+'.json'))
    assert a.sha(a.V/meta['model'])==fields['derivative_sha256']==meta['sha256']
    assert a.sha(a.V/'models'/SOURCE/'point_cloud.ply')==fields['source_sha256']
    proof_names=['sh0-delivery-final-ui.png','sh0-delivery-side-ui.png','sh0-delivery-original-ui.png',
                 'sh0-delivery-openfile-points.png','sh0-delivery-openfile-gaussian.png']
    proofs={name:a.sha(a.H/name) for name in proof_names}
    review=dict(updated=stamp,actual_ui_verified=True,viewer_version='2026.11.02.2',
        modelId=ID,model_sha256=meta['sha256'],source_sha256=fields['source_sha256'],
        point_count=56884,original_sh=3,derivative_sh=0,
        native_float_images_bitwise_equal_at_SH0=True,native_views_actually_reviewed=2,
        same_camera_original_SH3_versus_SH0_ui_verified=True,
        open_file_zip_verified=True,imported_camera_count=1,imported_camera='frame_000033',
        model_camera_SHA_binding_verified=True,gaussian_and_points_actually_viewed=True,
        query_SH_mode_verified=True,fit_framing_verified=True,
        existing_report_refreshed_direct_links_verified=True,tab15_untouched=True,
        findings='SH0 removes the strong chromatic speckles. The same surface still has shoulder/arm gaps, blurry hands and incompletely constrained back geometry. Colour improvement retained; geometry not accepted.',
        training_updates=0,quality_accepted=False,company00_verified=False,wps_cloud_verified=False,
        screenshot_sha256=proofs)
    a.save(OUT/'viewer-ui-review.json',review)
    note=a.read(OUT/'publication.json');note.update(updated=stamp,state='sh0_colour_delivery_ui_verified',
        actual_ui_verified=True,latest_entry_uses_SH0_derivative=True,old_SH3_preserved=True,
        viewerUrl='http://127.0.0.1:8870/?model='+ID+'&sh=0&framing=fit')
    a.save(OUT/'publication.json',note)
    for folder in (a.V,a.S):
        latest=a.read(folder/'latest-result.json')
        assert latest['modelId']==SOURCE and latest['sha256']==fields['source_sha256']
        a.save(OUT/('latest-before-'+('source' if folder==a.V else 'sync')+'.json'),latest)
        latest.update(round=ID,modelId=ID,sha256=meta['sha256'],revision=meta['sha256'],updated=stamp,
            points=56884,viewer_url='?model=latest&sh=0&framing=fit',download_url='models/'+ID+'/scene.zip?download=1',
            report=note['reportUrl'],status=meta['status'],qualityAccepted=False,compatibilityPassed=True,
            colourCorrectionPublished=True,colourDerivativeOf=SOURCE,retrained=False)
        latest['latestDiagnostic'].update(updated=stamp,status=note['state'])
        a.save(folder/'latest-result.json',latest)
        a.save(folder/'models'/ID/'viewer-ui-review.json',review)
        dest=a.lp(folder/'diagnostics/video10-global-initializer'/SID)
        for name in proof_names:(dest/name).write_bytes(a.lp(a.H/name).read_bytes())
        a.save(folder/'diagnostics/video10-global-initializer'/SID/'viewer-ui-review.json',review)
    for path in (a.R/'queue-state.json',a.R/'evidence-ledger.json',a.C/'local-research-status.json'):
        obj=a.read(path);obj.update(updated=stamp,sh0_visible_delivery=note,current_stage=note['state'],
            training_running=False,own_gpu_task_running=False)
        a.save(path,obj)
    for path in (a.R/'video10-global-initializer/index.html',a.V/'diagnostics/video10-global-initializer/index.html',a.S/'diagnostics/video10-global-initializer/index.html'):
        text=a.lp(path).read_text(encoding='utf-8')
        anchor='<p>后续方向：固定SH0，使用这份已改善的模型作显示基线'
        assert text.count(anchor)==1
        added=f'<p>实际查看器验收：原SH3/SH0同相机切换、独立SH0文件、ZIP打开后自动载入1个SHA绑定相机，以及Gaussian/points两种显示均已检查。默认“最新模型”现指向这份SH0颜色修复版；原SH3仍保留，几何质量状态未变。<a href="{SID}/viewer-ui-review.json">查看核验记录</a>。</p>'
        text=text.replace(anchor,added+anchor,1)
        a.lp(path).write_text(text,encoding='utf-8')
    assert all(a.sha(a.V/name)==a.sha(a.S/name) for name in ('viewer.js','viewer.css','index.html','latest-result.json'))
    a.save(OUT/'closeout.json',dict(updated=stamp,state=note['state'],latestId=ID,sourcePreserved=True,
        local_and_sync_folder_equal=True,training_updates=0,quality_accepted=False))
    print(json.dumps(dict(state=note['state'],viewer=note['viewerUrl'],latest=ID),ensure_ascii=False))


if __name__=='__main__':main()
