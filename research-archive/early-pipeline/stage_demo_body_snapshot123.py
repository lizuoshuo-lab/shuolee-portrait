"""Register an actually opened body preview; keep quality acceptance separate."""
import argparse
import json
import shutil
import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now


def main(step):
    run = a.REC/'demo-vhap-initialization/body-gaussian-upstream-offset'
    out = run/'delivery'/f'step-{step:06d}'
    snap = run/'gaussian-run'/f'step-{step:06d}'
    v, p, ui, ctx, result = [a.read(x) for x in (
        out/'validation.json', out/'package-receipt.json', out/'viewer-ui-review.json',
        out/'context.json', snap/'result.json')]
    cameras = a.read(out/'cameras.json')
    visual = a.read(snap/'visual-review.json')
    assert visual['actual_visual_review'] and visual['frames'] == 59 and visual['novel_views'] == 36
    assert visual['world_ply_sha256'] == result['world_ply_sha256']
    assert v['format_passed'] and v['native_rotated_render_parity']['passed'] and v['native_source_ply_parity']['passed']
    assert v['native_replay_uses_serialized_camera']
    assert ui['file_opened_through_UI'] and ui['actual_same_camera_review']
    assert set(ui['modes']) == {'Gaussian', 'points'}
    assert p['verifiedBinding'] and p['cameraCount'] == ui['camera_count'] == len(cameras) == 1
    assert ui['source_sha256'] == ctx['model']['sha256'] == v['sha256'] == a.sha(out/'point_cloud.ply')
    assert ui['serialized_camera_sha256'] == v['serialized_camera_sha256'] == a.sha(out/'cameras.json')
    assert ui['package_sha256'] == p['zip_sha256'] == a.sha(p['path'])
    assert v['source_sha256'] == result['world_ply_sha256'] == a.sha(snap/'world.ply')
    assert ui['observed_count'] == p['points'] == v['points'] == result['points']
    assert ui['step'] == p['step'] == v['step'] == result['step'] == step
    assert ui['reference_frame'] == p['referenceFrameId'] == v['reference_frame'] == cameras[0]['img_name']
    for shot in ui['screenshots']:
        assert a.sha(shot['path']) == shot['sha256']
    assert {s['mode'] for s in ui['screenshots']} == {'Gaussian', 'points'}
    ident = f'local-surface-123-demo-body-smplx-offset-s{step:06d}'
    url = 'diagnostics/video10-global-initializer/index.html#demo-body-gaussian-upstream'
    status = '全身阶段结果：固定第33帧姿态；正面可看，侧面手臂与背面仍有缺损，质量未验收。'
    meta = dict(label=ctx['label'], model=f'models/{ident}/point_cloud.ply', sha256=v['sha256'],
        gaussianCount=v['points'], initialGaussianCount=20908, steps=step, trained=True, fromScratch=True,
        oldCheckpointLoaded=False, representation='named_reference_static_snapshot',
        referenceFrameId=v['reference_frame'], referenceFrameIsPoseSnapshot=True, singlePlyIsNotDynamic=True,
        initialView=0, cameraSource='capture', cameras=cameras, focus=v['viewer_focus'],
        realFramesUsed=59, registeredCameras=1, interpolatedCameras=0, rendererContract=ctx['rendererContract'],
        coordinateSystem=v['coordinate_space'], cameraEstimateAccepted=False, experimentalCameraUse=True,
        cameraNote='固定第33帧。其他训练帧包含不同人体姿态，不冒充静态点云的相机轨迹。',
        filter=ctx['filter'], qualityAccepted=False, plyCompatibilityPassed=True, deliveryAccepted=True,
        diagnosticOnly=True, localAdaptation=True, upstreamCompleteReproduction=False,
        sharedVertexOffset=True, report=url, status=status)
    # Preflight every destination before publishing or updating latest.
    for base in (a.V, a.S):
        dest = base/'models'/ident
        for source, name in ((out/'point_cloud.ply', 'point_cloud.ply'), (p['path'], 'scene.zip')):
            if a.lp(dest/name).exists(): assert a.sha(dest/name) == a.sha(source)
        for path, value in ((dest/'cameras.json', cameras), (dest/'model-context.json', meta), (base/f'model-{ident}.json', meta)):
            if a.lp(path).exists(): assert a.read(path) == value
        js = a.lp(base/'viewer.js').read_text(encoding='utf-8')
        assert json.dumps(ident) in js or js.count('const BUILTINS = [') == 1
    stamp = now()
    for base in (a.V, a.S):
        dest = base/'models'/ident; a.lp(dest).mkdir(parents=True, exist_ok=True)
        for source, name in ((out/'point_cloud.ply', 'point_cloud.ply'), (p['path'], 'scene.zip')):
            if not a.lp(dest/name).exists(): shutil.copy2(a.lp(source), a.lp(dest/name))
        for path, value in ((dest/'cameras.json', cameras), (dest/'model-context.json', meta), (base/f'model-{ident}.json', meta)):
            if not a.lp(path).exists(): a.save(path, value)
        js = base/'viewer.js'; text = a.lp(js).read_text(encoding='utf-8')
        if json.dumps(ident) not in text:
            a.lp(js).write_text(text.replace('const BUILTINS = [', 'const BUILTINS = ['+json.dumps(ident)+', ', 1), encoding='utf-8')
        previous = a.read(base/'latest-result.json')
        a.save(base/'latest-result.json', dict(round=ident, modelId=ident, sha256=v['sha256'], points=v['points'],
            compatibilityPassed=True, qualityAccepted=False, updated=stamp, viewer_url='?model=latest',
            download_url=f'models/{ident}/scene.zip?download=1', report=url, candidateNotBest=True, diagnosticOnly=True,
            status=status, revision=v['sha256'], qualityReviewCompleted=True,
            latestDiagnostic=previous.get('latestDiagnostic')))
    receipt = dict(updated=stamp, modelId=ident, step=step, points=v['points'], reference_frame=v['reference_frame'],
        source_sha256=v['sha256'], package_sha256=p['zip_sha256'], package_path=p['path'],
        local_UI_verified=True, quality_accepted=False, quality_review_complete=True,
        training_continues=True, company00_verified=False, cloud_sync_verified=False,
        url='http://127.0.0.1:8870/?model=latest')
    a.save(out/'staged-delivery.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--step', type=int, required=True)
    main(p.parse_args().step)
