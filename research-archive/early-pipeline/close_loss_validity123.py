"""Record the observed UI boundary without claiming an unavailable screenshot."""
from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a

out = a.REC / 'demo-vhap-initialization/body-gaussian-smplify-eight-dc-reviewedfit/loss-validity-contract'
stamp = datetime.now(timezone.utc).isoformat()
model = 'local-surface-123-demo-body-smplx-offset-s120000'
digest = 'd151f6d4f47a8ff4fd869ea34d6842f7e9663f421c46397dfba0a75de130a7ad'
assert a.sha(a.V / 'models' / model / 'point_cloud.ply') == digest
for folder in (a.V, a.S):
    result = a.read(folder / 'latest-result.json')
    assert (result['modelId'], result['sha256'], result['points']) == (model, digest, 56884)
    assert result['latestDiagnostic']['round'] == '123-occlusion-loss'
receipt = dict(updated=stamp, browser_tab='36', report_DOM_text_verified=True,
    report_section='demo-occlusion-loss-contract', actual_screenshot_verified=False,
    screenshot_failure='Browser reported zero-width viewport after a click timeout. The section text and36-control table were read successfully.',
    open_in_codex_status='queued', original_model_sha256=digest,
    main_viewer_reloaded_this_turn=False, training_running=False,
    company00_UI_verified=False, WPS_cloud_verified=False, quality_accepted=False)
a.save(out / 'viewer-ui-review.json', receipt)
pub = a.read(out / 'publication.json')
pub.update(actual_ui_verified=False, report_DOM_verified=True, display_queued=True,
    ui_receipt=str(out / 'viewer-ui-review.json'))
a.save(out / 'publication.json', pub)
for path in (a.R / 'queue-state.json', a.C / 'local-research-status.json', a.R / 'protocol.json', a.R / 'evidence-ledger.json'):
    data = a.read(path)
    data['occlusion_loss_contract'] = pub
    if path.name in ('queue-state.json', 'local-research-status.json'):
        data['goal_turn_classification'] = 'progress'
        data['goal_turn_blocked_streak'] = 0
    a.save(path, data)
for folder in (a.R / 'video10-global-initializer', a.V / 'diagnostics/video10-global-initializer', a.S / 'diagnostics/video10-global-initializer'):
    dest = a.lp(folder / 'demo-occlusion-loss-contract')
    for name in ('viewer-ui-review.json', 'environment.json'):
        (dest / name).write_bytes(a.lp(out / name).read_bytes())
print('LOSS_INTERFACE_COMPLETE_REAL_MASKS_AND_VISUAL_UI_STILL_PENDING')
