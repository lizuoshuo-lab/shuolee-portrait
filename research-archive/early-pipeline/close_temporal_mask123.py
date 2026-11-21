"""Record actual UI verification and preserve the published historical model."""
from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a

out = a.REC / 'demo-vhap-initialization/body-gaussian-smplify-eight-dc-reviewedfit/sam2-temporal-mask-control'
stamp = datetime.now(timezone.utc).isoformat()
model = 'local-surface-123-demo-body-smplx-offset-s120000'
expected = 'd151f6d4f47a8ff4fd869ea34d6842f7e9663f421c46397dfba0a75de130a7ad'
assert a.sha(a.V / 'models' / model / 'point_cloud.ply') == expected
for folder in (a.V, a.S):
    result = a.read(folder / 'latest-result.json')
    assert (result['modelId'], result['sha256'], result['points']) == (model, expected, 56884)
    assert result['latestDiagnostic']['round'] == '123-temporal-mask'
assert a.sha(a.V / 'viewer.js') == a.sha(a.S / 'viewer.js')
assert a.sha(a.V / 'index.html') == a.sha(a.S / 'index.html')
shots = [a.H / name for name in ['temporal-mask-report-ui.png', 'temporal-mask-evidence-ui.png', 'temporal-mask-viewer-ui.png']]
receipt = dict(updated=stamp, actual_ui_verified=True,
    report_section='demo-temporal-mask-audit', summary_and_native_images_visible=True,
    latestDiagnostic_visible=True, modelId=model, sha256=expected, points=56884,
    file_SH=3, display_SH=0, model_replaced=False, original_inputs_unchanged=True,
    training_started=False, quality_accepted=False, end_to_end_15min_accepted=False,
    company00_UI_verified=False, WPS_cloud_verified=False,
    screenshots={str(p): a.sha(p) for p in shots})
a.save(out / 'viewer-ui-review.json', receipt)
publication = a.read(out / 'publication.json')
publication.update(actual_ui_verified=True, ui_receipt=str(out / 'viewer-ui-review.json'))
a.save(out / 'publication.json', publication)
for path in (a.R / 'queue-state.json', a.C / 'local-research-status.json', a.R / 'protocol.json', a.R / 'evidence-ledger.json'):
    data = a.read(path)
    data['temporal_mask_review'] = publication
    a.save(path, data)
for folder in (a.R / 'video10-global-initializer', a.V / 'diagnostics/video10-global-initializer', a.S / 'diagnostics/video10-global-initializer'):
    dest = a.lp(folder / 'demo-temporal-mask-audit')
    for p in [out / 'viewer-ui-review.json', *shots]:
        (dest / p.name).write_bytes(a.lp(p).read_bytes())
print('TEMPORAL_MASK_REVIEW_PUBLISHED_UI_VERIFIED_ORIGINAL_MODEL_UNCHANGED')
