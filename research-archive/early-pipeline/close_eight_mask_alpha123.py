from datetime import datetime,timezone
import audit_gaussianavatars_params123 as a

out=a.REC/'demo-vhap-initialization/body-gaussian-smplify-eight-dc-reviewedfit/eight-view-surface-audit'
stamp=datetime.now(timezone.utc).isoformat()
pub=a.read(out/'publication.json');pub['actual_ui_verified']=True
pub['ui_receipt']=str(out/'viewer-ui-review.json')
expected='d151f6d4f47a8ff4fd869ea34d6842f7e9663f421c46397dfba0a75de130a7ad'
model_id='local-surface-123-demo-body-smplx-offset-s120000'
assert a.sha(a.V/'models'/model_id/'point_cloud.ply')==expected
for folder in (a.V,a.S):
    d=a.read(folder/'latest-result.json')
    assert (d['modelId'],d['sha256'],d['points'])==(model_id,expected,56884)
receipt=dict(updated=stamp,actual_ui_verified=True,
    modelId=model_id,model_sha256=expected,point_count=56884,
    original_file_SH=3,display_SH=0,display_only=True,
    report_section='demo-eight-mask-alpha-audit',report_summary_and_image_visible=True,
    latestDiagnostic_visible=True,new_model_promoted=False,
    body_and_face_quality_accepted=False,company00_UI_verified=False,WPS_cloud_verified=False,
    screenshots={str(p):a.sha(p) for p in [a.H/'eight-mask-alpha-report-ui.png',a.H/'eight-mask-alpha-viewer-ui.png']})
a.save(out/'viewer-ui-review.json',receipt);a.save(out/'publication.json',pub)
for path in (a.R/'queue-state.json',a.C/'local-research-status.json',a.R/'protocol.json',a.R/'evidence-ledger.json'):
    d=a.read(path);d['eight_mask_alpha_review']=pub;a.save(path,d)
for folder in (a.R/'video10-global-initializer',a.V/'diagnostics/video10-global-initializer',a.S/'diagnostics/video10-global-initializer'):
    dest=a.lp(folder/'demo-eight-mask-alpha-audit')
    for p,name in [(out/'viewer-ui-review.json','viewer-ui-review.json'),(a.H/'eight-mask-alpha-report-ui.png','report-ui.png'),(a.H/'eight-mask-alpha-viewer-ui.png','viewer-ui.png')]:
        (dest/name).write_bytes(a.lp(p).read_bytes())
print('PUBLICATION_UI_VERIFIED_MODEL_UNCHANGED')
