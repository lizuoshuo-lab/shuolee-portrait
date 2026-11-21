"""Record the actual current-page verification; not a renderer release gate."""
from datetime import datetime,timezone
import audit_gaussianavatars_params123 as a


def main():
    now=datetime.now(timezone.utc).isoformat()
    out=a.REC/'demo-vhap-initialization/body-gaussian-upstream-offset/gaussian-mesh-guidance-audit/mesh-initialization-boundary'
    proof=a.H/'teacher-boundary-ui.png';assert a.lp(proof).is_file()
    documents=[a.R/'video10-global-initializer/index.html',
        a.V/'diagnostics/video10-global-initializer/index.html',a.S/'diagnostics/video10-global-initializer/index.html']
    assert len({a.sha(p) for p in documents})==1
    for folder in (a.V,a.S):
        latest=a.read(folder/'latest-result.json')
        assert latest['modelId']=='local-surface-123-demo-body-smplx-offset-s120000'
        assert latest['latestDiagnostic']['round']=='123-teacher-boundary'
    result=dict(updated=now,report_tab='36',viewer_tab='38',
        report_section='demo-gaussian-teacher-boundary',actual_report_DOM_verified=True,
        actual_viewer_latestDiagnostic_verified=True,current_model='local-surface-123-demo-body-smplx-offset-s120000',
        model_selector_and_title_verified=True,current_view_mode='points',model_replaced=False,
        screenshot=str(proof),screenshot_sha256=a.sha(proof),tab15_untouched=True,
        existing_viewer_notice='Preserve current model / Failed to fetch was already present; this verifies new diagnostic text, not a new import or viewer-wide regression test.',
        source_and_D00_document_sha256=a.sha(documents[0]),new_gaussian_or_mesh_promoted=False,
        company00_GUI_verified=False,WPS_cloud_sync_verified=False,quality_accepted=False)
    a.save(out/'viewer-ui-review.json',result)
    publication=a.read(out/'publication.json');publication.update(actual_ui_verified=True,ui_review=str(out/'viewer-ui-review.json'));a.save(out/'publication.json',publication)
    for path in (a.R/'queue-state.json',a.C/'local-research-status.json',a.R/'evidence-ledger.json'):
        data=a.read(path);data['teacher_boundary_diagnosis'].update(actual_ui_verified=True,ui_review=str(out/'viewer-ui-review.json'))
        a.save(path,data)
    print('BOUNDARY_UI_VERIFIED_PREVIEW_PRESERVED')


if __name__=='__main__':main()
