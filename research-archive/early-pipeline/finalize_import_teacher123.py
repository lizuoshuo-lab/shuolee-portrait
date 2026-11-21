"""Record actual local report/diagnostic UI verification; no model publication."""
from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a
from publish_import_teacher123 import OUT

now=datetime.now(timezone.utc).isoformat()
proof=a.H/'import-teacher-ui.png'
assert a.lp(proof).exists()
ui=dict(updated=now,report_tab=36,viewer_tab=38,actual_report_DOM_verified=True,
        actual_report_screenshot_verified=True,viewer_latestDiagnostic_verified=True,
        screenshot=str(proof),screenshot_sha256=a.sha(proof),
        scope='Local report text, figures and current viewer diagnostic only; no new model import or full viewer release.',
        current_model='local-surface-123-demo-body-smplx-offset-s120000',model_promoted=False,
        company00_deployment_verified=False,WPS_sync_verified=False,tab15_untouched=True,quality_accepted=False)
a.save(OUT/'viewer-ui-review.json',ui)
publication=a.read(OUT/'publication.json');publication.update(actual_ui_verified=True,ui_review=ui);a.save(OUT/'publication.json',publication)
for path in (a.R/'queue-state.json',a.C/'local-research-status.json',a.R/'evidence-ledger.json'):
    value=a.read(path);value['original_import_teacher_contract'].update(actual_ui_verified=True,ui_review=str(OUT/'viewer-ui-review.json'))
    a.save(path,value)
print('IMPORT_TEACHER_LOCAL_UI_VERIFIED')
