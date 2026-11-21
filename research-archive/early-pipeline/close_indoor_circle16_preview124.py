"""Reconcile completed local preview with the preserved failed strict-save gate."""
from datetime import datetime,timezone
import shutil
import audit_gaussianavatars_params123 as a
from publish_indoor_circle16_124 import atomic

def main():
    root=a.REC/'cross-subject-sh0-124/indoor-circle16-paired18k124'
    out=root/'delivery/step-018000';now=datetime.now(timezone.utc).isoformat()
    publication=a.read(root/'delivery/publication.json');assert publication['ui_checked']
    proof=root/'ui-review/delivered-report.png';assert a.lp(proof).is_file()
    r=a.REC/'cross-subject-sh0-124/indoor-circle16-pose124/report-publication.json'
    report=a.read(r);report.update(report_ui_verified=True,ui_proof=str(proof),ui_proof_sha256=a.sha(proof));atomic(r,report)
    receipt=dict(updated=now,stage='indoor_circle16_delivered_quality_open',publication=publication,
        policy=a.read(a.C/'view-selection-policy124.json'),
        train_exit_code=a.read(root/'process-exit.json')['exit_code'],original_strict_save_failed=True,
        geometry_and_camera_verified=True,native_strict_pixel_equivalence=False,
        one_alpha_cutoff_exception_diagnosed=True,read_only_export=True,additional_training_updates=0,
        local_UI_verified=True,company00_deployed=False,quality_accepted=False,
        remaining='Mask/geometry holes and pose residuals; same frozen profile across all four subjects; cold video-to-export15min not accepted.')
    atomic(root/'delivery/final-receipt.json',receipt)
    p=a.REC/'cross-subject-sh0-124/four-subject-delivery-status.json';status=a.read(p)
    status.update(updated=now,view_selection_policy124=receipt['policy'])
    status.setdefault('subsequent_deliveries',{})[publication['modelId']]=receipt
    status['next_action']='Indoor full-circle16delivered as a diagnostic preview; no more retries of this candidate. Verify sofa capture arc/quality roster before frozen ROI2x+paired-background+SH0+18k-clock transfer. Half-circle8/full-circle16, independent heldouts remain protected. Quality and15min objective remain open.'
    atomic(p,status)
    # Keep exact source versions alongside their numerical result for replay.
    src=a.H if hasattr(a,'H') else None
    from pathlib import Path
    here=Path(__file__).parent
    archive=a.lp(root/'export-roundoff-certificate124-v2/source');archive.mkdir(exist_ok=True)
    for n in ['native_export_roundoff124.py','certify_indoor_export_roundoff124.py','finish_indoor_circle16_preview124.py']:
        shutil.copyfile(here/n,archive/n)
    print('INDOOR16_RECEIPT_RECONCILED',publication['modelId'])

if __name__=='__main__':main()
