"""Record sampled temporal viewing separately from complete numerical auditing."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-street-surface-temporal-019'

def main():
    audit_path=OUT/'independent-audit/independent-audit.json'
    data=json.loads(audit_path.read_text())
    assert len(data['images'])==240 and len(data['transitions'])==228
    assert len(data['videos'])==8 and all(v['all_frames_decode'] for v in data['videos'])
    sheets=[s for s in data['contact_sheets'] if s['view']=='fixed-union']
    assert len(sheets)==4 and sum(len(s['frames']) for s in sheets)==16
    for s in sheets:
        assert hashlib.sha256((audit_path.parent/s['file']).read_bytes()).hexdigest()==s['sha256']
    report=dict(status='four_fixed_union_contact_sheets_sixteen_frames_reviewed_clarity_not_accepted',
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        independent_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        reviewed_contact_sheets=sheets,
        observations=[
            dict(window='pose-onset',finding='Distant people and their limbs remain soft; before SMPL validity onset the target uses the unchanged deformable fallback. Four sample frames alone cannot certify a smooth handoff.'),
            dict(window='approach',finding='Training frames108/117 still show blurred coat and legs compared with observations. The surface candidate has small local appearance changes, not clear cloth recovery.'),
            dict(window='side-occlusion',finding='Both train and heldout samples retain soft/fragmented target outline and unresolved bag/limb boundaries. Other people and frozen scene also contribute to the region residual.'),
            dict(window='rear-occlusion',finding='Rear coat/legs and bag remain soft; occlusion and disocclusion still require better geometry. Some samples contain very little of target at image boundary.')],
        window_metrics=data['window_summaries'],quality_accepted=False,
        boundaries=['All240PNGs and228adjacent differences audited;8videos fully decoded, not all frames manually viewed.',
            'Manual scope is16fixed-union contact-sheet frames, alongside separate13static heldout-crop review018.',
            'Fixed union includes large background/other people; improvements are not silhouette/3D/flow or isolated flicker accuracy.',
            '10Hz is source cadence; MP4 is a lossy viewing copy; no enhancement.',
            'Models use known observed-trajectory priors; this is not future prediction.'])
    destination=audit_path.parent/'visual-review.json'
    assert not destination.exists()
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    manifest=LAB/'manifests/dynamic-human-work.json'
    m=json.loads(manifest.read_text(encoding='utf8'))
    m['status']='surface_color_and_temporal_comparisons_complete_audited_clarity_still_failed'
    m['updated']=report['reviewed_at']
    m['completed'].append('Source016/color018/surface018 temporal comparison019:240images228adjacent residuals independently audited,8videos decoded,16sampled frames visually reviewed; near garment/limb clarity remains unaccepted')
    m['current_runs']=[]
    m['pending']=[
        'Restore near-person clothing and limb clarity; all completed street candidates still fail',
        'Training-only capacity diagnostic: compare fitting a fixed close training view with shared multi-frame fitting before escalating scene-wide training',
        'After an effective repair, repeat complete heldout and observed-motion/occlusion validation; preserve untouched future validation boundary']
    m['street_surface_temporal_audit']='outputs/dynamic-human-street-surface-temporal-019/independent-audit/independent-audit.json'
    m['street_surface_temporal_visual_review']='outputs/dynamic-human-street-surface-temporal-019/independent-audit/visual-review.json'
    manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=report['status'],frames=16,video_files=8)))

if __name__=='__main__':main()
