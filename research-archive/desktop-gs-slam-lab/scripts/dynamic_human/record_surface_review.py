"""Record the actual thirteen-view review; do not infer acceptance from scores."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-street-surface-refit-018'

def main():
    destination=OUT/'paired-audit/visual-review.json'
    assert not destination.exists(), 'Preserve completed visual reviews'
    index=json.loads((OUT/'paired-audit/figures.json').read_text())
    for row in index['figures']:
        path=destination.parent/row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    report=dict(status='all13_paired_heldout_target_crops_visually_reviewed_clarity_not_accepted',
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        columns=['original_observation','pose016_source','appearance_only018','bounded_surface_and_appearance018'],
        figures=index['figures'],
        inspected_regions=13,
        observations=[
            dict(frames_cameras=[[60,0],[70,0],[80,0],[80,2]], finding='Distant subject remains soft in all candidates; few source pixels constrain fine detail. Small color/surface changes are not clear garment recovery.'),
            dict(frames_cameras=[[90,0],[90,2],[100,2],[110,2]], finding='Original observations show separate legs, coat folds and bag edges. Reconstruction retains blurred legs and smeared coat/bag; surface branch changes local contrast and coverage but does not recover these structures.'),
            dict(frames_cameras=[[120,2],[130,4]], finding='Near side view still has severely soft and misplaced limb/bag boundaries and patchy clothing. Surface branch has locally altered highlights and thin dark marks, not a faithful sharp surface.'),
            dict(frames_cameras=[[140,4],[140,6],[150,6]], finding='Rear/occluded views retain soft coat and legs; bag becomes a diffuse bright patch. Other people and frozen background also remain imperfect.')],
        quality_accepted=False,
        boundaries=['All13 static paired target crops viewed; not every full image manually reviewed.',
            'Original anonymization preserved; no face reconstruction or sharpening.',
            'Projected boxes include background and other people; no independent silhouette or 3D surface ground truth.',
            'Joint surface changes several attributes, not an isolated scale intervention.',
            'Temporal comparison019 still requires independent numerical and visual review.'])
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    manifest=LAB/'manifests/dynamic-human-work.json'
    data=json.loads(manifest.read_text(encoding='utf8'))
    data['status']='matched_color_surface_refits_completed_audited_clarity_failed_temporal_comparison_running'
    data['updated']=report['reviewed_at']
    data['completed'].append('Surface control018 completed100fullbatch updates; independent105image435region and frozen-state audits passed; all13paired target crops visually reviewed, clothing and limbs remain blurred')
    data['pending']=[
        'Complete source016/color018/surface018 temporal comparison019 and visually assess observed motion/occlusion',
        'Restore near-person clothing and limb clarity; all completed candidates still fail',
        'Separate remaining pose/temporal correspondence error from fixed-topology surface fitting capacity using training-only diagnostics']
    data['current_runs']=[dict(name='Source/color/bounded-surface continuous replay',status='running_read_only_rendering',
        protocol='outputs/dynamic-human-street-surface-temporal-019/protocol.json',
        state='outputs/dynamic-human-street-surface-temporal-019/state.json',quality_claim=False)]
    data['street_surface_refit_audit']='outputs/dynamic-human-street-surface-refit-018/surface-retry-v2/surface_and_appearance/independent-image-audit.json'
    data['street_surface_refit_visual_review']='outputs/dynamic-human-street-surface-refit-018/paired-audit/visual-review.json'
    data['street_current_quality_issue']='Target13 mean PSNR source19.94/color20.10/surface20.30dB; all13 paired crops inspected, near coat folds, bag and limb boundaries remain blurred. No clarity acceptance.'
    manifest.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=report['status'],reviewed=13)))

if __name__=='__main__':main()
