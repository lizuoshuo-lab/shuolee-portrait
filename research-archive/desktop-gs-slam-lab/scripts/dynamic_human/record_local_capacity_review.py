"""Preserve finalization failure; record independent completion and actual viewing."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import shutil

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-street-local-capacity-020'

def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    assert not (OUT/'completion.json').exists() and not (OUT/'paired-audit/visual-review.json').exists()
    branches={}
    for name in ('single_view','adjacent_pair'):
        root=OUT/name
        state=json.loads((root/'independent-state-audit.json').read_text())
        images=json.loads((root/'independent-image-audit.json').read_text())
        assert state['updates']==1000 and state['all_poses_voxel_background_other_people_unchanged']
        assert len(images['images'])==105 and len(images['regions'])==435
        branches[name]=dict(state=state,summary=images['summary'],training=images['training'])
    previous=json.loads((OUT/'state.json').read_text())
    assert previous['status']=='failed_needs_inspection' and 'too many indices' in previous['error']
    save(OUT/'finalization-failure.json',dict(original_state=previous,
        cause='The name prior was reused for a scalar regularization tensor; final source-hash lookup attempted prior[key].',
        location='After both1000-update branches, complete training/heldout evaluation, frozen checks and checkpoint saves; before parent completion record.',
        recovery='Independent CPU state and PNG audits verify both saved candidates and unchanged source hash; finalize without retraining or changing candidates.',
        original_script='fit_street_local_capacity.py in this output folder remains the exact executed copy; future runner source corrected.'))
    figures=json.loads((OUT/'paired-audit/figures.json').read_text())['figures']
    seen={(129,4),(131,4),(110,2),(130,4),(140,4),(150,6)}
    reviewed=[f for f in figures if (f['frame'],f['camera']) in seen];assert len(reviewed)==6
    for row in reviewed:
        assert hashlib.sha256((OUT/'paired-audit'/row['file']).read_bytes()).hexdigest()==row['sha256']
    review=dict(status='two_training_and_four_heldout_comparisons_reviewed_overfit_candidates_rejected',
        reviewed_at=datetime.now(timezone.utc).isoformat(),figures=reviewed,
        observations=[
            '129 training view: single-view fit recovers recognizable coat folds, sleeve and bag; adjacent-pair fit also recovers much more clothing structure than the source. Feet/ground and occlusion remain imperfect.',
            '131 training view: single-view candidate produces severe colored/fragmented artifacts; fitting both129and131 recovers visible coat folds and trousers in both supervised views.',
            '130 heldout intermediate view: both candidates retain broken coat texture, bag/limb artifacts; double-view training is not clean temporal interpolation.',
            '110,140,150 heldout views: severe colored and fragmented surface artifacts in both local candidates; reject as scene-model replacements.'],
        branches=branches,quality_accepted=False,
        boundaries=['Manual review covers2training and4heldout paired views, not all13heldout crops or every full image.',
            'All210heldout full PNGs870regions independently verified; static diagnostics have not undergone new continuous-video acceptance.',
            'Local fit demonstrates existing representational capacity for some observed details, not recovered true3D geometry.',
            'Canonical geometry, opacity and SH colors change together; these results do not identify a single causal defect.',
            'Original anonymization preserved; no image enhancement; all existing models retained.'])
    save(OUT/'paired-audit/visual-review.json',review)
    completion=dict(status='both_capacity_diagnostics_completed_independently_audited_local_overfit_rejected',
        source_checkpoint_sha256=branches['single_view']['state']['source_checkpoint_sha256'],
        branches={k:dict(summary=v['summary'],selected_iteration=v['state']['selected_iteration']) for k,v in branches.items()},
        quality_accepted=False,recovery='finalization-failure.json',visual_review='paired-audit/visual-review.json')
    save(OUT/'completion.json',completion)
    save(OUT/'state.json',dict(status=completion['status'],no_running_process=True,quality_accepted=False))
    manifest=LAB/'manifests/dynamic-human-work.json';m=json.loads(manifest.read_text(encoding='utf8'))
    m['status']='local_capacity_diagnostics_complete_single_view_detail_recovered_temporal_generalization_failed'
    m['updated']=review['reviewed_at']
    m['completed'].append('Capacity020: two1000update target-only bounded fits, all210heldout PNGs870regions and actual frozen checkpoints independently audited;2train4heldout pairs viewed. Supervised coat detail recovers but adjacent/unseen times show severe artifacts; neither candidate accepted.')
    m['pending']=[
        'Diagnose geometry/opacity versus SH color contributions to cross-time artifacts with frozen read-only component interventions',
        'Develop a correction that retains detail across training times and improves independent heldout images, then verify continuous motion/occlusion',
        'Restore near-person clothing and limb clarity across the scene; single/two-view overfit is not the requested end state']
    m['current_runs']=[]
    m['street_local_capacity_review']='outputs/dynamic-human-street-local-capacity-020/paired-audit/visual-review.json'
    save(manifest,m)
    print(json.dumps(dict(status=completion['status'],reviewed_views=6,heldout_images=210,heldout_regions=870)))

if __name__=='__main__':main()
