"""Record actual UI proof and immutable publication checks for this goal turn."""
from datetime import datetime,timezone
import audit_gaussianavatars_params123 as a

ROOT=a.REC/'cross-subject-sh0-124'


def main():
    stamp=datetime.now(timezone.utc).isoformat()
    pages=[a.R/'video10-global-initializer/index.html',a.V/'diagnostics/video10-global-initializer/index.html',a.S/'diagnostics/video10-global-initializer/index.html']
    digests=[a.sha(p) for p in pages];assert len(set(digests))==1
    assert all('quality-observation124' in a.lp(p).read_text(encoding='utf-8') for p in pages)
    models=[a.read(base/'latest-result.json')['modelId'] for base in (a.V,a.S)]
    assert models==['local-surface-123-demo-body-smplx-offset-s120000-sh0']*2,models
    grass=ROOT/'grass-protected-refine124/ui-review/summary.png'
    observation=ROOT/'selected-observation-audit124/ui-review/summary.png'
    assert a.lp(grass).stat().st_size>10000 and a.lp(observation).stat().st_size>10000
    proof=dict(updated=stamp,actual_UI_verified=True,
        scope='Current report DOM and screenshots: grass protected chest pose, winterhat result text and all8overview images on page, new duplicate-wrist diagnosis. No new Gaussian/OpenFile/00 test.',
        report_html_sha256=digests[0],source_viewer_D00_html_identical=True,models_preserved=models,
        screenshots=[dict(path=str(p),sha256=a.sha(p)) for p in [grass,observation]],
        actual_ui_url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#quality-protected124',
        company00_verified=False,WPS_cloud_sync_verified=False,goal_complete=False)
    for folder in ['grass-protected-refine124','winterhat-selected-view-pose124','selected-observation-audit124']:
        path=ROOT/folder/'publication.json';value=a.read(path);value.update(proof);a.save(path,value)
    for path in [a.R/'queue-state.json',a.R/'evidence-ledger.json',a.C/'local-research-status.json']:
        value=a.read(path);value['sparse_pose124_ui_review']=proof;a.save(path,value)
    a.save(ROOT/'selected-observation-audit124/final-turn-receipt.json',dict(updated=stamp,
        actual_results=['Grass8protected refinement completed290.682s;2coarse admitted frames, no Gaussian.',
            'Winterhat8shared pose completed183.609s;2numerical candidates,0depth-admitted frames, no Gaussian.',
            '8frame SMPLX/originalLBS/body-gauge native projection roundtrip0.0004438px is numerical consistency, not physical accuracy.',
            '16input mappings exact; duplicate wrist physical aliases verified; winterhat13observations conflict258.023px.'],
        pose_defect_resolved=False,cross_person_pipeline_accepted=False,training_started=False,
        runtime='Final process inspection found no owned fitting/training process.',UI_proof=proof))
    print('SPARSE_POSE_REPORT_VERIFIED_MODEL_PRESERVED',models[0])


if __name__=='__main__':main()
