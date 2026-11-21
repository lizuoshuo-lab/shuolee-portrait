"""Record the already observed local model/package UI and preserve old aliases."""
import inspect
from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a
import finalize_indoor_clock_delivery124 as existing
from update_new_person126_state import main as report
from publish_indoor_circle16_124 import atomic


def main():
    text=inspect.getsource(existing.main)
    for old,new,count in (
        ('indoor-zoom-clock18k124','football-half12-paired18k126',3),
        ('training_frames=6','training_frames=12',1),
        ("'frame_000105'","'frame_000026'",1),
        ('INDOOR_CLOCK_DELIVERY_LOCAL_UI_VERIFIED_QUALITY_OPEN','FOOTBALL126_DELIVERY_LOCAL_UI_VERIFIED_QUALITY_OPEN',1),
    ):
        assert text.count(old)==count,old
        text=text.replace(old,new)
    namespace=dict(existing.main.__globals__,report=report)
    exec(compile(text,__file__+':existing-local-UI-delivery-checks','exec'),namespace)
    namespace['main']()
    root=a.REC/'cross-subject-sh0-124/football-half12-paired18k126'
    pub=a.read(root/'delivery/publication.json')
    v=a.read(root/'delivery/step-018000/validation.json')
    facing=a.read(root/'diagnostics/ref26-facing-proxy018000/result.json')
    assert facing['state']=='diagnostic_completed' and facing['optimizer_updates']==0
    now=datetime.now(timezone.utc).isoformat()
    mid=pub['modelId']
    for name,folder in [('local',a.V),('sync_folder',a.S)]:
        path=folder/'latest-result.json';prior=a.read(path)
        backup=root/f'delivery/previous-latest-{name}.json'
        assert not a.lp(backup).exists(),'Preserve the original alias receipt; do not re-finalize.'
        atomic(backup,prior)
        latest=dict(round=mid,modelId=mid,sha256=v['sha256'],points=v['points'],
            compatibilityPassed=True,qualityAccepted=False,updated=now,
            viewer_url='?model=latest&sh=0&framing=fit',
            download_url=f'models/{mid}/scene.zip?download=1',
            report='diagnostics/video10-global-initializer/index.html#new-person126',
            candidateNotBest=True,diagnosticOnly=True,revision=v['sha256'],qualityReviewCompleted=True,
            status='新足球人物12视角、18,000步SH0结果；134,101点。源视角更清晰，未拍背面仍有孔洞杂色，质量未验收。',
            retrained=True,fromScratch=True,latestDiagnostic=prior.get('latestDiagnostic'))
        atomic(path,latest)
    pub.update(updated=now,latest_alias_updated=True,historical_models_preserved=True,
        default_demo_model_unchanged=True,quality_accepted=False)
    atomic(root/'delivery/publication.json',pub)
    report()


if __name__=='__main__':main()
