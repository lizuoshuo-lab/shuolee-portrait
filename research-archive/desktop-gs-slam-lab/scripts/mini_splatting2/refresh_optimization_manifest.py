"""Refresh factual long-task state without replacing frozen evidence."""
import datetime
import json
from pathlib import Path

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'

def main():
    path=LAB/'manifests/mini-splatting2-optimization.json';m=json.loads(path.read_text(encoding='utf-8'))
    m['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    m['scope']='Same work video; both fixed-topology post-training controls and full native Mini-Splatting2 training. Development holdouts share SfM and are repeatedly used. No final independent test.'
    m['native_queues']={name:json.loads((ROOT/name).read_text()) for name in ['native-repeat-queue.json','native-mip-queue.json','native-depthmask-queue.json','native-depth-stage-queue.json','native-depth-repeat-queue.json','depth-repeat-acceptance-queue.json','native-spread-queue.json','native-spread-acceptance-queue.json'] if (ROOT/name).exists()}
    m['native_mip_runtime']=json.loads((ROOT/'native-mip-runtime/manifest.json').read_text())
    validation=ROOT/'native-mip-runtime/validation.json'
    if validation.exists():
        result=json.loads(validation.read_text());m['native_mip_runtime']['validation_status']=result['status']
        if result['status']=='passed':m['native_mip_runtime']['status']='validated';m['native_mip_runtime']['validation_evidence']=str(validation)
        (ROOT/'native-mip-runtime/manifest.json').write_text(json.dumps(m['native_mip_runtime'],indent=2))
    if (ROOT/'native-motion-summary.json').exists():
        s=json.loads((ROOT/'native-motion-summary.json').read_text());m['motion_repeats']=dict(status=s['status'],seeds=s['seeds'],aggregate=s['aggregate'])
    m['confirmed_user_method']='Mip-Splatting'
    m['next']=['finish matched guided repeats for both pre-existing depth weights',
        'evaluate new guided seeds on fixed extra video frames and sparse geometry; summarize paired seed differences',
        'investigate residual depth spread and wide-pitch artifacts; gated Difix changes were negligible and not accepted for3Dtraining',
        'independent real distance/pitch coverage validation; same-video frames cannot establish extrapolation']
    m['active_job']=[v.get('current') or v['status'] for v in m['native_queues'].values() if v['status']!='complete']
    m['local_viewer_models']=len(list((ROOT/'viewer').glob('model-*.json')))
    if (ROOT/'native-mip-summary.json').exists():m['native_mip_report']='http://127.0.0.1:8820/native-mip.html'
    extra=ROOT/'additional-frame-evaluation-001/summary.json'
    if extra.exists():m['additional_video_frames']=dict(summary=str(extra),protocol=str(extra.parent/'frozen-protocol.json'),frames=24,scope='same-video unused frames; fixed-map pose-only localization; temporal interpolation, not independent capture')
    package=ROOT/'native-package-receipt.json'
    if package.exists():m['native_package']=json.loads(package.read_text(encoding='utf-8-sig'))
    package=ROOT/'depth-package-receipt.json'
    if package.exists():m['depth_package']=json.loads(package.read_text(encoding='utf-8-sig'))
    if (ROOT/'native-depth-summary.json').exists():
        m['depth_guidance_stage']=dict(summary=str(ROOT/'native-depth-summary.json'),
            report='http://127.0.0.1:8820/native-depth.html',scope='frozen two-weight seed0 snapshot plus baseline3seeds; later guided repeats separate')
    for key,relative in [('native_depth_validation','native-depth-runtime/validation.json'),
                         ('depth_evaluation_queue','native-depth-evaluation-queue.json'),
                         ('viewer_capture_frame_fix','viewer-capture-frame-fix.json'),
                         ('matched_native_repeats','native-depth-matched-repeats.json'),
                         ('head_detail_depth003','head-detail-depth003-001.json'),
                         ('head_detail_depth010','head-detail-depth010-001.json')]:
        evidence=ROOT/relative
        if evidence.exists():m[key]=json.loads(evidence.read_text(encoding='utf-8-sig'))
    if (ROOT/'head-epipolar-audit-001/summary.json').exists():
        m['head_epipolar_audit']=dict(path=str(ROOT/'head-epipolar-audit-001/summary.json'),
            scope='image-only tracks vs fixed camera epipolar geometry; not proof of rigid or nonrigid motion')
    if (ROOT/'native-depth-repeat-review.json').exists():
        m['matched_repeat_review']=dict(path=str(ROOT/'native-depth-repeat-review.json'),
            report='http://127.0.0.1:8820/native-repeats.html',scope='3 matched seeds; same-video development comparison')
        m['next']=['finish native center-depth variance control and2strengths; no acceptance before real-image/geometry validation',
            'evaluate head fidelity, common-mask ray spread, extra24frames and extrapolation side by side',
            'repair training-only head attention drafts before considering a separate region-weighted RGB ablation',
            'independent real distance/pitch coverage validation; current evidence remains one-video development']
    if (ROOT/'repeat-audit-package-receipt.json').exists():
        m['repeat_audit_package']=json.loads((ROOT/'repeat-audit-package-receipt.json').read_text(encoding='utf-8-sig'))
    if (ROOT/'native-spread-runtime/manifest.json').exists():
        m['native_spread_stage']=dict(runtime=str(ROOT/'native-spread-runtime/manifest.json'),
            validation=str(ROOT/'native-spread-runtime/validation.json'),
            training_queue=str(ROOT/'native-spread-queue.json'),
            evaluation_queue=str(ROOT/'native-spread-acceptance-queue.json'),
            accepted=False,scope='Seed0 controlled auxiliary center-depth variance experiment; not a 2DGS reproduction')
    if (ROOT/'native-spread-candidate-review.json').exists():
        m['native_spread_stage']['review']=str(ROOT/'native-spread-candidate-review.json')
    path.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(active_job=m['active_job'],models=m['local_viewer_models'],updated=m['updated_at'])))

if __name__=='__main__':main()
