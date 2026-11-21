"""Refresh current state without rewriting frozen protocols or prior result evidence."""
import datetime
import json
from pathlib import Path
import time

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def dump(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def main():
    names=['native-resolution-cache-queue.json','native-resolution-cache-publication-queue.json','native-resolution-followup-queue.json','native-late-queue.json','native-late-publication-queue.json','native-training-fit-queue.json']
    queues={name:read(ROOT/name) for name in names if (ROOT/name).exists()}
    training=queues[names[0]];current=training.get('current');telemetry=None
    if current:
        log=ROOT/current/'model/memory-usage.jsonl'
        if log.exists() and log.stat().st_size:telemetry=json.loads(log.read_text().splitlines()[-1])
    manifest=LAB/'manifests/mini-splatting2-optimization.json';m=read(manifest)
    m.update(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),confirmed_user_method='Mip-Splatting',
        local_viewer_models=len(list((ROOT/'viewer').glob('model-*.json'))))
    m.setdefault('native_queues',{}).update(queues)
    m['active_job']=[q.get('current',name) for name,q in queues.items() if q['status'] not in ['complete','failed','complete_pending_browser_and_visual_review']]
    m['resolution_stage'].update(status=training['status'],accepted=False,latest_telemetry=telemetry,
        smoke=training.get('smoke'),primary_publication=queues.get(names[1],{}).get('status'),
        additional24=queues.get(names[2],{}).get('status'),
        sessions=dict(training=89155,publication=45577,followup=72697),
        review='native-resolution-final-review-001.json',
        qq_progress=dict(recipient='小润',sent_local_time='2026-10-19 04:38',visible_sent_bubble_verified=True,
            scope='17model package delivered and opened on computer00 port8860, photo comparison page;5100step smoke passed,18k training continues'))
    receipt=ROOT/'resolution-package-receipt.json'
    if receipt.exists():
        delivery=read(receipt)
        m['resolution_stage']['delivery']={key:delivery.get(key) for key in ['status','package','remote_path','remote_url','remote_report','remote_proof']}
        m['current_remote_delivery']=m['resolution_stage']['delivery']
        m['latest_report_url']='http://127.0.0.1:8820/native-resolution.html'
    late=queues.get('native-late-queue.json',{})
    if late:
        log=ROOT/late.get('current','native-late-full-18000-s0')/'model/memory-usage.jsonl'
        latest=json.loads(log.read_text().splitlines()[-1]) if log.exists() and log.stat().st_size else None
        m['late_topology_stage']=dict(status=late['status'],session=39397,publication_session=85947,fit_diagnostic_session=22618,current=late.get('current'),latest_telemetry=latest,
            protocol=str(ROOT/'native-late-protocol-001.json'),accepted=False,loss_terms_unchanged=True)
        review_path=ROOT/'native-late-final-review-001.json'
        if review_path.exists():
            review=read(review_path)
            m['late_topology_stage'].update(review=review_path.name,decision=review['decision'],
                primary_delta=review['primary_delta'],additional24_delta=review['additional24_delta'],
                training51_delta=review['training51_delta'],cost=review['cost'],
                registered_in_viewer=False,remote_snapshot_unchanged=True,
                qq_progress=dict(recipient='小润',sent_local_time='2026-10-19 05:07',visible_sent_bubble_verified=True,
                    scope='18k complete, limited head improvement and farther-view artifacts; keep17model remote snapshot; head weighting not trained'))
            m['latest_report_url']='http://127.0.0.1:8820/native-late.html'
    m['head_attention_preparation']=dict(status='CPU validation only; not integrated or trained',
        evidence=str(ROOT/'head-attention-preparation-001/validation.json'),used_in_resolution_experiment=False)
    failed=ROOT/'native-head-repeatability-validation-002/1-21-baseline.json'
    if failed.exists() and not read(failed)['sane']:
        m['head_attention_preparation'].update(status='Native validation held: original unmodified backward exceeds baseline repeatability gate',
            first_strict_gate='native-head-gradient-validation-001/partial.json',
            followup_gate=str(failed),head_loss_integrated=False,head_loss_trained=False,
            original_baseline_scale_relative_variation=read(failed)['envelope']['scaling']['relative_l2'])
    diagnosis=ROOT/'native-gradient-path-diagnosis-001'
    if diagnosis.exists():
        summary=diagnosis/'summary.json';partial=diagnosis/'partial.json'
        data=read(summary) if summary.exists() else read(partial) if partial.exists() else {}
        m['native_gradient_diagnosis']=dict(status=data.get('status','running'),session=66304,
            protocol=str(diagnosis/'protocol.json'),completed_modes=list(data.get('results',{})),
            model_updates=False,detached_gradient_approved_for_training=False)
        interpretation=diagnosis/'interpretation.json'
        if interpretation.exists():
            m['native_gradient_diagnosis'].update(status='reviewed',interpretation=str(interpretation),
                decision=read(interpretation)['decision'])
        if not summary.exists():m['active_job'].append('native-gradient-path-diagnosis-001')
    m['next']=['Complete exact neutral RGB/image-gradient invariants on remaining native fixtures; report existing Gaussian-gradient repeatability separately. Preserve failed strict gates.',
        'Keep late densification as an unaccepted local experiment: minor head gain but farther-view detached streaks; default model unchanged.',
        'Only integrate training-head weighting after native validation; compare on original schedule rather than combining interventions.',
        'Resolution snapshot already delivered and opened on computer00 port8860; preserve this frozen17model snapshot.',
        'Repeat useful candidates, then verify independent distance/pitch observations; no geometry acceptance without GT.']
    dump(manifest,m)
    runtime=ROOT/'native-resolution-cache-runtime/manifest.json';r=read(runtime)
    if training.get('smoke'):
        r.update(status='reference_cache_and_training_transition_validated',transition_smoke=training['smoke']['name']);dump(runtime,r)
    dump(ROOT/'source-resolution-rectified-002/training-status.json',dict(status=training['status'],current=current,
        current_queue=str(ROOT/names[0]),frozen_inventory_preserved=True,
        reference_cache_and_transition_passed=bool(training.get('smoke')),legacy_v1_evaluation_excluded=True))
    input_review=ROOT/'resolution-input-detail-001/visual-review.json'
    if not input_review.exists():
        dump(input_review,dict(status='reviewed',created=time.time(),images=['review-1.png','review-2.png'],frames=8,
            findings=['Original photos contain visible face, glasses and hair details.',
                'Area reduction and enlargement soften small edges; bicubic is closer in this resampling diagnostic.',
                'This does not establish reconstruction-error cause or a3Dquality bound.'],training_changed=False))
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    dump(ROOT/f'progress-checkpoint-{stamp}.json',dict(created=m['updated_at'],goal='active',resolution=m['resolution_stage'],
        queues=queues,viewer_models=m['local_viewer_models'],head_attention=m['head_attention_preparation'],late_topology=m.get('late_topology_stage'),
        native_gradient_diagnosis=m.get('native_gradient_diagnosis'),
        original_models_preserved=True,unrelated_asr_pid_18064_untouched=True,next=m['next']))
    print(json.dumps(dict(status=training['status'],current=current,iteration=telemetry['iteration'] if telemetry else None,
        viewer_models=m['local_viewer_models']),ensure_ascii=False))


if __name__=='__main__':main()
