"""Read saved wall-clock observations; no GPU, fitting, or interpolation."""
from datetime import datetime, timezone
import statistics
import audit_gaussianavatars_params123 as a

root=a.REC/'cross-subject-sh0-124'
names=['grass-selected-sh0-pipeline124','winterhat-selected-sh0-pipeline124',
       'sofa-selected-sh0-pipeline124','indoor-selected-sh0-pipeline124','winterhat-clock18k-cache124']
result=[]
for name in names:
    out=root/name
    run=a.read(out/'gaussian-run/result.json')
    assert a.read(out/'progress.json')['state']=='bounded_control_completed'
    log=[__import__('json').loads(s) for s in a.lp(out/'gaussian-run/loss.jsonl').read_text().splitlines()]
    dt=[];excluded=[]
    for prev,cur in zip(log,log[1:]):
        if cur['step']-prev['step']!=100:continue
        ms=(cur['seconds']-prev['seconds'])*10
        if prev['step'] in (1000,10000):
            excluded.append(dict(from_step=prev['step'],to_step=cur['step'],seconds=(cur['seconds']-prev['seconds'])))
        else:dt.append(ms)
    req=a.read(out/'review-1000-request.json');dec=a.read(out/'review-1000-decision.json')
    review_wait=(datetime.fromisoformat(dec['updated'])-datetime.fromisoformat(req['updated'])).total_seconds()
    final=a.read(out/'gaussian-run/step-018000/result.json')
    result.append(dict(subject=name,reported_total_seconds=run['seconds'],final_points=final['points'],
        review_request_to_decision_seconds=review_wait,
        total_minus_external_review_wait_seconds=run['seconds']-review_wait,
        steady_100step_interval_median_ms=statistics.median(dt),
        steady_100step_interval_min_ms=min(dt),steady_100step_interval_max_ms=max(dt),
        sampled_intervals=len(dt),excluded_snapshot_gate_intervals=excluded,
        input_fit_reused=True,cold_video_start=False,
        scope='Saved adapter wallclock includes initialization, training, safety checks, snapshots; request-to-decision subtraction is a labelled estimate, not a measured fresh end-to-end execution. Steady intervals include optimizer and any density events but exclude intervals after1000/10000 snapshots.',
        log_sha256=a.sha(out/'gaussian-run/loss.jsonl'),result_sha256=a.sha(out/'gaussian-run/result.json')))
a.save(root/'measured-stage-timing124.json',dict(updated=datetime.now(timezone.utc).isoformat(),results=result,
    quality_accepted=False,end_to_end_15min_accepted=False,
    causal_limit='New candidate changed the clock profile as well as exact cache; only the independent zero-update paired cache benchmark isolates cache speed. Do not call full-run speed difference a cache-only effect.'))
for r in result:print(r['subject'],round(r['reported_total_seconds'],2),round(r['total_minus_external_review_wait_seconds'],2),round(r['steady_100step_interval_median_ms'],2),r['final_points'])
