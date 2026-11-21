"""Preserve prior progress and append the current head-weighting experiment status."""
import json
import time
import datetime
from pathlib import Path

S=Path(__file__).parent;LAB=S.parents[1];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'


def main():
    target=LAB/'manifests/mini-splatting2-optimization.json'
    state=json.loads(target.read_text(encoding='utf-8'))
    queue=json.loads((ROOT/'native-head-queue.json').read_text())
    state['updated']=time.time()
    state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    state['confirmed_method']='Mip-Splatting'
    state['confirmed_user_method']='Mip-Splatting'
    state['head_weighting_stage']=dict(queue=queue, runtime='native-head-runtime-001/manifest.json',
        interface_validation='native-head-interface-validation-003/summary.json',
        neutral_loss_and_image_gradient_exact=True, native_parameter_repeatability_unresolved=True,
        original_failed_gates_preserved=True, head_weight=2, topology_schedule='original',
        baseline='native-resolution-cache-full-18000-s0', accepted=False)
    state.setdefault('native_queues',{})['native-head-queue.json']=queue
    publication_path=ROOT/'native-head-publication-queue.json'
    if publication_path.exists():
        publication=json.loads(publication_path.read_text())
        state['native_queues']['native-head-publication-queue.json']=publication
        state['head_weighting_stage']['publication_status']=publication['status']
    state['head_weighting_stage']['sessions']=dict(training=97808,publication=14103)
    state['head_weighting_stage']['qq_progress']=dict(recipient='小润',sent_local_time='2026-10-19 05:28',
        visible_sent_bubble_verified=True,scope='3300step smoke passed;18k candidate running; no quality claim; remote17model snapshot unchanged')
    state['head_attention_preparation'].update(status='Integrated; exact loss interface validated, native parameter repeatability remains unresolved',
        head_loss_integrated=True,head_loss_trained=bool(queue.get('smoke')),
        current_training_status=queue['status'],interface_validation='native-head-interface-validation-003/summary.json')
    state['active_job']=[queue['current']] if queue.get('current') and queue['status'] in ['smoke','training'] else []
    telemetry=ROOT/queue.get('current','native-head-full-18000-s0')/'model/head-training-audit.jsonl'
    if telemetry.exists() and telemetry.stat().st_size:
        state['head_weighting_stage']['latest_training_audit']=json.loads(telemetry.read_text().splitlines()[-1])
    state['next']='Complete controlled head-weighting experiment; review fixed8+24 real observations and paths before repetition or model promotion.'
    state['unrelated_asr_pid']=18064
    temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(target)
    checkpoint=ROOT/('progress-checkpoint-head-'+str(int(time.time()))+'.json')
    checkpoint.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=queue['status'],current=queue.get('current'),checkpoint=str(checkpoint))))


if __name__=='__main__':main()
