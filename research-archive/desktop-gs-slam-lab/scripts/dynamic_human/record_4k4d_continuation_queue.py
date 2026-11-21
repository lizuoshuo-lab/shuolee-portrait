from pathlib import Path
import json,datetime
LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-4k4d-official-035'
CHILD=OUT/'continuation-10000'
live=json.loads((CHILD/'training-live.json').read_text())
queue=json.loads((CHILD/'evaluation-queue-status.json').read_text())
assert live['status']=='training' and queue['status']=='waiting_for_training_completion'
path=LAB/'manifests/dynamic-human-work.json'
m=json.loads(path.read_text(encoding='utf8'))
m['current_runs']=[dict(name='official4k4d_native_singleframe_continuation',path='outputs/dynamic-human-4k4d-official-035/continuation-10000',session_id=82968,
    target_new_updates=9000,target_history_updates=10000,last_observed_new_updates=live['completed_updates_this_process'],last_observed_history_updates=live['history_updates'],
    evaluation_queue_pid=queue['pid'],evaluation_queue_launcher_pid=3104,evaluation_status=queue['status'])]
m['official4k4d_projection_check']='outputs/dynamic-human-4k4d-official-035/projection-cpu-diagnostic.json'
m['official4k4d_website_qa']='outputs/dynamic-human-4k4d-official-035/website-qa.json'
m['next_actions']=[
    'Await already-running continuation and its bounded automatic native-render/CPU-audit queue; do not start competing GPU training.',
    'When queue reports completion, visually inspect all6paired full-person panels and3fullnative outputs; assess hands, clothing, overlap, and dotted boundaries rather than acceptPSNRalone.',
    'Proceed to continuouswalking/occlusion only after sufficient single-frame fit; retain current failed results and same-camera development-set qualification.',
    'Complete background/personcomposition and physical calibration/time checks; CPUcoordinateconversioncheck is not camera/time calibration validation.'
]
m['goal_turn_classification']='progress: official4k4d1000pilot rendered/audited/reviewed and rejected; continuation and automatic evaluation queue live;33CPUprojection cases passed; website292linkedresources verified.'
m['last_recorded_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(history_updates=live['history_updates'],queue_pid=queue['pid'],website_resources=292)))
