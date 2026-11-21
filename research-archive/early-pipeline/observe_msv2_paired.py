import json, time, shutil
from pathlib import Path
run=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/msv2-video15-20261105/runs/video15-parent1-child4')
record=Path('D:/00/记录/2026-11-06_MSV2_video15_子母图联合训练')
while True:
 try:
  state=json.loads((run/'state.json').read_text(encoding='utf-8'))
  item={k:state.get(k) for k in ('state','phase','failures','training_completed')}
  progress=run/'swinir4x/progress.json'
  if progress.exists():
   d=json.loads(progress.read_text(encoding='utf-8'));item.update({k:d.get(k) for k in ('iteration','points','seconds','peak_reserved_gib','warmup_level')})
  elif (run/'swinir4x/superresolution.json').exists():
   d=json.loads((run/'swinir4x/superresolution.json').read_text(encoding='utf-8'));item['SR_processed']=d.get('processed')
  item['D_free_GiB']=round(shutil.disk_usage('D:/').free/1024**3,2)
  for name,out in [('job.json','训练_job.json'),('plan.json','训练_plan.json'),('state.json','训练_state.json'),('swinir4x/progress.json','训练进度.json')]:
   if (run/name).exists():shutil.copy2(run/name,record/out)
  print(json.dumps(item,ensure_ascii=False),flush=True)
  if state['state'] not in ('running','training'):break
 except (OSError,ValueError) as exc:print(repr(exc),flush=True)
 time.sleep(45)
