from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');REC=T/'shuolee/current/local-clothed-surface-123'
A=REC/'remaining-native-input-audit';H=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=next(s for s in read(A/'result.json')['subjects'] if s['subject']=='grass')
video=Path('D:/00/工作/数据集/video5.mp4');assert sha(video)==old['video_sha256']
review=read(A/'visual-review.json');assert any(s['subject']=='grass' for s in review['subjects'])
assert not (A/'grass-intake-plan.json').exists() and not (REC/'grass-preflight').exists()
extractor='/opt/gs-slam-lab/shuolee-local-surface-123/runtime/src/shuolee_portrait/video_sfm.py'
digest=subprocess.check_output(['wsl.exe','-d','Ubuntu-20.04','--','sha256sum',extractor],encoding='utf-8').split()[0]
plan=dict(registered=datetime.now(timezone.utc).isoformat(),subject='grass',video='/mnt/d/00/工作/数据集/video5.mp4',video_sha256=old['video_sha256'],
    source_frames=470,native_wh=[2704,1520],stride=5,selected_frames=94,selected_indices=old['predeclared123_indices'],shared_historical_indices=old['shared_indices'],
    output='/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/grass-preflight',
    run_root='/opt/gs-slam-lab/shuolee-local-surface-123/preflight/grass',project='/opt/gs-slam-lab/shuolee-local-surface-123/runtime',
    historical_observations='/opt/gs-slam-lab/shuolee-raw-domain-121/grass/observations.json',extractor_sha256=digest,
    review_slots=[0,23,47,70,93],previous_goal_turn='progress: A0c real fitter synthetic integration and initializer identity binding completed',
    hypothesis='The seventh registered person still lacks the complete94-frame native stream.89of94selected source indices were absent from historical24diagnostic, so that diagnostic cannot establish full-sequence support.',
    changed_information='Complete existing predeclared5-frame sampling. No changedK, detector weights, camera solver, prior geometry or new feature matching.',
    required_next='Review five actual native originals before deciding on a once-only same-settings region/landmark pass.',
    budget=dict(seconds=240,decode_passes=1,CPU_threads=4,GPU_jobs=0,new_downloads=0,feature_matching=0,cameras=0,SMPLX=0,training=0),
    stop='Any identity/raster/PTS/rotation/count failure stops. If decoder pixels differ, do not import old observations. No duplicate run, extra threshold search or automatic camera solve.',
    old_sources_preserved=True,training_allowed=False)
assert plan['selected_indices']==list(range(0,470,5)) and len(old['previously_unexamined123_indices'])==89
(A/'grass-intake-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(registered=True,frames=94,new_indices=89,training=False)))
