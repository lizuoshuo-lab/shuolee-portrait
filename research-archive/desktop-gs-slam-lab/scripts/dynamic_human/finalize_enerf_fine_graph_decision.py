"""Record completed 071 measurements and the panels actually reviewed."""
from pathlib import Path
from datetime import datetime, timezone
import json, shutil
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-fine-graph-071'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
k=read(R/'paired-comparison/summary.json');h=read(R/'heldout/paired-comparison/summary.json')
files=[f'paired-comparison/camera{c}-frame{f:06d}-full-person.png' for c in ['01','04','08','17'] for f in [745,751,755]]
files += ['paired-comparison/'+x['panel'] for x in k['details']]
files += [f'heldout/paired-comparison/camera{c}-frame{f:06d}-{r}.png' for c in ['00','06','12'] for f in [745,751,755] for r in ['full-person','dark_right_forearm','light_left_forearm']]
files += ['paired-comparison/camera05-frame000745-full-person.png']
assert len(set(files))==64 and all((R/x).is_file() for x in files)
narrative='机位04的衣物图案和部分前臂边缘稍有改善；手指、抬起的鞋和细衣纹仍偏软。新机位00与12的多数画面差别很小；机位06第745帧胸前的浅色竖向条纹、第755帧遮挡手部的拖影仍在。增加运动节点未带来稳定的新视角清晰度改善。'
v=dict(status='64fixed_and_worst_native_comparison_panels_actually_viewed',reviewed_utc=datetime.now(timezone.utc).isoformat(),reviewed_files=files,narrative_zh=narrative,decision_zh='不替换067开发基线；已知机位的增益未稳定延伸到新机位',adopt_as_development_baseline=False,limitations=['Development views and times reused; no final-test claim.','One seed and one actor pair only.','Some forearm regions contain occluding people.','No isolated actor layer audit for071; no actual video playback review.'],goal_complete=False)
save(R/'visual-review.json',v)
temporal={n:read(R/p)['mean_temporal_residual'] for n,p in [('known','temporal-comparison/summary.json'),('heldout','heldout/temporal-comparison/summary.json')]}
f=read(R/'metrics-summary.json');f.update(status='completed_graph_resolution_ablation_no_stable_newview_clarity_gain',visual_review='visual-review.json',temporal_difference_residual=temporal,adopt_as_development_baseline=False,narrative_zh=narrative,goal_complete=False)
save(R/'final-summary.json',f);shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps(dict(known=f['known'],heldout=f['heldout'],temporal=temporal,reviewed=len(files),adopt=False)))
