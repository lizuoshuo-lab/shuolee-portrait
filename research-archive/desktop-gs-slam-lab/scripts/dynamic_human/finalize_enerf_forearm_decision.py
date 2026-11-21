"""Save the completed072ablation and actual visual review, clarity not accepted."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-forearm-matte-repair-072'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
k=read(R/'paired-comparison/summary.json');h=read(R/'heldout/paired-comparison/summary.json');raw=read(R/'raw-forearm-evaluation/summary.json')
files=[f'paired-comparison/camera{c}-frame{t:06d}-full-person.png' for c in ['01','04','08','17'] for t in [745,751,755]]
files+=['paired-comparison/'+r['panel'] for r in k['details']]
files+=[f'heldout/paired-comparison/camera{c}-frame{t:06d}-{r}.png' for c in ['00','06','12'] for t in [745,751,755] for r in ['full-person','dark_right_forearm','light_left_forearm']]
assert (k['worst_change']['camera'],k['worst_change']['frame'])==('07',751)
assert (h['worst_change']['camera'],h['worst_change']['frame'])==('06',755)
files+=['paired-comparison/camera07-frame000751-full-person.png']
files+=['raw-forearm-evaluation/'+r['panel'] for r in raw['records']]
assert len(set(files))==68 and all((R/x).is_file() for x in files)
narrative='机位01的可见前臂内部颜色更接近原照片，四个预定皮肤内部区域合并PSNR提高0.861dB，三处改善、一处略退步；这些内部像素在基线中已接近完全不透明。完整人物与局部图仍显示手指边界、抬起的鞋和细衣纹偏软，新机位06胸前的浅色条纹及遮挡手部拖影仍在。修正三张训练matte没有带来明显的整体清晰度改善。'
next_actions=['保留067开发基线；071运动节点密度与072局部matte修正作为已完成对照。','先核查现有全身关键点中的手部位置、置信度、原生像素覆盖与跨机位一致性，再决定手部运动监督；不直接把预测点当真值。','针对机位06的胸前条纹和交叉手部检查表面与运动对应，不再把单纯增加节点或补不透明度作为默认办法。','清晰度通过后继续新的独立时段与大场景背景联合验证；当前灰底人物实验不是大场景完成。']
v=dict(status='68native_full_detail_and_raw_forearm_panels_actually_viewed',reviewed_utc=datetime.now(timezone.utc).isoformat(),reviewed_files=files,narrative_zh=narrative,decision_zh='不替换067开发基线；保留局部肤色收益与掩码漏标证据，整体人物清晰度仍未通过',adopt_as_development_baseline=False,next_actions=next_actions,limitations=['One seed, same actor pair and development views/times.','Raw1604pixel check excludes fingers, silhouette, hidden surfaces and newcamera skin.','Some projected forearm boxes contain the occluder.','No072actor layer rerender, no actual video playback review; seven videos were mechanically decoded.'],goal_complete=False)
save(R/'visual-review.json',v)
temporal={n:read(R/p)['mean_temporal_residual'] for n,p in [('known','temporal-comparison/summary.json'),('heldout','heldout/temporal-comparison/summary.json')]}
f=dict(status='completed_three_local_training_matte_ablation_no_material_overall_clarity_gain',known=k['validation_foreground'],heldout=h['validation_foreground'],known_per_camera=k['per_camera_validation_delta_db'],heldout_per_camera=h['per_camera'],known_worst=k['worst_change'],heldout_worst=h['worst_change'],raw_forearm=raw,temporal_difference_residual=temporal,training=read(R/'residual_graph/completion.json'),visual_review='visual-review.json',adopt_as_development_baseline=False,narrative_zh=narrative,next_actions=next_actions,goal_complete=False)
for path in ['training-protocol-audit.json','residual_graph/native-evaluation/independent-image-audit.json','heldout/native-evaluation/independent-image-audit.json','trained-graph-independent-audit.json']:
 assert (R/path).is_file(),path
save(R/'final-summary.json',f);shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps(dict(known=f['known'],heldout=f['heldout'],raw_gain=raw['pooled_delta_db'],reviewed=len(files),adopt=False)))
