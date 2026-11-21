"""Record completed measurements and the actual visual review, without accepting clarity."""
from pathlib import Path
from datetime import datetime, timezone
import json, shutil
L=Path(__file__).resolve().parents[2]
R=L/'outputs/dynamic-human-enerf-actor-supervision-070'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
known=read(R/'paired-comparison/summary.json'); novel=read(R/'heldout/paired-comparison/summary.json')
owners=[read(R/(s+'-training-ownership.json')) for s in ['baseline','candidate']]
reviews=[f'paired-comparison/camera{c}-frame{f:06d}-full-person.png' for c in ['01','04','08','17'] for f in [745,751,755]]
reviews += ['paired-comparison/'+x['panel'] for x in known['details']]
reviews += [f'heldout/paired-comparison/camera{c}-frame{f:06d}-{r}.png' for c in ['00','06','12'] for f in [745,751,755] for r in ['full-person','dark_right_forearm','light_left_forearm']]
reviews += ['paired-comparison/camera09-frame000755-full-person.png','heldout/paired-comparison/camera00-frame000747-full-person.png']
reviews += [f'actor-layers/camera{c}-frame{f:06d}-actors.png' for c,f in [('06',745),('06',751),('06',755),('00',751),('12',751)]]
reviews += [f'actor-layers/camera06-frame{f:06d}-layers.png' for f in [745,751,755]]
assert len(reviews)==73 and all((R/x).is_file() for x in reviews)
narrative='固定全身图和局部图中，两版外观多数近似；手指、抬起的鞋和细衣纹仍偏软。机位06第745帧胸前仍有浅色竖向条纹，第755帧遮挡手部仍有拖影。单独渲染黑衣人物时胸前缺口仍存在，白衣人物层仍带有重叠区域残留。监督改善了部分训练像素的人物归属，但未恢复明显的精细结构。'
next_actions=['保留067局部变形基线，070保留为已完成的弱实例监督消融。','核查原生训练分辨率与细部监督，再对运动模型、前景matte和局部清晰度作针对性控制；不根据整体PSNR的小幅上涨宣称清晰度通过。','保持未参与外观训练的机位00、06、12及时间验证；这些是反复使用的开发集，最终还需独立时段与联合场景。']
v=dict(status='fixed_native_panels_actually_viewed_clarity_not_accepted',reviewed_utc=datetime.now(timezone.utc).isoformat(),reviewed_files=reviews,narrative_zh=narrative,decision_zh='不替换067开发基线；归属损失仅保留为局部证据，清晰人体目标未通过',adopt_as_development_baseline=False,next_actions=next_actions,limitations=['No video playback review claimed; seven videos were mechanically decoded.','Projected forearm ROIs can contain the occluder, especially camera06.','No unobstructed ground truth for isolated hidden surfaces.','One training seed; small gains have no repeatability claim.'],goal_complete=False)
save(R/'visual-review.json',v)
temporal={name:read(R/path)['mean_temporal_residual'] for name,path in [('known','temporal-comparison/summary.json'),('heldout','heldout/temporal-comparison/summary.json')]}
ownership=dict(before_wrong_pixels=sum(x['wrong_owner_pixels'] for x in owners[0]['records']),after_wrong_pixels=sum(x['wrong_owner_pixels'] for x in owners[1]['records']),weak_label_pixels=sum(x['known_pixels'] for x in owners[0]['records']),qualification='Disagreement with predicted conservative training labels, not instance ground-truth error.')
f=dict(status='completed_weak_actor_supervision_ablation_no_material_visual_clarity_gain',known_camera_validation=known['validation_foreground'],heldout_camera_validation=novel['validation_foreground'],training_ownership=ownership,temporal_difference_residual=temporal,training=read(R/'residual_graph/completion.json'),visual_review='visual-review.json',adopt_as_development_baseline=False,narrative_zh=narrative,next_actions=next_actions,goal_complete=False)
for path in ['training-protocol-audit.json','residual_graph/native-evaluation/independent-image-audit.json','trained-graph-independent-audit.json','actor-layers/independent-audit.json']:
 assert (R/path).is_file(),path
save(R/'final-summary.json',f);shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps(dict(ownership=ownership,known=known['validation_foreground'],heldout=novel['validation_foreground'],reviewed=len(reviews),adopt=False),ensure_ascii=False))
