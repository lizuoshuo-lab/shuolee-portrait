"""Record the completed temporal-basis control and actual native-image review."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

L=Path(__file__).resolve().parents[2]
R=L/'outputs/dynamic-human-enerf-temporal-basis-078'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
known=read(R/'paired-comparison/summary.json')
other=read(R/'heldout/paired-comparison/summary.json')
assert read(R/'residual_graph/native-evaluation/independent-image-audit.json')['images']==102
assert read(R/'heldout/native-evaluation/independent-image-audit.json')['images']==18
assert read(R/'trained-graph-independent-audit.json')['trained_graph_optimizer_steps']==[15000]*4
files=read(R/'visual-review-plan.json')['files'].copy()
assert len(files)==37
for prefix,report in [('',known),('heldout/',other)]:
    row=report['worst_change']
    file=f"{prefix}paired-comparison/camera{row['camera']}-frame{row['frame']:06d}-full-person.png"
    if file not in files:files.append(file)
assert len(files)==39
expanded=sorted(p.relative_to(R).as_posix() for p in (R/'expanded-detail').glob('*.png'))
assert len(expanded)==12
files+=expanded
review=dict(status='51actual_native_comparison_images_reviewed',actual_images=len(files),
    images=[dict(path=f,sha256=sha(R/f)) for f in files],
    findings={
        'general':'Main bodies, faces, broad garment appearance and toys remain recognizable. Some shirt/toy boundaries look slightly more concentrated, but improvement is subtle and not uniform.',
        'hands':'Expanded01/04/00/12frame751 windows still show softened or fused fingers.06frame751 crossing hands retain duplicate/blurred boundaries; no reliable fingertip recovery.',
        'shoes':'01/04frame751 airborne shoes remain smeared compared with the original photograph;06later frames retain incorrect pale/green sole structure.',
        'occlusion':'06frame745 chest streaks persist.06frame755 hand artifacts and12frame745 hand/toy overlap remain defective.',
        'worst':'13frame755 known-camera and00frame747 other-camera regressions reviewed; neither resolves the existing hand, garment or overlap softness.',
    },
    scope='37preselected comparisons plus2additional worst images and12expanded native windows. Full-person tool displays may be downscaled; detailed panels viewed at original pixels. Seven videos encoded/decoded, not watched.',
    clarity_accepted=False,goal_complete=False)
save(R/'visual-review.json',review)
completion=read(R/'residual_graph/completion.json')
completion['status']='15000updates_120native_renders_and_comparisons_complete'
save(R/'residual_graph/completion.json',completion)
old=read(L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/final-summary.json')
summary=dict(status='temporal_basis_control_complete_partial_improvement_clarity_not_accepted',
    training=completion,native_images=dict(known=102,other_cameras=18),actual_images_reviewed=len(files),
    known_validation=known['validation_foreground'],other_camera_validation=other['validation_foreground'],
    context_against_retained067=dict(
        known_mean_delta_db=known['validation_foreground']['after_mean_psnr_db']-old['known_validation']['before_mean_psnr_db'],
        other_mean_delta_db=other['validation_foreground']['after_mean_psnr_db']-old['other_camera_validation']['before_mean_psnr_db'],
        note='Same repeatedly used development images. Corrected RGB preprocessing and temporal basis both differ from067; this is not an isolated comparison against067.'),
    temporal_difference_residual={key:read(R/path/'summary.json')['mean_temporal_residual'] for key,path in [('known','temporal-comparison'),('other_cameras','heldout/temporal-comparison')]},
    checkpoint_sha256=sha(R/'residual_graph/resume.pt'),
    intervention='Original cubic residual motion plus7reference-centered cubic B-spline terms per rotation/translation coordinate. Same105originalphotos,074motion,384spatialnodes,52109initialGaussians,seed73,sampling,loss,densityschedule and15000updates. Trainablegraphparameters6912to23040.',
    decision='局部时间基函数在两个开发集合上平均改善，保留为下一轮候选；手指、抬脚和交叉遮挡仍不合格。旧067参考完整保留，独立片段验证前不替换正式开发基线。',
    retain_as_temporal_candidate=True,adopt_as_development_baseline=False,goal_complete=False,
    limits='Single seed and repeatedly reused development times/cameras. Fixed body motion and spatial graph. Final Gaussian count changes with optimization. No proof that time basis alone explains blur; static-scene joint reconstruction incomplete.',
    next_focus='Validate the fixed temporal extension on an independent motion interval, retaining native hand/shoe/occlusion comparisons and the original cubic control. Investigate residual visibility/timing errors rather than raising pixel-loss weights again.')
save(R/'final-summary.json',summary)
p=L/'manifests/dynamic-human-work.json';m=read(p)
m.update(status='temporal_basis_control_complete_partial_improvement_clarity_pending',current_runs=[],
    outdoor_temporal_basis='outputs/'+R.name+'/final-summary.json',
    current_temporal_candidate='outputs/'+R.name+'/residual_graph',
    last_recorded_utc=datetime.now(timezone.utc).isoformat())
entry='078:15000updates223105points;120native renders independently checked;51actual comparisons viewed. Known+0.251dB72/90improve;other+0.073dB12/18improve. Temporal residuals lower, fingers/shoes/occlusion still fail; candidate retained pending independent interval.'
if entry not in m['completed']:m['completed'].append(entry)
save(p,m)
shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps({k:summary[k] for k in ['status','known_validation','other_camera_validation','actual_images_reviewed','goal_complete']},ensure_ascii=False))
