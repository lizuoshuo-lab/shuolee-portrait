"""Record actual 50-image review and completed detail-weight evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-detail-weight-076'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
known=read(R/'paired-comparison/summary.json');heldout=read(R/'heldout/paired-comparison/summary.json')
files=read(R/'visual-review-plan.json')['files'].copy()
assert len(files)==37
for root,report in [('',known),('heldout/',heldout)]:
    row=report['worst_change'];file=f"{root}paired-comparison/camera{row['camera']}-frame{row['frame']:06d}-full-person.png"
    if file not in files:files.append(file)
assert len(files)==38
expanded=sorted(p.relative_to(R).as_posix() for p in (R/'expanded-detail').glob('*.png'))
assert len(expanded)==12
files+=expanded
review=dict(status='50actual_native_comparison_images_reviewed',actual_images=len(files),
    images=[dict(path=f,sha256=sha(R/f)) for f in files],
    findings={
        'hands':'Expanded frame751 windows show fused or softened fingers in01/04/00/12;06 crossing hands retain ghosting. A few boundary changes are visible, no consistent finger recovery.',
        'shoes':'01/04 airborne shoe remains smeared; outsole and laces are not restored to the photograph.06later frames retain spurious pale/green lower shoe structure.',
        'occlusion':'06frame745 chest streaks persist;06frame755 hand boundary and12hand/toy overlap remain defective.',
        'general':'Main body, broad garment appearance and carried toys are reconstructed, but fine clothing texture and important moving extremities remain softened.',
        'worst':'05frame751 (known) and06frame755 (other camera) inspected; hand/occlusion and foot errors remain.',
    },
    scope='37preselected comparisons plus1additional worst image and12expanded native windows. Some full-person figures downscaled by viewing tool; expanded detail panels viewed at original pixels. Seven videos decoded numerically, not watched in this review.',
    clarity_accepted=False,goal_complete=False)
save(R/'visual-review.json',review)
completion=read(R/'residual_graph/completion.json');completion['status']='15000updates_120native_renders_and_comparisons_complete';save(R/'residual_graph/completion.json',completion)
base=read(L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/final-summary.json')
summary=dict(status='detail_weight_control_complete_clarity_not_accepted',
    training=completion,native_images=dict(known=102,other_cameras=18),actual_images_reviewed=50,
    known_validation=known['validation_foreground'],other_camera_validation=heldout['validation_foreground'],
    context_against_retained067=dict(
        known_mean_delta_db=known['validation_foreground']['after_mean_psnr_db']-base['known_validation']['before_mean_psnr_db'],
        other_mean_delta_db=heldout['validation_foreground']['after_mean_psnr_db']-base['other_camera_validation']['before_mean_psnr_db'],
        note='Same development views; different motion preprocessing and loss, not an isolated comparison against067.'),
    temporal_difference_residual={key:read(R/path/'summary.json')['mean_temporal_residual'] for key,path in [('known','temporal-comparison'),('other_cameras','heldout/temporal-comparison')]},
    checkpoint_sha256=sha(R/'residual_graph/resume.pt'),
    intervention='Extra0.2mean RGB-L1 on hand/shoe-adjacent valid native windows in70images;35global-only images.105originalphotos, same074motion,initialization,384nodegraph,schedule,seed73,15000updates.',
    decision='不替换067开发基线；保留正确RGB输入。局部权重只有小幅同机位收益，另机位总体退化，未恢复指尖、抬脚或遮挡细节。',
    adopt_as_development_baseline=False,goal_complete=False,
    limits='Single seed and repeatedly reused development cameras/times. Window supervision is not body-part GT. Weights change densification and final point count as part of optimization; not a fixed-point-count test. Static scene joint reconstruction incomplete.',
    next_focus='Investigate local deformation and multi-view consistency at the failed hand/shoe surfaces; do not increase pixel-loss weight again without evidence.')
save(R/'final-summary.json',summary)
mfile=L/'manifests/dynamic-human-work.json';m=read(mfile)
m.update(status='detail_weight_control_completed_clarity_not_accepted',current_runs=[],
    outdoor_detail_weight='outputs/'+R.name+'/final-summary.json',
    current_best_continuous_control='outputs/dynamic-human-enerf-residual-graph-067/residual_graph',
    last_recorded_utc=datetime.now(timezone.utc).isoformat())
# Keep the authoritative existing reference path if directory spelling differs.
old=read(mfile).get('current_best_continuous_control')
if old:m['current_best_continuous_control']=old
m['completed'].append('076:15000updates280237points;102known18other native images independently checked;50actual comparisons viewed. Known+0.051dB, other-0.117dB, fingers/shoes/occlusion still fail; not adopted.')
save(mfile,m);shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps({k:v for k,v in summary.items() if k not in ['limits','next_focus']},ensure_ascii=False))
