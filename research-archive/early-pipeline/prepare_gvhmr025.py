from pathlib import Path
import json,hashlib
from datetime import datetime
BASE=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
CODE=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\repos\GVHMR-shuolee\shuolee')
R24=BASE/'current/smplx-gvhmr-body-motion-fresh-024'
R25=BASE/'current/smplx-gvhmr-shared-world-fresh-025'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
now=datetime.now().astimezone().isoformat(timespec='seconds')
assert not (R25/'experiment-design.json').exists()
review=read(R24/'review/review-manifest.json')
conclusion=dict(created=now,state='relative_motion_improved_absolute_body_geometry_not_accepted',
    new_body_updates=2400,new_gaussian_updates=0,from_scratch=True,
    control023_exactly_reproduced=max(review['control023_replay_max_abs'].values())==0,
    statistics=review['statistics'],unoptimized_points=53,unoptimized_observations=930,
    body_geometry_accepted=False,independent_GT=False,
    findings=['Torso/leg diagnostic feature median24.23->2.97px; foot diagnostics1.36->1.31px.',
        'Visible body median8.71->10.06px; left knee7.58->17.71px/right7.996->12.73px. Aggregate median hides local regressions.',
        'Reference-frame side shape remains similar; relative transport becomes identity at reference, so low motion loss cannot establish absolute anatomy.',
        'Do not promote this fitting as complete or deploy a new Gaussian.'],
    visual_review=dict(full_source_frames=[23,45],fixed_world_views=['front','left90'],feature_frames=[45],
                       fixed_time_snapshot_views=['front','left90'],time_snapshots=[0,11,23,33,45,58],
                       full_continuous_video_accepted=False,all59_source_images_rendered_but_not_all_visually_reviewed=True),
    next='Independent shared-world body hypothesis with same fresh initialization in both arms; compare against free-world control. Do not initialize from024.',
    remote_deployed=False)
save(R24/'phase-conclusion.json',conclusion)
design=dict(created=now,state='prepared',hypothesis='Over-flexible roots and local lower-body/spine poses let single-view image fitting retain incorrect absolute geometry. A shared torso/lower-body configuration allows all59 real calibrated views to constrain one geometry.',
    arms=['free','shared'],steps_per_arm=1200,total_updates=2400,new_gaussian_updates=0,
    initialization='Both independently regenerate raw inputs; same SO3 means for world root and local joints1..11, median root position and shape, original per-frame neck/arm prediction. No learned person arrays read.',
    intervention='Share world root rotation/translation and local joint1..11 corrections across frames in shared arm. Free arm retains per-frame corrections from the same initial values.',
    held_fixed=['59 real cameras','all image/mask/hand observations','body and foot texture splits','loss weights','seed/schedule','scale and shape parameterization'],
    body_motion_assumption='Approximate stationary torso/lower body only; head, collar, arms and hands remain time-dependent. Test assumption against visible real images.',
    limits=['Measured skeleton GT absent; torso under loose clothing is not uniquely observed.', 'Texture diagnostics are source-conditioned, not independent GT.',
            '025 numerical result must not be attributed solely to sharing by comparison with024; compare free/shared arms with identical starting meshes.'],
    gate='Native same-camera overlay + fixed-world front/side + time consistency; reject if shared model creates limb/geometry regressions even if aggregate residual improves.',
    source_sha256=hashlib.sha256((CODE/'refine_shared_world_fresh.py').read_bytes()).hexdigest(),accepted=False,remote_deployed=False)
save(R25/'experiment-design.json',design)
state=read(BASE/'work-status.json');assert state['current_phase']=='gvhmr_body_motion_fresh_024'
state['updated']=now;state['current_phase']='gvhmr_shared_world_fresh_025'
state['current_decision']='024真实纹理运动改善但侧面几何未改、膝部图像残差变大，不作为完整人体采用。开始025两组相同原始平均初始化的共享世界躯干/下身对照，各1200步；不加载024学得参数，不启动高斯、不部署00。'
state['latest_local_execution']=dict(kind='gvhmr_shared_world_fresh_025',state='prepared',running=False,planned_steps_per_arm=1200,arms=['free','shared'],directory=str(R25),gaussian_updates=0,accepted=False)
state['gvhmr_body_motion_024']=dict(conclusion=str(R24/'phase-conclusion.json'),body_geometry_accepted=False,new_body_updates=2400,new_gaussian_updates=0)
save(BASE/'work-status.json',state)
print('FRESH_SHARED_WORLD_DESIGN_READY')
