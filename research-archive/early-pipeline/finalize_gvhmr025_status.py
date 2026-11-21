from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil

build=Path(__file__).resolve().parent
base=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
root=base/'current/smplx-gvhmr-shared-world-fresh-025'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now().astimezone().isoformat(timespec='seconds')
canonical=base/'reports/02_有限视角重建优化路线.pptx'
deck=read(build/'gvhmr025-deck-receipt.json')
cleanup=read(root/'cleanup-receipt.json')
assert cleanup['state']=='completed' and deck['canonical_replaced']
assert sha(canonical)==deck['final_sha256']
deck.update(retained_final_path=str(canonical),duplicate_exports_removed=True,
    retained_changed_slide=str(build/'gvhmr025-final-renders/55.png'),cleanup_receipt=str(root/'cleanup-receipt.json'))
save(build/'gvhmr025-deck-receipt.json',deck)
web=read(root/'web-publication.json')
proof=build/'gvhmr025-page-proof.png'
assert proof.is_file()
web.update(browser_verification_pending=False,verified_at=now,
    proof=str(proof),proof_sha256=sha(proof),after_sha256=sha(Path(web['page'])),
    images_loaded=True,formula_visually_checked=True,full_continuous_video_accepted=False)
save(root/'web-publication.json',web)
for trial,arms in [('smplx-gvhmr-body-motion-fresh-024',['control','body-texture']),('smplx-gvhmr-shared-world-fresh-025',['free','shared'])]:
    folder=base/'current'/trial
    conclusion=read(folder/'phase-conclusion.json')
    conclusion['retention']=dict(updated=now,cleanup_receipt=str(root/'cleanup-receipt.json'),
        fitted_files={arm:(folder/arm/'fitted.npz').is_file() for arm in arms},
        shared_025_retained_only_for_read_only_diagnostic=True,
        old_learned_states_not_allowed_as_new_training_initialization=True)
    save(folder/'phase-conclusion.json',conclusion)
    public_name='body-motion-conclusion.json' if trial.endswith('024') else 'shared-world-conclusion.json'
    shutil.copyfile(folder/'phase-conclusion.json',base.parent/'viewer/body-fitting-method/gvhmr-motion'/public_name)
for asset in web['assets']:
    asset['sha256']=sha(base.parent/'viewer/body-fitting-method/gvhmr-motion'/asset['name'])
save(root/'web-publication.json',web)
state=read(base/'work-status.json')
state['updated_at']=now
state['latest_report']=dict(updated=now,canonical_path=str(canonical),final_sha256=sha(canonical),
    final_bytes=canonical.stat().st_size,slides=55,changed_slides=[55],all55_rendered=True,
    first54_pixel_identical=True,slide55_visually_reviewed=True,nativePowerPointVerified=False,
    validated_output=str(canonical),receipt=str(build/'gvhmr025-deck-receipt.json'))
state['gvhmr_shared_world_025'].update(web=web,cleanup=dict(receipt=str(root/'cleanup-receipt.json'),files=cleanup['count'],bytes=cleanup['bytes']))
state['local_training_running']=False
state['next_action']='先在真实肩、肘、膝照片上比较原检测与ViTPose，核对裁切、像素坐标、时间与遮挡；以观测不确定性设计从头匹配对照。025共享拟合只作只读对照，不复用为训练起点。'
save(base/'work-status.json',state)
plan_file=base/'long-term-plan.md'
plan=plan_file.read_text('utf-8-sig')
anchor='[时序人体先验](http://127.0.0.1:8870/body-smplx.html#gvhmr-prior-019)'
assert plan.count(anchor)==1
plan=plan.replace(anchor,'[运动与绝对几何](http://127.0.0.1:8870/body-smplx.html#gvhmr-body-observation-025) · '+anchor,1)
plan_file.write_text(plan,'utf-8')
print(json.dumps(dict(priors_ready=True,report_sha256=sha(canonical),web_verified=True,
    cleanup_files=cleanup['count'],cleanup_bytes=cleanup['bytes'],body_geometry_accepted=False,
    new_gaussian_updates=0,goal_remains_active=True),ensure_ascii=False))
