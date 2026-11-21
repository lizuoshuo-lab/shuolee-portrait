"""Close only this registered four-person phase after concrete result reviews."""
from pathlib import Path
import json,hashlib,datetime,shutil
TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=TOP/'shuolee/current/new-capture-118/all-persons-20261027'
L=Path('D:/00/工作日志/2026-10-27/全部人物训练')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat();subjects=[]
for name in ['sofa','portrait']:
    q=read(R/f'{name}-quality-review.json');u=read(R/f'{name}-viewer-ui-review.json');v=read(R/f'{name}-export-validation.json');delivery=read(R/f'{name}-staged-delivery.json')
    assert q['actual_same_camera_review'] and q['continuous_novel_views_reviewed']==36
    assert u['file_opened_through_UI'] and set(u['modes'])=={'Gaussian','points'} and u['observed_count']==v['points']
    assert v['format_passed'] and q['source_sha256']==u['source_sha256']==v['sha256']==sha(Path(delivery['download_path']))
    assert all(sha(Path(s['path']))==s['sha256'] for s in u['screenshots'])
    subjects.append(dict(subject=name,steps=delivery['steps'],points=v['points'],format_passed=True,quality_accepted=q['quality_accepted'],summary=q['summary'],viewer_url=delivery['viewer_url'],download_url=delivery['download_url']))
    dest=L/f'{name}-final-current';dest.mkdir(exist_ok=True)
    for f in (R/f'{name}-final-review').iterdir():
        if f.suffix=='.png':shutil.copy2(f,dest/f.name)
    for suffix in ['quality-review.json','viewer-ui-review.json','export-validation.json','staged-delivery.json']:
        shutil.copy2(R/f'{name}-{suffix}',L/f'{name}-{suffix}')
assert read(R/'grass-fit-review.json')['allow_finite_research_training'] is False
grass=read(R/'grass-shared-point-audit-v2/result.json')
occ=read(R/'indoor118-external-occluders/actual-review.json');assert occ['allow_new_training'] is False and occ['actual_pixel_review']
subjects.extend([
    dict(subject='grass',steps=0,quality_accepted=False,summary='全新1200步拟合未通过：全身关节P90为框高26.55%，末端视角整体偏到背景。原帧与K/E映射数值一致；放开SMPL-X为自由静态3D点后留出P90仍26.98%，不能唯一归因于模板或相机。没有启动高斯。'),
    dict(subject='indoor118',steps=0,previous_steps=18000,quality_accepted=False,summary='保留既有18k失败结果。89帧家具分割候选未覆盖主要圆桌遮挡，也缺前后深度关系；本轮未应用候选、未原样重训。')])
result=dict(updated=now,state='bounded_phase_reviewed',all_four_subjects_reviewed=True,all_four_successful=False,
    admitted_quality_results=sum(x['quality_accepted'] for x in subjects),subjects=subjects,
    training_processes_expected=0,formal_calibration_accepted=False,automatic_followup='PAUSED_AFTER_PHASE_REVIEW',
    remaining_evidence=['FISHEYE_8官方公式/参数顺序及每段视频的导出、裁切、防抖关系','草地人物可靠对应与固定几何支持','人物后方及脚部真实拍摄覆盖','独立于人物mask的前景遮挡深度关系'],
    end_condition='The predeclared finite phase allows reviewed results or concrete non-admission evidence for every subject; this is not a claim that ghosting or all reconstructions are solved.',
    additional_training_started=False)
save(R/'phase-result.json',result);save(L/'本阶段结果.json',result)
ledger=read(R/'evidence-ledger.json');ledger['final_review']=result;save(R/'evidence-ledger.json',ledger)
save(L/'证据索引.json',dict(phase_result='本阶段结果.json',sofa='sofa-quality-review.json',portrait='portrait-quality-review.json',
    grass_fit=str(R/'grass-fit-review.json'),grass_mapping=str(R/'grass-shared-point-audit-v2/result.json'),indoor_occlusion=str(R/'indoor118-external-occluders/actual-review.json')))
print('Four-person finite phase reviews complete; quality success is false.')
