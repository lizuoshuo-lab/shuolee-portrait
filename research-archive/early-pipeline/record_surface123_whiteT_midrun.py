"""Handoff the actually reviewed checkpoints without touching the running worker."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,hashlib
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';R=T/'shuolee/current/robust-portrait-122'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123')
H=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)
now=datetime.now(timezone.utc).isoformat();prog=read(ROOT/'whiteT/progress.json')
assert prog['stage']=='training' and prog['step']>=12000
assert read(REC/'whiteT-train-012000-review/actual-review.json')['actual_visual_review']
next_action=read(REC/'next-action.json')
next_action.update(updated=now,observed_progress=prog,latest_actual_checkpoint_review='whiteT-train-012000-review/actual-review.json',
    completed_this_turn=['WhiteT6000 and12000 actual6camera fullbody/face/center reviews.',
    'Read-only frozen regional audit completed once; six preselected3view native material samples actually inspected.',
    'Automatic faceROI includes background, not material truth.47/54ROI candidates>=1px; no filtering or fitting.',
    'All6nativePLY snapshot roundtrips at12000 are pixel-identical; visible ghosting already exists beforeexport.'],
    next=['Verify WSL361 and publisher29516 realcommands andprogress; neverduplicate or edit runningcode.',
    'WhiteT18k inspect sixsame-camera plus36orderednovelviews. review_local_surface123.py --subject whiteT --phase train --step18000; review_surface123_novel.py --subject whiteT.',
    'Afteractualreview call record_surface123_final_review.py withsubject/summary/novelsummary andreviewedallimages; itdoesnot certifyquality.',
    'Stage whiteT; pack_local_surface123.mjs whiteT; check_scene123_cameras.mjs whiteT. ActualOpenFileZIP Gaussian/points inlocaltab31; save whiteT-openfile-gaussian-ui.png and whiteT-openfile-points-ui.png.',
    'Then record_surface123_actual_ui.py withactualobservedpoints,cameras,sourceframe,notes andobservedbothmodes; promoteafterUI; screenshot latest.',
    'PPT52lastpage: prepare_surface123_whiteT_deck.py prepared but notrun. Afterfinalquality+UI, runprepare, skillmarker exactlyoncebeforefirstauthor command inthatturn, update_deck_surface123_whiteT_final.mjs, actualviewfinal52, commit_surface123_whiteT_deck.py. CanonicalsourceSHA stillaea3eb43b0143133c30e77506dd82b90a0d7235ff19531b83b4a8dab9fde7c6a. First51preserved.',
    'WorkerstartsgrayT automatically. Review nextpersonfit,surface,1000; no extraexperimentalvariants oroldreruns.',
    'Updateexistingstatus/ledger/log; notify00onlyoncefornewactualdelivery. Goalactive, no timer.'])
save(REC/'next-action.json',next_action)
parent=read(R/'queue-state.json');parent.update(updated=now,phase='LOCAL_SURFACE123_whiteT12000_reviewed_training_continues',
    last_goal_turn=dict(classification='progress',time=now,current='WhiteT6000/12000 actually reviewed andpublished. Clothed centers present; facesstillghosting. FrozenROI/materialaudit andnativePLYparity separate limits documented.',quality_accepted=False),
    next='Finish samewhiteT18k andactualdelivery; grayTautomaticallynext. Followlocal123next-action.json.',goal_status='active',quality_accepted=False)
save(R/'queue-state.json',parent)
for dest in [T/'viewer/local-surface-123',Path('D:/00/工作/查看器/local-surface-123'),Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')]:
    for name in ['evidence-ledger.json','next-action.json']:shutil.copy2(REC/name,dest/name)
plan=T/'shuolee/long-term-plan.md';s=plan.read_text(encoding='utf-8-sig')
marker='<!-- LOCAL_SURFACE123_WHITET12000 -->'
if marker not in s:
    s+='\n\n'+marker+'\n白T恤人物6000和12000步已实际检查六个相机的全身、面部与点中心，衣物表面分布成立，但头部多重轮廓及肩袖溅射未消失。12000步六相机PLY原生回读与训练逐像素一致，重影已经存在于训练结果。冻结面部ROI候选54次中47次误差不小于1像素，六组预选纹理样本又发现背景横幅/网格/树叶混入，不能用自动ROI统计当真实人脸标定。继续同一有限18k队列，未改变正在执行的训练源码、相机或观察；其他人物依次独立训练。\n'
    plan.write_text(s,encoding='utf-8')
print(json.dumps(dict(step=prog['step'],actual_review=12000,quality_accepted=False,next_action=str(REC/'next-action.json')),ensure_ascii=False))
