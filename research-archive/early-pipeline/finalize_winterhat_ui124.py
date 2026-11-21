from pathlib import Path
from datetime import datetime,timezone
import json
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
ROOT=T/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124'
OUT=ROOT/'winterhat-selected-sh0-pipeline124';R=T/'shuolee/current/robust-portrait-122';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat();ui=OUT/'delivery/step-018000/ui-review'
for name in ['openfile-gaussian.png','openfile-points.png','viewer-final.png','report-final.png']:
    assert (ui/name).stat().st_size>1000
review=dict(updated=now,model='cross-person-124-winterhat-selected-sh0-pipeline124-sh0-s018000',
    actualOpenFileGaussian=True,actualOpenFilePoints=True,actualPresetRender=True,
    pairedCameras=1,points=24861,actualReportTextAndFirstImage=True,
    otherReportImagesLazyLoaded=True,qualityAccepted=False,company00Verified=False,
    source_and_export_black_parity=True,native_gray_parity_tested=False)
save(ui/'review.json',review)
for rel in ['delivery/publication.json','pipeline-publication.json']:
    p=OUT/rel;d=read(p);d.update(updated=now,company00_verified=False,quality_accepted=False)
    d['ui_checked' if rel.startswith('delivery/') else 'actual_UI_verified']=True;save(p,d)
remy=read(C/'remy00-delivery-20261103.json')
remy.update(updated=now,local_offline_package_import_rechecked=True,
    local_preset_rechecked=True,local_paired_cameras=174,local_gaussians=789005,
    model_select_404_negative_UI_tested=False,company00_verified=False,
    message_checks=str(H/'remy00-delivery-20261103/model-selection-404/message-check.json'))
save(C/'remy00-delivery-20261103.json',remy)
for p in [R/'protocol.json',R/'queue-state.json',R/'evidence-ledger.json',C/'local-research-status.json']:
    d=read(p);d['updated']=now;d['remy00_delivery_20261103']=remy
    winter=d['winterhat_sh0_pipeline124'];winter.update(viewer_UI_verified=True,
        modelId=review['model'],package=str(Path('D:/00/工作/查看器/exports')/'winterhat-selected-sh0-pipeline124_SH0_018000_with_camera.zip'),
        paired_cameras=1,quality_accepted=False,company00_verified=False)
    d['next_action']='Observe existing grass-selected-sh0-pipeline124 PID411/start121321. Its1000 review decision is continue; review10000 and18000, paired export after trainer exits, then actualUI. Preserve old grass files and do not repeat fitting. Indoor remains unresolved source-visibility input, not a new fit. Remy00 selector exact new/old label is pending human reply; directZIP is already locally verified.'
    save(p,d)
plan=R/'long-term-plan.md'
with plan.open('a',encoding='utf-8') as f:
    f.write('\n\n## 2026-11-03 冬帽完整交付与草地同配置训练\n\n冬帽8帧从头SH0完成18,000步，24,861点，累计1130.22秒（复用既有前处理和拟合，含快照与1000步审阅等待）。全部8个训练视角、原生面部/全身裁片及36连续未知视角已实际审阅；正面与帽子较清楚，未拍背面有明显空洞破碎。衣物网格偏移最大0.000512米、面部保护位移为零，无新增法向翻折；质量、约20万点及15分钟全流程目标仍未通过。点云同坐标单参考相机导出/原生回读/实际OpenFile高斯与点模式/预设模型均已核验，00尚未核验。\n\n草地白衣使用grass-screened-seven124的同一份共享体型/姿态输入，7个预登记视角和原有完整K/w2c保持不变，从头训练PID411/start121321；1000步实际审阅后继续同一优化器至最多18000。未增加姿态修复，不把原姿态错误改成验收通过。新记录文件名含“共享七帧124”，不覆盖旧草地证据。\n\nRemy00用户明确是选择模型后404；Windows ZIP白名单修复不等于该故障根因已确定。新旧Remy条目分别为261103_112806(789005点)和历史分享模型(727737点)，请求目录和同步文件未在00核实。页面2026.11.03.3只补充404缺文件路径/当前目录/配套ZIP导入提示。本机新条目和完整ZIP实际加载174相机；00可以直接打开工作/查看器/exports/Remy_261103_112806_点云与相机.zip。\n')
print('WINTERHAT_LOCAL_UI_DELIVERY_VERIFIED_GRASS_REMAINS_RUNNING')
