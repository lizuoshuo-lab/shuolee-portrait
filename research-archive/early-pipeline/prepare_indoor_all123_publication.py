from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';O=REC/'indoor-preflight/semantic-face-all'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
r=read(O/'review-result.json');pub=read(O/'publication.json');stamp=datetime.now(timezone.utc).isoformat()
dom=(O/'browser-report.txt').read_text(encoding='utf-8');expanded=(O/'browser-expanded.txt').read_text(encoding='utf-8')
assert '177帧2D特征与SMPL-X固定位置对应' in dom and '537可见、466遮挡、413未知' in dom
assert expanded.count('image 原生面部槽位')==15
pub.update(diagnostic_DOM_verified=True,all15_evidence_images_expanded_in_DOM=True,
    screenshot_verified=False,screenshot_limitation='Existing zero-width browser capture is not retried or bypassed; native source figures actually inspected separately.',updated=stamp)
save(O/'publication.json',pub)
save(O/'viewer-ui-review.json',dict(updated=stamp,current_existing_tab_reloaded=True,actual_binding_table_seen=True,
    actual_all15_image_elements_expanded=True,native_source_figures_actual_reviewed=True,browser_screenshot_verified=False,
    historical_models_preserved=True,new_model_published=False,quality_accepted=False))
save(O/'slide-review.json',dict(updated=stamp,actual_images_reviewed=True,training_started=False,quality_accepted=False,
    slide=dict(title='室内177帧二维点与SMPL-X对应',subtitle='固定解剖锚点与遮挡观测停用',
        result='1416处重点观测：537可见、466遮挡、413未知；相机与真实拟合尚未准入',
        caption='源帧0、280、440；上排原图，下排检测。背面仍有五官预测，不能作为SMPL-X约束。',
        figure='front-transition-back-native.png',figure_sha256=sha(O/'front-transition-back-native.png'),
        notes=[
            '用户目标是原图2D特征点约束同一解剖身份的SMPL-X三维表面，不是让模型自己的投影点互相吻合。',
            '177个原生2704×1520帧，51个固定官方三角面重心锚点/帧，保存9027条对应记录。沿用源帧、原生像素、连续逆映射，没有改变K或做二维扭曲。已有5帧复用，仅新增172帧检测。',
            '15张原生证据图逐幅实际审阅，重点8部位共1416观测：537部位可见，466遮挡，413未知；其余43部位/帧尚未逐点审阅。可见性不等于独立精确坐标真值。',
            '本页576×384图由原192×192裁片直接拼接：源帧0、280、440，上排原图下排FAN预测。侧后方和背面仍有眼鼻嘴预测落在耳和头发上，是必须停用的观测风险，不能唯一解释全部重影。',
            '固定锚点X=b0V0+b1V1+b2V2。拟合损失应比较完整K/R/t投影的X与原图u；左右按人物自身定义。有效观测的定位不确定性、独立留出和真实相机尚未通过，未进行此人物SMPL-X拟合或训练。',
            '本轮只启动一次室内177帧原生材料匹配；区别于历史24帧，173个源帧此前未检验。外观轨迹需逐物理材料验证后才可用于相机，不将背景ROI或循环一致自动当静止真值。',
            '原生review-result.json、visual-review.json与real-2d-smplx-bindings.json保存在indoor-preflight/semantic-face-all，查看器沿用历史点云。'
        ])))
s=(H/'update_deck_field_background123.mjs').read_text(encoding='utf-8')
repl={
    "tag='field-background123'":"tag='indoor-semantic-all123'",
    "/field-preflight/native-material/topology-review/background-full-review'":"/indoor-preflight/semantic-face-all'",
    "b+'/field-semantic-all123-deck-receipt.json'":"b+'/field-background123-deck-receipt.json'",
    "replace('177帧二维特征与SMPL-X表面绑定'":"replace('背景轨迹的完整观测核验'",
    "replace('固定解剖锚点与逐帧可见性'":"replace('同一区域的相似外观与固定物理点'",
    "replace('1416次重点部位审阅'":"replace('10条轨迹共102个观测已审阅'",
    "replace('上排原图，下排检测'":"replace('上排为树冠边界，下排为地面纹理'",
    '02_有限视角重建优化路线_123背景材料完整观测.pptx':'02_有限视角重建优化路线_123室内全帧二维三维对应.pptx',
    'preserve_deck_field_background123.py':'preserve_deck_indoor_semantic_all123.py'
}
for a,b in repl.items():
    assert s.count(a)>=1,(a,s.count(a));s=s.replace(a,b)
out=H/'update_deck_indoor_semantic_all123.mjs';assert not out.exists();out.write_text(s,encoding='utf-8')
print(json.dumps(dict(UI_report_saved=True,slide_review=str(O/'slide-review.json'),authoring_helper=str(out)),ensure_ascii=False))
