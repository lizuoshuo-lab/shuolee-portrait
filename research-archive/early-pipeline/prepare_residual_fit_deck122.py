from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import json
import hashlib

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/robust-portrait-122/native-residual-fit'
review=json.loads((O/'review.json').read_text(encoding='utf-8-sig'))
assert review['checks_passed']==29 and review['actual_image_review']
source=Path('D:/00/报告/02_有限视角重建优化路线.pptx')
with ZipFile(source) as z:
    parts=[n for n in z.namelist() if n.startswith(('ppt/slideMasters/','ppt/slideLayouts/')) and n.endswith('.xml')]
    records=[]
    for name in parts:
        root=ET.fromstring(z.read(name))
        records.append(dict(part=name,sha256=hashlib.sha256(z.read(name)).hexdigest(),
            placeholders=[e.attrib for e in root.iter() if e.tag.endswith('}ph')],
            canvas_names=[e.attrib.get('name') for e in root.iter() if e.tag.endswith('}cSld')]))
(H/'residualFit-122-template-inspection.json').write_text(json.dumps(dict(source=str(source),masters_and_layouts=records,policy='Edit existing lastslide only; no master/layout/theme modification.'),ensure_ascii=False,indent=2),encoding='utf-8')
s=(H/'update_deck_residualAdmission_122.mjs').read_text(encoding='utf-8')
s=s.replace('residualAdmission-122','residualFit-122').replace('/native-residual-admission','/native-residual-fit')
s=s.replace("const figure=root+'/residual-admission-control.png';","const figure=root+'/visibility-control.png';")
s=s.replace("build+'/residualCapacity-122-deck-receipt.json'","build+'/residualAdmission-122-deck-receipt.json'")
s=s.replace('review.checks_passed!==39','review.checks_passed!==29')
a=s.index("replace('完整畸变相机的全尺寸验证'");b=s.index('const images=',a)
s=s[:a]+"""replace('畸变相机接入材料对应验证','拟合与遮挡使用同一畸变相机');
replace('39项检查通过；人体拟合与遮挡判断仍需接入。','投影、连续自遮挡与衣物轮廓的29项检查通过。');
replace('实拍面部仍为13.049像素，尚未放行训练。','GPU合成投影差约0.000016像素，实拍仍未通过。');
replace('绿圈：观测；蓝点：完整畸变；红叉：忽略畸变（最高6.705像素）。','左：完整畸变。中：零畸变。右：差异位置。未启动人物训练。');
"""+s[b:]
s=s.replace('frame=image.frame,width=460','frame=image.frame,width=860')
s=s.replace('image.frame={left:410,top:190,width,height:width*3/4};','image.frame={left:210,top:240,width,height:width*128/576};')
s=s.replace('四台已知合成相机的48个留出目标，绿圈为观测、蓝点完整畸变预测、红叉忽略畸变；图内无文字','已知前后两平面的完整畸变深度、零畸变深度及原像素差异位置，保持原始比例，图内无文字')
a=s.index('p.resolve(slide.id).speakerNotes.text=');b=s.index(".concat([previous]).join('\\n\\n');",a)+len(".concat([previous]).join('\\n\\n');")
notes=['2026-10-28 R6拟合、连续自遮挡、衣物轮廓与叠加预览的完整输出相机适配。',*review['findings_zh'],'图：每格192×128，左完整畸变，中零畸变，右为深度差异的原始像素位置。绿色近平面，蓝色远平面。颜色与遮挡顺序已实际审阅，图中无文字。合成控制不代表实际人物改善。','证据：native-residual-fit/result-completed.json、review.json、serialization-repair.json。以下保留历史备注。']
s=s[:a]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[b:]
s=s.replace('preserve_deck_residualAdmission_122.py','preserve_deck_residualFit_122.py')
s=s.replace('02_有限视角重建优化路线_122畸变材料验证_v2.pptx','02_有限视角重建优化路线_122畸变拟合与可见性.pptx')
s=s.replace("receiptPath:build+'/residualFit-122-final-validation-v2.json'","receiptPath:build+'/residualFit-122-final-validation.json'")
assert not (H/'update_deck_residualFit_122.mjs').exists()
(H/'update_deck_residualFit_122.mjs').write_text(s,encoding='utf-8')
s=(H/'preserve_deck_residualAdmission_122.py').read_text(encoding='utf-8').replace('residualAdmission-122','residualFit-122')
(H/'preserve_deck_residualFit_122.py').write_text(s,encoding='utf-8')
print('R6_DECK_READY')
