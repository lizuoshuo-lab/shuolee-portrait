from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import json
import hashlib

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/robust-portrait-122/native-residual-admission'
review=json.loads((O/'review.json').read_text(encoding='utf-8'))
assert review['checks_passed']==39
review.update(actual_image_review=True,figure_review='All48targetrecords in four native-aspect views; correctD blue aligns withgreen knownobservations, ignoredD red differs. No selected residuals, warped image or magnified arrows.')
for parent in [O,T/'viewer/122-residual-admission',Path('D:/00/工作/查看器/122-residual-admission'),Path('D:/00/工作日志/2026-10-27/内参文件解析/122-residual-admission'),T/'viewer/new-capture-118/calibration-note/122-residual-admission',Path('D:/00/工作/查看器/new-capture-118/calibration-note/122-residual-admission')]:
    (parent/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source=Path('D:/00/报告/02_有限视角重建优化路线.pptx')
with ZipFile(source) as z:
    masters=[n for n in z.namelist() if n.startswith('ppt/slideMasters/') and n.endswith('.xml')]
    layouts=[n for n in z.namelist() if n.startswith('ppt/slideLayouts/') and n.endswith('.xml')]
    records=[]
    for name in masters+layouts:
        root=ET.fromstring(z.read(name))
        records.append(dict(part=name,sha256=hashlib.sha256(z.read(name)).hexdigest(),
            placeholders=[e.attrib for e in root.iter() if e.tag.endswith('}ph')],
            canvas_names=[e.attrib.get('name') for e in root.iter() if e.tag.endswith('}cSld')]))
(H/'residualAdmission-122-template-inspection.json').write_text(json.dumps(dict(source=str(source),masters_and_layouts=records,policy='Edit existing lastslide only; no master/layout/theme modification.'),ensure_ascii=False,indent=2),encoding='utf-8')
s=(H/'update_deck_residualCapacity_122.mjs').read_text(encoding='utf-8')
s=s.replace('residualCapacity-122','residualAdmission-122').replace('/native-residual-capacity','/native-residual-admission')
s=s.replace("const figure=root+'/r3-landscape-rgb.png';","const figure=root+'/residual-admission-control.png';")
s=s.replace("build+'/residualCamera-122-deck-receipt.json'","build+'/residualCapacity-122-deck-receipt.json'")
s=s.replace('review.checks_passed!==19','review.checks_passed!==39')
s=s.replace("const slide=snap[51];","const slide=snap[51];\nawait fs.writeFile(build+'/residualAdmission-122-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));")
a,b=s.index("replace('残余畸变"),s.index('const images=')
s=s[:a]+"""replace('完整畸变相机的全尺寸验证','畸变相机接入材料对应验证');
replace('20万高斯，双视角更新与实际界面回放已核验。','39项检查通过；人体拟合与遮挡判断仍需接入。');
replace('两路浮点回放差为0；实拍标定仍未通过。','实拍面部仍为13.049像素，尚未放行训练。');
replace('原生1520×2704与2704×1520，完整K、外参与畸变。','绿圈：观测；蓝点：完整畸变；红叉：忽略畸变（最高6.705像素）。');
"""+s[b:]
s=s.replace('20万完整协方差三阶SH高斯的2704×1520合成原生渲染，图内无文字','四台已知合成相机的48个留出目标，绿圈为观测、蓝点完整畸变预测、红叉忽略畸变；图内无文字')
s=s.replace('frame=image.frame,width=620','frame=image.frame,width=460')
s=s.replace('image.frame={left:330,top:190,width,height:width*1520/2704};','image.frame={left:410,top:190,width,height:width*3/4};')
a=s.index('p.resolve(slide.id).speakerNotes.text=');b=s.index(".concat([previous]).join('\\n\\n');",a)+len(".concat([previous]).join('\\n\\n');")
notes=['2026-10-28 R5完整畸变相机正式准入与材料对应验证。',*review['findings_zh'],'图：四格按相机0、1、2、3排序，每格320×240原生坐标。绿圈已知独立观测、蓝点完整模型预测、红叉忽略畸变预测。48个留出目标全部保留，误差线未放大。合成观测并非实拍人脸。','证据：native-residual-admission/result.json、review.json、upstream-projection-audit.json。以下保留历史备注。']
s=s[:a]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[b:]
s=s.replace('preserve_deck_residualCapacity_122.py','preserve_deck_residualAdmission_122.py').replace('02_有限视角重建优化路线_122畸变全尺寸回放.pptx','02_有限视角重建优化路线_122畸变材料验证_v2.pptx')
s=s.replace("receiptPath:build+'/residualAdmission-122-final-validation.json'", "receiptPath:build+'/residualAdmission-122-final-validation-v2.json'")
(H/'update_deck_residualAdmission_122.mjs').write_text(s,encoding='utf-8')
s=(H/'preserve_deck_residualCapacity_122.py').read_text(encoding='utf-8').replace('residualCapacity-122','residualAdmission-122')
(H/'preserve_deck_residualAdmission_122.py').write_text(s,encoding='utf-8')
print('R5_DECK_READY')
