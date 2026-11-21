from pathlib import Path
import json,shutil
H=Path(__file__).parent
old='surface123-face-observation';new='r7-pipeline122'
figure=json.loads((H/(old+'-figure.json')).read_text(encoding='utf-8'))
(H/(new+'-figure.json')).write_text(json.dumps(figure,ensure_ascii=False,indent=2),encoding='utf-8')
code=(H/'update_deck_surface123_face_observation.mjs').read_text(encoding='utf-8').replace(old,new)
code=code.replace("read(build+'/surface123-fixed-direction-deck-receipt.json')","read(build+'/surface123-face-observation-deck-receipt.json')")
code=code.replace("read(root+'/grayT-face-observation-cross-check/actual-review.json')","read(root+'/../robust-portrait-122/native-residual-pipeline/actual-review.json')")
a=code.index("replace('单一栏杆");b=code.index('const images=',a)
code=code[:a]+"""replace('脸部观测改善了误差，但仍不是静态对齐','静态对齐未通过；训练入口检查完成');
replace('当前眼部权重为0，结果含逐帧变化，不能当作静态验收。','36项工程检查通过；眼部仍缺约束，人像尚未对齐。');
"""+code[b:]
a=code.index('p.resolve(slide.id).speakerNotes.text=');b=code.index('const candidatePath=',a)
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-residual-pipeline')
d=json.loads((O/'conclusion.json').read_text(encoding='utf-8'))
notes=[d['summary'],d['next'],'本页保留此前真实人物证据图；新R7仅为已知96点非零畸变合成工程控制，并未修复或重训人物。','CPU检查22项，耗时27.57秒；GPU检查14项，耗时61.57秒。总12次小场景优化更新，少于登记16次和120秒CUDA预算。','缓存变量覆盖导致重复加载但像素相等，不能作为既有人脸错位根因。原始失败记录保留，修复后相同帧实际复用张量。','完整K/E/D传递，缺失系数拒绝；两视角梯度与独立联合损失最大差3.73e-9；同相机原生回读RGB/alpha逐值一致。','同一实际失败输入在新拟合/训练入口均返回3，保留102目标与13脸部观察、最大13.0488像素，GPU/初始化/优化器未启动。','证据：'+str(O/'conclusion.json')]
code=code[:a]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['此前研究记录：',previous]).join('\\n\\n');\n"+code[b:]
code=code.replace('123面部观测回放.pptx','122原生训练入口核验.pptx').replace('preserve_deck_surface123_face_observation.py','preserve_deck_r7_pipeline122.py')
(H/'update_deck_r7_pipeline122.mjs').write_text(code,encoding='utf-8')
for src,dst in [('preserve_deck_surface123_face_observation.py','preserve_deck_r7_pipeline122.py'),('commit_surface123_face_observation_deck.py','commit_r7_pipeline122_deck.py')]:
 (H/dst).write_text((H/src).read_text(encoding='utf-8').replace(old,new),encoding='utf-8')
print('Prepared minimal slide52 text update; existing person image preserved.')
