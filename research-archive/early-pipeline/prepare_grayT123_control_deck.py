from pathlib import Path
import hashlib,json,shutil
from PIL import Image
H=Path(__file__).parent;old='r7-pipeline122';new='grayT123-final-control'
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/grayT-final-control-review'
figure=O/'frame_057_input_static_deformed.png'
src=Path('D:/Projects/Shuolee-analysis-20261028/dynamic-face-fan-all-views/head-crops/frame_000057.png')
shutil.copy2(src,figure);im=Image.open(figure)
(H/(new+'-figure.json')).write_text(json.dumps(dict(path=str(figure),width=im.width,height=im.height,source=str(src),sha256=hashlib.sha256(figure.read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
code=(H/'update_deck_r7_pipeline122.mjs').read_text(encoding='utf-8').replace(old,new)
code=code.replace("read(build+'/surface123-face-observation-deck-receipt.json')","read(build+'/r7-pipeline122-deck-receipt.json')")
code=code.replace("read(root+'/../robust-portrait-122/native-residual-pipeline/actual-review.json')","read(root+'/grayT-final-control-review/visual-review.json')")
a=code.index("replace('脸部观测");b=code.index('const images=',a)
code=code[:a]+"""replace('静态对齐未通过；训练入口检查完成','逐帧改善了脸部，但统一姿态仍未通过');
replace('同一皮肤点35留出','95个训练视角：头部颜色误差下降19.35%，全身仅1.57%。');
replace('36项工程检查通过','眼眉未参与拟合；新视角仍有拉丝，尚未通过静态验收。');
replace('原图／RTMW有效点','第57帧：原图（闭眼）／静态对照／逐帧形变对照（仍张眼）。');
"""+code[b:]
a=code.index("alt:'");b=code.index("',fit:'contain'",a)
code=code[:a]+"alt:'第57帧：原图、静态对照、FAN逐帧头颈表情对照。原图闭眼而渲染仍张眼。原始工作分辨率裁片，保持比例、无新增图内文字。'"+code[b+1:]
a=code.index('p.resolve(slide.id).speakerNotes.text=');b=code.index('const candidatePath=',a)
con=json.loads((O/'conclusion.json').read_text(encoding='utf-8'))
notes=[con['summary'],*con['confirmed'],*con['unknown'],con['next'],
 '只读核验完成的00对照：95个训练相机、18000步、461598点。静态与逐帧对照使用同一初始化，不是新人物独立泛化测试。',
 '头部前景L1 0.0277799→0.0224055，84帧改善11帧退步；头部裁片mask L1 0.0822014→0.0531489，95帧改善；全身前景L1 0.0478256→0.0470740，71帧改善24帧退步。',
 '实际审阅95个头部三联图、6个全身三联图和12头部/12全身统一姿态新视角。连续角度图已查看，但未在本次原生交互界面验收，也没有新视角实拍真值。',
 '首次结果把899×1600的像素误写为native，已保留原报告并在unit-correction.json记录纠正。1520/899和2704/1600轴向比例给出严格上下界；相机参数和训练源未修改。',
 '此前原生训练入口R7的36项合成工程控制通过，不等于本人物物理对齐；本次没有新拟合、相机优化或训练。',
 '证据：'+str(O/'conclusion.json'),'图源：'+str(src)]
code=code[:a]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".join('\\n\\n');\n"+code[b:]
code=code.replace('122原生训练入口核验.pptx','123灰衣全帧与统一姿态核验.pptx')
code=code.replace('preserve_deck_r7_pipeline122.py','preserve_deck_grayT123_final_control.py')
(H/'update_deck_grayT123_final_control.mjs').write_text(code,encoding='utf-8')
for srcname,dst in [('preserve_deck_r7_pipeline122.py','preserve_deck_grayT123_final_control.py'),('commit_r7_pipeline122_deck.py','commit_grayT123_final_control_deck.py')]:
 (H/dst).write_text((H/srcname).read_text(encoding='utf-8').replace(old,new),encoding='utf-8')
print(json.dumps(dict(figure=str(figure),size=im.size,changed_slide=52)))
