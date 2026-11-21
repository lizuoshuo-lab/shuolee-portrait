from pathlib import Path
from PIL import Image
import hashlib,json,shutil
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
old='surface123-scene-support';new='surface123-fixed-direction'
source=R/'grayT-fixed-direction-review/frame_080_completed_raw_edges_fixed_cameras.png'
figure=H/(new+'-native-figure.png');shutil.copy2(source,figure)
w,h=Image.open(source).size
(H/(new+'-figure.json')).write_text(json.dumps(dict(path=str(figure),width=w,height=h,sources=[dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())]),ensure_ascii=False,indent=2),encoding='utf-8')
code=(H/'update_deck_surface123_scene_support.mjs').read_text(encoding='utf-8').replace(old,new)
code=code.replace("read(build+'/surface123-articulation-deck-receipt.json')","read(build+'/surface123-scene-support-deck-receipt.json')")
code=code.replace('grayT-scene-support-cross-run-review/actual-review.json','grayT-fixed-direction-review/actual-review.json')
a=code.index("replace('增加表情");b=code.index('const images=',a)
code=code[:a]+"""replace('头部附近场景验证仍不充分','单一栏杆方向无法验证全部外参');
replace('两组已有留出残差中位1.19和1.50像素，材料身份待核实。','两组留出方向RMS约0.57和0.59像素，仍不能保证人脸对齐。');
replace('局部区域实际BA支持为0条与1条，尚不能据此判定外参错误。','数学控制：绕栏杆方向额外旋转7°，线方向预测保持不变。');
replace('00复现的一条栏杆轨迹：54、58、61帧。原图保持比例。','原图／边缘测量／两套相机预测。最大线偏差2.35像素。');
"""+code[b:]
code=code.replace('00复现实际进入相机BA的同一栏杆候选轨迹，54、58、61帧的原生256像素方形裁片。青圈定位，没有图内文字，没有形变。材料身份未通过精细验证。','第80帧原生裁片，依次为原图、有效边缘测量、研究与00冻结相机方向预测。图内无文字，原图比例不变。预测线通过观测中心，不能检验绝对位置。')
a=code.index('p.resolve(slide.id).speakerNotes.text=');b=code.index('const candidatePath=',a)
d=json.loads((R/'fixed-direction-conclusion.json').read_text(encoding='utf-8'))
notes=[d['summary'],d['next'],
 '独立方向控制：56和72帧6条anchor边缘，64和80帧6条held边缘，共480个有效原生采样。只为各冻结相机求一个公共三维方向，没有优化相机或三维点。实际审阅4组完成图和4组初始测量图。',
 '第一轮第64帧的三个上端种子来自阴影边缘，测量在相机比较前停止。保留原始失败记录，只纠正3个起点x，复用其余9条测量。未改变物理边缘、行范围、搜索宽度、门槛或留出划分。',
 '研究侧held方向误差median0.321704 RMS0.569363 max2.199605像素；00为median0.332691 RMS0.592848 max2.350204像素。这里不是面部误差，也不是完整相机绝对重投影。',
 '7度公共方向旋转负对照对线系数影响小于7e-13；线方向不能识别这一旋转分量以及任何平移。合成2度俯仰控制被检出，只用于检查实现。',
 '上下半段边缘方向差0.120372至3.883690度，物理杆弯曲、光照、采样及残余图像扭曲未分离。端点正负0.5像素敏感度0.392970至0.658604度，不是统计置信区间。',
 '仅2个held视角。没有完成全部95帧相机或人脸小于1像素验收，不判定鱼眼、K、外参或微动中的某一项为唯一原因。正式静态训练未准入，历史人物点云保持。',
 '证据：'+str(R/'fixed-direction-conclusion.json')]
code=code[:a]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),'此前研究记录：',previous]).join('\\n\\n');\n"+code[b:]
code=code.replace('123场景支持核查.pptx','123固定方向核查.pptx').replace('preserve_deck_surface123_scene_support.py','preserve_deck_surface123_fixed_direction.py')
(H/'update_deck_surface123_fixed_direction.mjs').write_text(code,encoding='utf-8')
for a,b in [('preserve_deck_surface123_scene_support.py','preserve_deck_surface123_fixed_direction.py'),('commit_surface123_scene_support_deck.py','commit_surface123_fixed_direction_deck.py')]:
    (H/b).write_text((H/a).read_text(encoding='utf-8').replace(old,new),encoding='utf-8')
print('Prepared fixed direction slide52 edit; canonical unchanged.')
