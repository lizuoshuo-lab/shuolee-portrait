from pathlib import Path
import json, hashlib
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/material-support-gradient-110';R=B/'current/person-static-shared-face-fresh-098'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
result=read(O/'result.json');check=read(O/'geometry-only-check.json');probe=read(O/'opacity-probes.json');queries=read(O/'queries.json')
assert result['queries']==160 and check['check_passed'] and result['opacity_only_decreased']==7
rows=read(R/'data/training-inputs.json')['records']
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
canvas=Image.new('RGB',(1200,430),'white');draw=ImageDraw.Draw(canvas)
chosen=[(100,7429,'面部与眼镜'),(136,9609,'头顶'),(136,10412,'裤腿')]
evidence=[]
for col,(raw,tid,label) in enumerate(chosen):
    p=next(p for p in probe if p['raw']==raw and p['track_id']==tid)
    q=next(q for q in queries if q['raw']==raw and q['track_id']==tid)
    path=Path(rows[raw-1]['image'].replace('/mnt/c/','C:/'));im=Image.open(path).convert('RGB')
    u,v=q['observed_uv'];left=int(np.clip(round(u)-200,0,im.width-400));top=int(np.clip(round(v)-140,0,im.height-280))
    crop=im.crop((left,top,left+400,top+280));ink=ImageDraw.Draw(crop)
    x,y=u-left,v-top;ink.ellipse((x-5,y-5,x+5,y+5),outline='#28fca0',width=2)
    ink.line((x-9,y,x+9,y),fill='#28fca0',width=1);ink.line((x,y-9,x,y+9),fill='#28fca0',width=1)
    canvas.paste(crop,(400*col,42))
    draw.text((col*400+8,8),f'{label}  原视频{raw:06d}',font=font,fill='#173744')
    decrease=100*(1-p['after_penalty']/p['before_penalty'])
    draw.text((col*400+8,330),f"约束值 {p['before_penalty']:.3f}  降至 {p['after_penalty']:.3f}",font=font,fill='#173744')
    draw.text((col*400+8,360),f'点位置、形状未变，下降 {decrease:.2f}%',font=small,fill='#173744')
    evidence.append(dict(raw=raw,track_id=tid,source=str(path),source_sha256=sha(path),box=[left,top,left+400,top+280],probe=p))
draw.text((8,400),'实拍原比例裁片仅标示核验位置。数值来自原生高斯的不透明度反事实，不是训练改进。',font=small,fill='#344956')
canvas.save(O/'material-review-slide.png')
decreases=np.array([100*(1-p['after_penalty']/p['before_penalty']) for p in probe])
decision=dict(created=datetime.now().astimezone().isoformat(),status='geometry_only_auxiliary_verified_not_trained',
    summary='发现辅助几何约束可以通过调整贡献比例而下降。仅推动三维点位置的版本已通过160个原生查询的数值与梯度核验，尚未训练，不能宣称重影修复。',
    evidence=dict(queries=160,sample_views=9,opacity_cases=7,opacity_only_decreased=7,
                  opacity_decrease_percent_range=[float(decreases.min()),float(decreases.max())],
                  exact_objective_difference=check['values']['maximum'],center_gradient_max_relative_error=check['center_gradients']['maximum'],
                  auxiliary_scale_rotation_opacity_gradients_zero=check['noncenter_gradients_zero']),
    interpretation=['098实际重影尚未解决。110检查的是待采用的有限支撑目标及冻结模型，不是对098完整训练历史的因果重放。',
        '仅改不透明度可降低目标，证明该数值不是几何改善的充分证据。光度和透明度本身仍需学习，不能全局锁死不透明度。',
        '160个查询中，原目标中心梯度有2个与固定贡献几何校正方向相反。其余方向并未普遍反向，不能将所有误差归为错误梯度。',
        '原图审阅发现旧group3的torso名称过窄，实际包含面框外的头发。重分区得到面部框下半24、上半41、面框外头部11、肩臂躯干高度带42、腿鞋高度带42。高度带不等于语义分割。109肩臂躯干高度带单独重算为1.459到2.728像素，仍然退化。详见region-reporting-correction.json。',
        '用于诊断的材料点由图像与估计相机三角化，非真实三维标定。仍存在特征身份、人物静态假设和真实相机误差的不确定性。',
        '102的43个空观测中，有9个在原mask半值外，11个缺少既有合格入向匹配。空射线的梯度为零，也不能盲目用补点将不可靠观测都拟合进去。',
        '110最初整身坐标原点的FP32二次式发生相减消失。改以同一世界目标作为计算原点后，保持阈值不变通过；该数值问题不属于此前098的已证实根因。'],
    implementation='material_geometry_support.py增加独立辅助渲染，冻结当次光栅化贡献权重和局部度量的梯度，只将几何约束传给中心。常规颜色、透明度和尺度损失保留原渲染路径。当前尚未接入训练器。',
    next_bounded_work='准备全身材料点的独立拟合/验证划分，重新从原2D和074相机求共享XYZ；检查所有175视角的有效观测与全新200k初始化覆盖。随后才决定一次18k步从头训练。不能把110诊断XYZ或098模型用作训练初始化。',
    new_training=False,new_point_cloud=False,quality_accepted=False,goal_complete=False,
    actual_gaussian='098',source_cameras='074',visual_evidence=evidence)
decision['presentation_notes']=[decision['summary'],decision['implementation'],
    '9个预先指定相机中的160个材料点查询。更正区域标注后：面部框下半24、上半41、面框外头部11、肩臂躯干高度带42、腿鞋高度带42。每旧区域按原始轨迹ID均匀选择，未根据残差挑点。',
    '7个预先指定的仅不透明度反事实，logit最大改动0.05，点位及协方差固定。目标下降范围'+str(decision['evidence']['opacity_decrease_percent_range'])+'%，最大alpha改变量小于0.0007。未改变模型文件。',
    '新辅助梯度160/160通过，前向目标完全一致。中心梯度相对误差最大'+str(check['center_gradients']['maximum'])+'，不透明度、尺度及旋转梯度均为零。未接入或运行新训练。',
    *decision['interpretation'],decision['next_bounded_work'],
    '图源：原工作视频的去畸变实拍裁片；绿圈表示核验位置。详细证据 '+str(O/'result.json')+'；'+str(O/'geometry-only-check.json')]
save(O/'decision.json',decision)
print(json.dumps(dict(image=str(O/'material-review-slide.png'),decrease_range=decision['evidence']['opacity_decrease_percent_range'],next=decision['next_bounded_work']),ensure_ascii=False))
