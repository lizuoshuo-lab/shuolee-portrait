from pathlib import Path
import json,re
from PIL import Image
build=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
src=(build/'update_face047r2_slide.mjs').read_text(encoding='utf-8')
notes=json.loads(re.search(r'slide\.speakerNotes\.text=(".*?");',src,re.S).group(1))
notes=notes[:notes.index('五、实际作用与观察')]
notes=notes.replace('唯一优化变量差异为总目标增加0.2 L_face','048的两组都包含0.2 L_face')
notes=notes.replace('控制组计算但不回传此项。','两组的面部目标完全相同。')
notes=notes.replace('这是下一阶段需要检查的具体设计。','048新增项针对最终世界头颈位姿，原有修正量时序项仍保留。')
notes+='''五、最终世界位姿的连续约束
令R_t^root为根世界旋转，R_tk为已修正的局部关节旋转，C(j)为按父子关系排列的根到关节链，则Q_tj=R_t^root ∏_{k∈C(j)}R_tk。只取SMPL-X关节12颈和15头，父关系为9至12至15。x_tj为原生关节蒙皮后的世界位置，s是模型米到世界单位的共享尺度。原始采样时间来自视频时间戳，选取原视频第1、4、7直至175帧，共59帧，相邻间隔约0.12485秒。
令d_t=τ_(t+1)−τ_t，旋转残差a_tj=||Q_(t+1,j)−Q_tj||_F / (sqrt(2) d_t σ_R)，平移残差b_tj=||x_(t+1,j)−x_tj||_2 / (s d_t σ_x)。
新增目标L_seq=[2(T−1)]^-1 Σ_t Σ_{j∈颈,头}[ρ(a_tj)+ρ(b_tj)]，ρ(r)=sqrt(1+r²)−1。矩阵弦距离在小角度下近似旋转角度；代码没有把它误写成精确测地线距离。位置差除以s以后使用模型米。共同的世界平移在相邻差中抵消。
σ_R=0.24弧度/秒，σ_x=0.048模型米/秒。总目标增加0.10 L_seq，在300至500步线性渐增。前后帧均接收梯度。控制组权重0，候选组0.10，其他目标、初始化与优化日程完全一致。两组分别从原始预测重新开始，候选不加载控制结果。
这是针对短片中人物近似静止的有界假设。相机已经统一到世界坐标，不能把相机绕行直接作为人物头部旋转。对真实快速动作，该速度惩罚可能过度平滑，需要另外验证，不应无条件复用本片的尺度。
梯度通过有限差分核验，解析值−1.115237、数值值−1.115084。固定世界刚体变换后损失最大差约7.75e−7，恒定位姿损失为0。验证过程没有优化器更新且原变量逐值恢复。

六、受控结果
本页两幅图分别为33和45帧，都是未加入面部目标的诊断帧。每幅图三列为原图、已有鼻嘴约束、加入最终世界头颈连续项。诊断帧仍参与身体、鞋纹等其他目标，绝不是独立未见视角测试。
两组各1200步，实际初始化逐数组一致。新控制模型的文件哈希与047的面部约束模型一致，确认从头重现原对照。485个面部训练点中位残差2.989至3.158像素，50个诊断点12.333降至4.087像素，诊断点95分位20.355降至8.027像素。图中侧面鼻嘴位置更接近原图。
58对相邻帧的最终世界头部旋转变化中位1.873降至0.442度，95分位7.981降至2.105度；连接诊断帧的9对旋转变化中位6.356降至0.656度。以上用NumPy的常规中位数，偶数样本取中间两项均值，不能与训练日志Torch下中位数混用。模型头部位置速度中位0.05734降至0.01397模型米/秒。这些都是拟合模型的运动量，不是真实头部运动测量。
头部以下464个身体点中位残差10.102至10.668像素，可见耳部66点8.076至9.824像素。第23帧模型左膝屈曲13.994至14.145度、右膝9.398至9.755度。它们不是人体真值，也不证明侧面站姿正确。原生SMPL-X网格重放最大差0，只说明实现一致。

七、解决范围与不足
局部收益：在相同鼻嘴观测下，直接约束最终世界位姿，减小了面部诊断帧的对齐偏差和相邻帧的模型跳变。此前仅平滑修正量的设计不能消除原始姿态自身的波动。
仍未解决：已检查全部59张原图叠加、关键头部细节与7个固定视角，完整站姿、膝部、头颈轮廓没有得到明确改善。没有独立身体真值，没有连续动画验收。该人体模型没有通过高斯训练底模验收，不部署电脑00。
当前保留这一局部方法，候选参数只供只读诊断，下一轮仍从原始输入重新开始。继续使用已有VPoser、GVHMR、NLF与许可模板，SAM3DBody暂缓。先分解肩肘髋膝的观测与先验冲突，不凭穿衣轮廓强行拉直膝盖，不继续盲目增加头部权重。整体几何通过后才开展20万新高斯、18000步训练。本轮2400次身体更新，0次高斯更新，0次远端部署。

八、方法来源与本地证据
SMPL-X：https://github.com/vchoutas/smplx 。face-alignment：https://github.com/1adrianb/face-alignment 。NLF：https://github.com/isarandi/nlf 。VPoser：https://github.com/nghorbani/human_body_prior 。
51点表面对应及姿态、平移时序正则参考本地VHAP_smplx/vhap/model/tracker_smplx.py。官方参考使用其自身参数和时序安排；048的最终世界旋转弦距离、模型尺度归一化和双向梯度是本地适配，不宣称完整官方复现。
046保留原始FAN观测、对应与可见性选择。048的study-design、head-sequence-protocol、gradient contract、comparison、phase-conclusion记录试验设定、数学核验、实际数值和验收范围。网页为http://127.0.0.1:8870/body-smplx.html#head-sequence-048。
'''
(build/'head048-notes.txt').write_text(notes,encoding='utf-8')
s=src.replace('face047r2','head048')
s=s.replace('smplx-face-surface-fresh-047','smplx-head-sequence-fresh-048')
s=s.replace('5eed49a3377d6a656c1666feaca9df5ab8b7b272ba2760fa4f9b653a43a04b92','9f4402c574b250789bf3e74472248b0ff04f6bf4d95fb1e13561fa6c3d472727')
s=s.replace('face_surface_local_gain_geometry_not_accepted','head_temporal_local_gain_geometry_not_accepted')
s=s.replace('observable045r4-deck-receipt.json','face047r2-deck-receipt.json')
start=s.index('const replace=');end=s.index("await fs.writeFile(build+'/head048-edited-layout.json'",start)
phase=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/smplx-head-sequence-fresh-048/review')
positions=[]
for i,f in enumerate(['head-000033.jpg','head-000045.jpg']):
    w,h=Image.open(phase/f).size;W=[565,549][i];H=W*h/w
    positions.append(dict(left=[66,665][i],top=365-H/2,width=W,height=H))
block='''const replace=(before,after)=>{const e=row.elements.find(x=>x.text===before);if(!e)throw Error('Missing text '+before);p.resolve(e.id).text.replace(before,after);};
replace('面部表面约束与头颈对齐','头颈连续约束改善诊断帧的局部对齐');
replace('以鼻嘴表面点约束头部，联合检查相邻帧的姿态一致性。','在鼻嘴约束上，抑制世界坐标中最终头颈位姿的跳变。');
replace('每组：原图、身体约束、鼻嘴约束。左44帧拟合，右45帧诊断。','每组：原图、鼻嘴约束、增加连续约束。左33帧，右45帧。');
replace('有监督帧的鼻嘴投影改善。\\n留出帧仍有偏差，头颈及整体几何未通过。','诊断帧鼻嘴更贴合，相邻头部位姿更平滑。\\n整体站姿、膝部与头颈轮廓仍未通过。');
const images=row.elements.filter(x=>x.kind==='image').sort((a,b)=>a.position.left-b.position.left);
if(images.length!==2)throw Error('Expected two images');
const positions=POSITIONS;
for(const [i,file] of ['head-000033.jpg','head-000045.jpg'].entries()){
  const im=p.resolve(images[i].id);
  im.replace({blob:await fs.readFile(phase+'/review/'+file),contentType:'image/jpeg',fit:'contain',alt:'面部诊断帧的原图、鼻嘴约束和最终世界头颈连续约束'});
  im.crop={left:0,top:0,right:0,bottom:0};im.position=positions[i];
}
slide.speakerNotes.text=await fs.readFile(build+'/head048-notes.txt','utf8');
'''.replace('POSITIONS',json.dumps(positions))
s=s[:start]+block+s[end:]
s=s.replace('02_有限视角重建优化路线_面部对应_定稿2.pptx','02_有限视角重建优化路线_头颈连续_定稿.pptx')
(build/'update_head048_slide.mjs').write_text(s,encoding='utf-8')
print('Prepared focused edit of slide 55 with existing layout and verified render sources.')
