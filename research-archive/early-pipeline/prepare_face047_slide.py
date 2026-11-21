from pathlib import Path
import json
from PIL import Image
B=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
root=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/smplx-face-surface-fresh-047')
s=(B/'update_observable045_slide.mjs').read_text(encoding='utf-8')
s=s.replace('smplx-observable-landmark-fresh-045','smplx-face-surface-fresh-047')
s=s.replace('6c225c91fb2549d73c1d0db2454813f320efb0231c54bb7184569175ef5d79cc','5eed49a3377d6a656c1666feaca9df5ab8b7b272ba2760fa4f9b653a43a04b92')
s=s.replace("evidence.state!=='observable_landmarks_no_clear_alignment_gain'","evidence.state!=='face_surface_local_gain_geometry_not_accepted'")
s=s.replace("build+'/vposer043-deck-receipt.json'","build+'/observable045r4-deck-receipt.json'")
lo=s.index("replace('身体姿态先验");hi=s.index('const images=',lo)
s=s[:lo]+'''replace('身体拟合中的可见性与关节对应','面部表面约束与头颈对齐');
replace('先确认图像中真正可见的位置，再约束对应的模型表面。','以鼻嘴表面点约束头部，联合检查相邻帧的姿态一致性。');
replace('原图与三组拟合对照，以及同一世界相机下的侧面几何。','左：44帧参与面部拟合。右：45帧仅作面部诊断。');
replace('可见耳部投影略有改善，整体几何仍未通过。\\n手腕观测的分歧尚不能由删点解决。',
        '有监督帧的鼻嘴投影改善。\\n留出帧仍有偏差，头颈及整体几何未通过。');
'''+s[hi:]
s=s.replace("['source-000045.jpg','fixed-left90.jpg']","['head-000044.jpg','head-000045.jpg']")
s=s.replace("phase+'/review/'+file","phase+'/diagnosis/'+file")
s=s.replace("alt:i?'固定23帧同一侧面相机，三组身体拟合':'45帧实拍、原约束、可见耳部、可见耳部和单腕约束'",
            "alt:i?'45帧鼻嘴不参与拟合：原图、对照、面部约束':'44帧鼻嘴参与拟合：原图、对照、面部约束'")
pos=[]
for name,(x,y,w,h) in zip(['head-000044.jpg','head-000045.jpg'],[(66,174,565,382),(665,174,549,382)]):
    iw,ih=Image.open(root/'diagnosis'/name).size;factor=min(w/iw,h/ih)
    pos.append(dict(left=x+(w-iw*factor)/2,top=y+(h-ih*factor)/2,width=iw*factor,height=ih*factor))
a=s.index('  im.position=');b=s.index(';',a)
s=s[:a]+'  im.position='+json.dumps(pos)+'[i]'+s[b:]
notes='''一、观测与模型点的对应
身体先验VPoser、GVHMR和NLF均已在本地可用。先验提供姿态范围；个体与相机的实际对齐仍依赖图像观测。此前可见耳部筛选去除了52个不可见侧观测，但仅靠身体17点不足以可靠约束头颈。衣领边界不能代替位于人体内部的颈部骨架关节。
当前新增SMPL-X的51个静态面部表面点。每个点由一个指定三角面f_j及三项重心坐标b_jk定义，X_tj=Σ_k b_jk V_t,f_jk，Σ_k b_jk=1。V为完成蒙皮与世界变换的顶点。该定义与SMPL-X原生输出的最后51点逐点核对，最大差2.98×10^-8模型米。IBUG的0基索引17至67对应这些点，轮廓0至16不纳入本次目标。

二、原图检测与可见性
已有SFD先在原始头部附近512像素区域确定候选框，不进行外部缩放；将框平移回原图坐标后，3DFAN4在全分辨率原图中预测。检查59张图，55张有唯一有效头部；0、1、16、17帧缺失或存在歧义，排除。模型权重全部已有，没有新增下载。
对有明显鼻嘴可见性的12至34帧，采用0基鼻点29至35、外唇48至59；其余有效帧保守采用鼻嘴中线30、33、51、57。眉毛、眼周和轮廓因头发、眼镜及侧视遮挡不参与。此选择是本序列的人工可见性假设，不能视为通用真实标签。
535个观测中485点参与目标，11、23、33、45、58帧的50点只作面部诊断。这5帧仍参与身体、鞋纹等目标，因此不是独立测试集。模型检测响应不等于可见概率，裁切重放在个别帧最多有3像素差异，不宣称亚像素测量。

三、两组独立从头实验
两组均重新从原始NLF预测计算初始化，使用59张原图、固定相机、许可SMPLX_MALE模板、零体形、原始HaMeR局部手姿态与新Adam。没有复用旧个体网格、形变、高斯或优化器；初始姿态、根位姿、尺度与原始初始化逐数组相同。
每组1200步。1至300步优化根旋转、平移和共享尺度；301至700步开放共享姿态与体形；701至1200步增加有界逐帧姿态。学习率分别0.008、0.006、0.004，第1000步后乘0.4，梯度范数上限10。
姿态修正为c_t=0.30 tanh(a)+0.10 tanh(b_t)，单位弧度；R_t=Exp([c_t]×)R_t^0。a为共享参数，b_t为逐帧参数，R_t^0为原始预测。根旋转每轴界限0.35弧度；根平移每轴0.30倍初始尺度；尺度s=s0 exp(0.30 tanh(q))。10维体形从0开始、每维范围±1，下颌与表情固定。手部只采用HaMeR局部关节，不移植裁切相机的根旋转。
两组保留相同身体、脚部、手部、遮罩包含、正深度、接触、鞋纹、时序和NLF局部表面目标。身体、脚、手的稳健尺度20、18、10像素，权重1、0.2、0.5。NLF1024个表面点使用0.06模型米尺度，权重1；鞋纹尺度10像素、权重0.3。原始预测都不是三维真值。

四、面部表面目标
令e_tja=[π_t(X_tj)]_a−[u_tj]_a，其中π_t使用原标定相机，像素中心约定减0.5，a为横纵坐标。稳健函数ρ(r)=sqrt(1+r²)−1。
L_face=(1/|T|)Σ_t∈T (1/n_t)Σ_j m_tj Σ_a ρ(e_tja/σ_face)。m为保留指示，n_t为该帧保留点数，T为有面部监督的帧集合。逐帧归一化避免正面点数较多的帧自动占更大权重。对横纵坐标分别计算稳健项，与先算二维距离再计算一次稳健项不同。
唯一优化变量差异为总目标增加0.2 L_face，σ_face=8像素，第300至500步渐增。控制组计算但不回传此项。头颈姿态梯度通过有限差分核对，未改变初始化。
原时序项对21个关节的姿态修正求平均，形式包含mean ρ((c_t−c_t−1)/0.04)，总权重0.05，并附根位姿变化项。它平滑的是修正量，并非直接平滑最终真实头部旋转；原始姿态中的波动仍可能保留。这是下一阶段需要检查的具体设计。

五、实际作用与观察
本页左图为44帧，面部点参与优化；右图为相邻45帧，其面部点仅用于诊断。每幅图依次为原图、身体约束控制组、增加鼻嘴表面目标。全部来自原图和模型实际渲染。
485个面部训练点的中位误差15.827降至2.989像素，50个诊断点15.259降至12.333像素。训练帧的鼻嘴位置更接近原图。身体原始759点8.840升至12.345像素；拆分后，243个头部点6.779升至15.072像素，464个头部以下身体点9.902至10.102像素。误差增加主要在另一套头部观测上，不能据此宣称全身几何恶化。
RTMW身体鼻点与FAN鼻尖相差中位14.112像素；RTMW同一次预测内部的身体鼻点与面部鼻尖也相差6.897像素。模型回归鼻点与表面鼻尖在三维中相距约8.4模型毫米，投影约9.5像素，差异方向与观测不一致。改用原生头部顶点也不能自动消除冲突。所有观测均非真值，不能只选离当前模型近的检测。
两组局部头部修正差异在相邻面部训练帧间变化中位0.771度，而在连接面部诊断帧的相邻帧间为3.731度；颈部为0.559与1.976度。它们是两个拟合结果间修正的变化，不是真实运动测量。相邻帧缺少某类监督时，原先验与图像项可能把头部推向不同姿态。

六、仍未解决的问题与后续
有监督帧的局部鼻嘴投影改善。留出帧、头颈及整体站姿未通过，不能升级为高斯训练底模。脸部点也不能直接解决衣物下的膝肘位置。全部59帧和关键正侧面已审阅，尚无独立身体真值，未验收连续动画。原生网格重放0和面部投影重放小于0.00026像素，只确认计算链一致。
下一步继续核查RTMW133中的面部点与FAN对应、遮挡与像素误差方向，再从原始输入重新比较共享头颈修正和小幅时序变化。不得导入047个体参数作为新实验初值。既有VPoser、GVHMR、NLF继续可用，SAM3DBody暂缓。整体几何验收后才开始20万新高斯、18000步训练。本轮2400次身体更新、0次高斯更新、0次部署。

七、来源
SMPL-X：https://github.com/vchoutas/smplx 。face-alignment：https://github.com/1adrianb/face-alignment 。NLF：https://github.com/isarandi/nlf 。VPoser：https://github.com/nghorbani/human_body_prior 。
51点构造同时参考本地VHAP_smplx/vhap/model/tracker_smplx.py，不宣称完整复现其官方流程。模型模板由用户提供。当前结果是本地受控适配实验。
原始观测与对应依据保存在046的inference.json、mapping-contract.json、admission.json；047的protocol、comparison和diagnosis记录实际数据、对照结果与判定。'''
lo=s.index('slide.speakerNotes.text=');hi=s.index('await fs.writeFile',lo)
s=s[:lo]+'slide.speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+';\n'+s[hi:]
s=s.replace('observable045r4-', 'face047-')
# The previous receipt retains its original name.
s=s.replace("build+'/face047-deck-receipt.json','utf8'","build+'/observable045r4-deck-receipt.json','utf8'")
s=s.replace('02_有限视角重建优化路线_可见性_比例核验.pptx','02_有限视角重建优化路线_面部对应_定稿.pptx')
out=B/'update_face047_slide.mjs';assert not out.exists();out.write_text(s,encoding='utf-8')
print('Prepared slide55. Image aspect ratios preserved.')
