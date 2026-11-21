from pathlib import Path
import json
from PIL import Image
B=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
s=(B/'update_vposer043_slide.mjs').read_text(encoding='utf-8')
s=s.replace("const phase=base+'/current/smplx-nlf-vposer-soft-fresh-043';", "const phase=base+'/current/smplx-observable-landmark-fresh-045';")
s=s.replace("const contact=base+'/current/smplx-nlf-vposer-compatibility-042';", "const contact=base+'/current/smplx-observable-landmark-audit-044';")
s=s.replace('1c9c2169126570254e14999ed259f494fd47b02263519178e4a705a68aead06c','6c225c91fb2549d73c1d0db2454813f320efb0231c54bb7184569175ef5d79cc')
s=s.replace('soft_vposer_no_clear_alignment_gain','observable_landmarks_no_clear_alignment_gain')
s=s.replace('vposer043','observable045').replace('contact-shape041-deck-receipt.json','vposer043-deck-receipt.json')
begin=s.index("replace('身体拟合中的体形与接触约束'")
end=s.index('const images=',begin)
s=s[:begin]+'''replace('身体姿态先验的适配与软约束','身体拟合中的可见性与关节对应');
replace('先核查编码解码对关节的影响，再用原图约束个体姿态。','先确认图像中真正可见的位置，再约束对应的模型表面。');
replace('左组：实拍、原拟合目标、增加软约束。右组：同一侧面相机。','原图与三组拟合对照，以及同一世界相机下的侧面几何。');
replace('软约束改善姿态表示的一致性，整体对齐仍未通过。\\n两组从头拟合，图像关节残差均约 9 像素。',
        '可见耳部投影略有改善，整体几何仍未通过。\\n手腕观测的分歧尚不能由删点解决。');
'''+s[end:]
s=s.replace("i?'固定23帧同一侧面相机，对照与VPoser软约束':'45帧实拍及不加VPoser项、增加软约束的两组从头拟合'", "i?'固定23帧同一侧面相机，三组身体拟合':'45帧实拍、原约束、可见耳部、可见耳部和单腕约束'")
# Use the full evidence images without cropping. Allocate space in the existing image band.
anchor='  im.crop={left:0,top:0,right:0,bottom:0};'
phase=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/smplx-observable-landmark-fresh-045/review')
rects=[]
for name,(x,y,w,h) in zip(['source-000045.jpg','fixed-left90.jpg'],[(66,174,565,382),(665,174,549,382)]):
    iw,ih=Image.open(phase/name).size
    scale=min(w/iw,h/ih);rw=iw*scale;rh=ih*scale
    rects.append(dict(left=x+(w-rw)/2,top=y+(h-rh)/2,width=rw,height=rh))
s=s.replace(anchor,anchor+'\n  im.position='+json.dumps(rects)+'[i];')
begin=s.index('slide.speakerNotes.text=`')
end=s.index('await fs.writeFile(build+',begin)
notes='''一、为什么检查观测对应
当前身体先验VPoser、GVHMR、NLF已可运行。042仅做确定性VPoser均值编码解码时，原始姿态的部分关节位置发生变化。043用保持原始起点的软约束优化后，姿态更接近先验表示，但整体站姿没有明显改善。因此本轮转到实际图像中哪些观测应当参与拟合。
检测器输出的二维点、由顶点加权回归的关节点、蒙皮骨架中的旋转中心具有不同定义。把三个名称相同的点强行看成同一个物理位置，会造成相互冲突的监督。宽松衣物下的关节中心通常不能直接看见，衣袖或衣领的边界也不等于腕关节或颈关节的投影。

二、原图核查与可见性
044核查全部59帧头部，并查看0、13、15、18、23、25、34、37、45帧的头部和双手细节。0帧的远侧右耳检测落到近侧脸颊，45帧远侧左耳落到近侧耳颊附近。所有59帧仍然有鼻、双眼和双耳五个有效身体检测点，说明网络响应不能直接充当可见概率。
原始身体腕点与HaMeR腕点距离中位左17.92、右15.90像素。只读036模型中，COCO表面回归腕点与SMPL-X骨架腕点的投影距离本身已有左5.92、右4.28像素。不能仅依据观测分歧就宣布HaMeR或身体检测更准确。HaMeR提供的二维关键点也是预测，不是实测真值。
既有手部可见区为左0至33、右16至58，共77个观测，三种裁切尺度一致性筛选通过。相邻过渡帧的手仍部分可见，本轮保留这些帧原来的身体腕点，不把它们一律当作完全遮挡。
耳部保守保留左0至24、右18至58，共66个观测，屏蔽52个。边界帧可见性存在不确定，远侧眼点本轮未改动，以便限制变量数量。

三、三组独立从头实验
对照保留原约束。第二组只屏蔽上述52个耳部观测。第三组在第二组基础上，移除有有效HaMeR监督处的77个COCO腕观测，原HaMeR的21个手点完全保留。
三组均重新从原始NLF59帧预测、去畸变图像与相机、许可SMPLX_MALE模板、零体形、原始HaMeR局部手姿态和新Adam开始。没有加载旧个体网格、高斯、形变、体形或优化器。原始初始化、鞋纹输入以及实际初始姿态逐数组完全一致。旧036只用于原图语义诊断，不作为拟合起点。
每组1200步。1至300步拟合根旋转、平移、共享尺度，301至700步开放共享姿态和体形，701至1200步开放有界逐帧修正，1000步后学习率乘0.4。根学习率0.008，共享参数0.006，逐帧参数0.004，梯度范数上限10。
共享和逐帧关节修正分别以0.30、0.10弧度为每轴tanh界限，通过SO(3)左乘组合。根每轴修正0.35弧度，根位移每轴0.30倍初始尺度，尺度s=s0 exp(0.30 tanh(q))。10维体形从0开始、范围±1，体形项0.04 mean(β²)。这是本地有界试验，参数范围不是独立测量结论。

四、固定分母的可见性目标
对第t帧、第j点，图像观测记u_tj，对应三维模型点X_tj，原始权重w_tj，筛选指示v_tj。固定标定相机投影记π_t。定义第a个像素坐标的残差e_tja=[π_t(X_tj)]_a−[u_tj]_a，a为横、纵坐标。身体图像损失为L_image=Σ_tj v_tj w_tj Σ_a ρ(e_tja/σ)/Σ_tj w_tj，σ=20像素，ρ(r)=sqrt(1+r²)−1。实现对横纵两个坐标分别计算稳健项后求和，与先取欧氏距离再计算一次稳健项不同。
三组分母均使用原始权重总和，不因删点减少。这样只改变被屏蔽点的贡献，不会同时把其他点的归一化权重抬高。此设计用于控制变量，不代表所有实际系统都必须固定分母。
脚和手的稳健尺度18、10像素，权重0.2、0.5，身体权重1。保留遮罩包含、正深度、原始姿态/关节约束、时序平滑、脚底高度及对应顶点稳定。裸人体不强制填满外套裤子。
鞋纹相对运动权重0.3、10像素尺度、100至300步渐增。NLF1024个局部三维表面点约束同索引顶点，尺度0.06模型米，不确定性权重按负1.5次幂并逐帧归一化，总权重1、300至500步渐增。原始预测具有偏差，不作为真值。本轮不再增加VPoser权重，保留已有NLF局部先验。

五、图像与相同观测的结果
本页左图是45帧原图与三组拟合，右图为固定23帧和同一世界侧面相机。所有照片和网格均来自实际数据与拟合，没有生成补全。
三组在共同630个身体观测上的中位投影残差为9.458、9.050、9.039像素，95%分位35.847、35.872、35.009像素。共同66个耳部点中位8.569、7.713、7.531像素。相同77个HaMeR腕点14.634、14.623、14.581像素，1540个手指点6.292、6.282、6.395像素。每个比较均保持统计集合相同。
未优化鞋纹诊断中位1.393、1.397、1.427像素。固定23帧模型左膝角14.03、14.08、13.70度，右膝9.42、9.46、9.21度，均为模型量，没有独立解剖真值。
全部59帧联系表、关键原图、固定正侧面和俯视已审阅。整体头颈与站姿仍没有明显改善。原生SMPL-X重放最大差0，说明参数导出和模板计算一致，不是几何正确的证据。

六、作用、局限与下一步
耳部筛选去除了明确不可靠的侧脸约束，并使保留耳部的投影残差略降。整体几何仍未通过。删除重复腕点没有解决手部位置。52个耳部屏蔽是保守人工判断，过渡帧和远侧眼点仍不确定。所有观测都是同一训练序列或网络预测，不能作为独立精度评估。未验收连续动画。
下一步利用现有VHAP_smplx代码中的51个内脸表面标志点。每个点用一个三角面及其重心坐标定义，可与68点检测中去掉17个轮廓点后的索引对应。现有FAN68只有12帧，裁切逆变换、像素中心、遮挡和响应必须重新核验，文件未被自动认定为可训练真值。
先验证可见鼻部、眼角和嘴部对应，再进行独立从头拟合。脸部约束不能直接解决被衣服遮挡的膝肘，也不能把内部颈关节强行贴到衣领。整体几何通过后才用20万新高斯训练18000步。本轮3600次身体更新、0次高斯更新、0次00部署，三个未通过模型不升级。

七、来源与复核
SMPL-X官方代码：https://github.com/vchoutas/smplx 。VPoser：https://github.com/nghorbani/human_body_prior 。NLF：https://github.com/isarandi/nlf 。模板由用户提供，本轮仅本地研究验证。
本地VHAP_smplx/vhap/model/tracker_smplx.py的51点构造和gt_offset=17为后续参考，其代码可能包含已有本地修改，不宣称本轮完整复现官方算法。
044/audit.json与review-decision.json保存原图观测核查，045各组protocol.json、review/comparison.json与phase-conclusion.json保存实际输入、代码指纹、结果和不采用判定。完整方法页保留前阶段VPoser公式和对照。'''
s=s[:begin]+'slide.speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+';\n'+s[end:]
s=s.replace('02_有限视角重建优化路线_姿态先验_定稿.pptx','02_有限视角重建优化路线_可见性_比例核验.pptx')
s=s.replace('observable045-', 'observable045r4-')
(B/'update_observable045_slide.mjs').write_text(s,encoding='utf-8')
print('Prepared existing slide55 update, no other slides changed.')
