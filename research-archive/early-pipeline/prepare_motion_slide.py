from pathlib import Path
import re

build=Path(__file__).resolve().parent
text=(build/'update_floor_contact_slide.mjs').read_text('utf-8')
text=text.replace("const expected='90e34be744e409aca7b26c81d43d10cb0d378d110fa1c5119c6c65a3981c9061';","const expected='fd9ba40cd05f3b6d9200f940da156255a2a7c20efdb8efbe42e846acbbefb5aa';")
text=text.replace("base+'/current/smplx-floor-contact-fresh-016/review/review-manifest.json'","base+'/current/smplx-convergence-fresh-018/review/review-manifest.json'")
text=text.replace('review.visual_review_pending||!review.control_contact_initialization_identical||!review.control_exactly_reproduces_015B','review.visual_review_pending||!review.all_initialization_arrays_identical||!Object.values(review.milestone1400).every(v=>v.exact_017_reproduction)')
replacements="""const replacements=new Map([
 ['身体先验与场景接触约束','身体先验与运动一致性'],
 ['姿态先验约束关节配置；真实图像、地面与接触关系约束人物在场景中的位置。','关节姿态、场景接触与时间一致性共同约束人体，着地点稳定仍需配合全身几何。'],
 ['同一时刻、同一地面参考相机：脚掌有所放平，膝部弯曲基本未变。','同一时刻、同一侧面相机：延长拟合后膝部略有变化，整体姿态仍未通过。'],
 ['接触约束未使整体通过；身体关节图像残差由8.12增至9.25像素。\\n下一步核查跨视角姿态与整体运动，通过几何验收后再训练高斯。','脚部纹理传递残差由43.36降至7.32像素，整体姿态仍未通过。\\n后续用时序人体预测检查髋、膝、踝与整体位置的联合解释。'],
]);"""
text,n=re.subn(r'const replacements=new Map\(\[.*?\n\]\);',lambda _:replacements,text,count=1,flags=re.S);assert n==1
text=text.replace("base+'/current/smplx-floor-contact-fresh-016/review/fixed-left90.jpg'","base+'/current/smplx-convergence-fresh-018/review/fixed-left90.jpg'")
text=text.replace('同一人物时刻23及地面参考相机，无接触项、含接触项与真实纹理稀疏三维支持','同一人物时刻23及地面参考相机，1400步控制、5000步控制、5000步脚底稳定与真实纹理支持')
text=text.replace('382*1560/826','382*2080/826')
notes="""const previousNotes=slide.speakerNotes.text;
if(!previousNotes.includes('身体先验、真实观测与场景接触'))throw Error('Original method notes missing.');
slide.speakerNotes.text=previousNotes.replace('下一步核查原始单目根旋转、逐帧位置和腿部姿态是否随观察方向漂移，用真实纹理轨迹区分运动与检测变化。准备共享下身位置与受限逐帧运动的从头比较，同时允许头部、手臂真实运动；不得强制全身静止或将衣物点直接用作关节真值。','此后已完成运动一致性与优化预算对照，见下文当前结论。')+`

当前方法：场景接触与时间一致性

017从原始观测重新初始化三组，每组1400步。接触控制与016接触结果逐字节一致。第二组对整体旋转、平移及髋/膝/踝/足局部姿态施加跨帧离散惩罚，第三组只约束每个脚底顶点的世界位置稳定。三组所有初始数组相同，无旧个体拟合或高斯参数输入。约束根据该段视频双脚近似着地的特征提出，不是通用动态运动模型。

共享下身项：整体旋转矩阵与跨帧均值之差除以0.075，平移差除以初始化尺度乘0.01，十个髋/膝/踝/足局部姿态中的实际选取为八个关节[0,1,3,4,6,7,9,10]，各轴角与均值之差除以3度。分别用rho(x)=sqrt(1+x^2)-1，三部分取平均。这里以轴角残差衡量局部姿态，不是严格SO(3)测地距离。头部和手臂保留逐帧自由度。

脚底稳定项：每只脚仍使用99个模板脚底顶点。各点在世界坐标中除以原始初始化尺度，减去该点跨59帧的均值，再除以0.01模型米，用同一鲁棒函数并对全部点、坐标、帧和双脚取平均。权重1，250步后用200步渐入。平均位置参与优化，没有固定到旧模型或预设站姿。相比同时固定整体朝向，这种约束允许身体在稳定脚点上发生变化。

运动检查只使用014筛选出的66条稳定鞋面纹理，共1289次观测。每条纹理在参考时刻23分配到较近且不歧义的脚，用踝、膝、脚关节构成的局部坐标系跨帧传递。静态纹理投影残差中位数0.631px。017控制、共享下身、稳定脚底的传递残差分别46.636、28.794、6.886px。这是预选稳定子集及近似脚坐标系上的检查，相机、纹理、鞋面与裸脚关系都会影响结果，不是独立运动真值。脚底稳定与此检查有关联，不能把指标下降当作整体准确率。

018为新的预算对照：接触控制和脚底稳定分别从原始数据重新拟合5000步。两组在1400步输出与017对应模型逐字节一致，因而后续变化可在同一方法内归为额外优化。没有载入旧模型续训，未改变权重、初始数组或学习率规则。前250步仅优化整体，接着500步开放共享体形和关节，其余4250步开放受限逐帧参数，1000步后学习率乘0.35。

图中A为017的1400步接触控制，只读比较。B为018接触控制5000步，C为018脚底稳定5000步，最后是原图纹理的稀疏三维支持。均使用同一时刻23和同一地面参考相机，未为了显示调整人体。左右膝几何弯曲分别为A23.39/21.63度、B20.01/21.05度、C19.25/19.95度。真实关节角度未知，数字只是模型几何量。

018控制与稳定组的鞋面纹理传递残差中位数为43.36和7.32px。左右脚底平面与地面的夹角分别为10.08/8.35度和9.76/8.03度。整体旋转跨帧离散中位数仍约10.20和9.63度。原图侧面/正面叠加、固定侧面/俯视四张关键图均已审阅，膝部和全身姿态仍未通过。更多步数改善了部分训练残差和局部弯曲，没有建立可信的完整几何。其余生成图未逐张宣称验收，未执行完整新视角视频验收。

只读局部目标检查：重放018最终参数，顶点世界坐标最大差1.91e-6，投影最大差0.000733px。固定髋、踝、体形、尺度和整体运动，仅给两个膝关节局部X旋转施加小扰动，不运行优化器、不生成替代模型。稳定组各减少5度时，膝几何弯曲约变为14.55/15.04度，总目标从0.5110增至0.9355。主要增加量为足部图像0.1838、身体图像0.0758、接触0.0915、脚底稳定0.0666，而单目关节与旋转先验合计仅约0.000742。因此未支持单一的“先验锁住膝盖”解释。这是局部方向检查，不能排除髋/膝/踝及整体位置共同调整后的更好解。

下一步选择独立时序人体预测。已阅读GVHMR官方安装、demo及后处理源码，确认输出SMPL-X参数，提供静止关节的整体位移修正与CCD逆运动学，demo为渲染将SMPL-X顶点转换到SMPL。拟将它作为新的原始预测参考，而不是在018结果上继续修改。需要首先核查原视频时间基准、图像裁切、已有相机内参和旋转约定。不能把每3帧采样的59观测直接当作30fps，也不能将移动相机设为静止。默认demo估计焦距，接入本项目时要用真实相机内参。原始网络输出与后处理输出必须分开评估，后处理把最低关节移到地面只是启发式，并非恢复本场景的实测地面。该模型尚未在本素材运行，没有新的实验成绩。

SMPLest-X作为后备单帧先验，官方代码和Huge权重公开，README注明8.2GB。它可能提供不同初始化，但本身不能替代时序约束。当前没有下载该大权重，后续遵循00优先下载的要求并复用本地已有文件。两条参考的模型与数据许可需要沿用已有授权范围，不据代码公开推断所有权重商业许可。

本阶段017/018共14200次新的身体更新，0次高斯更新。暂不部署00。保留当前对照、真实观测和精简证据，清理已审阅的重复失败模型。身体通过相同相机、多方向和时间一致性后，再从20万新高斯及新优化器训练18000步，眼镜专项仍停止。

官方来源：https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md；https://github.com/zju3dv/GVHMR/blob/main/tools/demo/demo.py；https://github.com/zju3dv/GVHMR/blob/main/hmr4d/model/gvhmr/utils/postprocess.py；https://github.com/MotrixLab/SMPLest-X/blob/main/README.md。
当前本地证据：${base}/current/smplx-convergence-fresh-018/review/review-manifest.json、motion-audit.json、knee-objective-audit.json与phase-conclusion.json。
`;"""
notes=notes.replace('十个髋/膝/踝/足局部姿态中的实际选取为八个关节','八个髋/膝/踝/足局部关节')
text,n=re.subn(r'slide\.speakerNotes\.text=`.*?`;',lambda _:notes,text,count=1,flags=re.S);assert n==1
for a,b in [('floor-slide-before','convergence018-slide-before'),('floor-contact-candidate','convergence018-candidate'),
 ('floor-final-validation','convergence018-final-validation'),('floor-final-renders','convergence018-final-renders'),
 ('floor-deck-receipt','convergence018-deck-receipt'),('20261023_场景接触_定稿','20261023_运动一致性_定稿')]:text=text.replace(a,b)
(build/'update_convergence018_slide.mjs').write_text(text,'utf-8')
print('Prepared targeted slide55 edit.')
