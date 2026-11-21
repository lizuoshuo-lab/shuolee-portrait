import hashlib,json,re,shutil
from datetime import datetime
from pathlib import Path

LAB=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
OUT=BASE/'current/smplx-gvhmr-prior-fresh-019'
VIEWER=BASE.parent/'viewer'
BUILD=Path(__file__).parent
stamp=datetime.now().astimezone().isoformat(timespec='seconds')

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

assert (OUT/'prediction-receipt.json').exists()
review=read(OUT/'review/review-manifest.json')
conclusion=dict(created=stamp,state='independent_prior_reviewed_alignment_pending',
    accepted=False,keep_as_candidate=True,
    evidence='VPoser downloaded and CUDA verified earlier. This phase independently predicts SMPL-X with released GVHMR/HMR2/ViTPose weights.',
    observed_improvement='At source23, same pelvis-centered side orientation shows reduced knee flexion vs018 control. Raw13.69/10.53 degrees; official PP13.47/9.19; previous20.01/21.05.',
    residual_problem='Calibrated overlays still show displaced head, wrist and feet, especially right-side source45/58. Mean hand pose does not model observed fists. Body shape and world registration are not accepted.',
    comparison_limit='Geometry panels use common orientation and metric template units but each pelvis is translated to center; they cannot prove scene placement. Joint angles are model quantities, not subject ground truth.',
    visual_reviewed=['crop-review.jpg','review/geometry-left90.jpg','review/geometry-front.jpg','review/source-000000.jpg','review/source-000023.jpg','review/source-000045.jpg','review/source-000058.jpg'],
    not_yet_reviewed=['review/geometry-above25.jpg','review/source-000011.jpg','review/source-000033.jpg','full temporal continuity'],
    next='Fresh fixed-camera world registration from raw image observations and this official independent temporal prediction. First keep local bone poses fixed and optimize shared scale/root transforms; only then test bounded local correction. No old personalized result initializes it.',
    new_personalized_fit_steps=0,new_gaussian_steps=0,company_deployment=False,
    license='GVHMR source permits research/education/nonprofit; no additional commercial deployment authorization for this project is inferred from earlier other-model licensing.',
    weights='Pinned public mirror files verified against LFS SHA256; official independent release checksums were not available.')
save(OUT/'phase-conclusion.json',conclusion)
review.update(state='reviewed_candidate_world_alignment_pending',visual_review_pending=False,
              visual_review_scope=conclusion['visual_reviewed'],accepted=False)
save(OUT/'review/review-manifest.json',review)
save(OUT/'run-state.json',dict(created=stamp,state=conclusion['state'],running=False,
     old_person_fit_loaded_for_initialization=False,pretrained_prediction_frames=218,real_unique_frames=175,
     gaussian_training_steps=0,personalized_fit_steps=0,quality_accepted=False))

dest=VIEWER/'body-fitting-method/gvhmr-prior'
dest.mkdir(parents=True,exist_ok=True)
for name in ['geometry-left90.jpg','geometry-front.jpg','source-000000.jpg','source-000023.jpg','source-000045.jpg','source-000058.jpg','review-manifest.json']:
    shutil.copy2(OUT/'review'/name,dest/name)
shutil.copy2(OUT/'phase-conclusion.json',dest/'phase-conclusion.json')
section='''<!-- GVHMR_PRIOR_019_BEGIN -->
<section id="gvhmr-prior-019"><h2>时序人体先验：同时利用图像、关节与相机运动</h2>
<p>VPoser约束单个身体姿态是否符合训练分布；GVHMR进一步利用连续帧中的图像特征、二维关节点与相机相对旋转，联合预测人体姿态、体形和运动。相机运动与人物运动必须分开，绕人物拍摄不能按固定相机处理。</p>
<p>本地已从原视频独立运行GVHMR，没有读取旧拟合参数作起点。原视频约24.03帧/秒，按真实时间重采样为30帧/秒输入；重复采样没有增加新视角。已有相机内参与去畸变保持一致，59张标定图与重新解码的像素完全相同。</p>
<figure><a href="body-fitting-method/gvhmr-prior/geometry-left90.jpg"><img src="body-fitting-method/gvhmr-prior/geometry-left90.jpg" alt="原有拟合与GVHMR原始预测、官方后处理的侧面网格对照"></a><figcaption>实际网格：原有5000步拟合、GVHMR原始预测、官方后处理。固定时刻23，使用相同观察方向；各模型只将骨盆平移到共同中心，未逐个转正。这是局部形态比较，不能证明场景位置对齐。</figcaption></figure>
<p>这一时刻，新预测的膝部弯曲较小：原拟合左右约20°／21°，原始预测约14°／11°。这些是网格关节的几何角度，真实关节角度未知，不能把更小的数字直接当作精度提高。</p>
<h3>局部姿态更合理，仍要重新检查场景配准</h3>
<figure><a href="body-fitting-method/gvhmr-prior/source-000045.jpg"><img src="body-fitting-method/gvhmr-prior/source-000045.jpg" alt="同一实拍45与原有拟合、GVHMR原始预测和官方后处理的校准相机叠图"></a><figcaption>同一实拍与真实相机投影。GVHMR的头部、手腕和脚仍有位置偏差；默认手姿态也没有表达照片中的握拳。</figcaption></figure>
<p>官方后处理根据预测的静止关节修正位移，再用逆运动学调整关节。它有助于运动一致性，但不能代替本场景的标定和地面证据。当前保留GVHMR作为候选先验，先固定其局部姿态，从原始观测重新求共享尺度与整体变换，检查全部真实视角后再开放有限的关节修正。</p>
<div class="route-links"><a href="body-fitting-method/gvhmr-prior/source-000000.jpg">左侧叠图</a><a href="body-fitting-method/gvhmr-prior/source-000023.jpg">正面叠图</a><a href="body-fitting-method/gvhmr-prior/source-000058.jpg">右侧叠图</a><a href="body-fitting-method/gvhmr-prior/geometry-front.jpg">正面网格</a></div>
<details><summary>实验附录：输入与复现范围</summary><p>官方代码ee960bb，GVHMR核心247个状态张量严格匹配；HMR2.0a和ViTPose-H也严格加载。RTX5080，Torch2.7.1/CUDA12.8；使用独立环境兼容当前显卡。三个权重来自固定版本公开镜像，SHA256匹配镜像LFS记录，未取得作者独立哈希。</p><p>完整视频解码177帧；本次仅用标定覆盖的索引1至175。30fps输入218个位置对应175个不同实拍帧；相机旋转在已标定节点之间用球面插值，按每张实际图像的拍摄时间关联，插值相机不是真值。人物框由已有二维遮罩插值后按官方比例扩大，未使用YOLO或重新运行视觉里程计。因此属于有标定输入的本地适配，而非默认demo原样复现。</p><p>已检查六张关键网格/叠图和输入裁切；完整时间一致性尚未验收。0次个体优化、0次高斯训练，没有替换查看器或部署00。<a href="body-fitting-method/gvhmr-prior/phase-conclusion.json">精简判定</a> · <a href="https://github.com/zju3dv/GVHMR">官方论文与实现</a></p></details>
</section>
<!-- GVHMR_PRIOR_019_END -->
'''
page=VIEWER/'body-smplx.html'
html=page.read_text(encoding='utf-8')
assert 'GVHMR_PRIOR_019_BEGIN' not in html
html=html.replace('<!-- BODY_PRIOR_SHARED_REVIEW_BEGIN -->',section+'\n<!-- BODY_PRIOR_SHARED_REVIEW_BEGIN -->',1)
html=html.replace('下一步核验独立的时序人体先验，没有新增高斯训练或部署00。','GVHMR独立时序预测已完成并在上方展示，局部屈膝减轻但整体配准仍未通过，没有新增高斯训练或部署00。')
page.write_text(html,encoding='utf-8')

status=read(BASE/'work-status.json')
status.update(updated=stamp,current_phase=conclusion['state'],local_training_running=False,local_inference_running=False,
              current_decision='GVHMR独立预测局部屈膝减轻，右侧头部/手腕/脚配准仍未通过；保留为新的先验候选，先做从原始观测出发的整体配准。',
              next_action=conclusion['next'],goal_accepted=False)
status['latest_local_execution']=dict(kind='gvhmr_temporal_prior_fresh_019',state=conclusion['state'],running=False,
    frames=218,unique_source_frames=175,calibrated_observations=59,new_gaussian_training_steps=0,new_personalized_fit_steps=0,
    directory=str(OUT),review=str(OUT/'review/review-manifest.json'),next=conclusion['next'])
status['gvhmr_prior_019']=dict(conclusion=str(OUT/'phase-conclusion.json'),downloaded_verified=True,
   independent_prediction_completed=True,visual_review_completed=True,world_alignment_accepted=False)
save(BASE/'work-status.json',status)
planpath=BASE/'long-term-plan.md'
plan=planpath.read_text(encoding='utf-8')
old=re.search(r'下一步核验GVHMR官方时序人体预测作为独立起点。.*?\n\n',plan,re.S)
assert old
new='019已下载并核验GVHMR、HMR2.0a和ViTPose-H，运行218个30fps采样位置（175个独立实拍帧），没有使用旧个体参数初始化。59标定图与原视频去畸变像素完全一致，按真实图像时间关联移动相机。固定时刻23的左右膝弯曲由018控制20.01/21.05度变为独立原始预测13.69/10.53度，后处理13.47/9.19度；只表示模型形态改变，不是关节真值。右侧原图叠加仍有头、手腕及脚偏移，整体未通过。\n\n下一步020从原始观测与官方GVHMR预测重新建立共享尺度和场景根变换，先固定局部骨架，仅检验配准，再比较有界关节修正。不能把019的骨盆居中展示当成世界对齐，也不能用官方最低关节归零代替实测地面。未训练新高斯，不部署00。候选原始/后处理数组与必要证据暂时保留；SMPLest-X仍为后备，未下载8.2GB权重。\n\n'
plan=plan[:old.start()]+new+plan[old.end():]
plan=plan.replace('[身体先验](http://127.0.0.1:8870/body-smplx.html#vposer-ready) ·','[时序人体先验](http://127.0.0.1:8870/body-smplx.html#gvhmr-prior-019) · [身体先验](http://127.0.0.1:8870/body-smplx.html#vposer-ready) ·',1)
planpath.write_text(plan,encoding='utf-8')
print(json.dumps(dict(state=conclusion['state'],page=str(page),out=str(OUT)),ensure_ascii=False))
