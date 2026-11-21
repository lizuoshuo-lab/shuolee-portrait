from pathlib import Path
import json, hashlib, shutil, re
from datetime import datetime

LAB=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
VIEWER=BASE.parent/'viewer'
R53=BASE/'current/smplx-absolute-foot-audit-053'
R54=BASE/'current/smplx-foot-containment-fresh-054'
R55=BASE/'current/smplx-world-body-audit-055'
BUILD=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now().astimezone().isoformat(timespec='seconds')
assert not (R54/'phase-conclusion.json').exists()
d=read(R54/'review/comparison.json')
assert d['initial_arrays_equal'] and all(d['initial_arrays_equal'].values())
assert d['control_exactly_reproduces_052_reviewed_shoes'] and d['new_body_updates']==2400
assert d['new_gaussian_updates']==0 and not d['old_person_used_as_initializer']
assert read(R55/'audit.json')['all_sources_unchanged']
reviewed=['all-source-00.jpg','all-source-12.jpg','all-source-24.jpg','all-source-36.jpg','all-source-48.jpg',
          'source-000023.jpg','feet-000045.jpg','fixed-front.jpg','fixed-left45.jpg','fixed-right45.jpg',
          'fixed-left90.jpg','fixed-right90.jpg','fixed-above25.jpg','fixed-below25.jpg']
for n in reviewed:assert (R54/'review'/n).is_file()
next_action=('继续使用现有SMPL-X、NLF、HaMeR和已下载VPoser；SAM3DBody暂缓。'
 '足部单向轮廓的局部改善不足以解决站姿，不继续扫描足部权重。'
 '055已核验：现有身体平滑惩罚的是相对初始预测的修正量，而最终世界骨架仍随时间变化；'
 '固定第23帧投影到所有相机使759点残差中位数约翻倍，不能直接强制全身静止。'
 '下一项先把原始视频分段，对照实际可见肢体和最终世界骨架，区分真实微动、遮挡与逐帧先验不一致。'
 '据此选择一个有界的实际世界姿态连续性对照；不得将模型变化或与拍摄角度相关性直接当成误差真值。'
 '若此约束仍无整体收益，停止该先验拟合小项叠加，转查原始图像直接驱动的身体配准参考实现。'
 '所有新拟合仍重算原始预测初始化、零共享体形和新优化器；旧拟合只读。'
 '整体几何通过后再做20万新高斯、18000步。眼镜专项停止，本地未通过前不部署00，不同步E盘。')
findings=[
 '原图逐帧审阅显示，解剖学脚踝被裤腿遮盖，检测器位置不能当作直接观测的骨关节。11个代表性/高残差帧的原人物遮罩可用于排除轮廓外位置，但边缘有小孔、锯齿与轻微侵蚀；不把它称为精确鞋部语义分割。',
 '按中性姿态的蒙皮归属及踝高选择493个足部顶点，占全身8917个包含顶点的5.53%，却贡献当前全身单向包含代价的47.60%。新增项对左右脚分别按帧数和顶点数归一化，再等权平均；保留原全身约束和5像素容差、15像素稳健尺度，候选权重为2。',
 '两组各1200步，均从原始NLF预测、原59帧相机、零共享体形和新优化器开始，使用同样24条核验鞋部特征及11/13原划分。原始初始化所有数组完全相同；对照模型哈希精确复现052的核验鞋部组，唯一新项为足部单向包含。',
 '在原图分辨率重新计算距离场，足部493×59次投影中越出人物遮罩5像素的比例从7.8248%降至3.8540%，95分位越界距离8.0896降至3.9266像素。该统计使用训练原图遮罩，仅证明图像可行性改善，不证明真实深度或三维足部位置。',
 '身体759点残差中位数13.6767至13.7977像素，头部以下464点10.8240至10.8662像素；50个面部诊断点3.9550至3.9631像素。全体网格顶点两方案位移中位数仅0.347模型毫米，95分位2.311模型毫米。',
 '全部59帧接触表、完整第23帧、放大脚部第45帧与7个固定视角已审阅。足部边缘有局部变化，整体膝部、站姿和躯干仍没有明确改善，头手整体变化很小。未通过整体几何，不采用054作为新的高斯底模。',
 '055只读核验最终原生关节链：054候选脊柱末端相对第23帧最大旋转约12.99度，左膝位置最大变化约5.02模型厘米。这些是拟合模型的变化，可能包含真实运动或预测误差。将第23帧骨架冻结到全部相机，身体残差中位数13.80增至28.68像素；单一固定姿态没有得到现有图像支持。',
 '本轮共2400次身体拟合更新，0次高斯更新，0次新权重下载。已下载身体先验继续可用，整体几何尚未验收。']
con=dict(updated=now,state='local_foot_boundary_improved_whole_body_not_accepted',running=False,
 geometry_accepted=False,candidate_promoted=False,new_gaussian_updates=0,new_body_updates=2400,
 source_download_bytes=0,arms=['control','foot_containment'],steps_per_arm=1200,from_scratch=True,
 findings=findings,reviewed_images=reviewed,next_action=next_action,
 input_equal=True,control_exact_reproduction=True,fit_sha256=d['fit_sha256'],
 statistics=d['statistics'],displacement_between_candidates_model_mm=d['displacement_between_two_candidates_model_mm'],
 source_audit='current/smplx-absolute-foot-audit-053/audit.json',
 motion_audit='current/smplx-world-body-audit-055/audit.json',
 public_url='http://127.0.0.1:8870/body-smplx.html#foot-containment-054')
save(R54/'phase-conclusion.json',con)
d.update(state='reviewed_local_boundary_gain_whole_body_rejected',visual_review_completed=now,reviewed_images=reviewed)
save(R54/'review/comparison.json',d)
a=read(R55/'audit.json');a.update(state='interpreted_no_static_body_assumption_accepted',
 interpretation=findings[6],next_action=next_action);save(R55/'audit.json',a)

# Existing section is extended as one explanation, without a new daily log page.
p=VIEWER/'body-smplx.html';old=p.read_text(encoding='utf-8')
assert old.count('<div id="shoe-support-052">')==1 and 'id="foot-containment-054"' not in old
insert='''<div id="foot-containment-054">
<h6>单向轮廓约束如何保护足部投影</h6>
<p>衣物和鞋的外表面包住人体，不能把裸身顶点直接吸引到它们的边界。单向包含约束只惩罚投影落在人物轮廓外的部分；轮廓内部没有边界吸引力。原图中的脚踝被裤腿遮挡，因此这一约束提供可行范围，不能独立确定脚踝深度。</p>
<p>全身顶点平均会稀释小部位的贡献。本次保持全身约束，另将左右脚分别归一化，再等权平均。令 <var>D</var><sub>t</sub> 为第 <var>t</var> 帧原图人物遮罩外部的像素距离，内部为零；<var>π</var><sub>t</sub> 为相机投影，<var>V</var><sub>s</sub> 为一只脚的顶点集合，<var>T</var> 为帧数。</p>
<math display="block"><mrow><msub><mi>ℒ</mi><mtext>foot-contain</mtext></msub><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><munder><mo>∑</mo><mrow><mi>s</mi><mo>∈</mo><mo>{</mo><mi>L</mi><mo>,</mo><mi>R</mi><mo>}</mo></mrow></munder><mfrac><mn>1</mn><mrow><mi>T</mi><mo>|</mo><msub><mi>V</mi><mi>s</mi></msub><mo>|</mo></mrow></mfrac><munder><mo>∑</mo><mrow><mi>t</mi><mo>,</mo><mi>i</mi><mo>∈</mo><msub><mi>V</mi><mi>s</mi></msub></mrow></munder><mi>ρ</mi><mo>(</mo><mfrac><mrow><mo>max</mo><mo>(</mo><msub><mi>D</mi><mi>t</mi></msub><mo>(</mo><msub><mi>π</mi><mi>t</mi></msub><mo>(</mo><msub><mi>v</mi><mrow><mi>t</mi><mi>i</mi></mrow></msub><mo>)</mo><mo>)</mo><mo>−</mo><mi>ε</mi><mo>,</mo><mn>0</mn><mo>)</mo></mrow><mi>σ</mi></mfrac><mo>)</mo><mo>,</mo><mspace width="1em"/><mi>ρ</mi><mo>(</mo><mi>z</mi><mo>)</mo><mo>=</mo><msqrt><mrow><mn>1</mn><mo>+</mo><msup><mi>z</mi><mn>2</mn></msup></mrow></msqrt><mo>−</mo><mn>1</mn></mrow></math>
<p>本地对照中，容差 <var>ε</var> 为 5 像素，稳健尺度 <var>σ</var> 为 15 像素，新增损失权重为 2。容差用于保留遮罩边缘的不确定性；稳健函数降低较大残差的影响。足部顶点由模板蒙皮归属与脚踝高度预先确定，不根据拟合误差挑选。</p>
<figure><a href="body-fitting-method/nlf-prior/foot-containment-comparison.jpg"><img src="body-fitting-method/nlf-prior/foot-containment-comparison.jpg" alt="第45帧实拍与两组从头人体拟合，比较足部独立归一化约束"></a><figcaption>本地实测，第45帧。实拍、原约束、增加足部单向包含约束。</figcaption></figure>
<p>原图分辨率下，足部投影越出遮罩超过 5 像素的比例约减半。固定侧面和整体站姿仍没有明确改善：轮廓排除了部分不合理投影，但轮廓内的深度和关节位置依然不唯一。</p>
</div>
'''
anchor='<details><summary>实验附录：纹理归属核查与从头对照</summary>'
assert old.count(anchor)==1
new=old.replace(anchor,insert+anchor)
extra='''<p>足部归一化对照仍为两组各1200步、59帧，初始化数组完全相同。493个足部顶点占全身包含顶点的5.53%；在原图距离场复算，越界超过5像素比例7.8248%至3.8540%，越界距离95分位8.0896至3.9266像素。身体759点残差中位数13.6767至13.7977像素，网格位移中位数0.347模型毫米。全部59帧与7个固定视角已审阅，整体几何未通过。</p>
<p>后续世界姿态核验发现：身体平滑约束主要作用于相对先验的修正量；最终世界姿态仍有变化，但固定同一帧投影到全部相机使残差约翻倍。该现象不能独立区分真实微动与预测误差，不能据此强制人体完全静止。<a href="body-fitting-method/nlf-prior/foot-containment-side.jpg">固定侧面</a> · <a href="body-fitting-method/nlf-prior/foot-containment-source.jpg">完整实拍叠加</a> · <a href="body-fitting-method/nlf-prior/foot-containment-conclusion.json">本轮结论</a>。</p>
'''
pattern=r'(<details><summary>实验附录：纹理归属核查与从头对照</summary>.*?)(</details>)'
new,count=re.subn(pattern,lambda m:m.group(1)+extra+m.group(2),new,count=1,flags=re.S)
assert count==1
public=VIEWER/'body-fitting-method/nlf-prior'
mapping={'feet-000045.jpg':'foot-containment-comparison.jpg','fixed-left90.jpg':'foot-containment-side.jpg','source-000023.jpg':'foot-containment-source.jpg'}
for source,dest in mapping.items():
    shutil.copy2(R54/'review'/source,public/dest)
    assert sha(R54/'review'/source)==sha(public/dest)
save(public/'foot-containment-conclusion.json',con)
p.write_text(new,encoding='utf-8')
save(R54/'publication.json',dict(created=now,html_before_sha256=hashlib.sha256(old.encode()).hexdigest(),html_after_sha256=sha(p),
    files={str(public/d):sha(public/d) for d in [*mapping.values(),'foot-containment-conclusion.json']},browser_verified=False))

status_path=BASE/'work-status.json';status=read(status_path)
status.update(updated=now,current_phase='smplx_foot_containment_054_world_motion_audit_055',
 current_decision='足部轮廓可行性局部改善，整体几何未通过；已完成世界骨架只读核查，继续现有先验与原图运动核验。',
 local_training_running=False)
status['latest_local_execution']=dict(kind='smplx_foot_containment_054',state=con['state'],running=False,from_scratch=True,
    original_frames=59,arms=2,steps_per_arm=1200,gaussian_updates=0,accepted=False,directory=str(R54))
status['current_experiment'].update(status='foot_boundary_local_improvement_geometry_not_accepted',latest_fitting_trial='054-foot-containment',
    current_training_running=False,candidate_accepted=False,current_candidate_role='048保留为必要只读结果；054局部改善但整体几何未通过，不晋升。')
status['foot_containment_054']=dict(state=con['state'],conclusion_path=str(R54/'phase-conclusion.json'),
    review_complete=True,geometry_accepted=False,new_body_updates=2400,new_gaussian_updates=0,next_action=next_action)
status['world_body_audit_055']=dict(state=a['state'],audit_path=str(R55/'audit.json'),read_only=True,geometry_accepted=False)
status['latest_body_handoff']=dict(updated=now,state='reviewed_publication_cleanup_pending',running=False,geometry_accepted=False,
    gaussian_training_started=False,body_updates_this_pair=2400,new_download_bytes=0,evidence_url=con['public_url'],next_action=next_action)
save(status_path,status)
plan_path=BASE/'long-term-plan.md';plan=plan_path.read_text(encoding='utf-8')
section='<!-- FOOT_CONTAINMENT_054 -->\n## 当前结论：轮廓约束只改善了足部边界\n\n'
section+='\n\n'.join(findings)+'\n\n### 接续方向\n\n'+next_action+'\n\n'
section+='证据：[当前对照](http://127.0.0.1:8870/body-smplx.html#foot-containment-054)、[精简结论](current/smplx-foot-containment-fresh-054/phase-conclusion.json)、[世界骨架只读核验](current/smplx-world-body-audit-055/audit.json)。\n'
section+='<!-- END_FOOT_CONTAINMENT_054 -->'
plan,count=re.subn(r'<!-- SHOE_SUPPORT_052 -->.*?<!-- END_SHOE_SUPPORT_052 -->',lambda _:section,plan,count=1,flags=re.S)
assert count==1;plan_path.write_text(plan,encoding='utf-8')

notes='''【足部观测、刚体归属与轮廓包含】
穿鞋图像的脚尖、脚跟与裸脚模型端点定义不同。取消六个足部点会让部分脚趾投影越出鞋面；单纯移除这些约束不足以解决绝对配准。

纹理轨迹需先核对所处表面。66条图像低处轨迹中，原图审阅确认24条位于鞋面或鞋带，21条在裤脚，21条归属不确定。保留原ID划分后11条用于约束、13条作同源图像诊断。相对运动映射为 X_t=T_t T_r^(-1) X_r；所有帧共同右乘常量H，(T_t H)(T_r H)^(-1)=T_t T_r^(-1)。因此相对纹理项自身不能唯一确定骨架与鞋的绝对关系，仍需关节、接触和轮廓等约束。

单向轮廓包含允许裸脚位于鞋的外轮廓内，避免强迫皮肤贴到衣物表面。D_t为原图人物遮罩外的像素距离、内部为0；ρ(z)=√(1+z²)−1。每只脚分别按帧数T和足部顶点数|V_s|归一化，再对左右脚等权平均：L_f=½ Σ_s [1/(T|V_s|)] Σ_(t,i∈V_s) ρ(max(D_t(π_t(v_ti))−ε,0)/σ)。本次ε=5px、σ=15px、新增权重2。左右251/242个顶点由模板蒙皮与踝高确定，保留原全身包含项及其他损失。人物遮罩有小孔和锯齿，不把它称为精确鞋部真值。

两组各1200步，使用相同59帧原图、相机、24条核验鞋轨迹、原始NLF预测、零共享体形与新优化器。初始化数组完全相同；对照精确复现上一组模型。原分辨率复算493×59个足部投影，越界超过5px比例7.8248%降至3.8540%，95分位距离8.0896降至3.9266px。身体759点残差中位数13.6767至13.7977px；两组网格位移中位数仅0.347模型毫米。59帧实拍叠加与7个固定视角没有显示整体站姿突破。本地图像可行性改善不等于真实三维几何精度。

【世界姿态与近似静态假设】
现有身体平滑主要约束相对原始预测的修正量，不等价于最终世界骨架连续性。055以原生关节父子链只读复算，候选脊柱末端相对第23帧最大旋转约12.99度，左膝位置最大变化约5.02模型厘米。这些变化可能包含真实微动和预测误差。固定第23帧人体到全部相机，身体残差中位数从13.80增至28.68px，不能直接强制全身静止。下一步应由原始视频的可见运动核验约束强度，禁止把“膝盖更直”或单个指标降低当成独立三维验收。

适用范围：当前仅有水平顺序视频，无独立三维人体真值。足部边缘局部改善，整体站姿、膝部和头颈轮廓仍待解决。高斯尚未重训；整体几何通过后按20万新点、18000步执行。继续使用已有身体先验。

本地证据：current/smplx-shoe-absolute-support-051/semantic-admission.json；current/smplx-shoe-absolute-support-051/gauge-contract.json；current/smplx-absolute-foot-audit-053/audit.json及review-decision.json；current/smplx-foot-containment-fresh-054/review/comparison.json及phase-conclusion.json；current/smplx-world-body-audit-055/audit.json。全部路径相对于当前报告上一级shuolee目录。
'''
(BUILD/'foot054-notes.txt').write_text(notes,encoding='utf-8')
print(json.dumps(dict(state=con['state'],web=con['public_url'],reviewed_source_frames=59,fixed_views=7,notes_chars=len(notes)),ensure_ascii=False))
