from pathlib import Path
import datetime, hashlib, json, os, shutil

BASE=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
RUN=BASE/'current/smplx-identity-prior-fresh-033'
VIEWER=BASE.parent/'viewer'
NOW=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

assert not (RUN/'phase-conclusion.json').exists()
r=read(RUN/'review/comparison.json');st=r['statistics']
assert r['prior_initialization_arrays_identical'] and r['actual_new_fit_updates']==2400
next_action='停止仅调整GVHMR体形先验中心或扫描权重。已筛选NLF-L独立三维点预测路线，先核验官方权重、真实相机内参接口与SMPL-X输出，再对六张原图做从头预测和几何对照。SAM3DBody授权权重仍待用户提供。几何未通过前不启动新的高斯训练、不部署00。'
summary=dict(created=NOW,state='identity_controls_reviewed_geometry_not_accepted',
    hypothesis='The fitted identity may compensate pose errors; test freezing the official shape and zero-centered shape regularization.',
    fresh_arms=['fixed-prediction','absolute-prior'],steps_per_arm=1200,new_body_updates=2400,new_gaussian_updates=0,
    all59_original_images=True,initialization_arrays_equal_to_original023=True,
    old_person_initialization=False,control='Preserved023 readonly; exact replay validated in032, unchanged source hash.',
    shape_prior_definitions={'control':'mean((beta-beta0)^2), weight0.04',
       'fixed-prediction':'beta=beta0, shape not optimized',
       'absolute-prior':'mean(beta^2), weight0.04, same beta0+[-1,1] bounds'},
    body_2d_px={a:s['body_2d_px'] for a,s in st.items()},
    beta_rms={a:s['beta_rms'] for a,s in st.items()},
    frame23_knee_flexion_degrees={a:{k:s['model_angles_degrees_not_GT'][k]['frame23'] for k in ['left_knee','right_knee']} for a,s in st.items()},
    common_depth_rays=r['common_depth_rays'],common_depth_violations=r['common_depth_violations'],
    actual_visual_review='Six source overlays plus fixed frame23 world front/left90/right90 inspected. Fixed identity alters trunk proportions but leaves side-view articulation and shoulder/arm placement insufficient. Absolute prior is visually close to the control. No accepted geometry improvement.',
    conclusion='Changing the identity prior alone did not resolve the rejected body fit in these bounded tests. This does not prove identity is correct or that every shape prior is ineffective.',
    caveats=['2D detector coordinates, fixed SfM cameras and reconstructed clothing points are diagnostics, not independent anatomical ground truth.',
        'Clothing is an outer surface; a naked body need not reproduce its contour.',
        'Only one zero-centered prior weight was tested, with otherwise identical bounds and objectives.',
        'Fixed predicted identity is not measured identity.'],
    quality_accepted=False,viewer_model_replaced=False,remote_deployment=False,next=next_action,
    evidence=['review/six-source-comparison.jpg','review/fixed-world-comparison.jpg','review/fixed-left90.jpg','review/comparison.json'])
save(RUN/'phase-conclusion.json',summary)
save(RUN/'visual-review.json',dict(reviewed=NOW,inspected=['six-source-comparison.jpg','fixed-front.jpg','fixed-left90.jpg','fixed-right90.jpg','source-000000.jpg','source-000023.jpg','source-000045.jpg'],
    observations=summary['actual_visual_review'],quality_accepted=False))
r.update(state='visually_reviewed_not_promoted',quality_accepted=False);save(RUN/'review/comparison.json',r)
for arm in summary['fresh_arms']:
    p=RUN/arm/'run-state.json';d=read(p);d.update(state=summary['state'],running=False,accepted=False);save(p,d)
save(RUN/'run-state.json',dict(updated=NOW,state=summary['state'],running=False,new_body_updates=2400,new_gaussian_updates=0,accepted=False))

status=read(BASE/'work-status.json');status.update(updated=NOW,current_phase='identity_prior_033',local_training_running=False,
    current_decision=summary['conclusion'],next_action=next_action)
status['body_identity_prior_033']={**summary,'report':str(RUN/'phase-conclusion.json')}
status['latest_local_execution']=dict(kind='identity_prior_033',state=summary['state'],running=False,directory=str(RUN),
    from_scratch=True,arms=2,body_fit_updates=2400,gaussian_updates=0,accepted=False)
status['current_experiment'].update(latest_fitting_trial='033',latest_fitting_steps=1200,latest_fitting_arms=2,
    latest_fitting_gaussian_steps=0,candidate_accepted=False,current_training_running=False,status=summary['state'])
save(BASE/'work-status.json',status)
plan=BASE/'long-term-plan.md';text=plan.read_text(encoding='utf-8-sig')
anchor='通过裸网格原图叠加、固定时刻多方向与时间一致性后，再从20万新高斯和新优化器训练18000步。'
assert text.count(anchor)==1
paragraph='033针对体形是否补偿姿态误差，完成固定官方预测体形、零中心体形先验两组各1200步从头拟合。原始59帧、相机、姿态起点和其他目标不变，原初始化逐数组与023相同；023只作已在032精确重放的只读对照。原拟合/固定体形/零中心先验的可见二维残差中位数8.71/10.06/8.76像素，23帧左膝模型弯曲12.38/13.38/12.41度、右膝6.89/8.64/6.91度。实际六视图与固定正侧面仍未通过，未支持仅改变体形先验即可解决当前姿态。裸人体与宽松衣物轮廓不同，不能把这种差异全当误差；没有身体真值，结论限于这两个受控对照。新2400次身体更新、0次高斯更新。'+next_action+'\n\n'
plan.write_text(text.replace(anchor,paragraph+anchor),encoding='utf-8')

asset=VIEWER/'body-fitting-method/identity-prior';asset.mkdir(exist_ok=True)
for name in ['six-source-comparison.jpg','fixed-world-comparison.jpg','fixed-left90.jpg']:
    dest=asset/name;assert not dest.exists();shutil.copy2(RUN/'review'/name,dest);assert sha(dest)==sha(RUN/'review'/name)
save(asset/'conclusion.json',summary)
page=VIEWER/'body-smplx.html';html=page.read_text(encoding='utf-8-sig');before=sha(page)
marker='</details>\n</section>\n<!-- GVHMR_PRIOR_019_END -->';assert html.count(marker)==1
addition='''<h4 id="body-identity-prior-033">体形先验与姿态不能相互替代</h4>
<p>限制体形参数偏离初始预测，与限制体形偏离统计模型的中心，是两种不同约束。前者保留预测的个体比例，后者抑制过大的体形系数。两者都不能提供衣物下不可见关节的真实位置。</p>
<math display="block"><mrow><msub><mi>L</mi><mtext>增量</mtext></msub><mo>=</mo><mfrac><mn>1</mn><mi>d</mi></mfrac><msubsup><mo>∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>d</mi></msubsup><msup><mrow><mo>(</mo><msub><mi>β</mi><mi>k</mi></msub><mo>−</mo><msub><mi>β</mi><mrow><mn>0</mn><mo>,</mo><mi>k</mi></mrow></msub><mo>)</mo></mrow><mn>2</mn></msup><mspace width="2em"/><msub><mi>L</mi><mtext>体形</mtext></msub><mo>=</mo><mfrac><mn>1</mn><mi>d</mi></mfrac><msubsup><mo>∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>d</mi></msubsup><msubsup><mi>β</mi><mi>k</mi><mn>2</mn></msubsup></mrow></math>
<p>此处 <var>β</var> 是无量纲体形系数，<var>β<sub>0</sub></var> 为官方时序预测，维数 <var>d</var> 为10。固定体形组保持 <var>β = β<sub>0</sub></var>；零中心组将原体形增量项替换为右式，保留0.04权重及相同参数范围。两组均从原始观测重新拟合1200步。</p>
<figure><a href="body-fitting-method/identity-prior/fixed-left90.jpg"><img src="body-fitting-method/identity-prior/fixed-left90.jpg" alt="同一世界侧面相机下的原拟合、固定体形与零中心体形先验三组实际网格"></a><figcaption>本地实际网格渲染。固定同一23帧和世界相机，三列依次为原拟合、固定预测体形、零中心体形先验。未调整每列的大小或朝向。</figcaption></figure>
<p>固定体形改变了躯干比例，但没有解决侧面姿态问题，可见二维观测残差中位数由8.71增至10.06像素；零中心组为8.76像素，正侧面也与原拟合接近。这两个对照均不作为新的高斯训练底模。结果只说明当前有界试验没有获得整体改善，不能证明初始体形就是真值。</p>
<p><a href="body-fitting-method/identity-prior/six-source-comparison.jpg">真实六视图叠加</a> · <a href="body-fitting-method/identity-prior/fixed-world-comparison.jpg">固定正侧面完整对照</a> · <a href="body-fitting-method/identity-prior/conclusion.json">精简判定</a></p>
'''
page.write_text(html.replace(marker,addition+marker),encoding='utf-8')
save(RUN/'publication-receipt.json',dict(created=NOW,page=str(page),before_sha256=before,after_sha256=sha(page),
    url='http://127.0.0.1:8870/body-smplx.html#body-identity-prior-033',scope='Existing collapsed appendix',
    ppt_changed=False,viewer_model_changed=False,remote_deployment=False))
print(json.dumps(dict(state=summary['state'],body_2d_px=summary['body_2d_px']),ensure_ascii=False))
