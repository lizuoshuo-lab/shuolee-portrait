from pathlib import Path
import datetime,hashlib,json,os,shutil

BASE=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
ROOT=BASE/'current/nlf-camera-prior-034'
PREV=BASE/'current/smplx-identity-prior-fresh-033'
BUILD=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
VIEWER=BASE.parent/'viewer'
NOW=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)

assert not (ROOT/'phase-conclusion.json').exists()
protocol=read(ROOT/'inference-protocol.json');replay=read(ROOT/'parametric-replay-audit.json');r=read(ROOT/'review/comparison.json')
assert len(protocol['records'])==6 and not protocol['old_person_state_used']
assert replay['official_smplx_flat_hand_replay']['max_coordinate_error_m']<2e-5
next_action='NLF官方先验已经完成六视图独立预测及模板/相机重放，保留为当前候选。下一步先扩展到原始59帧，检查逐帧体形、遮挡关节和侧面一致性；只有值得进入拟合时才从原始观测及官方输出建立新的共享体形/时间约束实验。不能用当前六视图预测直接启动高斯训练。SAM3DBody官方授权权重仍待用户提供。'
summary=dict(created=NOW,state='official_prior_integrated_six_view_screened_geometry_pending',
    weight=dict(name='NLF-L multi v0.3.2 patch6',path='D:/GS-SLAM-Data/models/nlf/nlf_l_multi_0.3.2.patch6.torchscript',bytes=549527206,
        sha256=protocol['weight_sha256'],source='https://github.com/isarandi/nlf/releases/tag/v0.3.2',official_release_digest_matches=True),
    original_frames=protocol['frames'],actual_gpu='NVIDIA GeForce RTX 5080',old_person_state_used=False,
    nonparametric_points=1024,full_parametric_vertices=10475,full_parameter_dimensions=165,
    native_to_local_smplx_replay_max_m=replay['official_smplx_flat_hand_replay']['max_coordinate_error_m'],
    camera_projection_roundtrip_max_px=max(max(x['projection_replay_max_px'].values()) for x in protocol['records']),
    observed_findings=['Six-source contact sheet and full source0/23/45 overlays inspected. Gross body coverage is plausible, but wrist/hand locations differ from the photos.',
        'Fixed frame23 front/left90/right90 comparison inspected. NLF changes the side-view arm placement; it does not demonstrate an accepted whole-body pose correction.',
        'Naked-body shape is not the clothing surface. Hidden joints and shoe-covered feet have no independent geometric ground truth.',
        'The local SMPL-X flat-hand convention reproduces official vertices to less than one micrometer numerically; this validates conversion, not body accuracy.'],
    residual_diagnostic=r['same_6_images_body_landmark_residual_px'],
    residual_comparison_limit='NLF raw prediction has not optimized these keypoints;023 did. Their residuals are not a fair model accuracy ranking.',
    world_registration=r['registration'],world_scale_per_model_meter=r['world_scale_per_model_meter'],
    measured_physical_scale=False,body_fit_steps=0,new_gaussian_updates=0,
    official_parametric_conversion_iterations=3,quality_accepted=False,viewer_model_replaced=False,remote_deployment=False,
    model_license='Official README: models for noncommercial research use. This phase is local research; no company deployment clearance asserted.',
    next=next_action)
save(ROOT/'phase-conclusion.json',summary)
save(ROOT/'visual-review.json',dict(reviewed=NOW,inspected=['six-source-comparison.jpg','source-000000.jpg','source-000023.jpg','source-000045.jpg','fixed-world-comparison.jpg'],
    observations=summary['observed_findings'],quality_accepted=False))
state=read(ROOT/'run-state.json');state.update(updated=NOW,state=summary['state'],running=False,quality_accepted=False);save(ROOT/'run-state.json',state)

deck=read(BUILD/'identity033-deck-receipt.json');cleanup=read(PREV/'cleanup-receipt.json')
assert deck['canonical_replaced'] and sha(Path(deck['source']))==deck['final_sha256'] and cleanup['protected_hashes_unchanged']
pub=read(PREV/'publication-receipt.json');pub.update(ppt_changed=True,ppt_scope='Slide55 notes only; all55 rendered slides unchanged',
    ppt_sha256=deck['final_sha256'],cleanup_receipt=str(PREV/'cleanup-receipt.json'),browser_screenshot=str(BUILD/'identity033-page-proof.png'));save(PREV/'publication-receipt.json',pub)
status=read(BASE/'work-status.json');status.update(updated=NOW,current_phase='nlf_camera_prior_034',local_training_running=False,
    current_decision='身体先验已接入并完成六视图筛查，参数转换通过；整体人体几何仍待验收。',next_action=next_action)
status['nlf_camera_prior_034']={**summary,'report':str(ROOT/'phase-conclusion.json')}
status['latest_local_execution']=dict(kind='nlf_camera_prior_034',state=summary['state'],running=False,directory=str(ROOT),
    from_scratch=True,original_frames=6,body_fit_updates=0,official_conversion_iterations=3,gaussian_updates=0,accepted=False)
status['latest_report'].update(sha256=deck['final_sha256'],updated=NOW,changed_slides=[],changed_notes_slides=[55],slides=55)
status['current_experiment'].update(latest_fitting_trial='034-official-inference',latest_fitting_steps=0,
    latest_fitting_arms=0,latest_fitting_gaussian_steps=0,candidate_accepted=False,current_training_running=False,status=summary['state'])
status['body_identity_prior_033']['cleanup']=dict(files_deleted=cleanup['files_deleted'],bytes_removed=cleanup['bytes_removed'],
    protected_hashes_unchanged=True,receipt=str(PREV/'cleanup-receipt.json'))
save(BASE/'work-status.json',status)

plan=BASE/'long-term-plan.md';text=plan.read_text(encoding='utf-8-sig')
anchor='通过裸网格原图叠加、固定时刻多方向与时间一致性后，再从20万新高斯和新优化器训练18000步。'
assert text.count(anchor)==1
paragraph='034已从NLF官方GitHub发布下载约550MB的NLF-L v0.3.2 patch6权重，并与官方SHA-256摘要核对。RTX5080运行6张原图，输入原始相机内参与人体框，没有旧个体状态。原始1024点、参数化10475顶点及完整165维姿态均留存；1024点对应关系按模板、体形基与蒙皮权重核对，不能直接套完整网格拓扑。原图投影重放最大差0.000244像素，官方参数在本地SMPL-X flat-hand约定下重放最大坐标差7.16e-7米，仅说明转换一致。实际三张完整叠图、六视图联系表和同相机正侧面对照仍有手部位置与侧面姿态疑问，未作为高斯训练底模；六视图二维诊断21.29像素不能与已优化过同一观测的023的8.67像素当作模型准确率排名。模型官方权重声明非商业研究使用，当前仅本地验证。'+next_action+'\n\n'
plan.write_text(text.replace(anchor,paragraph+anchor),encoding='utf-8')

asset=VIEWER/'body-fitting-method/nlf-prior';asset.mkdir(exist_ok=True)
for name in ['source-000023.jpg','six-source-comparison.jpg','fixed-world-comparison.jpg']:
    dest=asset/name;assert not dest.exists();shutil.copy2(ROOT/'review'/name,dest);assert sha(dest)==sha(ROOT/'review'/name)
save(asset/'conclusion.json',summary)
page=VIEWER/'body-smplx.html';html=page.read_text(encoding='utf-8-sig');before=sha(page)
marker='</details>\n</section>\n<!-- GVHMR_PRIOR_019_END -->';assert html.count(marker)==1
addition='''<h4 id="nlf-body-prior-034">从图像预测三维点，再拟合身体先验</h4>
<p>NLF先根据图像与相机内参预测人体表面点和关节，再把参数化人体拟合到这些三维预测。当前官方模型接口预测1024个表面点，拟合后输出10475顶点的SMPL-X网格。两者有明确顶点对应关系，原始预测点不能直接当成完整网格。</p>
<p>官方拟合对不确定性较低的点赋予更高权重。以表面点为例，其归一化权重为：</p>
<math display="block"><mrow><msub><mi>w</mi><mi>i</mi></msub><mo>=</mo><mfrac><msubsup><mi>σ</mi><mi>i</mi><mrow><mo>−</mo><mfrac><mn>3</mn><mn>2</mn></mfrac></mrow></msubsup><mrow><mfrac><mn>1</mn><mi>N</mi></mfrac><msubsup><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></msubsup><msubsup><mi>σ</mi><mi>j</mi><mrow><mo>−</mo><mfrac><mn>3</mn><mn>2</mn></mfrac></mrow></msubsup></mrow></mfrac></mrow></math>
<p><var>σ<sub>i</sub></var> 是网络输出的不确定性，<var>N</var> 为表面点数。关节权重另行归一化。这个权重能降低不确定预测的影响，但无法保证遮挡处的预测就是真实关节。</p>
<figure><a href="body-fitting-method/nlf-prior/source-000023.jpg"><img src="body-fitting-method/nlf-prior/source-000023.jpg" alt="同一原图下实拍、NLF原始1024点、官方SMPL-X拟合完整对照"></a><figcaption>本地原始23帧。左：实拍；中：NLF原始点预测；右：官方参数化拟合。未使用旧人体模型初始化。</figcaption></figure>
<p>官方权重和本机GPU推理已核验，6张原图的相机投影与本地SMPL-X参数重放也通过了数值检查。它们验证的是数据与坐标接口。正侧面仍需检查腕部位置、肢体遮挡与逐帧一致性，当前预测尚未作为新的高斯训练底模。</p>
<p><a href="body-fitting-method/nlf-prior/six-source-comparison.jpg">六视图完整预测</a> · <a href="body-fitting-method/nlf-prior/fixed-world-comparison.jpg">同一世界相机下的正侧面对照</a> · <a href="body-fitting-method/nlf-prior/conclusion.json">核验结论</a></p>
<p><small>方法与权重来源：<a href="https://github.com/isarandi/nlf">NLF，NeurIPS 2024</a>；官方模型声明用于非商业研究，本轮只在本地验证。</small></p>
'''
page.write_text(html.replace(marker,addition+marker),encoding='utf-8')
save(ROOT/'publication-receipt.json',dict(created=NOW,page=str(page),before_sha256=before,after_sha256=sha(page),
    url='http://127.0.0.1:8870/body-smplx.html#nlf-body-prior-034',scope='Existing collapsed appendix',
    report_note='Canonical report updated with033 controlled results;034 remains a screening candidate.',
    viewer_model_changed=False,remote_deployment=False))
print(json.dumps(dict(state=summary['state'],next=next_action),ensure_ascii=False))
