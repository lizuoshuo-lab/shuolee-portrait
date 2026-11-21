from pathlib import Path
import datetime,json,hashlib
LAB=Path(r'C:/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
RUN=BASE/'current/mhr-smplx-official-conversion-027'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')

r=read(RUN/'conversion-report.json')
assert r['smplx_parameter_forward_replay_max_m']==0
assert r['official_error_independent_check_max_m']<2e-6
assert len(r['subjects'])==4
assert read(RUN/'model-download-receipt.json')['official_digest_matched']
assert (RUN/'review/public-examples-front-side.jpg').is_file()
review={
  'reviewed_at':now,'image':'review/public-examples-front-side.jpg',
  'scope':'All four public examples at two matched camera angles; actual mesh rasterizations.',
  'observations':['Major limb articulation and overall orientation retained in all four conversions.','Local torso, head, hand and toe shapes differ between MHR and SMPL-X; conversion is not lossless.','No source photograph or independent ground-truth body was used for this example audit.'],
  'public_example_conversion_usable':True,'user_subject_quality_accepted':False,
  'next':'Run authorized SAM3DBody inference on original subject images with supplied calibrated intrinsics; inspect raw MHR geometry before any SMPL-X conversion or Gaussian training.'
}
save(RUN/'visual-review.json',review)
r.update(state='public_example_conversion_verified',visual_review_pending=False,visual_review=str(RUN/'visual-review.json'))
save(RUN/'conversion-report.json',r)
rr=read(RUN/'review/render-receipt.json');rr['visual_review_pending']=False;rr['reviewed_at']=now;save(RUN/'review/render-receipt.json',rr)
save(RUN/'run-state.json',{'state':'public_example_conversion_verified','running':False,'pid':None,'fit_seconds':r['fit_seconds'],'subject_inference_started':False,'gaussian_steps':0})
p=read(RUN/'protocol.json');p['runtime']='Working: isolated Conda Python 3.12.15, PyMomentum 0.1.99 CPU, PyTorch 2.8.0 CPU; rendering uses existing verified GPU runtime.';p['compatibility_resolution']='PyPI PyMomentum 0.1.114.post0 crashed in Character.with_blend_shape; switched to official recommended Conda packaging, consistent with upstream issue 44. Existing fitting environments were not modified.';save(RUN/'protocol.json',p)
summary={
 'state':'conversion_ready_subject_weights_pending','completed':now,
 'functional_conversion_verified':True,'user_subject_quality_accepted':False,
 'public_example_count':4,'optimizer_steps_per_sample':460,'gaussian_steps':0,
 'mean_mapped_vertex_error_mm_range':[min(s['mapped_vertex_mean_mm'] for s in r['subjects']),max(s['mapped_vertex_mean_mm'] for s in r['subjects'])],
 'body_region_mean_error_mm_range':[min(s['body_mask_mean_mm'] for s in r['subjects']),max(s['body_mask_mean_mm'] for s in r['subjects'])],
 'parameter_replay_max_m':r['smplx_parameter_forward_replay_max_m'],
 'unit_contract':'SAM pred_vertices + pred_cam_t once, meters; official MHR conversion input centimeters; fitted SMPL-X output meters.',
 'camera_contract':'Released 3-D/2-D keypoints agree after inferred intrinsics with RMS below 0.000031 px. These are predictions, not camera GT.',
 'identity_contract':'Four different public subjects fit independently. Subject video must explicitly choose a shared identity strategy.',
 'template_contract':'Official conversion uses flat_hand_mean=True; GVHMR conventions differ. Save template settings with parameters, never silently mix.',
 'next':'Prepare a six-view original-image SAM3DBody adapter with calibrated intrinsics and mask boxes. Execute only when authorized weights are available. Compare raw MHR predictions before conversion; do not start Gaussians from unaccepted geometry.',
 'remote_deployment':False
}
save(RUN/'phase-conclusion.json',summary)
st=read(BASE/'work-status.json');st.update(updated=now,current_phase='mhr_smplx_official_conversion_027',local_training_running=False,current_decision='Official MHR to SMPL-X public-example conversion verified. User-subject geometry remains unaccepted; SAM3DBody authorized image-predictor weights pending.',next_action=summary['next']);st['body_conversion_027']={**summary,'report':str(RUN/'conversion-report.json'),'review':str(RUN/'review/public-examples-front-side.jpg')};save(BASE/'work-status.json',st)
plan=BASE/'long-term-plan.md';txt=plan.read_text(encoding='utf-8-sig')
old='等待权重期间可用官方公开示例检查单位/坐标与转换误差。'
assert txt.count(old)==1
txt=txt.replace(old,'027已用四个官方公开样例实际完成转换核验。MHR v1.0.1资产的SHA-256与官方发布摘要一致；PyPI包在模板加载处崩溃，改用官方建议的独立Conda环境后成功。每个样例从零SMPL-X参数开始，按官方40+80+40+300步优化；输入不含旧个体拟合状态。平均对应顶点差8.70–9.43毫米，身体区域12.05–14.07毫米；参数重放最大差0，四人正侧面保持主要姿态但局部形状有差异。这是模型转换误差，不是人物重建精度。下一步准备六视角原图、真实内参与遮罩的SAM3DBody输入，等授权权重到位后先检查原始MHR预测。')
plan.write_text(txt,encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
