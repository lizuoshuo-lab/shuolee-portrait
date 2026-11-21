from pathlib import Path
import datetime, hashlib, json, shutil
from PIL import Image

LAB=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
RUN=BASE/'current/multihmr-camera-prior-030'
VIEWER=BASE.parent/'viewer'
NOW=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

assert not (RUN/'phase-conclusion.json').exists()
protocol=read(RUN/'protocol.json');audit=read(RUN/'checkpoint-audit.json')
review=read(RUN/'review/review-manifest.json');rigid=read(RUN/'root-only/diagnostic.json')
assert sha(RUN/'prediction.npz')==protocol['prediction_sha256']
assert protocol['old_learned_subject_loaded'] is False
assert not audit['missing_network_keys'] and not audit['unexpected_unreviewed_keys']
assert len(rigid['records'])==12 and all(r['success'] for r in rigid['records'])
next_action='先核对原始稳定三维纹理能否约束绝对表面位置，区分衣物表面与裸人体关节，避免再重复相对运动损失。SAM3DBody授权权重到位后运行原图预测；先验与整体几何未通过前不启动新的高斯训练。'
summary=dict(state='camera_aware_prior_not_adopted',completed=NOW,model='Multi-HMR 896-L, official SMPL-X output',
    official_repository='https://github.com/naver/multi-hmr',repo_commit=audit['model_commit'],
    weight=str(Path('D:/GS-SLAM-Data/models/multihmr/multiHMR_896_L.pt')),weight_sha256=audit['sha256'],
    sources=protocol['selected'],new_neural_image_predictions=6,neural_parameters_updated=False,
    source_inference='Original uncropped RGB plus calibrated K transformed through official contain/pad; full float32',
    standard_smplx_replay_max_m=max(x['standard_smplx_replay_max_m'] for x in protocol['sources']),
    observation='Raw overlays show floating feet and pose mismatch. Rigid-only placement reduces the frame23/33 foot displacement, but frozen knee flexion and arm/occluded-limb mismatch remain. No clear overall improvement sufficient to adopt.',
    diagnostic=dict(rigid_problems=12,parameters_per_problem=6,shape_and_articulation_frozen=True,
       total_scipy_reported_function_evaluations=sum(r['nfev'] for r in rigid['records']),
       gaussian_updates=0,world_registration=False,
       frame23={r['model']:{k:r[k] for k in ['median_before_px','median_after_px','rotation_delta_degrees','max_joint_distance_change_m']} for r in rigid['records'] if r['frame']==23}),
    frame23_model_knee_degrees={k:{n:v[2] for n,v in d.items() if 'knee' in n} for k,d in review['model_joint_bends_degrees'].items()},
    limitations=['Detector residuals are fitting diagnostics, not joint ground truth.',
       'Source-camera-relative side views do not establish world gravity or anatomical lean.',
       'Six sequential source frames are not simultaneous multi-camera capture.',
       'Rigid placement cannot fix wrong local pose, shape, or hidden-side geometry.'],
    next=next_action,quality_accepted=False,gaussian_updates=0,current_viewer_model_replaced=False,
    remote_deployment=False,license='NAVER non-commercial; this local research result is not cleared for company deployment',
    evidence=['review/six-source-comparison.jpg','review/frame23-pelvis-centered-geometry.jpg',
       'root-only/six-source-comparison.jpg','root-only/frame23-pelvis-centered-geometry.jpg','placement-comparison.jpg'])
save(RUN/'phase-conclusion.json',summary)
save(RUN/'visual-review.json',dict(reviewed=NOW,accepted=False,
    inspected=['Both six-source contact sheets','Native and rigid frame23 same-camera relative geometry',
        'Full-size native and rigid source23/source45 overlays'],
    observations=['Correct red-jacket subject selected despite other detected people.',
        'Native frame23 and33 feet project above the shoes; rigid fit brings them down.',
        'Frame23 local knee flexion remains after rigid adjustment, as required by the frozen articulation.',
        'Hands and far-side limbs do not match consistently; no accepted full-body initialization.'],
    changed_quality_gate=False))
for folder,filename in [('review','review-manifest.json'),('root-only','diagnostic.json')]:
    q=RUN/folder/filename;d=read(q);d['state']='visually_reviewed_not_adopted';d['accepted']=False;save(q,d)
save(RUN/'run-state.json',dict(state=summary['state'],running=False,new_neural_image_predictions=6,rigid_problems=12,gaussian_updates=0))

# Unmodified actual render comparisons, placed side by side without cropping.
im1=Image.open(RUN/'review/source-000023.jpg').convert('RGB')
im2=Image.open(RUN/'root-only/source-000023.jpg').convert('RGB')
sheet=Image.new('RGB',(im1.width+24+im2.width,max(im1.height,im2.height)),'white')
sheet.paste(im1,(0,0));sheet.paste(im2,(im1.width+24,0));sheet.save(RUN/'placement-comparison.jpg',quality=95)

decision=dict(created=NOW,selected='Multi-HMR calibrated-input bounded test completed, not adopted',
    hypotheses=[dict(name='VPoser',role='Pose regularization, not an image-conditioned identity or camera estimator',available=True),
        dict(name='Multi-HMR',role='Camera-ray-conditioned image-to-SMPL-X prior',available=True,result=summary['state']),
        dict(name='CameraHMR',role='Camera-aware human estimation',repository='https://github.com/pixelite1201/CameraHMR',
            commit='65d11a0640d23c60f55ec82ae54298e9e7e549e0',weights='Separate registration required, no weight download or inference'),
        dict(name='SMPLest-X',role='Larger image-conditioned whole-body predictor',repository='https://github.com/MotrixLab/SMPLest-X',
            weights='8.2GB according to official README; deferred in favor of this smaller bounded candidate'),
        dict(name='SAM 3D Body',role='Independent image-conditioned geometry with MHR-to-SMPL-X conversion',
            weights='Authorized weights pending; public conversion and calibrated input preflight completed')],
    next=next_action)
save(RUN/'research-decision.json',decision)
status=read(BASE/'work-status.json')
status.update(updated=NOW,current_phase='multihmr_camera_prior_030',local_training_running=False,current_decision=summary['observation'],next_action=next_action)
status['body_camera_prior_030']={**summary,'report':str(RUN/'phase-conclusion.json')}
status['latest_local_execution']=dict(kind='multihmr_camera_prior_030',state=summary['state'],running=False,
    directory=str(RUN),image_predictions=6,rigid_placements=12,gaussian_updates=0,accepted=False)
save(BASE/'work-status.json',status)
plan=BASE/'long-term-plan.md';text=plan.read_text(encoding='utf-8-sig')
anchor='通过裸网格原图叠加、固定时刻多方向与时间一致性后，再从20万新高斯和新优化器训练18000步。'
assert text.count(anchor)==1
para='030下载并按官方LFS摘要核验Multi-HMR 896-L，真实运行六张完整原图与标定相机内参。官方源码固定版本，网络可用参数全部加载，许可SMPL-X模板单独加载；头部位移转换成骨盆参考后，标准SMPL-X重放最大差小于0.5微米。原始预测仍有明显脚部悬空。进一步对GVHMR和Multi-HMR各六帧仅优化整体旋转和平移，共12个独立六自由度问题，体形与所有关节保持不变。23帧Multi-HMR可见二维残差中位数83.60降至16.66像素，脚部投影明显回落；但局部屈膝和手臂/遮挡侧偏差仍在，整体未采用。该残差是对不确定检测点的拟合诊断，非三维准确率；相机相对侧视不能当作世界重力方向。未新增高斯训练、未替换查看器、未部署00。NAVER许可限制商业用途，本次仅本地研究。'+next_action+'\n\n'
text=text.replace(anchor,para+anchor);plan.write_text(text,encoding='utf-8')

asset=VIEWER/'body-fitting-method/multihmr-camera-prior';asset.mkdir(exist_ok=True)
for rel in summary['evidence']:
    name=rel.replace('/','-');dest=asset/name;assert not dest.exists();shutil.copy2(RUN/rel,dest);assert sha(dest)==sha(RUN/rel)
save(asset/'conclusion.json',summary)
page=VIEWER/'body-smplx.html';html=page.read_text(encoding='utf-8-sig');before=sha(page)
marker='</details>\n</section>\n<!-- GVHMR_PRIOR_019_END -->';assert html.count(marker)==1
addition='''<h4 id="body-camera-prior-030">整体配准与身体姿态要分开检查</h4>
<p>Multi-HMR把相机内参转换为射线方向，参与人体预测，并直接输出SMPL-X。本次独立读取六张完整原图和真实内参；图像缩放、黑边与像素中心的变化一并作用于内参。头部参考的预测位移已转换成标准SMPL-X的骨盆参考，网格重放误差小于0.5微米。这证明接口一致，不证明人物几何正确。</p>
<figure><a href="body-fitting-method/multihmr-camera-prior/placement-comparison.jpg"><img src="body-fitting-method/multihmr-camera-prior/placement-comparison.jpg" alt="左侧原始预测，右侧冻结体形与关节后仅调整整体朝向和位置；每组依次为原图、GVHMR和Multi-HMR"></a><figcaption>本地实际23帧。左：原始预测；右：仅优化整体旋转和平移。每组三列依次为实拍、GVHMR、Multi-HMR。体形与关节不变，脚部悬空明显减轻。</figcaption></figure>
<p>刚体配准只改变整个人在相机中的摆放，不改变膝盖弯曲、肢体长度或胸腹形状。对两种方法各六帧作同样的六自由度诊断后，部分投影对齐改善，但侧面屈膝、手臂和遮挡侧偏差仍在。Multi-HMR暂不作为更好的初始化，保持现有查看器。二维检测点不是关节真值，残差下降不能替代三维形态检查。</p>
<figure><a href="body-fitting-method/multihmr-camera-prior/root-only-frame23-pelvis-centered-geometry.jpg"><img src="body-fitting-method/multihmr-camera-prior/root-only-frame23-pelvis-centered-geometry.jpg" alt="整体配准后，GVHMR与Multi-HMR的同相机相对正侧面几何"></a><figcaption>本地网格实际渲染，固定23帧，骨盆居中并施加相同相机相对旋转；图中方向并非独立测量的世界重力方向。</figcaption></figure>
<p><a href="https://github.com/naver/multi-hmr">Multi-HMR官方项目</a> · <a href="body-fitting-method/multihmr-camera-prior/conclusion.json">精简判定</a>。NAVER许可限制商业用途，本次仅作本地研究；此模型未部署公司电脑。</p>
'''
html=html.replace(marker,addition+marker);page.write_text(html,encoding='utf-8')
save(RUN/'publication-receipt.json',dict(page=str(page),before_sha256=before,after_sha256=sha(page),
    url='http://127.0.0.1:8870/body-smplx.html#body-camera-prior-030',scope='Existing collapsed appendix',
    ppt_changed=False,gaussian_viewer_model_changed=False,remote_deployment=False))
print(json.dumps(dict(state=summary['state'],next=next_action),ensure_ascii=False))
