"""Record source feasibility and concrete next checks without starting a fit."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import re
import shutil

TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
B=TOP/'shuolee'; O=B/'current/portrait-alignment-open-source-116'; R=O/'hsfm'; V=TOP/'viewer'; WEB=V/'body-camera116'
now=dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(p)
check=read(R/'source-contract-check.json')
assert check['state']=='source_contract_issues_confirmed_CPU_only'
files=[
    ('HMR2.0a','D:/GS-SLAM-Data/models/gvhmr/hmr2/epoch=10-step=25000.ckpt','Prior029 records strict full-head loading ofGVHMR adaptation. Needs HSfM-fork config/head comparison before reuse. Old subject outputs will not be loaded.'),
    ('ViTPose-H COCO','D:/GS-SLAM-Data/models/gvhmr/vitpose/vitpose-h-multi-coco.pth','Different output/schema from HSfM ViTPose huge wholebody133; not interchangeable by filename.'),
    ('RTMW wholebody','D:/GS-SLAM-Data/models/rtmw/rtmw-l-384x288.onnx','Potential locally available133-keypoint provider; would be an explicitly labeled detector adaptation, requiring new raw-frame inference and name/index check.'),
    ('SMPL neutral','C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl','Licensed user template, not a previous subject fit. Does not substitute for SMPL-X head/hands geometry.')]
assets=[dict(name=n,path=p,exists=Path(p).is_file(),bytes=Path(p).stat().st_size if Path(p).exists() else None,scope=s) for n,p,s in files]
preflight=dict(updated=now,state='source_contract_reviewed_adapter_and_dependency_check_pending',official_full_runtime_verified=False,
    pinned_commit='75f835e91c1b1dc97713ed7ab78b2d311f3b173a',source_files=10,source_manifest='source-manifest.json',
    source_only_tests='source-contract-check.json',source_immutable=True,new_camera_candidate=False,new_gaussian=False,
    root_cause_of_prior_ghosting_resolved=False,setup_started_at=now,setup_budget_hours=2,smoke_gpu_budget_minutes=30,
    source_findings=[
        dict(id='principal_point',file='align_world_env_and_smpl_hsfm_optim.py',lines=[1022,1026],finding='One first-view focal and hardcoded principal point256,144 passed to all initializers. Current portrait preprocess yields288x512. Per-view K in the same processed coordinate convention is required.'),
        dict(id='root_anchor',file='align_world_env_and_smpl_hsfm_optim.py',lines=[330,336,385,386,471,614],finding='COCO main-body first element is nose; initializer calls it pelvis, and returned translation later represents pelvis. Constant-depth source helper fixture returns nose offset0.6m instead of root. This is an initializer check, not measured subject error.'),
        dict(id='bbox_area',file='align_world_env_and_smpl_hsfm_optim.py',lines=[660,663,981,986],finding='xyxy is preserved after affine transform, while weight usesx2*y2. Equal-size translated boxes receive different weights. Need width*height and valid-box guard.'),
        dict(id='pixel_origin',file='get_world_env_dust3r_for_hsfm.py',lines=[97,130],finding='Resize affine requires explicit edge/center convention; official PIL-style center conversion differs by~2.14rawpx in this portrait fixture. This is a new interface risk, not proof of historical bug.'),
        dict(id='static_shared_body',file='align_world_env_and_smpl_hsfm_optim.py',lines=[580,634],finding='A single pose/shape/root per person is projected to all views. No temporal deformation network. Sequential video staticity remains an input assumption to test with independent image evidence.'),
        dict(id='dependency_and_cost',file='get_world_env_dust3r_for_hsfm.py',lines=[193,219],finding='Needs DUSt3R dense predictions and forked cloud optimizer. Main selects complete graph;175views would create30450directedpairs. Do not launch that blindly under the30min smoke budget.')],
    licenses=dict(HSfM='MIT repository code',DUSt3R='Fork image.py header CC BY-NC-SA4.0. Keep research use and any company redistribution/deployment license scope distinct.',body_templates='User reports related commercial authorization; files/templates remain private.',weights='Each asset and model-specific terms remain separate; availability does not imply license compatibility.'),
    known_asset_inventory=assets,
    missing_dependency_scope='DUSt3R not present as a named folder in the inspected D:/GS-SLAM-Data/models registry; its exact checkpoint has not been located. No global scan, no claim about all disks/00. No extra weights downloaded.',
    stage_plan=[
        'Read original fork camera and image coordinate semantics; freeze per-view K and pixel convention adapters using analytical fixtures before a fit.',
        'Verify compatible known HMR2/RTMW/template assets without loading prior subject fits. A detector substitution must be disclosed, not called unmodified official HSfM.',
        'Preregister one fixed-frame paired scene-only/joint diagnostic with equal observations and gauge, source074 immutable, no ROI distortion. Quantify shared-body residual, reject for motion/mapping violations. Do not sweep parameters.',
        'Only after source contract and dependency checks, run one bounded GPU diagnostic; evaluate independent face/body/background observations before any175-frame candidate or fresh Gaussian training.'
    ])
save(R/'preflight.json',preflight)
q=read(O/'queue-state.json');q.update(updated=now,state='P3_source_contract_reviewed_input_adapter_pending',stage='S1_P3_preflight',active_pid=None,active_process_kind=None,candidate_index=3,
    next_action='HSfM source audit found portrait principal-point, nose/pelvis initialization andxyxy area contracts requiring a separate explicit adapter. Read hsfm/preflight.json and source-contract-check.json. Next verify fork camera/pixel conventions and compatible known HMR2/RTMW assets, preregister one fixed scene-only/joint diagnostic. Do not repeatP1/P2 or source tests; no unmodifiedrun_hsfm.sh, automatic extra large weights, dynamic model or Gaussian train.',
    p3_preflight='hsfm/preflight.json',p3_source_checks='hsfm/source-contract-check.json',p3_setup_started_at=now,
    training_started=False,quality_accepted=False)
q['no_repeat']=list(dict.fromkeys(q['no_repeat']+['P3_same_source_contract_fixtures']))
save(O/'queue-state.json',q)
p=read(O/'protocol.json');p.update(updated=now,status=q['state'])
for stage in p['stages']:
    if stage['id']=='S1':stage['state']='P3_source_contract_reviewed_registration_of_runtime_pending'
    if stage['id']=='S2':stage['state']='P1_rejected_P2_not_adopted_P3_preflight'
save(O/'protocol.json',p)
c=read(O/'candidates.json')
for x in c:
    if x['id']=='P3':x.update(state=preflight['state'],source_reviewed=True,local_runtime_verified=False,preflight='hsfm/preflight.json',source_contract_checks='hsfm/source-contract-check.json')
save(O/'candidates.json',c)
rs=read(O/'research-sources.json');rs['runtime_updated']=now
for x in rs['sources']:
    if x.get('url')=='https://github.com/hongsukchoi/HSfM_RELEASE':x.update(source_reviewed=True,source_audit='hsfm/preflight.json',source_contract_tests='hsfm/source-contract-check.json',runtime_verified=False,runtime_scope='CPU source-helper fixtures only; full joint scene/human fitting not run')
save(O/'research-sources.json',rs)
l=read(O/'evidence-ledger.json');l['updated']=now
entry=dict(id='P3_HSfM_source_contract_preflight',updated=now,evidence=['hsfm/preflight.json','hsfm/source-contract-check.json','hsfm/source-manifest.json'],
    observation='Three source contracts confirmed with CPU fixtures before subject fitting: hardcoded landscape principal point, nose/pelvis initialization anchor, xyxy area weighting. Shared static body model is explicit. NoHSfMsubjectfit.',
    limit='This candidate has not been used in historical training. These findings do not establish a root cause of old ghosting. Actual micro-motion, physicalK/R/t and material identity remain unresolved.',
    next=q['next_action'])
l['entries']=[x for x in l['entries'] if x['id']!=entry['id']]+[entry];save(O/'evidence-ledger.json',l)
plan=B/'long-term-plan.md';txt=plan.read_text(encoding='utf-8-sig')
extra='P3 HSfM的10份公开源码已核验，CPU固定例确认竖幅主点、鼻点/骨盆初始平移和xyxy框面积三处接入约定问题；它从未用于历史训练，因此不是旧重影根因证据。已核对现有HMR2、ViTPose、RTMW与SMPL文件，但检测器/头部配置不能直接混用；下一步先验证隔离适配和已知依赖，再登记有界联合拟合。'
txt,n=re.subn(r'^(当前进度补充：.*)$',lambda m:m.group(1)+' '+extra,txt,count=1,flags=re.M);assert n==1
plan.write_text(txt,encoding='utf8')
w=read(B/'work-status.json');w.update(updated=now,current_phase='116-P3源码约定已核验，隔离适配与依赖检查待推进',local_training_running=False,local_inference_running=False)
w['next_bounded_work']=dict(run=str(O),phase='P3_input_adapter_pending',new_training_started=False,active_pid=None,process_state=q['state'],remaining=q['next_action'])
save(B/'work-status.json',w)

block='''<!-- P3_SOURCE_REVIEW_BEGIN -->
<h3 id="alignment116-hsfm">HSfM接入核验：先统一主点、人体根节点与检测框约定</h3>
<p>已核查固定版本的10份公开源码，并运行不加载权重的CPU例子。<strong>尚未执行HSfM的人物与相机联合拟合。</strong>它为每个人维护一套共享姿态、体形和根节点平移，再投影到各相机；顺序视频中的微动仍是需要检验的输入假设。</p>
<ul><li>初始化把主点写为(256, 144)，而本段竖图处理后为288×512。必须使用对应处理坐标下的实际内参，不能直接沿用横图常量。</li>
<li>传入初始化的第一个COCO身体点是鼻子，后续平移参考却是骨盆。已用已知深度的合成骨架复现这一参考点差异；这不是实拍人物的测量误差。</li>
<li>检测框保留左上、右下角形式，权重却使用右下角坐标的乘积。相同大小的框换个位置就得到不同权重，需要明确宽与高。</li></ul>
<p>这些是下一候选的源码约定问题，<strong>不能解释为此前重影的已知根因</strong>。下一步在独立适配中逐项核验，再检查已有HMR2与全身关键点模型的接口兼容。保持原074相机，不加载旧人物拟合，不直接运行会自动下载并展开完整图像对的总脚本。</p>
<p><a href="body-camera116/hsfm-preflight.json">源码与依赖核查</a> · <a href="body-camera116/hsfm-source-contract-check.json">CPU数值例子</a> · <a href="https://github.com/hongsukchoi/HSfM_RELEASE/tree/75f835e91c1b1dc97713ed7ab78b2d311f3b173a">固定源码版本</a></p>
<!-- P3_SOURCE_REVIEW_END -->
'''
page=V/'body-smplx.html';html=page.read_text(encoding='utf8')
anchor='<h3>什么条件下才开始下一轮训练？</h3>'
assert '<!-- P3_SOURCE_REVIEW_BEGIN -->' not in html and html.count(anchor)==1
page.write_text(html.replace(anchor,block+anchor),encoding='utf8')
for name in ['preflight.json','source-contract-check.json']:shutil.copyfile(R/name,WEB/('hsfm-'+name))
for name in ['protocol.json','queue-state.json','candidates.json','evidence-ledger.json']:shutil.copyfile(O/name,WEB/name)
pub=read(O/'publication.json');pub.update(updated=now,p3_preflight='hsfm/preflight.json',latest_diagnostic=q['state']);save(O/'publication.json',pub)
print(json.dumps(dict(state=q['state'],new_fitting=False,new_training=False,preflight=str(R/'preflight.json')),ensure_ascii=False))
