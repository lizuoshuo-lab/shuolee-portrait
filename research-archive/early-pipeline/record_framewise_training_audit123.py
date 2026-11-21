"""Bounded source audit and next interface plan; zero model execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, shutil

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
REC=T/'shuolee/current/local-clothed-surface-123'
ROUTE=REC/'smplifyx-reference-control/framewise-canonical-route'
D=ROUTE/'shared-three-frame/final-state-diagnosis'
O=D/'root-policy-control'
A=ROUTE/'training-interface-audit'
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
SYNC=Path('D:/00/工作/查看器')
def lp(p): return Path('\\\\?\\'+str(Path(p).resolve()))
def read(p): return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def save(p,v): lp(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(lp(p).read_bytes()).hexdigest()
A.mkdir(exist_ok=True)
stamp=datetime.now(timezone.utc).isoformat()
files=[P/'src/shuolee_portrait'/x for x in ['training.py','clothed_training.py','sequence_fitting.py','runtime.py','cli.py']]
files += [P/'src/avatar_core'/x for x in ['scene/smplx_gaussian_model.py','scene/gaussian_model.py','gaussian_renderer/__init__.py']]
external=T.parents[1]/'repos/GaussianAvatars_SMPLX-shuolee/scene/smplx_gaussian_model.py'
assert external.exists(),external
assert sha(external)==sha(P/'src/avatar_core/scene/smplx_gaussian_model.py')
summary='当前正式训练入口仍是单静态网格；底层本地SMPL-X适配已能逐帧运输高斯，但缺少真实帧ID、相机、网格姿态及参考导出的完整训练连接。此处确认的是新路线的工程缺口，不是历史重影的唯一根因。'
findings=[
 dict(id='static_entry',file=str(files[0]),lines=[11,20,25,80,83,120,131,132,173,175,201],
      finding='All cameras use timestep=0; train calls _train_static; differing frame meshes are explicitly rejected; render loop does not select frame mesh; export selects pose 0.',
      implication='Keep the static guard. Do not remove it to silently accept a sequence.'),
 dict(id='existing_transport',file=str(P/'src/avatar_core/scene/smplx_gaussian_model.py'),lines=[23,33,39,99,109],
      finding='Loads vertices_world sequence and selects triangle-local frames by integer index; parent get_xyz/get_scaling/get_rotation apply mesh frame. No real-frame-ID to camera mapping contract.',
      implication='Reuse the verified mathematical component; do not claim transport is absent.'),
 dict(id='fit_snapshot',file=str(P/'src/shuolee_portrait/sequence_fitting.py'),lines=[72,79,82,167,179],
      finding='Uses zero-based integer pixel centres; camera_to_world transforms vertices with inverse E; snapshot contains per-frame world vertices plus named reference, gaussian_export=False.',
      implication='No body averaging found. Snapshot is not end-to-end Gaussian training.'),
 dict(id='free_clothing_entry',file=str(P/'src/shuolee_portrait/clothed_training.py'),lines=[24,27,39,47],
      finding='Static mesh assertion; free world centres and binding=None; body_attraction=False and per_triangle_quota=False.',
      implication='Do not silently replace free-clothing route with old per-triangle body sampling and attraction.'),
 dict(id='pixel_bridge',file=str(P/'src/shuolee_portrait/training.py'),lines=[11,16,25,28],
      finding='Old camera matrix parity computes projected pixel minus 0.5, whereas new fitting declares integer-centre pixels. Training resizing also changes K.',
      implication='New interface must declare/convert coordinate convention including resizing explicitly. This source contrast is not proof of an already executed erroneous bridge.'),
 dict(id='renderer_ownership',file=str(P/'src/avatar_core/gaussian_renderer/__init__.py'),lines=[19,101],
      finding='Renderer reads Gaussian properties and does not itself select a mesh timestep.',
      implication='Frame selection belongs in the explicit training/evaluation interface.')]
audit=dict(updated=stamp,state='read_only_interface_gap_confirmed',summary=summary,
           sources=[dict(path=str(f),sha256=sha(f)) for f in files+[external]],findings=findings,
           independently_reviewed_by='route_contract_audit; root reviewed camera and static-entry source',
           user_sources=[dict(path=p,sha256=sha(p)) for p in ['D:/00/思路.md','D:/00/记录/规则.md']],
           scope='Known local public CLI, static/clothed trainers, sequence snapshot, embedded runtime and corresponding local adapter only; not a claim about every historical experiment.',
           historical_transport_control=str(ROUTE/'result.json'),historical_control_rerun=False,
           new_model_forwards=0,new_real_fits=0,training_steps=0,source_files_changed=False,
           unique_real_ghosting_cause_identified=False,real_quality_passed=False,new_PLY=False)
save(A/'result.json',audit)
plan=dict(updated=stamp,state='registered_engineering_only_not_training',summary='先建立显式帧身份和像素约定接口，再设计自由衣物点的姿态运输；保留旧静态入口及准入保护。',
          engineering_steps=[
            'Read existing bundle schema, tests and pixel bridges before adding a new minimal frame-binding module.',
            'Join image/mask hash, complete K/E, native dimensions and named mesh pose by unique frame ID; reject missing/duplicate/cross-video IDs; never infer by array order.',
            'Declare integer-centre native pixels and any renderer/resize conversion. No new camera estimates or silent source K changes.',
            'Specify one shared Gaussian field and optimizer; explicit pose selection for training AND evaluation; export one named reference frame with full covariance and SH convention.',
            'Preserve free clothing-surface initialization, no body-attraction/per-triangle quota. Existing mesh-relative transport is a component, not automatic acceptance of a new clothing deformation model.'
          ],
          first_validation=dict(kind='CPU pure contract tests only; no SMPL-X/VPoser/NN forwards',
             cases=['shuffled frame order resolves same named records','missing and duplicate IDs fail','wrong-video/hash mismatch fails','invalid K/E or unnamed reference fails','explicit pixel/resize coordinate contract matches analytic projection'],
             budget='One new interface implementation and meaningful CPU tests; fix concrete implementation errors only.'),
          stop_before=['real data fitting','GPU training','new camera optimization','new weights','changing legacy static guards','promoting a PLY'],
          not_claimed=['subpixel real calibration','physical material correspondence','historical ghosting cause','complete upstream reproduction'],
          prerequisite_for_real_training='All-frame registration, actual fitted-body quality and independent face/body/background safeguards plus reviewed clothing transport; existing failures remain failures.',
          expected_image_delivery='No synthetic engineering image substitutes in D:/00/记录. Add real same-frame stage images there only when produced.')
save(A/'next-interface-plan.json',plan)
nexttext='下一步按training-interface-audit/next-interface-plan.json建立显式视频帧ID—相机—网格姿态连接和像素约定检查；先做纯CPU接口验证，保留静态入口及自由衣物点要求。暂不运行新实拍拟合、相机优化或高斯训练。'
stage='framewise_training_interface_gap_confirmed'
for base in [R,REC]:
    p=base/'queue-state.json';v=read(p);v.update(updated=stamp,current_stage=stage,phase=stage,reason=summary,
        next=nexttext,next_action=nexttext,next_plan=str(A/'next-interface-plan.json'),latest_evidence=str(A/'result.json'),
        running=None,worker_pid=None,training_running=False,training_allowed=False,goal_active=True);save(p,v)
    p=base/'evidence-ledger.json';v=read(p);v['updated']=stamp
    if base==REC:
        if not any(e.get('id')=='framewise_training_interface_20261029' for e in v['entries']):
            v['entries'].append(dict(id='framewise_training_interface_20261029',summary=summary,evidence=str(A/'result.json'),quality_passed=False,gaussian_steps=0))
    else:
        if summary not in v['confirmed']:v['confirmed'].append(summary)
        if str(A/'result.json') not in v['artifacts']:v['artifacts'].append(str(A/'result.json'))
    save(p,v)
for base,name in [(R,'protocol.json'),(REC,'next-action.json')]:
    p=base/name;v=read(p);v.update(updated=stamp,current_stage=stage,next=nexttext,next_action=nexttext,next_plan=str(A/'next-interface-plan.json'),latest_result=str(A/'result.json'));save(p,v)
section='<section id="framewise-training-interface"><h2>新路线的训练连接：已确认接口缺口</h2><p>'+summary+'</p><p>逐帧拟合快照没有把不同身体平均；本地SMPL-X适配器已有网格相对运输能力。当前训练入口则明确拒绝不同帧网格，全部相机使用姿态0。保留这一保护，新增明确的帧ID、相机和姿态选择接口。衣物自由点路线也需要适配，不能悄悄恢复裸身体吸附或逐面配额。</p><p>新拟合使用整数像素中心，旧训练矩阵自检包含半像素约定；接入前需明确转换。这是尚未接通的接口边界，不能直接归因于历史重影。</p><p>'+nexttext+'</p><p><a href="training-interface-audit/result.json">源码核验</a> · <a href="training-interface-audit/next-interface-plan.json">接口验证范围</a></p></section>'
for dest in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',SYNC/'diagnostics/video10-global-initializer']:
    (dest/'training-interface-audit').mkdir(exist_ok=True)
    for name in ['result.json','next-interface-plan.json']:shutil.copy2(A/name,dest/'training-interface-audit'/name)
    p=dest/'index.html';s=p.read_text(encoding='utf-8')
    if 'id="framewise-training-interface"' in s:s=re.sub(r'<section id="framewise-training-interface">.*?</section>',lambda _:section,s,count=1,flags=re.S)
    else:s=s.replace('<section id="root-policy-control">',section+'<section id="root-policy-control">',1)
    p.write_text(s,encoding='utf-8')
p=H.parent/'shuolee-coordination/local-research-status.json';v=read(p)
v.update(updated_at=stamp,current_stage=stage,latest_evidence=str(A/'result.json'),next=nexttext,next_action=nexttext,
         next_plan=str(A/'next-interface-plan.json'),source_queue_updated=stamp,latest_image_count=81,
         training_started=False,training_running=False,company_installation_unchanged=True)
v['framewise_training_interface']=dict(evidence=str(A/'result.json'),runtime_modified=False,training_started=False)
save(p,v)
p=T/'shuolee/long-term-plan.md';s=p.read_text(encoding='utf-8');heading='### 2026-10-29 逐帧训练调用链核验'
if heading not in s:p.write_text(s+'\n\n'+heading+'\n\n'+summary+'\n\n'+nexttext+'\n',encoding='utf-8')
print(json.dumps(dict(audit=str(A/'result.json'),next_plan=str(A/'next-interface-plan.json'),source_modified=False,training_started=False),ensure_ascii=False))
