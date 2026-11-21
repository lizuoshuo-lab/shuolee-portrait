"""Reconcile the larger109background experiment before launching any new work."""
from pathlib import Path
import datetime as dt,hashlib,json,re,shutil
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');O=B/'current/portrait-causal-diagnosis-117';P=B/'current/background-architecture-109';V=B.parent/'viewer'
now=dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
c=read(P/'admitted-coverage.json');s=read(P/'solver-protocol.json');d=read(P/'decision.json')
assert c['tracks']==2684 and len(c['per_frame'])==175 and c['coverage_pass']
history=dict(updated=now,source=str(P),tracks=2684,fit_tracks=2127,held_tracks=557,frames=175,
 min_fit_tracks_per_frame=c['fit_min'],min_held_tracks_per_frame=c['validation_min'],
 prior_variable=s['sole_variable'],decision=d['status'],prior_candidate_accepted=False,
 review_scope=read(P/'material-review.json')['identity_scope'],
 scope_correction='D0 audited only the116bank. Its13-frame coverage is NOT a statement that all historical static observations cover only13frames.109has2684 admitted scene tracks across175frames.',
 next='Reuse raw109observations and frozen splits for conditioning/depth/identity audit, without oldoptimizedXYZ or cameras as new initial state. Inspect the109source before authorizing any fit.',
 source_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['solver-protocol.json','admitted-tracks.npz','solver-input.npz','admitted-coverage.json','decision.json']})
save(O/'historical-background-audit.json',history)
pr=read(O/'protocol.json');pr['amended_before_any_fit']=now;pr['historical_background']='historical-background-audit.json'
stage=pr['stages'][1]
stage['preparation']='First inspect original109admitted/solver-input observations, fixedfit/heldsplit, source-context review and original074cameras.109alreadycovered175frames with2684tracks. Reuse rawmeasurements, not oldoptimized3Dpoints or cameras. Audit correspondence ambiguity, per-frame spatial/depth coverage and intrinsic/extrinsic conditioning. Only extract new near-person rigid objects if a specific missing depth/material constraint is documented; no blanket repeat of175frame extraction.'
stage['parameters']='First run one read-only observability/conditional-intrinsic sensitivity audit with original074extrinsics. Preregister exact freeKparameters, pixelmodel and gauge before any optimization. At most2 newfits with a demonstrably new causal variable: K-only with074extrinsicsfixed; and only if justified, extrinsic refinement with the acceptedKfixed. No unchanged109extrinsic-only rerun and no jointallvariable explanation of singlefactorcause.'
stage['admission']='Reuse109frozenfit/heldsplit and keep116bank secondary. Check spatialspread, near/far geometry, trackgraph and localparameterconditioning; count/coverage alone does not prove calibratedK. If missing support is documented, freeze source-only admission for one newobjecttrackpass before residuals. Weak or degenerate K stays fixed and is labeled unobservable.'
stage['stop']='One read-only historicalconditioning audit, at most one documented missing-object track pass plus one implementationrepair, at most2 preregistered new-variable fits <=45min each. No unchanged109repeat, threshold sweep or relaxed gates.'
pr['rules'].append('109already tested static-background-only extrinsic correction and failed. Reconcile historical-background-audit.json before newcamera work; do not infer globalinsufficientcoverage from the smallD0bank.')
save(O/'protocol.json',pr)
q=read(O/'queue-state.json');q.update(updated=now,state='D1_historical_anchor_conditioning_next',stage='D1_preparation',
 historical_background='historical-background-audit.json',next_action='Read109source/input and frozen split. Audit actual correspondence ambiguity and near/far/sharedKconditioning using raw2684tracks with074fixed, before considering any new camera fit. No unchanged109rerun.');save(O/'queue-state.json',q)
r=read(O/'D0/result.json');r['history_context']=history;r['interpretation'].append(history['scope_correction']);save(O/'D0/result.json',r)
review=read(O/'D0/visual-review.json');review['history_context']='historical-background-audit.json';review['findings'].append(history['scope_correction']);save(O/'D0/visual-review.json',review)
led=read(O/'evidence-ledger.json');led['entries'].append(dict(id='historical109_scope_correction',updated=now,
 evidence=['historical-background-audit.json'],observation='109alreadyadmitted2684static-background tracks across175frames and testedextrinsic-only correction; itfailedhumanprotection.',
 implication='Do not blanketreextract or repeat109. Newscope firstauditsKidentifiability and depth/identity support. D0smallbanknotglobalcoverage.'));save(O/'evidence-ledger.json',led)
for src,dst in [(O/'D0/result.json','result.json'),(O/'D0/visual-review.json','visual-review.json'),(O/'protocol.json','protocol.json'),(O/'historical-background-audit.json','historical-background-audit.json')]:shutil.copyfile(src,V/'body-camera117'/dst)
page=V/'body-smplx.html';html=page.read_text(encoding='utf-8');start=html.index('<!-- ALIGNMENT117_BEGIN -->');end=html.index('<!-- ALIGNMENT117_END -->')+len('<!-- ALIGNMENT117_END -->');sec=html[start:end]
sec=sec.replace('175帧相机全部存在。已有人工确认的65条静止背景验证轨迹只涉及13帧，并集中在后段视角；只有3帧达到本轮预设的点数与图像分布门槛。这说的是<strong>独立验证覆盖</strong>，并非其余162帧没有相机。',
 '175帧相机全部存在。此前109轮已有2,684条背景轨迹覆盖全部175帧，其中2,127条拟合、557条整条留出；仅修正外参后，背景改善没有转化为全身改善，候选未采用。<strong>不能把固定点方法当成尚未尝试过，也不能原样重跑。</strong>本轮先核查这批观测的真实对应、近远景条件和共享内参是否受到充分约束。')
sec=sec.replace('沿用相同原图和074相机，背景留出投影误差中位数为0.44像素，下半脸为3.93像素。但背景与面部轨迹的视差分布不同，不能仅凭这组数字判断“相机正确”或“人物在动”。下一步加入人物附近和较远处的多层静止特征，检查空间约束是否充分。',
 '本次D0另复核了116轮较小的验证集：其中65条静止背景轨迹覆盖13帧，背景与下半脸留出误差中位数分别为0.44和3.93像素。这个13帧范围仅适用于该验证集，不能代表全部历史固定点资料。不同深度与视差的误差不可直接据以断言相机正确或人物在动。只在核查发现明确缺口时补采图像特征。')
sec=sec.replace('仅用审阅过的静止场景点求解，保留独立验证点；先固定内参检查外参，再以相同条件检验共享内参修正是否必要。',
 '先复核109轮固定点与留出划分，检查内参和不同深度的约束强弱。优先固定外参检验共享内参；有证据支持后，再检验固定新内参时的外参修正。每项只改变一个主要变量。')
sec=sec.replace('<a href="body-camera117/result.json">本轮诊断数据</a>', '<a href="body-camera117/result.json">本轮诊断数据</a> · <a href="body-camera117/historical-background-audit.json">109轮固定点历史核查</a>')
html=html[:start]+sec+html[end:];page.write_text(html,encoding='utf-8')
latest=read(V/'latest-result.json');latest['latestDiagnostic'].update(updated=now,title='固定场景点约束相机：复核内参与深度条件',status=q['state'],
 summary='109已用2684条背景轨迹覆盖175帧，外参单独修正未通过。此次先复核对应与内参/深度可辨识性，随后分开检查微动；未改变点云。');save(V/'latest-result.json',latest)
plan=B/'long-term-plan.md';text=plan.read_text(encoding='utf-8');a=text.index('<!-- PORTRAIT_CAUSAL_117_BEGIN -->');z=text.index('<!-- PORTRAIT_CAUSAL_117_END -->');part=text[a:z]
part=part.replace('已完成D0：复核已有231条材料轨迹。65条人工确认背景点的独立验证只覆盖13帧且集中于后段；这是验证覆盖不足，不是175相机缺失。原图和相机不变；背景与脸部视差分布不同，尚不能由局部误差确定唯一根因。',
 '已完成D0：复核116小验证集的231条材料轨迹，其中65条背景点覆盖13帧；该范围不代表全部固定点资料。继续查证发现109已有2684条背景轨迹覆盖全部175帧、外参单独修正未通过。优先复用原始观测核查对应与内参/深度可辨识性，不原样重跑109。原图与相机不变，根因仍未知。')
part=part.replace('下一步建立175帧静止场景锚点清单，优先人物附近的桌柜、地面及不同深度的建筑结构，排除其他人物、反射与遮挡。先冻结拟合/验证划分与停止条件，只做两项有界相机变量对照；面部、衣物先独立验收。',
 '下一步检查109原始固定点清单、既有拟合/验证划分、图像对应歧义和近远景约束。仅在明确缺口下补充人物附近桌柜等固定物体。最多两项有界新变量验证：固定074外参检验共享K；若通过，再固定新K检验外参。每项先登记因果变量、验证与停止条件；面部和衣物独立验收。')
plan.write_text(text[:a]+part+text[z:],encoding='utf-8')
work=read(B/'work-status.json');work['latest_alignment_execution'].update(state=q['state'],next_action=q['next_action']);save(B/'work-status.json',work)
print(json.dumps(dict(history_checked=True,old109_static_tracks=2684,actual_frames=175,newfit=False),ensure_ascii=False))
