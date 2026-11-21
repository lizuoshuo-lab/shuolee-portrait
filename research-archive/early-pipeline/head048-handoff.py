from pathlib import Path
import json,hashlib,datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/smplx-head-sequence-fresh-048';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
paths=[R/'phase-conclusion.json',B/'work-status.json',V/'body-fitting-method/nlf-prior/head-sequence-conclusion.json']
con=read(paths[0]);old=con['next_action']
extra=('下一项诊断针对当前048结果，先核对各损失在髋、膝和胸椎参数上的梯度方向、跨帧一致性，以及NLF表面预测在服装遮挡区的不确定性；'
 '这一只读诊断不更新个体参数。先查阅026检测均值与方向不确定性、032衣物深度顺序、043VPoser的既有负结果，避免重复同一方案。')
new=old.replace('下一步先用原始身体观测和NLF预测分解肩、肘、髋、膝的几何冲突，',extra+'随后用原始身体观测和NLF预测分解肩、肘、髋、膝的几何冲突，')
con['next_action']=new;save(paths[0],con)
s=read(paths[1]);s['next_action']=new;s['smplx_head_sequence_048']['next_action']=new;save(paths[1],s)
pub=read(paths[2]);pub['next_action']=new
pub['retention']=con['retention'];pub['current_candidate_retained_for_readonly_diagnosis']=True;save(paths[2],pub)
p=B/'long-term-plan.md';t=p.read_text(encoding='utf-8');assert t.count(old)==1;p.write_text(t.replace(old,new),encoding='utf-8')
receipt=read(Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update/head048-deck-receipt.json'))
check=dict(created=datetime.datetime.now().isoformat(),state=s['current_experiment']['status'],local_training_running=s['local_training_running'],goal_accepted=s['goal_accepted'],
    canonical_ppt_sha256=hashlib.sha256(Path(receipt['source']).read_bytes()).hexdigest(),
    current_fit_sha256=hashlib.sha256((R/'head_sequence/fitted.npz').read_bytes()).hexdigest(),
    duplicate_control_removed=not(R/'control/fitted.npz').exists(),next_action=new)
assert check['canonical_ppt_sha256']==receipt['final_sha256']
assert check['current_fit_sha256']==con['fit_hashes']['head_sequence']
assert check['duplicate_control_removed'] and not check['local_training_running'] and not check['goal_accepted']
save(R/'handoff.json',check)
print(json.dumps({k:v for k,v in check.items() if k!='next_action'},ensure_ascii=False))
