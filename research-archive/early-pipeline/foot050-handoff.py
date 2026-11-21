from pathlib import Path
import json,hashlib,datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/smplx-foot-semantics-fresh-050';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
con=read(R/'phase-conclusion.json');old=con['next_action']
new=old.replace('下一项先核对已有身份/姿态范围对照与本轮参数界限诊断，区分求解范围不足、原始身体先验偏差及穿衣观测与裸身体表定义不一致。',
 '本轮只读界限检查显示，膝部修正各分量最大0.167/0.175弧度，低于组合上限0.4；两组均没有绝对值超过0.98的体形系数（上限1）。这没有支持“膝部主要被硬范围卡住”，不因此扩大自由度。下一项优先核对原始三维鞋面支持、裸脚端点和脚踝之间的绝对位置关系，并先查阅014原始三维支持、023相对运动、025共享姿态和032单向深度的既有结论，避免重复。')
con['next_action']=new;save(R/'phase-conclusion.json',con)
s=read(B/'work-status.json');s['next_action']=new;s['smplx_foot_semantics_050']['next_action']=new;save(B/'work-status.json',s)
pub=read(V/'body-fitting-method/nlf-prior/foot-correspondence-conclusion.json');pub['next_action']=new;pub['retention']=con['retention'];save(V/'body-fitting-method/nlf-prior/foot-correspondence-conclusion.json',pub)
p=B/'long-term-plan.md';t=p.read_text(encoding='utf-8');assert t.count(old)==1;p.write_text(t.replace(old,new),encoding='utf-8')
receipt=read(Path(__file__).parent/'foot050-deck-receipt.json')
current=Path(s['current_experiment']['current_candidate'])
check=dict(created=datetime.datetime.now().isoformat(),state=s['current_experiment']['status'],local_training_running=s['local_training_running'],
    goal_accepted=s['goal_accepted'],current_readonly_fit=str(current),current_fit_sha256=sha(current),
    canonical_ppt_sha256=sha(Path(receipt['source'])),both050_models_removed=all(not(R/a/'fitted.npz').exists() for a in ['control','no_foot_points']),
    cleanup=read(R/'cleanup-receipt.json'),next_action=new,
    implementation_note='PPT55 notes include049/050 controlled evidence. All55 final slide renders match prior canonical; no new slide. Whitespace normalized by artifact import, text preserved.')
assert check['canonical_ppt_sha256']==receipt['final_sha256']==s['latest_report']['sha256']
assert check['current_fit_sha256']=='92235ec41ab5eafca615ea4f87f452af74f0726d10eb188c62731d9820e01707'
assert check['both050_models_removed'] and not check['local_training_running'] and not check['goal_accepted']
save(R/'handoff.json',check)
print(json.dumps({k:v for k,v in check.items() if k not in ['next_action','cleanup','implementation_note']},ensure_ascii=False))
