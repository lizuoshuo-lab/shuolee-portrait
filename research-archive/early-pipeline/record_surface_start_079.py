from pathlib import Path
import json,datetime,hashlib
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/static-depth-mixture-audit-078';R=B/'current/person-static-surface-consistency-fresh-079'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
assert read(O/'surface-cuda-preflight.json')['numerical_preflight_passed']
progress=read(R/'gaussian-run/progress.json');assert progress['running']
s=read(B/'work-status.json');s.update(updated=now,current_phase='079 fresh surface-concentration comparison running',
    current_decision='全图相机与人体初始化流程保持。078完整对应库检查已排除未采样解释；新增表面分散计算已通过原渲染和梯度一致性核验。079从头20万点、1.8万步对照运行中，质量仍未验收。',
    current_user_scope='shared static clothed person; whole original images for geometric camera estimation; body initialization only; no dynamic or glasses-specific branch',
    local_training_running=True)
s['latest_diagnostic']=dict(round='078',directory=str(O),completed=True,type='full bank and surface CUDA preflight',
    full_bank_result=str(O/'full-bank/result.json'),preflight=str(O/'surface-cuda-preflight.json'))
s['latest_local_execution'].update(round='079',run=str(R),state='fresh_static_surface_training',running=True,steps=18000)
save(B/'work-status.json',s)
done=read(O/'completion.json');done.update(updated=now,full_bank_diagnostic_complete=True,surface_cuda_preflight_passed=True,
    new_candidate_round='079',new_candidate_started=True,source_model_unchanged=True,goal_turn_classification='progress',
    preceding_turn_classification='progress',same_blocking_condition_count=0)
save(O/'completion.json',done)
plan=B/'long-term-plan.md';text=plan.read_text(encoding='utf8')
marker='<!-- CAMERA_REGISTRATION_CURRENT_END -->';assert marker in text
block='<!-- SURFACE_079 -->\n079当前单项对照：保留175帧整图SfM及BA相机，重新做人体粗拟合、全部轮廓与衣物特征初始化20万点。完整对应库158,714组均被采样至少5次，不能再归因为库未采到。新增不透明度加权点到平面分散约束，原渲染前向与旧梯度已核验；固定权重0.01，从第3,001步生效，总计18,000步。保持共享静态人物，不恢复动态或眼镜专项。结果必须通过鼻口同对应误差、175帧主体保护、真实图像和连续新视角审阅才接受。当前运行中，没有质量改善结论。\n'
assert '<!-- SURFACE_079 -->' not in text
plan.write_text(text.replace(marker,block+marker,1),encoding='utf8')
print(json.dumps(dict(recorded=True,round='079',running=True,step_observed=progress['step'])))
