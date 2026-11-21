import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now
paths=[a.H/'head-viewer-filter-audit123.json',a.H/'head-training-quality-audit123.json',a.H/'head-original-pruning-audit123.json',a.H/'flame-original-interface-audit123.json']
assert all(a.lp(p).is_file() for p in paths)
root=a.REC/'demo-vhap-initialization';long=root/'head-gaussian-long300k'
p=a.read(long/'progress.json')
entry=dict(updated=now(),stage='head-original-source-and-interface-audit',
    evidence=[dict(path=str(x),sha256=a.sha(x)) for x in paths],
    findings=['Confirmed viewer filter divergence fixed with actual same-camera UI comparison.',
              'Original missed size-prune affects only2 eligible18k saved points; not evidence of primary floaters cause, trajectories not reconstructed.',
              '59frame original FLAME forward matches VHAP exactly onCPU, cached projection difference<.0001px. Existing fullK camera adapter remains essential.',
              'Current Gaussian run freezesmesh; original optimizer pose/trans/expression path can be connected via verified data mapping after current run review. No new solver needed.'],
    training_source_modified=False,optimizer_updates_in_audit=0,physical_quality_accepted=False,
    next_action='Finish the already authorized300k run and actual quality review. Keep any original pose/trans/expr optimizer activation as a separately reviewed consequential change; no automatic second training.',
    live_job=dict(pid=p['pid'],step=p['step'],source=str(long)))
a.save(root/'head-runtime-audit-summary.json',entry)
ledger=a.R/'evidence-ledger.json';d=a.read(ledger)
if isinstance(d,list):d.append(entry)
else:d.setdefault('entries',[]).append(entry)
a.save(ledger,d)
for f in [a.R/'queue-state.json',a.REC/'queue-state.json',a.C/'local-research-status.json']:
    d=a.read(f);d['latest_head_interface_audit']=entry;a.save(f,d)
plan=a.R/'long-term-plan.md';text=a.lp(plan).read_text(encoding='utf-8');tag='## 2026-10-30 原库接口与预览差异核查'
if tag not in text:
    a.lp(plan).write_text(text+'\n\n'+tag+'\n\n已修复当前头部最新模型与原包导入的滤波配置差异，原包字节不变。同相机实际UI对照通过，模型质量仍未通过。独立30万步进程保持运行。\n\n原版FLAME与当前VHAP参数在59帧CPU零更新前向完全一致，缓存投影误差小于0.0001px；必须保留完整K适配，原FoV reader不能替代偏心主点。尺寸剪枝缺失在保存的18k状态只涉及2点，不能贸然认作主因。长训练完成后先实际审阅，再评估接回原版姿态/平移/表情微调的具体风险，不另写全序列求解器，不自动追加第二轮。头部验证不能替代全身质量与15分钟端到端验收。\n',encoding='utf-8')
print(dict(step=p['step'],points=p['points'],pid=p['pid'],evidence_files=4,goal_complete=False))