from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil

TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=TOP/'shuolee/current/new-capture-118'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
LOG=Path('D:/00/工作日志/2026-10-27/新人物118')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(ROOT/'export/validation.json');now=datetime.now(timezone.utc).isoformat()
ply=LOG/'点云/shuolee_118_fixedK_s18000.ply'
assert sha(ply)==v['sha256'] and v['points']==239602
screens=[]
for mode,name in [('Gaussian','fixedk-ui-gaussian.png'),('points','fixedk-ui-points.png')]:
    p=REC/name;assert p.exists() and p.stat().st_size>50000
    screens.append(dict(mode=mode,path=str(p),sha256=sha(p)))
ui=dict(updated=now,file_opened_through_UI=True,source_path=str(ply),source_sha256=v['sha256'],
    observed_count=239602,modes=['Gaussian','points'],screenshots=screens,
    application='Existing Shuolee viewer at localhost:8870 in the Codex in-app browser',
    action='Clicked Open file, selected the local daily-log PLY through file chooser, observed count and degree3 SH, switched Gaussian and points.',
    desktop_executable_tested=False,pc00_tested=False,quality_accepted=False,
    observed='Both modes loaded. Body appearance matches the exported orbit; fragmented legs and edge halos remain visible.')
save(REC/'fixedk-viewer-ui-review.json',ui);save(LOG/'查看器实际检查.json',ui)
quality=read(REC/'fixedk-gaussian-quality-review.json');quality['viewer_ui_review']='completed_local_OpenFile_Gaussian_and_points';save(REC/'fixedk-gaussian-quality-review.json',quality);save(LOG/'本轮质量审阅.json',quality)
q=dict(updated=now,state='completed_bounded_run_quality_failed',stage='delivered_after_actual_ui',
    run_root=str(ROOT),training_running=False,steps=18000,step_budget=18000,
    initial_points=200000,points=239602,sha256=v['sha256'],formal_training_admitted=False,
    quality_accepted=False,format_passed=True,local_viewer_open_file_verified=True,
    pc00_deployed=False,automatic_additional_training=False,
    summary=quality['summary'],remaining_evidence=quality['next_evidence'])
save(REC/'queue-state.json',q);save(LOG/'本轮最终状态.json',q)
plan=read(REC/'long-run-plan.json');plan.update(updated=now,training_running=False,
    state='completed_bounded_run_quality_failed',steps_completed=18000,
    final_quality_accepted=False,final_delivered_sha256=v['sha256'],automatic_additional_training=False)
save(REC/'long-run-plan.json',plan)
ledger=REC/'evidence-ledger.json';old=read(ledger) if ledger.exists() else {}
old['fixedk18000_final']=dict(updated=now,run_root=str(ROOT),quality_review='fixedk-gaussian-quality-review.json',
    export_validation='fixedk-export-validation.json',ui_review='fixedk-viewer-ui-review.json',
    calibration_note='D:/00/工作日志/2026-10-27/内参文件解析/01_内参文件解析与待核实项.html',
    training_completed=True,quality_accepted=False,geometry_root_cause_proven=False,
    numerical_export_issue='Float32 body-root rotation plus recentering exceeded strict pixel parity. An exact proper scene-axis change passed without threshold relaxation or geometry edits.')
save(ledger,old)
longplan=TOP/'shuolee/long-term-plan.md'
text=longplan.read_text(encoding='utf-8')
marker='<!-- CAPTURE118_FINAL_DELIVERY -->'
if marker not in text:
    text+='\n'+marker+'\n\n2026-10-27：新人物118已完成固定厂家K假设下的全新人体拟合与200000初始高斯、18000步有限研究训练。最终239602个高斯，完整协方差/SH、原生回读、实际打开文件Gaussian/points通过；面部散点、衣边光晕和下肢破碎仍在，质量未通过，未认定相机标定或重影根因解决。新模型ID为new-capture-118-fixedk，旧1000步探针保留原身份。内参解析和待核实清单已存入D:/00/工作日志/2026-10-27/内参文件解析，过程图片和约59MB PLY存于同日新人物118。公司00未实测部署。本轮预算结束，关闭本次自动跟进，不无限追加训练；下一轮须先补齐/验证镜头模型、导出裁切防抖及材料对应与遮挡证据。\n'
    longplan.write_text(text,encoding='utf-8')
print(json.dumps(q,ensure_ascii=False))
