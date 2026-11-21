from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib

P=Path('D:/00/工作/人体重建')
TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
I=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-portrait-clean-002')
LOG=Path('D:/00/工作日志/2026-10-26/服务器独立环境验收001')
receipt=json.loads((I/'temporary-environment-cleanup.json').read_text())
assert receipt['state']=='complete' and not (I/'env').exists()
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
paths=[I/'install-evidence/summary.json',LOG/'002_一次性安装验收.json',
       P/'provenance/installation-check-002.json',TOP/'viewer/server-install002/summary.json',
       Path('D:/00/工作/查看器/server-install002/summary.json')]
for path in paths:
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    data.update(temporary_second_environment_removed_after_acceptance=True,
                installation_evidence_retained=True,
                active_runtime_environment='/opt/gs-slam-lab/shuolee-portrait-clean-001/env',
                cleanup_logical_bytes=receipt['logical_bytes'],
                cleanup_includes_training_data_or_models=False)
    save(path,data)
save(LOG/'002_临时环境清理.json',receipt)
save(P/'provenance/installation-check-002-cleanup.json',receipt)
ui_dir=Path('D:/00/工作日志/2026-10-26/查看器验证')
images=['私有Git全新安装002_验收.png','查看器_002工程状态更新.png']
if all((ui_dir/name).is_file() for name in images):
    ui=dict(reviewed=datetime.now(timezone.utc).isoformat(),
            installation_section_actually_viewed=True,viewer_actually_viewed=True,
            displayed_model='whole-body-avatar114-smplx',
            displayed_gaussians=215693,model_quality_accepted=False,
            diagnostic_updated=True,new_model_promoted=False,
            remote_00_verified=False,source_ui='local browser at port8890',
            screenshots=[dict(file=name,sha256=hashlib.sha256((ui_dir/name).read_bytes()).hexdigest()) for name in images])
    save(ui_dir/'002-ui-review.json',ui)
print('Recorded actual cleanup; first runtime, inputs, models and results retained.')
