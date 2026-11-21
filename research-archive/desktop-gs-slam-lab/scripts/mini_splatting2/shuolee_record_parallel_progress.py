"""Record verified desktop, diagnostic and report-delivery observations."""
import json
from pathlib import Path
from datetime import datetime, timezone

home = Path(__file__).resolve().parents[2] / 'outputs/mini-splatting2-mip-depth-001/shuolee'
now = datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p, obj): p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

server = dict(observed_at=now, evidence='Live ToDesk 00 WinSCP authenticated Linux command output',
    route='local ToDesk -> 00 WinSCP Company-Training -> Linux server',
    account='zhouhaoyi', remote_home='/home/zhouhaoyi',
    login_verified=True, command_execution_verified=True,
    server_workspace='/home/zhouhaoyi/shuolee', server_workspace_created=True,
    marker={'owner': 'shuolee'}, directories=['code','data','runs','best'],
    gpu={'name': 'NVIDIA GeForce RTX 3090', 'total_mib':24576,
         'used_mib_at_preflight':15, 'compute_processes_at_preflight':0},
    disk_free_gib_at_preflight=634.68,
    nvcc_paths=['/usr/local/cuda-11.4/bin/nvcc', '/usr/local/cuda/bin/nvcc'],
    gpp_path='/usr/bin/g++',
    runtime_3dgs={'python':'3.8.18','torch':'1.10.0+cu113','torch_cuda':'11.3','cuda_available':True},
    direct_local_ssh='Both saved gs-00 route and direct company-server route timed out during banner exchange; ToDesk/WinSCP route verified separately.',
    input_transport='Short commands copied from a local text editor and pasted via WinSCP input context menu; Unicode key injection was not reliable.',
    training_started=False, training_accepted=False)
(home/'current/company-server').mkdir(exist_ok=True)
write(home/'current/company-server/live-preflight.json', server)

status = read(home/'work-status.json')
status.update(updated=now, server_workspace_created=True, server_training_started=False,
    current_phase='Parallel local retention experiment and company-server environment preparation',
    latest_server_preflight='current/company-server/live-preflight.json')
status['pending'] = [v for v in status.get('pending',[]) if v != 'user completes existing Company-Training login']
status['company_cleanup']['desktop_reconnection']='recovered; WinSCP Linux commands verified'
status['parallel_work_after_access']={
    'local':'Completed bounded 8200-step node audit; prepare budget-matched first-simplification retention comparison',
    'company_server':'RTX3090 24GiB verified idle at preflight; isolated shuolee namespace created; deploy code and verify compiler/runtime before independent training'}
status['diagnostics']={
    'motion':'current/company-problem-audit/motion-diagnosis/assessment.json',
    'roi_contribution':'current/company-problem-audit/density-diagnosis/result.json',
    'node_audit':'current/company-problem-audit/node-audit/assessment.json',
    'node_audit_completed_iteration':8200, 'node_audit_full_18000_complete':False,
    'new_best_model_accepted':False}
write(home/'work-status.json', status)

manifest=read(home/'reports/report-manifest.json')
manifest['remote'].update(status='updated_report_transfer_succeeded', verified_at=now,
    evidence='ToDesk target C:\\Users\\zerozero\\Desktop\\报告 shows 02 at 16.71MB and transfer success; replacement dialog showed source 17523165 bytes and old target 14619138 bytes.',
    delivered_source_sha256='59eeb25b0053997b5d388c2f90da6ede1459d6fad0ad17cfeb65469c5f0b77f6',
    delivered_source_bytes=17523165, remote_sha256_verified=False,
    native_powerpoint_open_verified=False)
manifest['updated']=now
write(home/'reports/report-manifest.json',manifest)
print(json.dumps({'recorded': True, 'server_training_started':False,'report_transfer':'succeeded'},ensure_ascii=False))
