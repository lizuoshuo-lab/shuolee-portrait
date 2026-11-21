"""Refresh the bounded setup queue from actual processes, not stale status files."""
from pathlib import Path
import datetime as dt
import json
import subprocess

O = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116')
P = O/'pixsfm'
now = dt.datetime.now(dt.timezone.utc).isoformat()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))

def save(p, value):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    tmp.replace(p)

code = r'''
import json,subprocess
from pathlib import Path
out=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if 'python3 -c ' in cmd: continue
        if 'setup_pixsfm116_' not in cmd and '/shuolee-pixsfm116/' not in cmd: continue
        raw=(p/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
        out.append({'pid':int(p.name),'ppid':int(f[1]),'start_ticks':int(f[19]),'command':cmd})
    except (OSError,ValueError): pass
g=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
print(json.dumps({'processes':out,'gpu_compute_processes':g.stdout.strip(),'gpu_query_exit':g.returncode}))
'''
run = subprocess.run(['wsl','-d','Ubuntu-20.04','-u','root','--','python3','-c',code],capture_output=True,text=True,encoding='utf8',errors='replace',timeout=20)
if run.returncode:
    raise RuntimeError('Read-only WSL process verification failed: '+run.stderr[-500:])
snapshot = json.loads(run.stdout.strip())
snapshot['updated'] = now
save(P/'process-check.json', snapshot)
setup = read(P/'setup-state.json')
live = next((p for p in snapshot['processes'] if p['pid']==setup['pid'] and p['command'].startswith('bash ') and 'setup_pixsfm116_' in p['command']),None)
wrapper = int((P/'windows-wrapper.pid').read_text(encoding='utf-8-sig').strip())
queue = read(O/'queue-state.json')
if setup['state']=='running' and live:
    state='pixsfm_isolated_setup_in_progress'
    next_action='Inspect this same build process and logs; do not launch a duplicate. Once native imports pass, verify official feature weights before the preregistered fixed-camera smoke test. DefaultS2DNet download is not verified; an explicitly labeled official VGGNet backend may reuse the known pretrained VGG16 cache, never randomweights. No newGaussian training before alignment gates.'
elif setup['state']=='running':
    state='pixsfm_setup_process_missing_requires_investigation'
    next_action='The recorded setup process is absent. Diagnose logs and deadline before any repair; do not infer successful setup.'
elif setup['state']=='failed':
    state='pixsfm_setup_failed_requires_diagnosis'
    next_action='Preserve the exact dependency failure; repair only within the original2h setup budget. Do not launch subject training.'
else:
    state='pixsfm_setup_ready_for_feature_verification'
    next_action='Review runtime-check.json and exact importedmodule paths. Verify pretrainedfeature provenance, then preregister and run only the fixed-camera smoke test within30minutes; sixviewcoverage does not establish facealignment.'
queue.update(updated=now,state=state,stage='S1_P1_setup',candidate_index=1,
             active_pid=setup['pid'] if live else None,windows_wrapper_pid=wrapper,
             active_process_kind='isolated native dependency setup; noGaussian training',
             setup_state=setup,process_evidence='pixsfm/process-check.json',
             setup_deadline_utc='2026-10-25T08:16:20+00:00',
             budget_enforcement={'mechanism':'GNUtimeout --signal=TERM --kill-after=10 around native setup',
                                 'deadline_utc':'2026-10-25T08:16:20+00:00',
                                 'verified_timeout_processes':[p for p in snapshot['processes'] if p['command'].startswith('timeout ')]},
             historical_first_setup_guard=read(P/'setup-guard.json'),
             training_started=False,quality_accepted=False,next_action=next_action,
             setup_repair_evidence=['pixsfm/setup-build-repair.json','pixsfm/setup-metis-repair.json'],
             active_launch_evidence='pixsfm/native-setup-launch.json',
             active_log='pixsfm/setup-native.log',active_stderr='pixsfm/setup-native-stderr.log')
queue.pop('budget_guard',None)
save(O/'queue-state.json', queue)
statefile=O.parents[1]/'work-status.json'
status=read(statefile)
status['updated']=now
status['next_bounded_work'].update(active_setup_pid=setup['pid'] if live else None,
    process_state=state,process_evidence=str(P/'process-check.json'),
    remaining=next_action)
save(statefile,status)
print(json.dumps({'state':state,'pid':setup['pid'] if live else None,
                  'stage':setup['stage'],'gpu_compute_processes':snapshot['gpu_compute_processes'],
                  'new_training_started':False}))
