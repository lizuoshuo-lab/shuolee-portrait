from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,time
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport'
OUT=D/'full-sequence-fit'
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def sha(p):
    h=hashlib.sha256()
    with lp(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def save(p,j):lp(p).write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def wsl(p):
    s=str(Path(p).resolve()).replace('\\','/');return '/mnt/'+s[0].lower()+s[2:]
def now():return datetime.now(timezone.utc).isoformat()
def prepare():
    assert not lp(OUT).exists(),'One registered full sequence trial only.'
    handoff=read(D/'sequence-streaming-handoff.json')
    assert handoff['CPU_checks']==55 and handoff['readback']['images_and_masks_fully_decoded']==188
    for path,h in handoff['sources'].items():assert sha(path)==h,path
    for check in handoff['tests']:assert sha(check['path'])==check['sha256']
    files=[H/'fit_full_sequence123.py',H/'run_smplify_reference123.py',H/'smplify_score_contract123.py',D/'full-sequence-inputs.json',D/'sequence-streaming-handoff.json']
    files += [P/'src/shuolee_portrait'/n for n in ['sequence_fitting.py','sequence_optimizer.py','sequence_input_adapter.py','alignment_inputs.py','camera_world_units.py']]
    assets=Path('D:/GS-SLAM-Data/shuolee-portrait-runtime-assets')
    files += [assets/'smplx/SMPLX_NEUTRAL.npz',assets/'vposer/V02_05/V02_05.yaml',assets/'vposer/V02_05/snapshots/V02_05_epoch=08_val_loss=0.03.ckpt']
    checks={wsl(p):sha(p) for p in files}
    lp(OUT).mkdir()
    manifest=read(D/'full-sequence-inputs.json')
    plan=dict(registered=now(),mode='one_full_sequence_research_adaptation',max_real_trials=1,frames=94,
        inputs=wsl(D/'full-sequence-inputs.json'),output=wsl(OUT),source_sha256=checks,
        reference_frame_id=manifest['reference_frame_id'],evaluation_mode='streamed',
        camera_length_contract=manifest['camera_length_contract'],
        fresh_state='One zero shared10-beta, per-frame original incam GVHMR prior encoded anew once to VPoser; zero jaw/expression, prior-only hands. No old optimized subject geometry.',
        objective='Unchanged five SMPLify-X schedules, fullK native camera-domain projection, shared shape prior once, mean per-frame terms. Same LBFGS; exact frame-local accumulation replaces dense Jacobian allocation.',
        face_review='Only existing 3-frame27FIT7HELD semantic review; remaining91 faces unknown/inactive. This is whole94 body/pose adaptation, not full94 face acceptance.',
        scale_scope='Explicit arbitrary common model-conditioned source-camera length gauge, not physical scene calibration. Camera-domain fitting does not depend on E; same declared E is used once for world output.',
        budget=dict(seconds=1800,max_evaluations=6000,export_seconds=120,external_timeout_seconds=2040),
        stops=['source/input modification','nonfinite or behind-camera geometry','registered evaluation or wall budget'],
        admission='User authorized progressing SMPL-X Gaussian reconstruction and exploratory training. This one fit extends actual roster with explicit unknown face semantics; no Gaussian training before output visual review.',
        old_models_preserved=True,new_detector=False,new_camera_estimation=False,new_weights=False,
        GPU_jobs=1,gaussian_steps=0,goal_complete=False)
    save(OUT/'plan.json',plan)
    print(json.dumps(dict(plan=str(OUT/'plan.json'),source_count=len(checks),budget=plan['budget'])))
def run():
    plan=read(OUT/'plan.json');assert not lp(OUT/'launch.json').exists()
    commands=['wsl.exe','-d','Ubuntu-20.04','--','/usr/bin/env','CUDA_VISIBLE_DEVICES=0','OMP_NUM_THREADS=4','PYTHONUNBUFFERED=1',
        'timeout',str(plan['budget']['external_timeout_seconds']),'/opt/gs-slam-lab/dynamic-human/.venv/bin/python','-B',wsl(H/'fit_full_sequence123.py'),'--plan',wsl(OUT/'plan.json')]
    save(OUT/'launch.json',dict(started=now(),windows_pid=os.getpid(),command=commands,plan_sha256=sha(OUT/'plan.json')))
    with lp(OUT/'run.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(commands,stdout=log,stderr=subprocess.STDOUT)
    save(OUT/'exit.json',dict(finished=now(),returncode=result.returncode,result_exists=lp(OUT/'result.json').exists(),new_gaussian_steps=0))
    print(json.dumps(dict(returncode=result.returncode,result_exists=lp(OUT/'result.json').exists(),log=str(OUT/'run.log'))))
    raise SystemExit(result.returncode)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);a=ap.parse_args()
    prepare() if a.mode=='prepare' else run()
