"""Launch a pinned, unchanged Hierarchical3DGS training stage in isolation.

Official scripts end on a saved forward/backward iteration before its optimizer
step: I requested iterations imply I-1 parameter-update calls, unlike gsplat.
This small-scene check disables depth and exposure optimization, and uses
normalized units. It is a workflow validation, not a full paper reproduction.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'repos' / 'hierarchical-3d-gaussians'
COMMIT = '596dc7081ee74e57a66502f4746328911f763b34'

def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['coarse','chunk'],required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--input',default='truck-four-chunks-001')
    p.add_argument('--chunk')
    p.add_argument('--scaffold-run')
    p.add_argument('--iterations',type=int,default=30000)
    p.add_argument('--port',type=int,default=6018)
    a=p.parse_args()
    for name in [a.run,a.input]:
        if not name.replace('-','').isalnum():
            raise ValueError('Invalid run identifier')
    if a.iterations<2 or not 1024<=a.port<=65535:
        raise ValueError('Invalid iteration budget or local viewer port')
    inputs=ROOT/'outputs'/'large-scene-gs-001'/'official-inputs'/a.input
    manifest=json.loads((inputs/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='ready_for_official_scaffold_and_chunk_training':
        raise RuntimeError('Derived inputs not ready')
    source=REPO/f'train_{"coarse" if a.stage=="coarse" else "single"}.py'
    actual_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    canonical=subprocess.check_output(['git','-C',str(REPO),'show',f'{COMMIT}:{source.name}'])
    if actual_commit!=COMMIT or source.read_bytes().replace(b'\r\n',b'\n')!=canonical.replace(b'\r\n',b'\n'):
        raise RuntimeError('Official training source changed')
    runs=ROOT/'outputs'/'large-scene-gs-001'/'official-training'
    out=runs/a.run
    out.mkdir(parents=True,exist_ok=False)
    data=inputs/'aligned' if a.stage=='coarse' else inputs/'chunks'/str(a.chunk)
    command=[sys.executable,'-u',str(source),'-s',str(data),'-i',manifest['images_absolute'],
        '-m',str(out/'model'),'-r','1','--data_device','cpu','--eval',
        '--iterations',str(a.iterations),'--save_iterations','-1','--port',str(a.port),
        '--exposure_lr_init','0','--exposure_lr_final','0',
        '--depth_l1_weight_init','0','--depth_l1_weight_final','0']
    if a.stage=='coarse':
        command+=['--skybox_num','100000']
        train_names=manifest['train_names']
    else:
        if not a.chunk or a.chunk not in [c['name'] for c in manifest['chunks']]:
            raise ValueError('Unknown chunk')
        if not a.scaffold_run or not a.scaffold_run.replace('-','').isalnum():
            raise ValueError('Scaffold run required')
        coarse=runs/a.scaffold_run
        receipt=json.loads((coarse/'outcome.json').read_text(encoding='utf-8'))
        if receipt['status']!='completed':
            raise RuntimeError('Scaffold training incomplete')
        scaffold=Path(receipt['saved_model_directory'])
        command+=['--scaffold_file',str(scaffold),'--bounds_file',str(data),'--skybox_locked','--disable_viewer']
        train_names=next(c for c in manifest['chunks'] if c['name']==a.chunk)['train_names']
    if set(train_names)&set(manifest['test_names']):
        raise RuntimeError('Training/test name overlap')
    configuration={
        'official_commit':COMMIT,'trainer_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'launcher_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_changed':False,'command_argv':command,'cwd':str(REPO),'stage':a.stage,'chunk':a.chunk,
        'input_manifest_sha256':hashlib.sha256((inputs/'manifest.json').read_bytes()).hexdigest(),
        'training_names':train_names,'frozen_global_test_names':manifest['test_names'],
        'requested_iterations':a.iterations,'expected_optimizer_update_calls':a.iterations-1,
        'snapshot_timing':'Official save precedes final optimizer update; final iteration exits without update',
        'training_scope':'Official small-scene workflow, fixed shared global coordinate frame; no local BA, monocular depth or exposure optimization',
        'partition_units':'Normalized scene coordinates; not metres',
        'schedule':'Official defaults, shortened only for explicit preflight; no adjusted density/SH schedule',
        'gpu_runtime':'Torch2.7.1+cu128, RTX5080; source build compatibility receipts stored separately',
    }
    (out/'launcher-source.py').write_bytes(Path(__file__).read_bytes())
    save(out/'configuration.json',configuration)
    start=time.perf_counter()
    outcome={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'stage':a.stage,
             'requested_iterations':a.iterations,'log_path':str(out/'training.log')}
    save(out/'outcome.json',outcome)
    print(json.dumps(outcome),flush=True)
    with (out/'training.log').open('wb') as log:
        code=subprocess.call(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    saved=out/'model'/'point_cloud'/f'iteration_{a.iterations}'
    ply=saved/'point_cloud.ply'
    outcome.update(exit_code=code,elapsed_seconds=time.perf_counter()-start,status='failed')
    if code==0 and ply.is_file() and (saved/'pc_info.txt').is_file():
        from plyfile import PlyData
        import numpy as np
        pc=PlyData.read(ply)
        vertices=pc['vertex'].data
        finite=all(np.isfinite(vertices[name]).all() for name in vertices.dtype.names)
        quaternion=np.stack([vertices[f'rot_{i}'] for i in range(4)],axis=1)
        if not finite or np.any(np.linalg.norm(quaternion,axis=1)<=0):
            raise RuntimeError('Official saved model has invalid parameters')
        saved_info=(saved/'pc_info.txt').read_text(encoding='utf-8').strip().splitlines()
        model_audit={'all_saved_parameters_finite':finite,'rotation_order':'wxyz',
                     'nonzero_quaternions':True,'pc_info_lines':saved_info,
                     'saved_fields':list(vertices.dtype.names)}
        if a.stage=='coarse':
            import numpy as np
            sys.path.insert(0,str(REPO/'preprocess'))
            import read_write_model as colmap
            _,_,points=colmap.read_model(str(data/'sparse'/'0'),ext='.bin')
            original_xyz=np.stack([p.xyz for p in points.values()]).astype(np.float32)
            saved_xyz=np.stack([vertices[n] for n in ['x','y','z']],axis=1)[100000:]
            delta=float(np.max(np.abs(saved_xyz-original_xyz)))
            model_audit.update(skybox_prefix_count=100000,scaffold_positions_max_abs_change=delta,
                               position_freeze_passed=delta==0.,max_SH_degree=1,
                               xyz_optimizer_note='Learning rate zero fixes positions; finite-gradient Adam moments may still accumulate')
            if delta!=0.:
                raise RuntimeError('Coarse scaffold positions moved')
        save(out/'model-audit.json',model_audit)
        training_text=(out/'training.log').read_text(encoding='utf-8',errors='replace')
        logged_peak=[int(n) for n in re.findall(r'(?:Peak memory[=:]|peak memory:)\s*(\d+)',training_text)]
        outcome.update(status='completed',saved_model_directory=str(saved),
            gaussian_count=len(pc['vertex'].data),point_cloud_sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),
            optimizer_update_calls=a.iterations-1,training_image_updates=a.iterations-1,
            logged_peak_allocated_bytes=max(logged_peak) if logged_peak else None,
            memory_scope='Largest recorded official torch.cuda.max_memory_allocated; reserved unavailable',
            model_audit=str(out/'model-audit.json'))
    save(out/'outcome.json',outcome)
    print(json.dumps(outcome),flush=True)
    if outcome['status']!='completed':
        raise SystemExit(code or 1)

if __name__=='__main__':
    main()
