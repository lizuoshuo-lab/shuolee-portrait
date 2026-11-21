from pathlib import Path
import json,hashlib,numpy as np
win=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
L=win if win.exists() else Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
B=L/'outputs/mini-splatting2-mip-depth-001/shuolee/current';R=B/'person-static-mask-foreground-fresh-070';A=B/'person-static-pgsr-fresh-064'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for rel in ['clothing-init/points.npz','clothing-init/motion.npz','data/observations.npz']:
    x=np.load(A/rel);y=np.load(R/rel);assert x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files),rel
assert sha(A/'data/training-inputs.json')==sha(R/'data/training-inputs.json')
assert read(R/'mask-loss-preflight.json')['passed']
assert sha(R/'source-final/shuolee_static.py')==sha(L/'repos/PGSR-shuolee/shuolee_static.py')
cfg=R/'gaussian-run/config.json';final=(R/'gaussian-run/result.json').exists()
if cfg.exists():
    a=read(A/'gaussian-run/config.json');b=read(cfg);assert a['options']==b['options']
    assert b['foreground_photometric']['enabled'] and not b['alpha_composite']['enabled']
    assert a['max_height']==b['max_height']==1536 and a['steps']==b['steps']==18000
    assert read(A/'gaussian-run/training-cameras.json')==read(R/'gaussian-run/training-cameras.json')
if final:
    assert read(R/'gaussian-run/result.json')['steps']==18000
    assert np.array_equal(np.load(R/'gaussian-run/sampling-schedule.npy'),np.load(B/'person-static-no-scale-fresh-065/gaussian-run/sampling-schedule.npy'))
proof=dict(reference='064',candidate='070',initial_points=200000,steps=18000,
    freshly_regenerated_initial_arrays_equal=True,raw_input_arrays_equal=True,
    same_options_and_cameras_verified=cfg.exists(),final_sampling_order_checked=final,
    single_main_variable='photometric support and normalization only',
    mask_loss_outside_invariance_passed=True,quality_accepted=False)
(R/'controlled-protocol.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8');print(proof,flush=True)
