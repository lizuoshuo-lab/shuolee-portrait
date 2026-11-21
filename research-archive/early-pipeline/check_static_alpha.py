from pathlib import Path
import json,numpy as np,hashlib
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current')
A=B/'person-static-pgsr-fresh-064';R=B/'person-static-alpha-fresh-069'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for rel in ['clothing-init/points.npz','clothing-init/motion.npz','data/observations.npz']:
    a=np.load(A/rel);b=np.load(R/rel);assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files),rel
assert sha(A/'data/training-inputs.json')==sha(R/'data/training-inputs.json')
a=read(A/'gaussian-run/config.json');b=read(R/'gaussian-run/config.json')
assert a['options']==b['options']
assert read(A/'gaussian-run/training-cameras.json')==read(R/'gaussian-run/training-cameras.json')
assert b['alpha_composite']['enabled'] and b['representation']=='shared_static_world' and b['from_scratch']
assert b['adapter_sha256']==sha(R/'source-final/shuolee_static.py')
final=(R/'gaussian-run/result.json').exists()
if final:
    assert read(R/'gaussian-run/result.json')['steps']==18000
    assert np.array_equal(np.load(R/'gaussian-run/sampling-schedule.npy'),np.load(B/'person-static-no-scale-fresh-065/gaussian-run/sampling-schedule.npy'))
proof=dict(reference='064',candidate='069',initial_points=200000,steps=18000,from_scratch=True,old_personalized_model_loaded=False,
    exact_fresh_initial_arrays=True,raw_images_masks_cameras_unchanged=True,all_geometry_options_unchanged=True,
    changes=['associated foreground RGB resampling','paired black and white composites','ROI and contour-normalized alpha supervision'],
    final_sampling_order_checked=final,not_a_single_term_ablation=True,representation='shared_static_world',dynamic_route_abandoned=True)
(R/'controlled-protocol.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8');print(json.dumps(proof))
