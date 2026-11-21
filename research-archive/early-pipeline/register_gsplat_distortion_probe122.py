from research_gsplat_distortion122 import H,G,R,O,sha,save
from datetime import datetime,timezone
import json
q=json.loads((R/'queue-state.json').read_text(encoding='utf-8-sig'))
if q['running'] is not None or (O/'probe-plan.json').exists():raise RuntimeError('Inspect existing registration/process')
files=[H/n for n in ['probe_gsplat_distortion122.py','run_gsplat_distortion_probe122.sh','native_portrait_renderer122_v2.py']]
files += [G/n for n in ['gsplat/rendering.py','gsplat/cuda/_wrapper.py','gsplat/cuda/include/Cameras.cuh','gsplat/cuda/_backend.py']]
wsl=lambda p:'/mnt/'+str(p).replace('\\','/')[0].lower()+str(p).replace('\\','/')[2:]
plan=dict(registered=datetime.now(timezone.utc).isoformat(),scope='synthetic API/gradient capability research',
    hypothesis='Installed gsplat1.5.3 can render a perspective camera with residual Brown radial/tangential/prism coefficients, but native eval3d backward does not estimate those coefficients or K/E. Localv2 adapter blocks this branch.',
    variables='One known six-Gaussian scene, exact K/E, one fixed nonzero coefficient set vs zero. No camera fitting or real-person observation.',
    controls=['OpenCV twelve-coefficient projection agrees with separately packed6+2+4 formula.',
              'Actual CUDA distorted render changes pixels, supplies finite Gaussian parameter gradients.',
              'Read actual K/D gradients and explicit pose-backward exception; unsupported gradients not fabricated.',
              'Current immutable adapter rejects Brown coefficients; demonstrates integration gap only.'],
    budget=dict(attempts=1,GPU_wall_seconds=120,Gaussians=6,width=96,height=64,optimizer_steps=0),
    stop='Retain any failed control; do not change coefficients/thresholds or retry physical calibration. Never infer face alignment from synthetic success.',
    source_hashes={wsl(p):sha(p) for p in files},
    no_new_weights=True,no_human_data=True,no_environment_upgrade=True,
    prerequisite_note='Initial standalone import omitted worker PYTHONPATH and failed before CUDA. Probe uses existing worker environment.')
save(O/'probe-plan.json',plan)
q.update(updated=plan['registered'],phase='gsplat_distortion_capability_registered',running=None,
         next='Inspect single120s synthetic Brown rendering/gradient capability probe; no camera estimation or person training. T10 CVO binding not started.')
save(R/'queue-state.json',q)
print('GSPLAT_DISTORTION_PROBE_REGISTERED')
