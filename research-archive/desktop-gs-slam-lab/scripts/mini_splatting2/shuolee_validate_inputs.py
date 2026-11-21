"""Validate retained inputs and a rebuilt view after retiring the Difix artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from local_refinement import ROOT, read_model, render
from run_difix_distill_pilot import camera

out = ROOT/'viewextrapolator-pilot-001'
protocol=json.loads((out/'protocol.json').read_text())
verified=[]
for name, clip in protocol['clips'].items():
    p=out/clip['file']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==clip['sha256']
    with np.load(p) as data:
        assert data['rgb'].shape==(25,1024,576,3)
        assert data['mask'].shape==(25,1024,576)
        assert np.isfinite(data['mask']).all()
    verified.append(name)
base=json.loads((ROOT/'native-resolution-cache-full-18000-s0/manifest.json').read_text())
model=read_model(Path(base['baked_ply']))
with torch.no_grad():
    rgb,alpha,_=render(model,camera(protocol['clips']['pitch_up']['cameras'][0]),mode='mip')
    generated=(rgb.clamp(0,1)*255).round().byte().cpu().numpy()
    with np.load(out/'pitch_up.npz') as data:
        delta=np.abs(generated.astype(float)-data['rgb'][0].astype(float))
        alpha_delta=np.abs(alpha[...,0].cpu().numpy()-data['mask'][0])
    assert delta.max()<=1 and delta.mean()<0.01
    assert alpha_delta.max()<1e-5
result=dict(status='pass',owner='shuolee',lossless_clips=verified,frames=50,
            rebuilt_rgb_max_byte_error=float(delta.max()),rebuilt_rgb_mean_byte_error=float(delta.mean()),
            rebuilt_alpha_max_error=float(alpha_delta.max()),source_model_sha256=base['baked_ply_sha256'],
            svd_inference_run=False,training_started=False)
(ROOT/'shuolee/active-input-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
