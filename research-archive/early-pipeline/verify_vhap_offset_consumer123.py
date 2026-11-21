"""Verify the saved-offset loader in the actual research consumer, no training."""
from pathlib import Path
import fcntl
import sys
import time
import numpy as np
import torch
from run_official_gaussians123 import PROJECT,ASSETS,read,save,sha,now


def main():
    root=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization')
    folder=root/'smplify-vhap-direct-body-preflight'
    destination=folder/'actual-offset-consumer-verification.json'
    assert not destination.exists(),'Do not repeat an already passed source verification.'
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic()
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime
    runtime.configure(ASSETS)
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    source=PROJECT/'src/avatar_core/scene/smplx_gaussian_model.py'
    source_hash=sha(source)
    cases=[]
    for name in ('fitted-nonzero-offset-fixture.npz','initial-zero-offset-fixture.npz'):
        path=folder/name
        with np.load(path,allow_pickle=False) as z:
            expected=z['static_offset'][0] if 'static_offset' in z else np.zeros((10475,3),np.float32)
        model=SMPLXGaussianModel(3);model.load_fresh_meshes(path)
        receipt=model.enable_shared_static_offset(path,ASSETS/'smplx/SMPLX_NEUTRAL.npz')
        assert np.array_equal(model.static_offset.detach().cpu().numpy(),expected)
        errors=[]
        with torch.no_grad():
            for i in range(59):errors.append(float((model.shared_offset_vertices(i)-model.vertices_sequence[i]).abs().max()))
        assert max(errors)<2e-6
        cases.append(dict(input=name,input_sha256=sha(path),frames=59,maximum_vertex_error_m=max(errors),
                          initial_offset_max_abs_m=float(np.abs(expected).max()),offset_preserved=True,receipt=receipt))
    assert sha(source)==source_hash and time.monotonic()-start<120
    save(destination,dict(updated=now(),state='passed_actual_consumer',source=str(source),source_sha256=source_hash,
        cases=cases,optimizer_updates=0,gaussian_training_started=False,quality_accepted=False,
        seconds=time.monotonic()-start,scope='Actual research shared-offset replay, not the company frozen-vertex deployment path. No model quality admission.'))
    print(destination.read_text(),flush=True)


if __name__=='__main__':main()
