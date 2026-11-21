"""Build isolated Mip-filtered native kernels; never modify the baseline runtime."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
BASE=Path('/opt/gs-slam-lab/mini-splatting2-native-001')
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')

def replace_once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)

def main():
    out=ROOT/'native-mip-runtime';out.mkdir(exist_ok=False)
    RUNTIME.mkdir(exist_ok=False);repo=RUNTIME/'repo';deps=RUNTIME/'deps'
    shutil.copytree(BASE/'repo',repo,ignore=shutil.ignore_patterns('.git','build','*.egg-info','__pycache__','*.so'))
    deps.mkdir()
    shutil.copy2(Path(__file__).with_name('native_mip_filter.py'),repo/'native_mip_filter.py')
    for module in ['diff-gaussian-rasterization','diff-gaussian-rasterization_ms']:
        for filename in ['forward.cu','backward.cu']:
            p=repo/'submodules'/module/'cuda_rasterizer'/filename
            text=p.read_text();assert text.count('+= 0.3f;')==2
            p.write_text(text.replace('+= 0.3f;','+= 0.1f;'))
    p=repo/'gaussian_renderer/__init__.py';text=p.read_text()
    text=text.replace('import math\n','import math\nfrom native_mip_filter import apply_filter\n')
    old='        rotations = pc.get_rotation'
    assert text.count(old)==4
    text=text.replace(old,old+'\n        scales, opacity = apply_filter(pc, viewpoint_camera, scales, rotations, opacity, scaling_modifier)')
    text=text.replace('        cov3D_precomp = pc.get_covariance(scaling_modifier)',
        '        raise NotImplementedError("Mip experiment uses scale/quaternion covariance path")')
    p.write_text(text)
    for name in ['train.py','train_motion_rgb.py']:
        p=repo/'msv2'/name;text=p.read_text()
        text=replace_once(text,'    gaussians.training_setup(opt)',
            '    gaussians.training_setup(opt)',) if text.count('    gaussians.training_setup(opt)')==1 else text
        text=replace_once(text,'    scene = Scene(dataset, gaussians, resolution_scales=[1,2])',
            '    scene = Scene(dataset, gaussians, resolution_scales=[1,2])\n    gaussians._mip_cameras = scene.getTrainCameras()')
        text=replace_once(text,'        gaussians.update_learning_rate(iteration)',
            '        gaussians._mip_iteration = iteration\n        gaussians.update_learning_rate(iteration)')
        text=replace_once(text,'                scene.save(iteration)',
            '                scene.save(iteration)\n                from native_mip_filter import save_baked\n                save_baked(gaussians, scene.model_path, iteration)')
        p.write_text(text)
    manifest=dict(status='building',started=time.time(),runtime=str(RUNTIME),builds=[],
        source_commit='5ec9202c2db7d900728bfc09733e85467cff885e',
        mip_reference_commit='dda02ab5ecf45d6edb8c540d9bb65c7e451345a9',
        implementation='3D smoothing and differentiable 2D determinant compensation in PyTorch; native forward/backward covariance floor0.1',
        baseline_runtime_untouched=True,validation_status='pending')
    mp=out/'manifest.json'
    def save():mp.write_text(json.dumps(manifest,indent=2))
    save();env=os.environ.copy();env.update(TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    for name in ['diff-gaussian-rasterization','diff-gaussian-rasterization_ms']:
        start=time.time()
        cmd=[sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','--target',str(deps),str(repo/'submodules'/name)]
        with (out/f'{name}.log').open('w') as log:result=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        manifest['builds'].append(dict(module=name,returncode=result.returncode,seconds=time.time()-start));save()
        if result.returncode:manifest['status']='failed';save();return result.returncode
    manifest.update(status='built_validation_pending',finished=time.time(),
        helper_sha256=hashlib.sha256((repo/'native_mip_filter.py').read_bytes()).hexdigest())
    save();print(json.dumps(manifest),flush=True)

if __name__=='__main__':raise SystemExit(main())
