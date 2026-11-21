"""Prepare an isolated Python-only depth-guidance checkout using verified kernels."""
import hashlib
import json
from pathlib import Path
import shutil

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')

def main():
    validation=json.loads((ROOT/'native-depth-runtime/validation.json').read_text());assert validation['status']=='passed'
    source=RUNTIME/'repo-depthmask';repo=RUNTIME/'repo-depthguide'
    shutil.copytree(source,repo,ignore=shutil.ignore_patterns('submodules','__pycache__','*.so','.git'))
    helper=Path(__file__).with_name('native_depth_guidance.py');shutil.copy2(helper,repo/helper.name)
    script=repo/'msv2/train_depth_guided.py';text=(repo/'msv2/train_motion_rgb.py').read_text()
    def replace_once(a,b):
        nonlocal text
        assert text.count(a)==1,a
        text=text.replace(a,b)
    replace_once('from random import randint','from native_depth_guidance import DepthGuidance\nfrom random import randint')
    replace_once('    gaussians._mip_cameras = scene.getTrainCameras()',
        "    gaussians._mip_cameras = scene.getTrainCameras()\n    depth_guidance = DepthGuidance(os.environ['MINIGS_DEPTH_PRIORS'],scene.getTrainCameras(),dataset.model_path,float(os.environ['MINIGS_DEPTH_WEIGHT']))")
    replace_once('        loss.backward()',
        '        loss = loss + depth_guidance(viewpoint_cam,gaussians,iteration,valid,gaussians._culling[:,viewpoint_cam.uid])\n        loss.backward()')
    replace_once("    print('Num of Guassians: %d'%(gaussians._xyz.shape[0]))",
        "    depth_guidance.finish()\n    print('Num of Guassians: %d'%(gaussians._xyz.shape[0]))")
    replace_once('    safe_state(args.quiet)',
        "    safe_state(args.quiet)\n    import random as depth_seed\n    seed=int(os.environ['MINIGS_DEPTH_SEED'])\n    depth_seed.seed(seed)\n    np.random.seed(seed)\n    torch.manual_seed(seed)")
    script.write_text(text)
    files={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [script,repo/helper.name,repo/'scene/gaussian_model.py']}
    for path in [script,repo/helper.name]:shutil.copy2(path,ROOT/'native-depth-runtime'/path.name)
    result=dict(status='gradient_validated_not_training_accepted',repo=str(repo),files=files,
        kernels='previously validated native Mip CUDA kernels, unchanged',
        source='repo-depthmask',coefficients_predeclared=[.003,.01],seeds_predeclared=[0],
        baseline_repeats=[1,2],warmup_start=500,ramp_steps=1000,
        acceptance='report all candidates; require appearance, sparse-depth agreement and visual evidence; no depth-GT claim',
        evaluation_scope='8 development views and previously inspected24 same-video frames are development evidence, not a fresh final test')
    (ROOT/'native-depth-runtime/manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
