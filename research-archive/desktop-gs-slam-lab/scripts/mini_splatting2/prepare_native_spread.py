"""Isolated checkout for one new variance intervention; old runs stay unchanged."""
import hashlib
import json
from pathlib import Path
import shutil

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')


def main():
    assert json.loads((ROOT/'native-spread-runtime/validation.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'depth-repeat-acceptance-queue.json').read_text())['status']=='complete'
    repo=RUNTIME/'repo-depthspread';source=RUNTIME/'repo-depthguide'
    shutil.copytree(source,repo,ignore=shutil.ignore_patterns('submodules','__pycache__','*.so','.git'))
    helper=Path(__file__).with_name('native_depth_spread.py');shutil.copy2(helper,repo/helper.name)
    script=repo/'msv2/train_depth_spread.py';text=(repo/'msv2/train_depth_guided.py').read_text()
    def replace(a,b):
        nonlocal text
        assert text.count(a)==1,a;text=text.replace(a,b)
    replace('from native_depth_guidance import DepthGuidance','from native_depth_guidance import DepthGuidance\nfrom native_depth_spread import DepthSpread')
    line="    depth_guidance = DepthGuidance(os.environ['MINIGS_DEPTH_PRIORS'],scene.getTrainCameras(),dataset.model_path,float(os.environ['MINIGS_DEPTH_WEIGHT']))"
    replace(line,line+"\n    spread_guidance = DepthSpread(os.environ['MINIGS_DEPTH_PRIORS'],scene.getTrainCameras(),dataset.model_path,float(os.environ['MINIGS_SPREAD_WEIGHT']))")
    replace('        loss.backward()',
        '        loss = loss + spread_guidance(viewpoint_cam,gaussians,iteration,valid,gaussians._culling[:,viewpoint_cam.uid])\n        loss.backward()')
    replace('    depth_guidance.finish()','    depth_guidance.finish()\n    spread_guidance.finish()')
    script.write_text(text)
    frozen={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [script,repo/helper.name,repo/'native_depth_guidance.py',repo/'scene/gaussian_model.py']}
    for path in [script,repo/helper.name]:shutil.copy2(path,ROOT/'native-spread-runtime'/path.name)
    (ROOT/'native-spread-runtime/manifest.json').write_text(json.dumps(dict(status='numerically_validated_not_training_accepted',
        repo=str(repo),files=frozen,mean_depth_weight=.003,variance_weights=[0.,.003,.01],seeds=[0],
        start=4000,ramp_steps=1000,steps=18000,smoke_schedule=dict(steps=600,start=100,ramp=200),
        evidence='3 matched prior-depth seeds increased center-z dispersion while head fidelity did not improve',
        required_acceptance=['head/shoulder true-image fidelity','common-mask depth spread','extra24frames',
            'sparse geometry and visual extrapolation paths','repeat any useful candidate; do not claim independent depthGT'],
        unchanged=['RGB loss','existing mean-depth pass','motion masks','camera calibration','native simplification','Mip kernels']),indent=2))
    print(json.dumps(dict(repo=str(repo),status='prepared')))


if __name__=='__main__':main()
