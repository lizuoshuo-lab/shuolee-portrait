"""Run native Mip experiments only after kernel/gradient validation passes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')
BASE=Path('/opt/gs-slam-lab/mini-splatting2-native-001')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--masked',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    assert json.loads((ROOT/'native-mip-runtime/validation.json').read_text())['status']=='passed'
    repo=RUNTIME/'repo';steps=32 if args.smoke else 18000
    name=f'native-{"motion-" if args.masked else ""}mip-{steps}-s0'
    out=ROOT/name;out.mkdir(exist_ok=False)
    # The renderer receives only train cameras when reconstructing the 3D filter.
    rp=repo/'render.py';text=rp.read_text()
    needle='        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)'
    if 'gaussians._mip_cameras' not in text:
        assert text.count(needle)==1;rp.write_text(text.replace(needle,needle+'\n        gaussians._mip_cameras = scene.getTrainCameras()'))
    src=repo/'msv2'/('train_motion_rgb.py' if args.masked else 'train.py')
    (out/src.name).write_text(src.read_text());(out/'native_mip_filter.py').write_text((repo/'native_mip_filter.py').read_text())
    config=out/'config';config.write_text((repo/'config/fast').read_text().replace('iterations=18_000',f'iterations={steps}'))
    env=os.environ.copy();env.update(PYTHONPATH=str(RUNTIME/'deps')+':'+str(repo)+':'+str(BASE/'deps'),OMP_NUM_THREADS='4',MINIGS_MOTION_MASKS=str(ROOT/'motion-masks-001'))
    cmd=[sys.executable,str(src),'-s',str(BASE/'dataset'),'-m',str(out/'model'),'-r','1','--eval',
        '--imp_metric','indoor','--config_path',str(config),'--test_iterations','-1','--port','6120']
    manifest=dict(status='running',started=time.time(),seed=0,steps=steps,command=cmd,
        method='native Mini-Splatting2 topology and optimizer + Mip-Splatting 3D/2D filters',
        motion_rgb_mask=args.masked,training_views=51,development_holdouts=8,
        native_covariance_floor=.1,filter_from_training_cameras_only=True,
        validation_sha256=hashlib.sha256((ROOT/'native-mip-runtime/validation.json').read_bytes()).hexdigest(),
        note='additional PyTorch compensation has overhead; not a speed benchmark')
    mp=out/'manifest.json';mp.write_text(json.dumps(manifest,indent=2))
    with (out/'train.log').open('w') as log:res=subprocess.run(cmd,cwd=repo/'msv2',env=env,stdout=log,stderr=subprocess.STDOUT)
    manifest.update(status='complete' if res.returncode==0 else 'failed',returncode=res.returncode,elapsed_seconds=time.time()-manifest['started'])
    if res.returncode==0:
        raw=out/f'model/point_cloud/iteration_{steps}/point_cloud.ply';baked=raw.with_name('point_cloud_mip3d_baked.ply')
        assert raw.exists() and baked.exists()
        manifest.update(ply=str(raw),baked_ply=str(baked),ply_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),baked_ply_sha256=hashlib.sha256(baked.read_bytes()).hexdigest())
    mp.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
    return res.returncode

if __name__=='__main__':raise SystemExit(main())
