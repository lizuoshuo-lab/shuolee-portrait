"""Run a bounded hardware probe or the single active PGSR surface experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from shuolee_prepare_pgsr import RUNTIME,OUT,HOME,sha,dump


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['probe','train']);args=parser.parse_args()
    adaptation=json.loads((OUT/'adaptation.json').read_text())
    assert sha(RUNTIME/'repo/train.py')==adaptation['training_sha256']
    assert sha(RUNTIME/'repo/shuolee_pgsr_support.py')==adaptation['support_sha256']
    if args.mode=='train':
        probe=json.loads((OUT/'probe/manifest.json').read_text())
        assert probe['status']=='complete'
        assert probe['validation']['geometry_terms_active'] and probe['validation']['finite_losses']
        assert probe['validation']['peak_allocated_mib']<11000
    folder=OUT/args.mode;folder.mkdir(exist_ok=False)
    steps=4 if args.mode=='probe' else 30000
    command=[sys.executable,str(RUNTIME/'repo/train.py'),'-s',adaptation['dataset'],'-m',str(folder/'model'),
        '-r','1','--eval','--data_device','cpu','--iterations',str(steps),'--max_abs_split_points','0',
        '--opacity_cull_threshold','0.05','--max_all_points','1000000','--test_iterations','-1','--save_iterations',str(steps)]
    if args.mode=='probe':command+=['--single_view_weight_from_iter','0','--multi_view_weight_from_iter','0']
    else:command+=['--checkpoint_iterations']+[str(i) for i in range(1000,30001,1000)]
    protocol=dict(owner='shuolee',mode=args.mode,started=time.time(),command=command,
        paper_method='PGSR official train+renderer, motion-mask adaptation recorded separately',
        source_commit='de24f1a38b350387e8d8fe381b2cd70c1ae946e7',
        source_images='Existing lossless rectified full-resolution1514x2694',training_frames=51,development_frames=8,
        camera_geometry='Same frozen COLMAP poses and sparse initialization',resolution=[1514,2694],
        iterations=steps,random_seed=0,seed_note='Official safe_state overrides module-level seed22 with seed0 before Scene initialization',
        geometry_start=0 if args.mode=='probe' else 7000,
        density_until=15000,max_points=1000000,abs_split=0,opacity_pruning=.05,
        ncc_samples=102400,pretrained_depth=False,mip=False,diffusion=False,exposure_compensation=False,
        method_comparison='Different representation, rasterizer, densification and training budget. Not a single-module causal ablation against MiniGS2.',
        gate='After complete training compare real person/background, native depth and identical continuous paths. Do not promote on PSNR alone.',
        training_sha256=adaptation['training_sha256'],support_sha256=adaptation['support_sha256'])
    dump(folder/'protocol.json',protocol)
    env=os.environ.copy();env.update(PYTHONPATH=f'{RUNTIME}/deps:{RUNTIME}/repo:/opt/gs-slam-lab/mini-splatting2-native-001/deps',
        SHUOLEE_MOTION_MASKS=str(HOME.parent/'motion-masks-001'),OMP_NUM_THREADS='4')
    manifest=dict(status='starting',protocol=protocol)
    dump(folder/'manifest.json',manifest)
    with (folder/'train.log').open('w') as log:
        process=subprocess.Popen(command,cwd=RUNTIME/'repo',env=env,stdout=log,stderr=subprocess.STDOUT)
        manifest.update(status='running',pid=process.pid);dump(folder/'manifest.json',manifest)
        print(json.dumps(dict(status='running',mode=args.mode,pid=process.pid)),flush=True)
        code=process.wait()
    manifest.update(status='complete' if code==0 else 'failed',returncode=code,elapsed_seconds=time.time()-protocol['started'])
    if code==0:
        rows=[json.loads(s) for s in (folder/'model/progress.jsonl').read_text().splitlines()]
        first=rows[0]
        active=[r for r in rows if r['normal'] is not None and r['multiview_geometry'] is not None and r['multiview_ncc'] is not None]
        manifest['validation']=dict(geometry_terms_active=bool(active),finite_losses=True,
            peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
            min_neighbor_count=min(first['neighbor_counts'].values()),gaussians=rows[-1]['gaussians'])
        for name,digest in adaptation['source_calibration_sha256'].items():
            assert sha(Path(adaptation['data_source'])/'sparse/0'/name)==digest
        model=folder/f'model/point_cloud/iteration_{steps}/point_cloud.ply'
        assert model.exists()
        import numpy as np
        from plyfile import PlyData
        vertex=PlyData.read(str(model))['vertex'].data
        assert all(np.isfinite(vertex[k]).all() for k in vertex.dtype.names)
        manifest['validation']['saved_parameters_finite']=True
        manifest.update(model=str(model),model_sha256=sha(model),accepted_as_best=False)
        if args.mode=='train':
            dump(folder/'model/progress.json',dict(status='training_complete_pending_evaluation',iteration=steps,accepted_as_best=False))
    dump(folder/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)
    assert code==0,'Inspect train.log; do not restart a live or unverified process'


if __name__=='__main__':main()
