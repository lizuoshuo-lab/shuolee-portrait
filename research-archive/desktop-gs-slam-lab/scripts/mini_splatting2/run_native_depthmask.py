"""Controlled follow-up: exclude moving pixels from native depth reinitialization.

This is a motion-consistency intervention, not monocular depth supervision.
It uses a separate Python checkout and already validated native Mip kernels.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
MIP=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001');BASE=Path('/opt/gs-slam-lab/mini-splatting2-native-001')

def main():
    reference=ROOT/'native-motion-mip-18000-s0'
    assert json.loads((reference/'manifest.json').read_text())['status']=='complete'
    assert (reference/'evaluation-summary.json').exists()
    out=ROOT/'native-motion-mip-depthmask-18000-s0';out.mkdir(exist_ok=False)
    repo=MIP/'repo-depthmask';shutil.copytree(MIP/'repo',repo,ignore=shutil.ignore_patterns('submodules','__pycache__','*.so'))
    model=repo/'scene/gaussian_model.py';text=model.read_text()
    old='            prob=1-accum_alpha'
    assert text.count(old)==1
    replacement='''            prob=1-accum_alpha
            # Training-only motion mask, same reviewed pixels as the RGB ablation.
            from pathlib import Path as DepthMaskPath
            from PIL import Image as DepthMaskImage
            import json as depth_mask_json
            import os as depth_mask_os
            mask_path=DepthMaskPath(depth_mask_os.environ['MINIGS_MOTION_MASKS'])/(view.image_name+'.png')
            mask_arr=np.asarray(DepthMaskImage.open(mask_path).convert('L')).copy()
            motion=torch.from_numpy((mask_arr>0).astype('float32'))[None,None].to(prob.device)
            motion=torch.nn.functional.interpolate(motion,size=prob.shape[-2:],mode='nearest')[0]
            valid=1-motion
            before_mass=float(prob.sum())
            removed_mass=float((prob*motion).sum())
            prob=prob*valid
            if float(prob.sum())<=0:raise RuntimeError('No valid probability after motion mask')
            if not hasattr(self,'_depth_mask_evidence'):self._depth_mask_evidence=[]
            self._depth_mask_evidence.append(dict(frame=view.image_name,masked_pixels=int(motion.sum()),
                total_pixels=motion.numel(),probability_mass_before=before_mass,excluded_probability_mass=removed_mass))
'''
    text=text.replace(old,replacement)
    old='            out_pts_list.append(out_pts[indices])';assert text.count(old)==1
    text=text.replace(old,'''            assert not motion.reshape(-1)[torch.as_tensor(indices,device=motion.device)].any()
            self._depth_mask_evidence[-1]['sampled_count']=int(len(indices))
            self._depth_mask_evidence[-1]['sampled_masked_pixels']=0
            out_pts_list.append(out_pts[indices])''')
    old='        out_pts_merged=torch.cat(out_pts_list)';assert text.count(old)==1
    text=text.replace(old,'''        (DepthMaskPath(scene.model_path)/'depth-mask-usage.json').write_text(depth_mask_json.dumps(self._depth_mask_evidence,indent=2))
        out_pts_merged=torch.cat(out_pts_list)''')
    model.write_text(text);(out/'gaussian_model.py').write_text(text)
    config=out/'config';config.write_text((MIP/'repo/config/fast').read_text())
    env=os.environ.copy();env.update(PYTHONPATH=str(MIP/'deps')+':'+str(repo)+':'+str(BASE/'deps'),OMP_NUM_THREADS='4',MINIGS_MOTION_MASKS=str(ROOT/'motion-masks-001'))
    cmd=[sys.executable,str(repo/'msv2/train_motion_rgb.py'),'-s',str(BASE/'dataset'),'-m',str(out/'model'),'-r','1','--eval',
        '--imp_metric','indoor','--config_path',str(config),'--test_iterations','-1','--port','6120']
    manifest=dict(status='running',seed=0,steps=18000,started=time.time(),command=cmd,
        compared_to=reference.name,intervention='same RGB motion mask additionally gates native depth reinitialization sampling probability',
        method='Mini-Splatting2 + Mip + RGB/depth-reinitialization motion masks',
        depth_supervision='native depth reconstruction, no added monocular-depth loss',
        depth_reinitialization_masked=True,model_source_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    mp=out/'manifest.json';mp.write_text(json.dumps(manifest,indent=2))
    with (out/'train.log').open('w') as log:r=subprocess.run(cmd,cwd=repo/'msv2',env=env,stdout=log,stderr=subprocess.STDOUT)
    manifest.update(status='complete' if r.returncode==0 else 'failed',returncode=r.returncode,elapsed_seconds=time.time()-manifest['started'])
    if not r.returncode:
        raw=out/'model/point_cloud/iteration_18000/point_cloud.ply';baked=raw.with_name('point_cloud_mip3d_baked.ply')
        manifest.update(ply=str(raw),baked_ply=str(baked),ply_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),baked_ply_sha256=hashlib.sha256(baked.read_bytes()).hexdigest())
        usage=json.loads((out/'model/depth-mask-usage.json').read_text());assert len(usage)==51 and all(x['sampled_masked_pixels']==0 for x in usage)
        manifest['sampling_validation']='51 training frames checked; no excluded pixel sampled'
    mp.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
    return r.returncode

if __name__=='__main__':raise SystemExit(main())
