"""Read-only native rasterizer comparison of trained geometry and depth priors."""
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB = Path(__file__).resolve().parents[2]
ROOT = LAB / 'outputs/mini-splatting2-mip-depth-001'
BASE = LAB / 'outputs/mini-splatting2-computer00-001'
OUT = ROOT / 'shuolee/current/geometry-diagnosis'
RUNTIME = Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')
REPO = RUNTIME / 'repo-depth-cache'
sys.path[:0] = [str(RUNTIME/'deps'), str(REPO), '/opt/gs-slam-lab/mini-splatting2-native-001/deps']
import torch
from scene import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render_imp
from native_depth_guidance import expected_depth
from shuolee_geometry_diagnosis import stats, sample, write_json, sha


def make_camera(c):
    R = np.asarray(c['rotation']); C = np.asarray(c['position']); t = -R.T @ C
    arr = np.asarray(Image.open(BASE / f'hover-sfm-001/dataset/images/{c["img_name"]}.jpg').convert('RGB')).copy()
    rgb = torch.from_numpy(arr).permute(2, 0, 1).float()/255
    return Camera(c['id'], R, t, 2*math.atan(c['width']/(2*c['fx'])),
        2*math.atan(c['height']/(2*c['fy'])), rgb, None, c['img_name'], c['id'], data_device='cpu')


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    assert (OUT/'summary.json').exists()
    result_file = OUT/'native-support.json'
    assert not result_file.exists(), 'Inspect existing audit before rerunning'
    region_file = ROOT/'training-head-regions-003/manifest.json'
    boxes = {r['frame']:r['box'] for r in json.loads(region_file.read_text())['rows']}
    cams = sorted(json.loads((BASE/'viewer/cameras.json').read_text()), key=lambda c:c['img_name'])
    cams = [c for c in cams if c['img_name'] in boxes]
    observations = json.loads((OUT/'sparse-observations.json').read_text())
    byframe = {c['img_name']:[x for x in observations if x['frame']==c['img_name']] for c in cams}
    models = ['native-motion-mip-depthmask-18000-s0', 'native-depthguide003-18000-s0', 'native-resolution-cache-full-18000-s0']
    protocol = dict(owner='shuolee', created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), source_sha256=sha(Path(__file__)),
        models=models, resolution=[757,1347], camera_frames=[c['img_name'] for c in cams],
        depth_semantics='Native alpha-normalized expected Gaussian-center camera z, not surface intersection.',
        culling='All-false diagnostic culling for every model; stored final 3D filter; native Mip 2D filter.',
        baseline_role='motion-depthmask has masked depth reinitialization but no monocular-depth loss; other two have weight0.003.',
        training=False, independent_geometry_GT=False)
    write_json(OUT/'native-protocol.json', protocol)
    background = torch.zeros(3, device='cuda')
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    results = {}
    for model_name in models:
        start = time.time()
        folder = ROOT/model_name/'model/point_cloud/iteration_18000'
        raw = folder/'point_cloud.ply'; digest=sha(raw)
        pc = GaussianModel(3); pc.load_ply(str(raw))
        pc._mip_iteration=18000; pc._mip_cameras=[]
        pc._mip_filter=torch.load(folder/'filter_3d.pt',map_location='cuda',weights_only=True)
        pc._mip_cache_key=(id(pc._xyz),len(pc.get_xyz),180); pc._mip_refreshes=0
        culled=torch.zeros(len(pc.get_xyz),device='cuda',dtype=torch.bool)
        rows=[]; pooled={'head_crop':[], 'outside_head_crop':[]}
        for ci,c in enumerate(cams):
            camera=make_camera(c); name=c['img_name']; h,w=c['height'],c['width']
            depth,alpha=expected_depth(camera,pc,culled)
            d,a=depth.cpu().numpy(),alpha.cpu().numpy()
            with np.load(ROOT/f'depth-priors/{name}.npz') as prior:
                inv=prior['inverse_depth'].astype('float32');conf=prior['weight'].astype('float32')
            motion=np.asarray(Image.open(ROOT/f'motion-masks-001/{name}.png').convert('L'))>0
            head=np.zeros((h,w),bool);x0,y0,x1,y1=boxes[name];head[y0:y1,x0:x1]=True
            eligible=(inv>0)&(conf>.25)&(~motion)
            supported=eligible&(a>.95)
            dense_ratio=(1/np.maximum(inv,1e-6))/np.maximum(d,.2)
            row=dict(frame=name, regions={})
            for label,mask in [('head_crop',head),('outside_head_crop',~head)]:
                row['regions'][label]=dict(prior_eligible_fraction=float(eligible[mask].mean()),
                    alpha_gated_fraction=float(supported[mask].mean()),
                    fraction_of_eligible_rejected_by_alpha=float((eligible&mask&(a<=.95)).sum()/max((eligible&mask).sum(),1)),
                    prior_over_native_depth=stats(dense_ratio[supported&mask]))
            obs=byframe[name]
            xy=np.array([o['xy'] for o in obs],dtype='float32')
            rd,ra=sample(d,xy),sample(a,xy)
            for j,o in enumerate(obs):
                if not o['prior_fit_reserved'] or o['confidence']<=.85 or ra[j]<=.95:
                    continue
                label='head_crop' if o['head_crop'] else 'outside_head_crop'
                pooled[label].append(dict(frame=name,point_id=o['point_id'],
                    native_relative_error=float(abs(rd[j]/o['z']-1)),
                    native_signed_ratio=float(rd[j]/o['z']),
                    prior_relative_error=o['relative_depth_error'],prior_signed_ratio=o['signed_depth_ratio']))
            if model_name==models[-1] and int(name) in [1,14,28,49]:
                image=render_imp(camera,pc,pipe,background,culling=culled)['render']
                pred=np.uint8(np.round(image.clamp(0,1).permute(1,2,0).cpu().numpy()*255))
                original=np.uint8(np.round(camera.original_image.permute(1,2,0).numpy()*255))
                # Actual source/render crops, plus eligible supervision pixels.
                display=original.copy();display[~supported]=display[~supported]//5
                pad=20; crop=(max(0,x0-pad),max(0,y0-pad),min(w,x1+pad),min(h,y1+pad))
                tiles=[]
                for title,arr in [('Source photograph',original),('Native trained render',pred),('Active depth supervision',display)]:
                    im=Image.fromarray(arr).crop(crop); im.thumbnail((320,380),Image.Resampling.LANCZOS)
                    tile=Image.new('RGB',(340,430),'white');tile.paste(im,((340-im.width)//2,40))
                    ImageDraw.Draw(tile).text((12,12),title,fill='black');tiles.append(tile)
                panel=Image.new('RGB',(1020,430),'white')
                for j,tile in enumerate(tiles):panel.paste(tile,(340*j,0))
                panel.save(OUT/f'head-evidence-{name}.png')
            rows.append(row)
            if ci%15==0:print(model_name,name,'head alpha-gated',round(row['regions']['head_crop']['alpha_gated_fraction'],3),flush=True)
            del camera,depth,alpha
        aggregate={}
        for label,obs in pooled.items():
            aggregate[label]=dict(observations=len(obs),unique_points=len({x['point_id'] for x in obs}),
                native_relative_error=stats([x['native_relative_error'] for x in obs]),
                prior_relative_error=stats([x['prior_relative_error'] for x in obs]),
                native_signed_ratio=stats([x['native_signed_ratio'] for x in obs]),
                prior_signed_ratio=stats([x['prior_signed_ratio'] for x in obs]),
                mean_prior_eligible_fraction=float(np.mean([r['regions'][label]['prior_eligible_fraction'] for r in rows])),
                mean_alpha_gated_fraction=float(np.mean([r['regions'][label]['alpha_gated_fraction'] for r in rows])))
        assert sha(raw)==digest
        results[model_name]=dict(ply_sha256=digest,gaussians=len(pc.get_xyz),elapsed_seconds=time.time()-start,
            aggregate=aggregate,rows=rows,reserved_sparse_observations=pooled)
        write_json(result_file,dict(status='running',protocol=protocol,models=results))
        print('finished',model_name,json.dumps(aggregate),flush=True)
        del pc;torch.cuda.empty_cache()
    write_json(result_file,dict(status='complete',protocol=protocol,models=results,models_unchanged=True,training_started=False))


if __name__=='__main__':main()
