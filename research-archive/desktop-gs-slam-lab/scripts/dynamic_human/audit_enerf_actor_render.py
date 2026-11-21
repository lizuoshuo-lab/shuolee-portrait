"""Rebuild node-local centers through declared transforms and render them."""
from pathlib import Path
import json,argparse
import numpy as np,torch
from PIL import Image
from gsplat import rasterization
from train_enerf_first_frame import tensor
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);contract=json.loads((out/'actor-interface/scene-contract.json').read_text());index=np.load(out/'actor-interface/gaussian-instance-index.npz');actor=index['foreground_instance'];assert np.array_equal(index['gaussian_id'],np.arange(len(actor)))
    manifest=json.loads((out/'offset-index/appearance-manifest.json').read_text());c=manifest['cameras']['06'];reports=[]
    with torch.no_grad():
        for t in [34,46,58]:
            state=torch.load(out/'offset-index/motion'/f'frame-{t:03d}.pt',weights_only=False,map_location='cpu');original=state['means'].numpy();rebuilt=np.empty_like(original)
            for k in [0,1,2]:
                tr=next(q for q in contract['node_trajectories'][str(k)] if q['common_frame']==t);center=np.array(tr['translation_world']);assert tr['root_rotation_wxyz']==[1,0,0,0];mask=actor==k;local=original[mask].astype('float64')-center;rebuilt[mask]=(local+center).astype('float32')
            assert np.array_equal(rebuilt,original);state['means']=torch.from_numpy(rebuilt);s={k:v.cuda() for k,v in state.items()};r,al,_=rasterization(s['means'],s['quats'],s['scales'].exp(),s['opacities'].sigmoid(),torch.cat([s['sh0'],s['shN']],1),tensor(c['w2c'])[None],tensor(c['K'])[None],1920,1080,sh_degree=2,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
            pred=((r[0]+.5*(1-al[0])).clamp(0,1).cpu().numpy()*255+.5).astype('uint8');reference=np.array(Image.open(out/'evaluation'/f'camera06-frame{t:03d}-offset-index.png'));difference=np.abs(pred.astype(int)-reference.astype(int));assert difference.max()<=1
            reports.append(dict(camera='06',common_frame=t,maximum_channel_difference=int(difference.max()),mean_channel_difference=float(difference.mean()),all_gaussians_preserved=True))
    result=dict(status='three_actual_scene_node_renders_match_source',views=reports,meaning='Node indexing, translations, local deformation positions and common render ordering can reconstruct the actual source render; this is not an OmniRe checkpoint import or proof of true object geometry')
    (out/'actor-interface/render-equivalence.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
