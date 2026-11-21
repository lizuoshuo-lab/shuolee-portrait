"""Conservative training-only depth contradiction audit; priors are not GT.

Thresholds are fixed before inspecting holdout performance. Preserve all input
models. A derived candidate suppresses opacity; it does not erase evidence.
"""
import hashlib
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from local_refinement import ROOT,read_model,cameras,render,save_image,write_json,export_ply,evaluate
from evaluate_geometry import restore

OUT=ROOT/'free-space-audit-001'

@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    protocol=dict(source='mip-depth-refine-s0',minimum_reliable_views=8,minimum_front_fraction=.9,
        maximum_surface_support_views=1,prior_confidence_min=.85,
        front_depth_fraction=.8,extent='camera z + 3 * maximum filtered standard deviation',
        surface_support_tolerance=.15,suppressed_opacity=1e-6,
        calibration='training cameras and depth priors only; thresholds fixed before holdout evaluation',
        meaning='contradiction with predicted depth, not measured free-space truth',
        topology='preserved; only opacity suppressed in a new candidate')
    write_json(OUT/'protocol.json',protocol)
    model,mode,filt=restore(protocol['source']);cams=cameras()
    n=len(model['means']);device='cuda'
    observed=torch.zeros(n,device=device);front=torch.zeros_like(observed);surface=torch.zeros_like(observed)
    filtered_scales=(model['scales'].exp().square()+filt.square()).sqrt()
    extent=3*filtered_scales.max(-1).values
    rows=[]
    for c in [c for c in cams if not c['is_test']]:
        with np.load(ROOT/'depth-priors'/f'{c["img_name"]}.npz') as data:
            prior=torch.from_numpy(np.stack([data['inverse_depth'],data['weight']]).astype('float32')).to(device)[None]
        V=c['viewmat'][0];xyz=model['means']@V[:3,:3].T+V[:3,3];z=xyz[:,2]
        u=xyz[:,0]/z.clamp_min(.001)*c['fx']+c['width']/2
        v=xyz[:,1]/z.clamp_min(.001)*c['fy']+c['height']/2
        grid=torch.stack([2*u/(c['width']-1)-1,2*v/(c['height']-1)-1],-1)[None,None]
        samples=F.grid_sample(prior,grid,align_corners=True,padding_mode='zeros')[0,:,0]
        depth=samples[0].clamp_min(1e-5).reciprocal()
        valid=(z>.2)&(u>3)&(u<c['width']-4)&(v>3)&(v<c['height']-4)&(samples[1]>.85)
        ahead=valid&((z+extent)<.8*depth)
        support=valid&((z-depth).abs()<.15*depth)
        observed+=valid;front+=ahead;surface+=support
        rows.append(dict(frame=c['img_name'],reliable_projected=int(valid.sum()),front=int(ahead.sum()),surface=int(support.sum())))
    selected=(observed>=8)&(front/observed.clamp_min(1)>=.9)&(surface<=1)
    np.savez_compressed(OUT/'gaussian-evidence.npz',observed=observed.cpu().numpy(),front=front.cpu().numpy(),
        surface=surface.cpu().numpy(),selected=selected.cpu().numpy())
    count=int(selected.sum())
    audit=dict(protocol=protocol,count=n,suppressed_count=count,suppressed_fraction=count/n,views=rows,
        source_checkpoint_sha256=hashlib.sha256((ROOT/protocol['source']/'checkpoint.pt').read_bytes()).hexdigest(),
        original_opacity_mean_selected=float(model['opacities'][selected].sigmoid().mean()) if count else None)
    write_json(OUT/'audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k!='views'}),flush=True)
    # Freeze the contradiction mask before examining any development holdout result.
    candidate=OUT/'opacity-suppressed';candidate.mkdir()
    # Render before and after for representative already-defined paths only.
    paths=json.loads((ROOT/'geometry-evaluation/paths.json').read_text())['frames']
    for p in [p for p in paths if p['index'] in [0,15,45]]:
        c=dict(cams[24]);R=np.array(p['rotation']);P=np.array(p['position']);V=np.eye(4);V[:3,:3]=R.T;V[:3,3]=-R.T@P
        c['viewmat']=torch.tensor(V,dtype=torch.float32,device=device)[None]
        before,_,_=render(model,c,divisor=2,mode=mode,filt=filt)
        save_image(OUT/f'{p["kind"]}-{p["index"]:02d}-before.png',before)
    model['opacities'][selected]=math.log(1e-6/(1-1e-6))
    export_ply(candidate/'point_cloud.ply',model,filt)
    torch.save(dict(params={k:v.detach().cpu() for k,v in model.items()},filter_3d=filt.cpu(),
                    config=dict(mode=mode,protocol=protocol)),candidate/'checkpoint.pt')
    metrics=evaluate(model,cams,candidate/'evaluation',mode,filt)
    baseline=json.loads((ROOT/protocol['source']/'evaluation/metrics.json').read_text())['aggregate']
    deltas={d:{k:metrics[d][k]-baseline[d][k] for k in metrics[d]} for d in metrics}
    for p in [p for p in paths if p['index'] in [0,15,45]]:
        c=dict(cams[24]);R=np.array(p['rotation']);P=np.array(p['position']);V=np.eye(4);V[:3,:3]=R.T;V[:3,3]=-R.T@P
        c['viewmat']=torch.tensor(V,dtype=torch.float32,device=device)[None]
        after,_,_=render(model,c,divisor=2,mode=mode,filt=filt)
        save_image(OUT/f'{p["kind"]}-{p["index"]:02d}-after.png',after)
    write_json(OUT/'status.json',dict(status='complete',suppressed_count=count,metrics=metrics,deltas=deltas,
        validated_floater_removal=False,limitations='predicted depth contradiction only; no independent dense depth; novel poses have no GT'))
    print(json.dumps(deltas),flush=True)

if __name__=='__main__':main()
