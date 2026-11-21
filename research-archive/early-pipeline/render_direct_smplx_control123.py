"""Same-camera native-pixel diagnostic of the known synthetic fitting control."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image,ImageDraw
OUT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/direct-smplx-bindings')
z=dict(np.load(OUT/'synthetic-known-correspondences.npz'))
assert json.loads((OUT/'result.json').read_text())['all_known_synthetic_errors_lt1']
assert not (OUT/'synthetic_native_comparison.png').exists()
g=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
ctx=dr.RasterizeCudaContext();faces=torch.tensor(z['faces'],device='cuda',dtype=torch.int32)
E=g(z['E'][1]);K=g(z['K_integer'][1]);W,H=1520,2704
def render(vertices):
    world=g(vertices);v=world@E[:3,:3].T+E[:3,3];depth=v[:,2]
    # Explicit conversion from integer-centre CV K to raster pixel-edge K.
    cx=K[0,2]+.5;cy=K[1,2]+.5
    clip=torch.stack((2*K[0,0]/W*v[:,0]+(2*cx/W-1)*depth,
                      -2*K[1,1]/H*v[:,1]+(1-2*cy/H)*depth,
                      1.00002*depth-.0200002,depth),-1)
    rast,_=dr.rasterize(ctx,clip[None].contiguous(),faces,(H,W))
    tri=v[faces.long()];norm=torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1)
    vertnorm=torch.zeros_like(v)
    for j in range(3):vertnorm.index_add_(0,faces[:,j].long(),norm)
    vertnorm=torch.nn.functional.normalize(vertnorm,dim=-1)
    attr,_=dr.interpolate(vertnorm[None].contiguous(),rast,faces)
    light=g([-.35,-.45,-1]);light/=light.norm()
    shade=(torch.nn.functional.normalize(attr,dim=-1)*light).sum(-1).abs()
    color=(.48+.40*shade)[...,None]*g([.84,.88,.91])
    rgb=torch.where((rast[...,3]>0)[...,None],color,torch.ones_like(color))
    return Image.fromarray((rgb[0].flip(0).clamp(0,1)*255).byte().cpu().numpy())
target=z['target_uv'][1]
xy=np.r_[target.min(0)-[70,80],target.max(0)+[70,55]].round().astype(int)
xy=np.maximum(xy,[0,0,0,0]);xy=np.minimum(xy,[W,H,W,H])
panels=[]
for name,mesh,projected in [('initial',z['initial_mesh'],z['initial_uv'][1]),('fitted',z['final_mesh'],z['final_uv'][1])]:
    im=render(mesh);draw=ImageDraw.Draw(im)
    for measured,predicted in zip(target,projected):
        a=tuple(measured);b=tuple(predicted)
        draw.line([a,b],fill=(100,100,100),width=1)
        x,y=a;draw.ellipse((x-2,y-2,x+2,y+2),outline=(0,110,255),width=1)
        x,y=b;draw.ellipse((x-1,y-1,x+1,y+1),fill=(220,25,35))
    crop=im.crop(tuple(xy));crop.save(OUT/f'synthetic_{name}_native.png');panels.append(crop)
comparison=Image.new('RGB',(panels[0].width*2,panels[0].height),'white')
for i,p in enumerate(panels):comparison.paste(p,(i*p.width,0))
comparison.save(OUT/'synthetic_native_comparison.png')
(OUT/'visual-source.json').write_text(json.dumps(dict(known_synthetic_only=True,unseen_camera=1,
    source_resolution=[W,H],crop_xyxy=xy.tolist(),panels=['before_fit','after_fit'],
    resampled=False,text_added=False,blue='known target',red='SMPL-X bound point projection',
    figure_sha256=hashlib.sha256((OUT/'synthetic_native_comparison.png').read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps(dict(image=str(OUT/'synthetic_native_comparison.png'),wh=comparison.size)))
