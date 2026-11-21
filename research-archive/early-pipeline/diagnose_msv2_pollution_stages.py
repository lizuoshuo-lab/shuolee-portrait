"""Locate pollution onset on frozen checkpoints without optimizer updates."""
from pathlib import Path
import importlib.util,math,json
from types import SimpleNamespace

BASE=Path('/opt/gs-slam-lab/msv2-demo-swinir4x-20261104')
OUT=Path('/opt/gs-slam-lab/msv2-pollution-diagnostic-20261104')
spec=importlib.util.spec_from_file_location('frozen_base',BASE/'experiment.py')
ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
torch=ex.runtime()
from scene import GaussianModel
from gaussian_renderer import render

def degree(p):
    count=0
    with p.open('rb') as f:
        for line in f:
            if line.startswith(b'property float f_rest_'):count+=1
            if line.strip()==b'end_header':break
    return int(round(math.sqrt(count/3+1)-1))

OUT.mkdir(exist_ok=True)
rows=ex.read(BASE/'native-inputs.json')['records']
pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
sources=[(f'black_{step}',BASE/f'model/point_cloud/iteration_{step}/point_cloud.ply') for step in [1000,3000,8000,18000]]
sources += [(f'{kind}_1000',Path('/opt/gs-slam-lab')/run/'model/point_cloud/iteration_1000/point_cloud.ply') for kind,run in [('alpha_l1','msv2-demo-transparent-20261104'),('alpha_bce','msv2-demo-transparent-bce-20261104')]]
result=[]
with torch.no_grad():
    for label,p in sources:
        original_hash=ex.sha(p);d=degree(p);g=GaussianModel(d);g.load_ply(str(p));g.active_sh_degree=d
        checks=[]
        for i in [0,27,58]:
            c=ex.Camera(rows[i],i);m=torch.as_tensor(c.mask.copy(),device='cuda',dtype=torch.float32)/255
            rgb=render(c,g,pipe,torch.zeros(3,device='cuda'))['render']
            a=render(c,g,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones_like(g.get_xyz))['render'][0]
            outside=m<.01
            ring=torch.nn.functional.max_pool2d((m>.01).float()[None,None],17,1,8)[0,0]>.5
            ys=torch.arange(c.height,device='cuda')[:,None].expand_as(m)
            regions={}
            for name,region in [('top_half',ys<c.height*.5),('lower_body',(ys>=c.height*.5)&(ys<c.height*.85)),('feet',ys>=c.height*.85)]:
                pixels=outside&region
                regions[name]=dict(opacity_equivalent_pixels=float(a[pixels].sum()),opaque_pixels=int((a[pixels]>.9).sum()))
            checks.append(dict(frame=c.image_name,outside_alpha_mean=float(a[outside].mean()),outside_alpha_gt_01=float((a[outside]>.1).float().mean()),
                far_outside_alpha_mean=float(a[outside&~ring].mean()),inside_alpha_mean=float(a[m>.99].mean()),
                false_positive_equivalent_pixels=float(a[outside].sum()),regions=regions))
            ex.save_image(OUT/f'{label}_{c.image_name}_gray.png',rgb+.5*(1-a[None]))
            if label=='black_1000':
                gt=torch.as_tensor(c.rgb.copy(),device='cuda',dtype=torch.float32).permute(2,0,1)/255
                ex.save_image(OUT/f'input_{c.image_name}.png',gt*m+.5*(1-m))
        assert ex.sha(p)==original_hash
        result.append(dict(label=label,source=str(p),sha256=original_hash,points=len(g.get_xyz),checks=checks))
        print(label,json.dumps(checks),flush=True)
        del g;torch.cuda.empty_cache()
ex.save(OUT/'stages.json',dict(optimizer_updates=0,models_unchanged=True,comparisons=result,
    caveat='Checkpoint comparisons identify when error appears, not which simultaneous upstream operation caused it. Native resolution and masks are identical.'))
