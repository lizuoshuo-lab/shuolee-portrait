"""Zero-update diagnostics for the stopped transparent MSV2 run."""
from pathlib import Path
import importlib.util, json, sys

RUN=Path('/opt/gs-slam-lab/msv2-demo-transparent-20261104')
sys.path.insert(0,str(RUN))
spec=importlib.util.spec_from_file_location('stopped_experiment',RUN/'experiment.py')
ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
torch=ex.runtime()
import transparent_hooks as rgba
from scene import GaussianModel
from gaussian_renderer import render,render_simp
from scene.gaussian_model import init_cdf_mask
from types import SimpleNamespace

path=RUN/'model/point_cloud/iteration_1000/point_cloud.ply'
sha=ex.sha(path)
g=GaussianModel(0);g.load_ply(str(path));g.active_sh_degree=0
pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
rows=ex.read(RUN/'native-inputs.json')['records']
checks=[]
for i in [0,27,58]:
    c=ex.Camera(rows[i],i);m=torch.as_tensor(c.mask.copy(),device='cuda',dtype=torch.float32)[None]/255
    a,_=rgba.render_alpha(c,g,pipe)
    with torch.no_grad():
        native=render(c,g,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones_like(g.get_xyz))['render'][0:1]
        max_diff=float((a-native).abs().max())
        near=torch.nn.functional.max_pool2d((m>.01).float()[None],17,1,8)[0]>.5
        far=(m<.01)&(~near)
        imp=render_simp(c,g,pipe,torch.zeros(3,device='cuda'))
        culled=~init_cdf_mask(importance=imp['accum_weights'],thres=.99)
        ac,_=rgba.render_alpha(c,g,pipe,culling=culled)
        del imp
    terms={
        'l1':.5*((a-m).abs()*m).sum()/m.sum()+.5*((a-m).abs()*(1-m)).sum()/(1-m).sum(),
        'bce':-.5*(m*a.clamp_min(1e-6).log()).sum()/m.sum()-.5*((1-m)*(1-a).clamp_min(1e-6).log()).sum()/(1-m).sum()
    }
    gradients={}
    for key,value in terms.items():
        da,dop=torch.autograd.grad(.2*value,(a,g._opacity),retain_graph=True)
        gradients[key]=dict(loss=float(value),outside_alpha_gradient_mean=float(da[m<.01].abs().mean()),
                           opaque_outside_gradient_mean=float(da[(m<.01)&(a>.9)].abs().mean()),
                           opacity_logit_gradient_l1=float(dop.abs().sum()),finite=bool(torch.isfinite(dop).all()))
    checks.append(dict(frame=c.image_name,ms_native_alpha_max_difference=max_diff,
       outside_alpha_mean=float(a[m<.01].mean()),far_outside_alpha_mean=float(a[far].mean()),
       outside_alpha_mass_far_fraction=float(a[far].sum()/a[m<.01].sum()),
       outside_opaque_pixel_count=int(((m<.01)&(a>.9)).sum()),
       recomputed_visibility_culled_fraction=float(culled.float().mean()),
       recomputed_culled_outside_mean=float(ac[m<.01].mean()),
       gradients=gradients))
    del c,m,a,ac,terms
    torch.cuda.empty_cache()
assert ex.sha(path)==sha
ex.save(RUN/'gate-diagnostic.json',dict(optimizer_updates=0,model_unchanged=True,source_sha256=sha,checks=checks,
    caveat='Visibility culling was recomputed on the frozen checkpoint, not recovered from the original optimizer state. Gradients are local diagnostics, not proof of successful training.'))
print(json.dumps(checks),flush=True)
