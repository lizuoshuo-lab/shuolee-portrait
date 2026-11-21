"""Probe the actual pinned temporal initialization without changing saved models."""
import json,torch
from adapter import OUT,DATA,load_official

tr,h=load_official();md=json.loads((DATA/'basketball/train_meta.json').read_text())
params={k:torch.nn.Parameter(v.cuda()) for k,v in torch.load(OUT/'dynamic/frame-0000.pt',map_location='cpu',weights_only=True).items()}
variables={'scene_radius':1.0};optimizer=tr.initialize_optimizer(params,variables)
for value in params.values():value.grad=torch.zeros_like(value)
optimizer.step();optimizer.zero_grad(set_to_none=True)
variables=tr.initialize_post_first_timestep(params,variables,optimizer)
params,variables=tr.initialize_per_timestep(params,variables,optimizer)
reference=variables['prev_col'];independent=params['rgb_colors'].detach().clone()
shared=reference.data_ptr()==params['rgb_colors'].data_ptr()
with torch.no_grad():params['rgb_colors'][0,0]+=0.25
published_loss=h.l1_loss_v2(params['rgb_colors'],reference)
cloned_loss=h.l1_loss_v2(params['rgb_colors'],independent)
grad=torch.autograd.grad(published_loss,params['rgb_colors'])[0]
result=dict(source_commit='7dbbd4dec404308524ff402756bdb8143a2589b0',scope='separate in-memory probe of actual initialize_per_timestep; saved training weights unchanged',previous_color_shares_parameter_storage=shared,controlled_parameter_increment=.25,published_color_loss_after_increment=float(published_loss),independent_previous_color_loss_after_increment=float(cloned_loss),published_color_gradient_l1=float(grad.abs().sum()),finding='detach aliases the updated parameter; the intended previous-frame color term is identically zero during in-place Adam updates')
assert shared and float(published_loss)==0 and float(cloned_loss)>0 and float(grad.abs().sum())==0
(OUT/'color-reference-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
