"""CPU-only bound/gradient probe on the actual saved target, no training mutation."""
import json
import math
import torch
from torch import nn
from torch.nn.utils import parametrize
import refit_street_human_surface as task


def main():
    torch.set_num_threads(3);torch.manual_seed(181)
    protocol=json.loads((task.OUT/'protocol.json').read_text())
    current_hash=task.sha(task.Path(task.__file__))
    p=task.OUT/'parameterization-sign-preserving-smoke.json'
    assert not p.exists(),'Preserve previous probe'
    saved=torch.load(task.SOURCE/'checkpoint-refined.pt',map_location='cpu',mmap=True,weights_only=False)
    state=saved['models']['SMPLNodes'];indices=torch.where(state['points_ids'][:,0]==20)[0]
    assert len(indices)==6890
    rows=[]
    for field in task.COLOR+task.SURFACE:
        original=state[field].clone()
        module=nn.Module();module.register_parameter('value',nn.Parameter(original.clone(),requires_grad=False))
        change=task.TargetChange(field,original,indices)
        parametrize.register_parametrization(module,'value',change)
        assert torch.equal(original,module.value),field
        weight=torch.randn_like(original[indices])
        (module.value[indices]*weight).sum().backward()
        assert torch.isfinite(change.delta.grad).all() and change.delta.grad.norm()>0,field
        grad=float(change.delta.grad.norm())
        with torch.no_grad():change.delta.copy_(torch.randn_like(change.delta)*100)
        changed=module.value.detach()
        mask=torch.zeros(len(original),dtype=torch.bool);mask[indices]=True
        assert torch.equal(original[~mask],changed[~mask]) and torch.isfinite(changed).all()
        bound={}
        if field=='_means':
            value=(changed[mask]-original[mask]).norm(dim=-1);assert value.max()<=.030001
            bound['max_center_displacement_m']=float(value.max())
        if field=='_scales':
            value=(changed[mask]-original[mask]).exp();assert value.min()>=.49999 and value.max()<=2.00001
            bound.update(min_scale_ratio=float(value.min()),max_scale_ratio=float(value.max()))
        if field=='_quats':
            a=nn.functional.normalize(changed[mask].double(),dim=-1);b=nn.functional.normalize(original[mask].double(),dim=-1)
            degrees=torch.acos((a*b).sum(-1).abs().clamp(max=1))*360/math.pi
            assert degrees.max()<=20.001
            bound['max_rotation_degrees']=float(degrees.max())
        with torch.no_grad():change.delta.zero_()
        parametrize.remove_parametrizations(module,'value',leave_parametrized=True)
        assert torch.equal(original,module.value)
        rows.append(dict(field=field,initial_exact=True,finite_nonzero_gradient_norm=grad,non_target_rows_exact=True,large_latent_probe=bound,rollback_exact=True))
    result=dict(status='actual_checkpoint_target_bounds_gradients_and_exact_identity_passed',device='cpu',source_script_sha256=current_hash,
        fields=rows,live_training_touched=False,optimized_steps=0)
    p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
