"""Read-only integrity and optimizer checks at native topology transitions."""
import json
from pathlib import Path
import torch

@torch.no_grad()
def audit_topology(model,mask_blur,iteration,camera,output,loss):
    count=len(model.get_xyz)
    assert count>100 and bool(torch.isfinite(loss)), 'Empty topology or non-finite loss'
    for name in ['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation',
                 'xyz_gradient_accum','denom','max_radii2D','_culling','factor_culling']:
        tensor=getattr(model,name)
        assert tensor.shape[0]==count,(iteration,name,tensor.shape,count)
        assert bool(torch.isfinite(tensor).all()),(iteration,name,'nonfinite')
    assert len(mask_blur)==count,(iteration,'blur mask shape',len(mask_blur),count)
    checked=0
    for group in model.optimizer.param_groups:
        for param in group['params']:
            assert param.shape[0]==count,(iteration,group['name'],'optimizer parameter shape')
            state=model.optimizer.state.get(param,{})
            for key in ['exp_avg','exp_avg_sq']:
                if key in state:
                    assert state[key].shape==param.shape,(iteration,group['name'],key)
                    assert bool(torch.isfinite(state[key]).all())
                    checked+=1
    with (Path(output)/'late-topology-audit.jsonl').open('a') as file:
        file.write(json.dumps(dict(iteration=iteration,gaussians=count,width=camera.image_width,
            height=camera.image_height,optimizer_moment_checks=checked,loss=float(loss),
            allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20,
            status='passed'))+'\n')
