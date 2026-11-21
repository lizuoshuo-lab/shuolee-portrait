from pathlib import Path
import torch,json
from fourkd_source_support import normalize_supported_weights
w=torch.tensor([.2,.3,.5],dtype=torch.float64).reshape(1,3,1,1).repeat(1,1,3,1).requires_grad_()
support=torch.tensor([[1.,0.,1.],[0.,0.,1.],[1.,0.,1.]],dtype=torch.float64).reshape(1,3,3,1)
out=normalize_supported_weights(w,support)
assert torch.allclose(out[0,:,0,0],torch.tensor([2/7,0,5/7],dtype=torch.float64))
assert torch.equal(out[0,:,1,0],w[0,:,1,0])
assert torch.equal(out[0,:,2,0],w[0,:,2,0])
assert torch.allclose(out.sum(1),torch.ones_like(out.sum(1)))
loss=(out*torch.tensor([.7,.8,.2],dtype=torch.float64).reshape(1,3,1,1)).sum();loss.backward()
assert torch.isfinite(w.grad).all() and w.grad[0,1,0,0]==0
p=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-4k4d-source-gate-training-044/support-weight-check.json')
p.write_text(json.dumps(dict(status='normalization_exclusion_fallback_and_gradient_verified',all_sources_supported_unchanged=True,no_supported_source_fallback=True,excluded_source_gradient_zero=True),indent=2)+'\n')
print('source-support mathematical checks passed')
