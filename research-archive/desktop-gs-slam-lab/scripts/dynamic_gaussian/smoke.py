from pathlib import Path
import sys
import torch
import warp as wp
wp.init()
import mpm_solver_warp.mpm_solver_warp as solver
# Legacy direct-pointer wrapping uses an obsolete ownership keyword. The public
# adapter preserves tensor ownership and zero-copy semantics on modern Warp.
solver.torch2warp_vec3 = lambda t, dvc='cuda:0', **kw: wp.from_torch(t, dtype=wp.vec3)
solver.torch2warp_float = lambda t, dvc='cuda:0', **kw: wp.from_torch(t, dtype=wp.float32)
solver.torch2warp_mat33 = lambda t, dvc='cuda:0', **kw: wp.from_torch(t.reshape(-1,3,3), dtype=wp.mat33)
p=torch.rand((64,3),device='cuda')*.1+.95
vol=torch.full((64,),1e-5,device='cuda')
m=solver.MPM_Simulator_WARP(64,n_grid=24,grid_lim=2.)
m.load_initial_data_from_torch(p,vol,n_grid=24,grid_lim=2.)
m.set_parameters_dict(dict(E=1000.,nu=.3,material='jelly',density=200.,g=[0,0,0]))
m.finalize_mu_lam()
for i in range(3):m.p2g2p(i,1e-4)
x=m.export_particle_x_to_torch()
assert torch.isfinite(x).all()
print('SMOKE PASS',float((x-p).abs().max()))
