"""Resettable Ficus environment: mass-proportional external force on a fixed region.

The additional kernel is an action adapter, not a change to official MPM kernels.
Observed state is privileged mean x displacement and x velocity in four z bands.
"""
from pathlib import Path
import copy,json
import numpy as np
import torch
import warp as wp
from plyfile import PlyData
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
import mpm_solver_warp.mpm_solver_warp as solver
from mpm_solver_warp.warp_utils import MPMStateStruct
from utils.decode_param import decode_param_json,set_boundary_conditions
from ficus_experiment import quatmat,unpack,tensor

LAB=Path(__file__).resolve().parents[2]
MODEL=LAB/'outputs/dynamic-gaussian-001/model/ficus_whitebg-trained'
CONFIG=LAB/'repos/PhysGaussian/config/ficus_config.json'

@wp.kernel
def add_acceleration(state:MPMStateStruct,mask:wp.array(dtype=int),acceleration:float,dt:float):
    p=wp.tid()
    if mask[p]==1:
        state.particle_v[p]=state.particle_v[p]+wp.vec3(acceleration*dt,0.0,0.0)

class FicusEnv:
    def __init__(self,dt=1e-4,frame_dt=.04,action_scale=2.):
        wp.init()
        solver.torch2warp_vec3=lambda t,dvc='cuda:0',**kw:wp.from_torch(t,dtype=wp.vec3)
        solver.torch2warp_float=lambda t,dvc='cuda:0',**kw:wp.from_torch(t,dtype=wp.float32)
        solver.torch2warp_mat33=lambda t,dvc='cuda:0',**kw:wp.from_torch(t.reshape(-1,3,3),dtype=wp.mat33)
        self.material,self.bc,self.timing,self.prep,_=decode_param_json(str(CONFIG))
        self.bc=[b for b in self.bc if b['type']!='particle_impulse']
        self.dt=dt;self.frame_dt=frame_dt;self.nsub=round(frame_dt/dt);self.action_scale=action_scale
        assert abs(self.nsub*dt-frame_dt)<1e-12
        self.material['grid_v_damping_scale']=.9999**(dt/1e-4)
        data=PlyData.read(MODEL/'point_cloud/iteration_60000/point_cloud.ply')['vertex'].data
        def fields(names):return np.stack([data[n] for n in names],-1)
        xyz=tensor(fields(['x','y','z']));opacity=torch.sigmoid(tensor(data['opacity']));mask=opacity>.02
        xyz=xyz[mask];self.opacity=opacity[mask]
        rot=quatmat(tensor(fields([f'rot_{i}' for i in range(4)]))[mask])
        scales=torch.exp(tensor(fields([f'scale_{i}' for i in range(3)])))[mask]
        L=rot*scales[:,None,:]
        lo=xyz.min(0).values;hi=xyz.max(0).values;self.center=(lo+hi)/2;self.scale=1/(hi-lo).max()
        self.x0=((xyz-self.center)*self.scale+1).contiguous();self.cov0=(L@L.transpose(1,2))*self.scale**2
        self.cov6=self.cov0[:,[0,0,0,1,1,2],[0,1,2,1,2,2]].contiguous()
        dc=tensor(fields([f'f_dc_{i}' for i in range(3)]))[mask,None,:]
        rest=tensor(fields([f'f_rest_{i}' for i in range(45)]))[mask].reshape(-1,3,15).transpose(1,2)
        self.sh=torch.cat([dc,rest],dim=1).contiguous()
        dx=.04;cell=torch.floor(self.x0/dx).long();flat=cell[:,0]*2500+cell[:,1]*50+cell[:,2]
        counts=torch.bincount(flat,minlength=50**3);self.volume=(dx**3/counts[flat]).float().contiguous()
        self.region_tensor=(self.x0[:,2]>1.15).to(torch.int32).contiguous()
        self.region=wp.from_torch(self.region_tensor,dtype=wp.int32)
        boundaries=[.8,.975,1.15,1.325,2.]
        self.bands=[(self.x0[:,2]>=a)&(self.x0[:,2]<b) for a,b in zip(boundaries[:-1],boundaries[1:])]
        self.band_counts=[int(b.sum()) for b in self.bands]
        self.observation_weights=np.zeros(8);self.observation_weights[4]=self.band_counts[2]/sum(self.band_counts[2:]);self.observation_weights[6]=self.band_counts[3]/sum(self.band_counts[2:])
        self.lower=self.x0[:,2]<.7
        cam=json.loads((MODEL/'cameras.json').read_text())[0]
        c2w=torch.eye(4,device='cuda');c2w[:3,:3]=tensor(cam['rotation']);c2w[:3,3]=tensor(cam['position'])
        self.view=torch.linalg.inv(c2w)[None];self.campos=c2w[:3,3];self.focal=cam['fx']/800
        self.reset()

    def reset(self):
        self.sim=solver.MPM_Simulator_WARP(10)
        self.sim.load_initial_data_from_torch(self.x0.clone(),self.volume,self.cov6,n_grid=50,grid_lim=2.)
        self.sim.set_parameters_dict(copy.deepcopy(self.material));set_boundary_conditions(self.sim,self.bc,self.timing);self.sim.finalize_mu_lam()
        self.index=0;self.last_action=0.;return self.observe()

    def observe(self):
        wp.synchronize()
        x=self.sim.export_particle_x_to_torch();v=self.sim.export_particle_v_to_torch()
        return torch.stack([z for b in self.bands for z in [(x[b,0]-self.x0[b,0]).mean(),v[b,0].mean()]]).cpu().numpy().astype(np.float64)

    def step(self,action,disturbance_delta_v=0.):
        action=float(action)
        if not np.isfinite(action) or abs(action)>1+1e-8:raise ValueError('action must be finite and in [-1,1]')
        if disturbance_delta_v:
            wp.launch(add_acceleration,dim=len(self.x0),inputs=[self.sim.mpm_state,self.region,float(disturbance_delta_v),1.],device='cuda:0')
        for j in range(self.nsub):
            wp.launch(add_acceleration,dim=len(self.x0),inputs=[self.sim.mpm_state,self.region,action*self.action_scale,self.dt],device='cuda:0')
            self.sim.p2g2p(self.index*self.nsub+j,self.dt)
        self.index+=1;self.last_action=action
        obs=self.observe();x=self.sim.export_particle_x_to_torch();F=wp.to_torch(self.sim.mpm_state.particle_F_trial)
        finite=bool(torch.isfinite(x).all() and torch.isfinite(F).all());detmin=float(torch.linalg.det(F).min())
        bounds=[float(x.min()),float(x.max())]
        if not finite or detmin<=0 or bounds[0]<.06 or bounds[1]>1.94:raise RuntimeError(f'Invalid state frame {self.index}: finite={finite} det={detmin} bounds={bounds}')
        info=dict(index=self.index,time=self.index*self.frame_dt,solver_time=self.sim.time,action=action,applied_acceleration_x=action*self.action_scale,disturbance_delta_v=disturbance_delta_v,min_det_F=detmin,position_bounds=bounds,lower_max_displacement=float((x[self.lower]-self.x0[self.lower]).norm(dim=1).max()),foliage_mean_dx=float(self.observation_weights@obs))
        return obs,info

    def render(self,size=320):
        if self.index==0:x=self.x0;cv=self.cov0;inverse=torch.eye(3,device='cuda').expand(len(x),-1,-1)
        else:
            x=self.sim.export_particle_x_to_torch();cv=unpack(self.sim.export_particle_cov_to_torch().reshape(-1,6));inverse=self.sim.export_particle_R_to_torch().reshape(-1,3,3)
        world=(x-1)/self.scale+self.center;d=world-self.campos;d=d/d.norm(dim=1,keepdim=True);d=torch.bmm(inverse,d[:,:,None]).squeeze(-1).contiguous()
        colors=(spherical_harmonics(3,d,self.sh)+.5).clamp_min(0)
        K=tensor([[self.focal*size,0,size/2],[0,self.focal*size,size/2],[0,0,1]])[None]
        rgb,_,_=rasterization(world,None,None,self.opacity,colors,self.view,K,size,size,covars=cv/self.scale**2,packed=False,backgrounds=torch.ones((1,3),device='cuda'))
        return (rgb[0].clamp(0,1).cpu().numpy()*255+.5).astype(np.uint8)

    def snapshot(self):
        wp.synchronize()
        return {name:wp.to_torch(getattr(self.sim.mpm_state,name)).cpu().numpy().copy() for name in ['particle_x','particle_v','particle_F_trial','particle_C']}

    def contract(self):
        return dict(particles=len(self.x0),material_bands_z=[.8,.975,1.15,1.325,2.],band_counts=self.band_counts,feature_order=[f'band{i}_{name}' for i in range(4) for name in ['mean_dx','mean_vx']],foliage_readout=self.observation_weights.tolist(),action_region='initial normalized z > 1.15, immutable material-particle mask',selected_action_particles=int(self.region_tensor.sum()),action='u in [-1,1]; mass-proportional force m_p * 2u along x, applied in every substep of 0.04 interval',action_scale=self.action_scale,dt=self.dt,frame_dt=self.frame_dt,physical_scale_calibrated=False,observation='privileged MPM state aggregation, not estimated from RGB',render_camera='author camera 0; 320x320 unless requested otherwise',internal_filling=False)
