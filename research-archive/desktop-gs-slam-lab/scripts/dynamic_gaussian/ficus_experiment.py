"""Bounded PhysGaussian Ficus intervention; official MPM, gsplat rendering adapter."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import torch
import warp as wp
from plyfile import PlyData
from PIL import Image
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
import mpm_solver_warp.mpm_solver_warp as solver
from utils.decode_param import decode_param_json, set_boundary_conditions

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-gaussian-001'
MODEL=OUT/'model/ficus_whitebg-trained'
SOURCE=LAB/'repos/PhysGaussian'

def save(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def tensor(a):return torch.as_tensor(np.array(a),device='cuda',dtype=torch.float32).contiguous()
def unpack(c):
    x=torch.zeros((len(c),3,3),device=c.device)
    x[:,0,0]=c[:,0];x[:,0,1]=x[:,1,0]=c[:,1];x[:,0,2]=x[:,2,0]=c[:,2]
    x[:,1,1]=c[:,3];x[:,1,2]=x[:,2,1]=c[:,4];x[:,2,2]=c[:,5]
    return x
def quatmat(q):
    q=q/torch.linalg.norm(q,dim=1,keepdim=True);w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)

def main():
    a=argparse.ArgumentParser();a.add_argument('--case',choices=['zero','half','full','soft','half_dt'],required=True);a.add_argument('--frames',type=int,default=30);a.add_argument('--render-only',action='store_true');args=a.parse_args()
    folder=OUT/args.case;folder.mkdir(exist_ok=True);(folder/'frames').mkdir(exist_ok=True)
    wp.init()
    solver.torch2warp_vec3=lambda t,dvc='cuda:0',**kw:wp.from_torch(t,dtype=wp.vec3)
    solver.torch2warp_float=lambda t,dvc='cuda:0',**kw:wp.from_torch(t,dtype=wp.float32)
    solver.torch2warp_mat33=lambda t,dvc='cuda:0',**kw:wp.from_torch(t.reshape(-1,3,3),dtype=wp.mat33)
    config=json.loads((SOURCE/'config/ficus_config.json').read_text())
    factor={'zero':0.,'half':.5,'full':1.,'soft':1.,'half_dt':1.}[args.case]
    config['boundary_conditions'][1]['force'][0]*=factor
    if args.case=='soft':
        config['E']*=.5
        for region in config['additional_material_params']:region['E']*=.5
    if args.case=='half_dt':
        config['substep_dt']*=.5
        config['boundary_conditions'][1]['num_dt']=2
        config['grid_v_damping_scale']=config['grid_v_damping_scale']**.5
    config['frame_num']=args.frames;config['move_camera']=False
    save(folder/'config.json',config)
    material,bc,timing,prep,camera=decode_param_json(str(folder/'config.json'))
    ply=MODEL/'point_cloud/iteration_60000/point_cloud.ply';data=PlyData.read(ply)['vertex'].data
    def fields(names):return np.stack([data[n] for n in names],-1)
    xyz=tensor(fields(['x','y','z']));opacity=torch.sigmoid(tensor(data['opacity']))
    mask=opacity>prep['opacity_threshold'];xyz=xyz[mask];opacity=opacity[mask]
    q=tensor(fields([f'rot_{i}' for i in range(4)]))[mask];s=torch.exp(tensor(fields([f'scale_{i}' for i in range(3)])))[mask]
    rot=quatmat(q);L=rot*s[:,None,:];cov=L@L.transpose(1,2)
    dc=tensor(fields([f'f_dc_{i}' for i in range(3)]))[mask,None,:]
    rest=tensor(fields([f'f_rest_{i}' for i in range(45)]))[mask].reshape(-1,3,15).transpose(1,2)
    sh=torch.cat([dc,rest],dim=1).contiguous()
    lo=xyz.min(0).values;hi=xyz.max(0).values;center=(lo+hi)/2;scale=prep['scale']/(hi-lo).max()
    x0=((xyz-center)*scale+1).contiguous();cov0=cov*scale**2
    cov6=cov0[:,[0,0,0,1,1,2],[0,1,2,1,2,2]].contiguous()
    dx=material['grid_lim']/material['n_grid']
    cell=torch.floor(x0/dx).long();flat=cell[:,0]*material['n_grid']**2+cell[:,1]*material['n_grid']+cell[:,2]
    counts=torch.bincount(flat,minlength=material['n_grid']**3)
    volume=(dx**3/counts[flat]).float().contiguous()
    sim=solver.MPM_Simulator_WARP(10)
    sim.load_initial_data_from_torch(x0.clone(),volume,cov6,n_grid=material['n_grid'],grid_lim=material['grid_lim'])
    sim.set_parameters_dict(material);set_boundary_conditions(sim,bc,timing);sim.finalize_mu_lam()
    mass=wp.to_torch(sim.mpm_state.particle_mass).clone()
    cam=json.loads((MODEL/'cameras.json').read_text())[0]
    c2w=torch.eye(4,device='cuda');c2w[:3,:3]=tensor(cam['rotation']);c2w[:3,3]=tensor(cam['position'])
    view=torch.linalg.inv(c2w).unsqueeze(0);width=height=640
    K=tensor([[cam['fx']*.8,0,width/2],[0,cam['fy']*.8,height/2],[0,0,1]]).unsqueeze(0)
    def render(x,cv,R_inverse,index):
        world=(x-1)/scale+center;dirs=world-c2w[:3,3];dirs=dirs/torch.linalg.norm(dirs,dim=1,keepdim=True)
        dirs=torch.bmm(R_inverse,dirs[:,:,None]).squeeze(-1).contiguous()
        colors=(spherical_harmonics(3,dirs,sh)+.5).clamp_min(0)
        rgb,alpha,meta=rasterization(world,None,None,opacity,colors,view,K,width,height,covars=cv/scale**2,packed=False,backgrounds=torch.ones((1,3),device='cuda'))
        image=(rgb[0].clamp(0,1).cpu().numpy()*255+.5).astype(np.uint8)
        Image.fromarray(image).save(folder/'frames'/f'{index:04d}.png')
        return image
    I=torch.eye(3,device='cuda').expand(len(x0),-1,-1)
    render(x0,cov0,I,0)
    if args.render_only:print('RENDER ONLY PASS');return
    rows=[];start=time.perf_counter();dt=timing['substep_dt'];nsub=round(timing['frame_dt']/dt)
    probes=torch.arange(0,len(x0),max(1,len(x0)//2000),device='cuda')
    # Upper foliage: this diagnostic group remains fixed in material coordinates.
    foliage=x0[:,2]>1.15
    lower=x0[:,2]<.7
    for frame in range(1,args.frames+1):
        tick=time.perf_counter()
        for sub in range(nsub):sim.p2g2p((frame-1)*nsub+sub,dt)
        wp.synchronize()
        # The author's renderer exports covariance from the post-G2P trial F,
        # whereas export_particle_F_to_torch returns the preceding stress F.
        x=sim.export_particle_x_to_torch().clone();F=wp.to_torch(sim.mpm_state.particle_F_trial).clone()
        cv=unpack(sim.export_particle_cov_to_torch().reshape(-1,6));R=sim.export_particle_R_to_torch().reshape(-1,3,3)
        assert torch.isfinite(x).all() and torch.isfinite(F).all() and torch.isfinite(cv).all()
        determinants=torch.linalg.det(F);assert determinants.min()>0,'deformation inverted'
        assert x.min()>.06 and x.max()<1.94,'near grid boundary'
        expected=F@cov0@F.transpose(1,2)
        discrepancy=float((cv-expected).abs().max())
        relative=float(torch.linalg.norm(cv-expected)/torch.linalg.norm(expected))
        assert relative<1e-5
        displacement=x-x0
        speed=sim.export_particle_v_to_torch().clone()
        row=dict(frame=frame,time=sim.time,substeps=frame*nsub,foliage_mean_dx=float(displacement[foliage,0].mean()),lower_mean_displacement=float(displacement[lower].norm(dim=1).mean()),rms_displacement=float(torch.sqrt((displacement**2).sum(1).mean())),max_displacement=float(displacement.norm(dim=1).max()),min_det_F=float(determinants.min()),max_det_F=float(determinants.max()),covariance_relative_error=relative,covariance_max_error=discrepancy,kinetic_energy=float((.5*mass*(speed**2).sum(1)).sum()),simulation_seconds=time.perf_counter()-tick)
        rows.append(row)
        render(x,cv,R,frame)
        if frame in {1,args.frames//2,args.frames}:
            np.savez_compressed(folder/f'state-{frame:04d}.npz',position=x.cpu().numpy(),F=F.cpu().numpy(),velocity=speed.cpu().numpy(),covariance=cv.cpu().numpy(),mass=mass.cpu().numpy(),initial_position=x0.cpu().numpy(),initial_covariance=cov0.cpu().numpy())
        save(folder/'measurements.json',rows)
        print(args.case,frame,round(row['foliage_mean_dx'],6),round(row['simulation_seconds'],2),flush=True)
    summary=dict(case=args.case,status='completed',particles=len(x0),source_particles=len(data),ply_sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),frames=args.frames+1,simulated_seconds=sim.time,substeps=args.frames*nsub,dt=dt,frame_dt=timing['frame_dt'],force_per_selected_particle=bc[1]['force'],impulse_duration=dt*bc[1]['num_dt'],total_mass=float(mass.sum()),foliage_particles=int(foliage.sum()),lower_particles=int(lower.sum()),wall_seconds=time.perf_counter()-start,torch=torch.__version__,warp=wp.__version__,gpu=torch.cuda.get_device_name(),all_frames_finite=True,all_frames_det_F_positive=True,max_covariance_relative_error=max(r['covariance_relative_error'] for r in rows),max_foliage_mean_displacement=max(abs(r['foliage_mean_dx']) for r in rows),final=rows[-1],scope='official MPM kernels and ficus material/BC; modern Warp public Torch adapter; equivalent occupied-cell volume via Torch; gsplat renderer, camera 0, SH rotated with F polar rotation; no retraining, no interior filling')
    save(folder/'summary.json',summary)
    print('COMPLETED',args.case,summary['wall_seconds'],flush=True)

if __name__=='__main__':
    with torch.no_grad():main()
