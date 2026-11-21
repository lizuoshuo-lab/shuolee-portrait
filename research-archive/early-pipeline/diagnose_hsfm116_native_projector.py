"""Investigate failed native round trip from preserved pair cache, no fitting."""
from pathlib import Path
import ast,datetime as dt,json,subprocess,sys
import numpy as np
H=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/hsfm');J=H/'joint-diagnostic-reserved';O=J/'projector-investigation'
R=Path('/opt/gs-slam-lab/shuolee-hsfm116');sys.path[:0]=[str(R/'dust3r'),str(R/'croco'),str(R/'python')]
assert not O.exists();O.mkdir()
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True);assert gpu.returncode==0 and not gpu.stdout.strip()
import torch
from dust3r.cloud_opt.optimizer import PointCloudOptimizer
from dust3r.utils.geometry import geotrf
torch.manual_seed(116);torch.set_num_threads(2)
rows=json.loads((H/'input-pilot/inputs.json').read_text())['records'];ratio=rows[0]['K_processed_centers'][1][1]/rows[0]['K_processed_centers'][0][0]
tree=ast.parse((Path(__file__).parent/'run_hsfm116_joint_diagnostic.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');cls=next(n for n in main.body if isinstance(n,ast.ClassDef))
exec(compile(ast.Module(body=[cls],type_ignores=[]),'<unchanged-portrait-projector>','exec'),globals())
out=torch.load(J/'native-fit/native-156-pair-output.pt',weights_only=False,map_location='cpu')
scene=PortraitOptimizer(*[out[k] for k in ('view1','view2','pred1','pred2')],verbose=False).cuda()
with torch.no_grad():
    pp=torch.tensor(np.array([r['K_processed_centers'] for r in rows])[:,:2,2],device='cuda',dtype=torch.float32);scene.im_pp.copy_((pp-scene._pp)/10);scene.init_default_mst()
    c=scene.get_im_poses();K=scene.get_intrinsics();dep=scene.get_depthmaps(raw=True);idx=torch.arange(0,dep.shape[1],dep.shape[1]//64,device='cuda')
    xyz=scene.get_pts3d(raw=True)[:,idx];X=geotrf(torch.linalg.inv(c),xyz);uv=torch.einsum('nij,nkj->nki',K,X);uv=uv[...,:2]/uv[...,2:];e=(uv-scene._grid[:,idx]).abs()
    ix=np.unravel_index(e.argmax().item(),e.shape)
    gpu_settings=dict(allow_tf32=torch.backends.cuda.matmul.allow_tf32,precision=torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    xyz_no=scene.get_pts3d(raw=True)[:,idx];Xn=geotrf(torch.linalg.inv(c),xyz_no);uvn=torch.einsum('nij,nkj->nki',K,Xn);uvn=uvn[...,:2]/uvn[...,2:]
    grid=scene._grid[:,idx].double();dd=dep[:,idx,None].double();kd=K.double();cd=c.double()
    rel=torch.cat([dd*(grid-kd[:,:2,2][:,None,:])/torch.stack([kd[:,0,0],kd[:,1,1]],dim=1)[:,None,:],dd],dim=-1)
    xd=geotrf(cd,rel);back=geotrf(torch.linalg.inv(cd),xd);uvd=torch.einsum('nij,nkj->nki',kd,back);uvd=uvd[...,:2]/uvd[...,2:]
    local=K@torch.eye(3,device='cuda')[None]
    result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),state='numerical_investigation_only',settings=gpu_settings,
        original_float32_max_px=float(e.max()),tf32_disabled_max_px=float((uvn-scene._grid[:,idx]).abs().max()),full_float64_max_px=float((uvd-grid).abs().max()),
        max_case=dict(raw_frame=rows[ix[0]]['raw_frame'],grid=grid[ix[0],ix[1]].cpu().tolist(),depth=float(dep[ix[0],idx[ix[1]]]),
                      camera_position=c[ix[0],:3,3].cpu().tolist(),world_point=xyz[ix[0],ix[1]].cpu().tolist(),projected=uv[ix[0],ix[1]].cpu().tolist()),
        depth_quantiles=np.quantile(dep.cpu().numpy(),[0,.01,.5,.99,1]).tolist(),camera_translation_max=float(c[:,:3,3].abs().max()),
        network_reinference=False,camera_fitting=False,old_ghosting_cause_proven=False)
    torch.save(scene.state_dict(),O/'native-mst-state.pt')
    (O/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)
