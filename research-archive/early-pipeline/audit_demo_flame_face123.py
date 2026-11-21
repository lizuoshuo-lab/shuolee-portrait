"""Read-only completed face checkpoint audit; no optimizer or candidate changes."""
from pathlib import Path
import os, fcntl, json, time
import numpy as np
import torch
import fit_demo_flame_face123 as f
from run_official_gaussians123 import read,save,sha,now

def main():
    out=f.OUT/'postfit-audit.json';assert not out.exists()
    save(f.OUT/'postfit-audit-plan.json',dict(updated=now(),single_readonly_pass=True,
        samples=[0,12,23,28,32,35,46,58],scope='Original loss components, parameter/gradient activity and per-region errors at frozen completed checkpoint',
        optimizer_updates=0,parameter_change=False,input=sha(f.OUT/'epoch-30/face-parameters-and-points.npz')))
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(f.WORK);torch.set_num_threads(4)
    t,rows=f.make_tracker()
    with np.load(f.OUT/'epoch-30/face-parameters-and-points.npz',allow_pickle=False) as z:data=dict(z)
    with torch.no_grad():
        for name in ['shape',*f.PER_FRAME,'scene_scale']:
            getattr(t,name).copy_(torch.tensor(data[name],device='cuda'))
    t.get_train_parameters('lmk_global_tracking')
    stats=[];regions=dict(brows=list(range(17,27)),nose=list(range(27,36)),eyes=list(range(36,48)),mouth=list(range(48,68)))
    for i in [0,12,23,28,32,35,46,58]:
        sample=t.dataset.one(i)
        loss,logs,*rest=t.compute_energy(sample,stage='lmk_global_tracking')
        params=[t.rotation,t.neck_pose,t.shape,t.expr]
        total_grad=torch.autograd.grad(loss,params,retain_graph=True)
        landmark_grad=torch.autograd.grad(logs['lmk'],params)
        row=dict(frame=rows[i]['frame'],losses={k:float(v.detach()) for k,v in logs.items()},
            root_rotation_rad=data['rotation'][i].tolist(),neck_rotation_rad=data['neck_pose'][i].tolist(),regions={})
        for n,idx in regions.items():
            e=np.linalg.norm(data['landmarks2d_native'][i,idx]-t.dataset.obs['keypoints'][i,np.array(idx)+23],axis=1)
            weights=t.dataset.samples[i]['lmk2d'][idx,2].numpy()
            row['regions'][n]=dict(valid_FIT_points=int((weights>0).sum()),relative_weight_sum=float(weights.sum()),
                median_px=float(np.median(e)),max_px=float(e.max()),error_by_IBUG=dict(zip(map(str,idx),map(float,e))))
        row['gradients']={n:dict(landmark_norm=float(g.norm()),total_norm=float(allg.norm()),
            cosine=float((g*allg).sum()/(g.norm()*allg.norm()+1e-15))) for n,g,allg in zip(['root_rotation','neck','shape','expression'],landmark_grad,total_grad)}
        stats.append(row)
    save(out,dict(updated=now(),state='readonly_complete',optimizer_updates=0,network_template_used=False,
        shape_l2=float(np.linalg.norm(data['shape'])),max_abs_shape=float(abs(data['shape']).max()),
        max_abs_neck_radians=float(abs(data['neck_pose']).max()),observed_root_rotation_range_radians=np.ptp(data['rotation'],axis=0).tolist(),
        source_line='vhap/model/tracker.py:366-413: masked L1 landmark residual, nose points multiplied by10',
        warning='Relative detector scores are not visibility. FIT/HELD split is semantic, not independent measured 3D. Loss and gradient sizes do not uniquely identify cause.',
        rows=stats))
    print(json.dumps(read(out),ensure_ascii=False))

if __name__=='__main__':main()
