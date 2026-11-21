from run_calibration_model_090 import O,S,read,save,sha,stamp,rw
import numpy as np
import re

def proj(v,cam):
    p=cam.params;xy=v[:,:2]/v[:,2,None];x,y=xy.T;r=x*x+y*y
    if cam.model=='SIMPLE_RADIAL':f,cx,cy,k=p;return xy*(1+k*r[:,None])*f+[cx,cy]
    fx,fy,cx,cy,k1,k2,p1,p2=p;rad=1+k1*r+k2*r*r
    return np.stack([fx*(x*rad+2*p1*x*y+p2*(r+2*x*x))+cx,fy*(y*rad+p1*(r+2*y*y)+2*p2*x*y)+cy],1)
report={}
for arm in ['simple_radial','opencv']:
    root=O/arm;seedc,seedi,_=rw.read_model(str(root/'seed'),ext='.bin');tc,ti,tp=rw.read_model(str(root/'triangulated'),ext='.bin');rc,ri,rp=rw.read_model(str(root/'refined'),ext='.bin')
    cam=next(iter(rc.values()));points=np.full((max(rp)+1,3),np.nan)
    for k,p in rp.items():points[k]=p.xyz
    residual=[];negative=0
    for im in ri.values():
        good=im.point3D_ids>=0;v=points[im.point3D_ids[good]]@rw.qvec2rotmat(im.qvec).T+im.tvec
        e=np.linalg.norm(proj(v,cam)-im.xys[good],axis=1);residual.extend(e.tolist());negative+=int((v[:,2]<=0).sum())
    pose_diff=max(max(np.max(abs(ti[i].qvec-seedi[i].qvec)),np.max(abs(ti[i].tvec-seedi[i].tvec))) for i in ti)
    intr_diff=float(np.max(abs(next(iter(tc.values())).params-next(iter(seedc.values())).params)))
    assert pose_diff<1e-12 and intr_diff<1e-12
    text=(O/'logs'/f'{arm}_joint_ba.log').read_text(encoding='utf8');term=re.findall(r'Termination\s*:\s*([^\r\n]+)',text)[-1].strip()
    report[arm]=dict(registered=len(ri),points=len(rp),source_point_count=0,triangulation_pose_difference=float(pose_diff),triangulation_intrinsics_difference=intr_diff,
      native_solver_termination=term,point_error_field_is_cached_pre_BA=True,actual_final_observation_count=len(residual),actual_final_reprojection=dict(mean=float(np.mean(residual)),median=float(np.median(residual)),p90=float(np.percentile(residual,90))),negative_depth_observations=negative)
state=read(O/'state.json')
for row in state['completed_arms']:
    row['cached_point_error_pre_BA_not_final_loss']=row.pop('native_training_reprojection_mean')
    row['actual_final_observation_reprojection']=report[row['arm'].lower()]['actual_final_reprojection']
save(O/'state.json',state)
save(O/'native-model-audit.json',dict(updated=stamp(),arms=report,cached_field_correction='COLMAP points3D.bin error fields were unchanged by bundle_adjuster. Final errors recomputed from saved3Dpoints,cameras andrawSIFTpositions; cached means must not be called finalBAerrors.',source_sha256=sha(__file__),not_validation_GT=True))
print({k:v['actual_final_reprojection'] for k,v in report.items()})
