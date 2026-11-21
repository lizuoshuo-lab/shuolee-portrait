"""One preregistered actual SMPL-X optimization on known synthetic observations."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,traceback
import numpy as np
import cv2
import torch
import smplx
from smplx_direct_correspondence123 import face_bindings,face_world,project_integer_centres,DirectFaceObjective

REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT=REC/'direct-smplx-bindings'
MODEL=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
def save(name,v):
    (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()

def main():
    start=time.monotonic()
    assert not (OUT/'result.json').exists()
    plan=json.loads((OUT/'plan.json').read_text())
    torch.manual_seed(6060);np.random.seed(6060);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda';dtype=torch.float32
    g=lambda x:torch.as_tensor(x,device=device,dtype=dtype)
    cpu=lambda x:x.detach().cpu().numpy()
    model=smplx.SMPLX(str(MODEL),num_betas=10,num_expression_coeffs=10,use_pca=False,
                      flat_hand_mean=False,use_face_contour=False,batch_size=1).to(device)
    # Only explicit optimization parameters should retain gradients.
    for p in model.parameters():p.requires_grad_(False)
    bindings=face_bindings(model)
    save('bindings.json',dict(template_sha256=sha(MODEL),anchors=bindings,
                             contours_0_to_16='not a fixed semantic surface binding'))
    E=[]
    for yaw in [-.40,0,.40]:
        center=np.array([3*np.sin(yaw),1.45,3*np.cos(yaw)])
        forward=np.array([0,1.45,0])-center;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
        down=np.cross(forward,right);rot=np.stack([right,down,forward])
        ee=np.eye(4);ee[:3,:3]=rot;ee[:3,3]=-rot@center;E.append(ee)
    E=np.array(E);K=np.broadcast_to(np.array([[2400.,0.,760.35],[0.,2380.,1353.8],[0.,0.,1.]]),(3,3,3)).copy()
    e,k=g(E),g(K)
    gt_pose=np.zeros((1,63),np.float32);gt_pose[0,33:36]=[.025,.04,-.02];gt_pose[0,42:45]=[.04,.10,.015]
    gt_expr=np.zeros((1,10),np.float32);gt_expr[0,:3]=[.35,-.22,.12]
    truth=model(body_pose=g(gt_pose),jaw_pose=g([[.08,.01,-.005]]),expression=g(gt_expr),
                global_orient=g([[.02,-.035,.01]]),transl=g([[.018,-.012,.015]]))
    ref=cpu(truth.joints[0,76:127])
    bound=face_world(truth.vertices,model)[0]
    binding_error=float(torch.max(torch.abs(bound-truth.joints[0,76:127])))
    target=np.stack([cv2.projectPoints(ref.astype(np.float64),cv2.Rodrigues(ee[:3,:3])[0],ee[:3,3],kk,None)[0][:,0] for ee,kk in zip(E,K)])
    calc,_=project_integer_centres(bound[None].expand(3,-1,-1),e,k)
    projection_error=float(np.max(np.abs(cpu(calc)-target)))
    assert binding_error<1e-6 and projection_error<.002
    fit_eligible=np.ones((3,51),bool);fit_eligible[1]=False # camera1 unseen by face optimizer
    split=np.full((3,51),'FIT',dtype='<U4');split[:,::7]='HELD' # whole landmark families held
    obj=DirectFaceObjective(target,np.ones((3,51))*2,np.ones((3,51)),fit_eligible,split,device=device)
    unknown=target.copy();unknown[1]+=10000;unknown[:,::7]-=20000
    masked_obj=DirectFaceObjective(unknown,np.ones((3,51))*2,np.ones((3,51)),fit_eligible,split,device=device)
    params={key:torch.nn.Parameter(torch.zeros(shape,device=device)) for key,shape in [
        ('root',(1,3)),('translation',(1,3)),('neck_head',(1,6)),('jaw',(1,3)),('expression',(1,10))]}
    body_ids=[0,1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,21]
    body_target=np.stack([cv2.projectPoints(cpu(truth.joints[0,body_ids]).astype(float),cv2.Rodrigues(ee[:3,:3])[0],ee[:3,3],kk,None)[0][:,0] for ee,kk in zip(E,K)])
    def state():
        pose=torch.zeros((1,63),device=device)
        pose[:,33:36]=params['neck_head'][:,:3];pose[:,42:45]=params['neck_head'][:,3:]
        m=model(body_pose=pose,jaw_pose=params['jaw'],expression=params['expression'],
                global_orient=params['root'],transl=params['translation'])
        face=face_world(m.vertices,model)
        uv,z=project_integer_centres(face.expand(3,-1,-1),e,k)
        buv,bz=project_integer_centres(m.joints[:,body_ids].expand(3,-1,-1),e,k)
        return m,uv,buv,z
    def objective():
        if time.monotonic()-start>plan['validation']['budget_seconds']:
            raise RuntimeError('Preregistered bounded validation time exhausted.')
        m,uv,buv,z=state();loss,terms=obj(uv)
        body=((buv[[0,2]]-g(body_target)[[0,2]])/2).square().mean()
        return loss+body,uv,terms
    initial=state();initial_uv=cpu(initial[1]);initial_mesh=cpu(initial[0].vertices[0])
    loss,uv,regions=objective()
    g0=torch.autograd.grad(loss,list(params.values()),retain_graph=True)
    gradient={key:float(v.norm()) for key,v in zip(params,g0)}
    assert all(np.isfinite(v) and v>0 for v in gradient.values())
    eye_grad=torch.autograd.grad(regions['eyes'],params['neck_head'],retain_graph=True)[0]
    assert float(eye_grad.norm())>0
    # Inactive and HELD targets must not change either value or derivatives.
    face_a=obj(uv)[0];face_b=masked_obj(uv)[0]
    ga=torch.autograd.grad(face_a,params['neck_head'],retain_graph=True)[0]
    gb=torch.autograd.grad(face_b,params['neck_head'])[0]
    masked_difference=float(torch.max(torch.abs(ga-gb)))
    assert float(torch.abs(face_a-face_b))==0 and masked_difference==0
    history=[]
    optimizer=torch.optim.Adam(list(params.values()),lr=.008)
    for step in range(250):
        optimizer.zero_grad(set_to_none=True);loss,uv,terms=objective();loss.backward();optimizer.step()
        if step%50==0:
            rec=dict(updated=stamp(),state='synthetic_optimization',pid=os.getpid(),step=step,
                     loss=float(loss.detach()),seconds=time.monotonic()-start,training_started=False)
            history.append(rec);save('progress.json',rec);print(json.dumps(rec),flush=True)
    lbfgs=torch.optim.LBFGS(list(params.values()),lr=1,max_iter=150,tolerance_grad=1e-8,
                          tolerance_change=1e-10,line_search_fn='strong_wolfe')
    calls=0
    def closure():
        nonlocal calls
        calls+=1;lbfgs.zero_grad(set_to_none=True);loss,_,_=objective();loss.backward();return loss
    lbfgs.step(closure)
    final=state();final_uv=cpu(final[1]);errors=np.linalg.norm(final_uv-target,axis=-1)
    region_stats={name:dict(initial_max_px=float(np.max(np.linalg.norm(initial_uv-target,axis=-1)[:,sl])),
                           final_max_px=float(errors[:,sl].max())) for name,sl in obj.regions.items()}
    np.savez_compressed(OUT/'synthetic-known-correspondences.npz',E=E,K_integer=K,
                        target_uv=target,initial_uv=initial_uv,final_uv=final_uv,error_px=errors,
                        known_points_world=ref,initial_mesh=initial_mesh,
                        final_mesh=cpu(final[0].vertices[0]),faces=model.faces,
                        fit_weight=cpu(obj.fit_weight),**{key:cpu(value) for key,value in params.items()})
    save('history.json',history)
    passed=bool(errors.max()<1.)
    result=dict(updated=stamp(),state='synthetic_direct_fit_passed' if passed else 'synthetic_direct_fit_failed',
        actual_optimizer_executed=True,real_subject_fit=False,known_synthetic_only=True,
        cameras=3,camera1_unseen_in_fit=True,landmark_families_held=list(range(17,68,7)),
        original_binding_max_error_world=binding_error,OpenCV_projection_max_abs_px=projection_error,
        initial_all_max_px=float(np.linalg.norm(initial_uv-target,axis=-1).max()),
        final_all_max_px=float(errors.max()),unseen_camera_max_px=float(errors[1].max()),
        held_family_max_px=float(errors[:,::7].max()),regions=region_stats,
        gradient_norms=gradient,eye_to_head_gradient=float(eye_grad.norm()),
        masked_and_held_target_gradient_change=masked_difference,
        Adam_updates=250,LBFGS_closure_calls=calls,elapsed_seconds=time.monotonic()-start,
        all_known_synthetic_errors_lt1=passed,training_started=False,quality_accepted=False,
        physical_camera_accepted=False,source_sha256=sha(Path(__file__)),
        module_sha256=sha(Path(__file__).with_name('smplx_direct_correspondence123.py')))
    save('result.json',result);save('progress.json',dict(updated=stamp(),state=result['state'],pid=None))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        save('failure.json',dict(updated=stamp(),error=repr(exc),traceback=traceback.format_exc(),training_started=False,quality_accepted=False))
        raise
