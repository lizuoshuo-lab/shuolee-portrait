"""Read-only native8 mask/mesh/alpha audit. No fitting, parameter updates or model export."""
import argparse,fcntl,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from run_official_gaussians123 import read,save,sha,now,PROJECT,ASSETS
from train_demo_body_upstream123 import DEMO,UP,module

BASE=DEMO/'body-gaussian-smplify-eight-dc-reviewedfit'
SNAP=BASE/'gaussian-run/step-018000'
OUT=BASE/'eight-view-surface-audit'
RECORD=Path('/mnt/d/00/记录/demo')


def check():
    p=read(OUT/'plan.json')
    for path,h in p['files'].items():
        path=path.replace('\\','/')
        if path[1:3]==':/':path='/mnt/'+path[0].lower()+'/'+path[3:]
        assert sha(path)==h,path
    return p


def cpu():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (OUT/'cpu-result.json').exists()
    import cv2
    cv2.setNumThreads(1);started=time.monotonic();p=check()
    rows=read(BASE/'data/training-inputs.json')['records']
    review={r['frame']:r for r in read(SNAP/'review/result.json')['records']}
    with np.load(BASE/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z:mesh=dict(z)
    assert list(fit['frame_ids'])==list(mesh['frame_ids'])==[r['frame'] for r in rows]
    records=[]
    for i,row in enumerate(rows):
        if row['frame'] not in p['frame_ids']:continue
        frame=row['frame'];assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        raw=np.asarray(Image.open(row['image']).convert('RGB'));mask=np.asarray(Image.open(row['mask']).convert('L'))
        M=mask>127;assert raw.shape[:2]==M.shape==(1520,2704)
        v=mesh['vertices_camera'][i];f=mesh['faces'];K=np.asarray(row['K_integer_centers']);q=v@K.T;uv=q[:,:2]/q[:,2:]
        assert (v[:,2]>.05).all()
        silhouette=np.zeros(M.shape,np.uint8)
        for tri in np.rint(uv[f]).astype(np.int32):cv2.fillConvexPoly(silhouette,tri,1)
        np.savez_compressed(OUT/(frame+'-silhouette.npz'),mesh=silhouette)
        keypoints=fit['observed_native_keypoints'][i];pred=fit['joints_projected_native'][i]
        errors=np.linalg.norm(pred-keypoints[:,:2],axis=-1)
        regions={}
        for name,ids in [('body',np.arange(25)),('face',np.arange(67,118))]:
            weight=keypoints[ids,2]**2
            regions[name]=dict(weighted_rms_px=float(np.sqrt((weight*errors[ids]**2).sum()/weight.sum())),
                observations=int((keypoints[ids,2]>.25).sum()),scope='Detector/SMPLify targets, not independent3D material truth')
        body=next(x for x in review[frame]['images'] if x['region']=='body');box=body['crop_integer_xyxy']
        trip=Image.open(body['path']).convert('RGB');w=box[2]-box[0];height=box[3]-box[1]
        assert trip.size==(3*w,height)
        mask_rgb=Image.fromarray(mask).convert('RGB').crop(box)
        masked=Image.fromarray((raw*(mask[:,:,None]/255)).astype(np.uint8)).crop(box)
        sheets=[trip.crop((0,0,w,height)),mask_rgb,masked,trip.crop((w,0,2*w,height)),trip.crop((2*w,0,3*w,height))]
        page=Image.new('RGB',(5*w,height))
        for n,img in enumerate(sheets):page.paste(img,(n*w,0))
        dest=RECORD/('分析_8帧覆盖_'+frame+'_原图_mask_mask内人像_mesh_SH0.png');page.save(dest)
        S=silhouette>0
        records.append(dict(frame=frame,image=str(dest),image_sha256=sha(dest),box=box,regions=regions,
            mesh_mask_iou=float((S&M).sum()/(S|M).sum()),mask_pixels=int(M.sum()),
            mesh_outside_mask_pixels=int((S&~M).sum()),mask_uncovered_by_mesh_pixels=int((M&~S).sum()),
            mask_values=np.unique(mask).tolist(),mesh_silhouette_scope='CPU rounded native triangles, not exact GPU edge coverage'))
        assert time.monotonic()-started<120
    check();assert len(records)==8
    save(OUT/'cpu-result.json',dict(updated=now(),state='completed',records=records,seconds=time.monotonic()-started,quality_accepted=False))
    print('EIGHT_VIEW_CPU_COMPLETE',len(records),flush=True)


def gpu():
    assert not (OUT/'gpu-result.json').exists()
    from run_offsetlr_idle123 import gpu_pids
    p=check();locks=[open(x,'a+') for x in ['/tmp/shuolee-native122-gpu.lock','/tmp/shuolee-body-export123.lock']]
    for f in locks:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not gpu_pids(),'Another GPU task is live'
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS)
    from shuolee_portrait.training import camera,image
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from smplify_direct_surface123 import enable,vertices
    from gaussian_renderer import render
    official=module(UP/'scene/gaussian_model.py','eight_alpha_original_model')
    class Body(official.GaussianModel):
        select_mesh_by_timestep=SMPLXGaussianModel.select_mesh_by_timestep
        get_covariance=SMPLXGaussianModel.get_covariance
        shared_offset_vertices=vertices
    with np.load(BASE/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    with np.load(SNAP/'binding.npz',allow_pickle=False) as z:binding=dict(z)
    model=Body(0);model.static_offset=None;model.num_timesteps=59
    model.vertices_sequence=torch.as_tensor(fit['vertices_world'],device='cuda');model.faces=torch.as_tensor(fit['faces'],dtype=torch.long,device='cuda')
    model.load_ply(str(SNAP/'bound-local.ply'));enable(model,BASE/'fit/fitted.npz',ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    with torch.no_grad():model.static_offset.copy_(torch.as_tensor(binding['static_offset'],device='cuda'))
    assert np.array_equal(model.binding.cpu().numpy(),binding['triangle_ids'])
    pars=[model._xyz,model._features_dc,model._features_rest,model._scaling,model._rotation,model._opacity,model.static_offset]
    frozen=[x.detach().clone() for x in pars]
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
    white=torch.ones((len(model._xyz),3),device='cuda');black=torch.zeros_like(white)
    rows=read(BASE/'data/training-inputs.json')['records'];records=[];started=time.monotonic()
    save(OUT/'gpu-progress.json',dict(state='running_read_only',pid=os.getpid(),frames=0,optimization_updates=0))
    with torch.no_grad():
        for i,row in enumerate(rows):
            if row['frame'] not in p['frame_ids']:continue
            frame=row['frame'];model.select_mesh_by_timestep(i);cam,_=camera(row,i,pose_index=i)
            rgb=render(cam,model,pipe,bg)['render'];saved=np.asarray(Image.open(SNAP/(frame+'_gaussian.png')).convert('RGB')).astype(np.int16)
            parity=float(np.abs(np.asarray(image(rgb)).astype(np.int16)-saved).mean()/255);assert parity<2e-5
            a=render(cam,model,pipe,bg,override_color=white)['render'];assert float((a-a[:1]).abs().max())<1e-6
            complement=None
            if not records:
                inv=render(cam,model,pipe,torch.ones_like(bg),override_color=black)['render']
                complement=float((a+inv-1).abs().max());assert complement<1e-5
            alpha=a[0].cpu().numpy();assert np.isfinite(alpha).all() and alpha.min()>=0 and alpha.max()<=1.00001
            M=np.asarray(Image.open(row['mask']).convert('L'))>127
            A=alpha>=p['alpha_occupancy_threshold'];S=np.load(OUT/(frame+'-silhouette.npz'))['mesh']>0
            np.savez_compressed(OUT/(frame+'-alpha.npz'),alpha=alpha)
            photo=np.asarray(Image.open(row['image']).convert('RGB'),dtype=np.float32)/255
            err=np.abs(rgb.permute(1,2,0).cpu().numpy()-photo).mean(-1)
            records.append(dict(frame=frame,alpha_mask_iou=float((A&M).sum()/(A|M).sum()),
                alpha_missing_mask_fraction=float((M&~A).sum()/M.sum()),alpha_outside_mask_pixels=int((A&~M).sum()),
                alpha_mean_in_mask=float(alpha[M].mean()),alpha_mean_on_mesh_only=float(alpha[S&~M].mean()),
                rgb_error_high_alpha=float(err[M&(alpha>=.95)].mean()),rgb_error_low_alpha=float(err[M&(alpha<.95)].mean()),
                replay_MAE=parity,alpha_complement_max_error=complement))
            save(OUT/'gpu-progress.json',dict(state='running_read_only',pid=os.getpid(),frames=len(records),optimization_updates=0))
            assert time.monotonic()-started<120 and not set(gpu_pids())-{os.getpid()}
    assert all(torch.equal(x,y) for x,y in zip(pars,frozen)) and len(records)==8
    check();save(OUT/'gpu-result.json',dict(updated=now(),state='completed',records=records,seconds=time.monotonic()-started,
        optimizer_created=False,optimization_updates=0,parameter_tensors_bitwise_unchanged=True,quality_accepted=False))
    save(OUT/'gpu-progress.json',dict(updated=now(),state='completed',frames=8,optimization_updates=0))
    print('EIGHT_VIEW_ALPHA_COMPLETE',len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu']);a=p.parse_args()
    try:globals()[a.mode]()
    except BaseException as error:
        import traceback
        save(OUT/(a.mode+'-failure.json'),dict(updated=now(),error=repr(error),traceback=traceback.format_exc()))
        raise
