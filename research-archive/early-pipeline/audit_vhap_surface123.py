"""Read-only audit of the selected-reference fit, its pixels and surface signal.

No optimizer is constructed and no fitted parameter or source input is changed.
"""
import argparse,os,time,fcntl,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from vhap_demo_adapter123 import ROOT,DATA,REPO,make_tracker
from continue_vhap_demo123 import PARAMS
from run_official_gaussians123 import read,save,sha,now

OUT=ROOT/'vhap-selected-reference/surface-audit'
SLOTS=[0,12,23,32,35,46,58]

def stats(v):
    v=np.asarray(v);return dict(n=int(v.size),median=float(np.median(v)),p90=float(np.quantile(v,.9)),max=float(v.max()))

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',choices=['vhap-selected-reference','vhap-selected-reference-white'],default='vhap-selected-reference')
    args=parser.parse_args()
    OUT=ROOT/args.run/'surface-audit'
    import torch
    from review_full_sequence123 import crop_box,joined
    assert not OUT.exists(),'Review existing audit, never run twice.'
    assert read(OUT.parent/'result.json')['full_sequence_complete']
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(6060)
    checkpoint=OUT.parent/'global-epoch-30/tracked-smplx-params.npz'
    save(OUT/'plan.json',dict(updated=now(),state='read_only_diagnostic',optimizer_updates=0,
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),all_frames=59,gradient_slots=SLOTS,
        checks=['Native CPU/K and CUDA projection closure', 'Actual CUDA silhouette against person masks',
                'Shared-offset region statistics','Original loss gradients without optimizer updates',
                'Target-background uncovered-area error versus supported white-background diagnostic'],
        no_new_fit=True,no_camera_changes=True,max_seconds=240))
    try:
        t,rows=make_tracker(OUT)
        if args.run=='vhap-selected-reference-white':
            control=read(OUT.parent/'single-variable-control.json')
            assert control['original_config_except_background_unchanged']
            t.cfg.render.background_train='white'
        with np.load(checkpoint,allow_pickle=False) as z,torch.no_grad():
            z=dict(z)
            assert list(z['frame_ids'])==[r['frame'] for r in rows]
            for key in PARAMS:getattr(t,key).copy_(torch.tensor(z[key],device='cuda'))
        hashes_before={k:sha_bytes(getattr(t,k)) for k in PARAMS}
        t.get_train_parameters('rgb_global_tracking')
        weights=t.smplx.lbs_weights.detach().cpu().numpy()
        dominant=weights.argmax(1)
        # Coarse model skinning regions, explicitly not a clothing segmentation.
        regions={'trunk':np.isin(dominant,[0,3,6,9,12]),'legs':np.isin(dominant,[1,2,4,5,7,8,10,11]),
                 'arms':np.isin(dominant,[13,14,16,17,18,19]),'head':np.isin(dominant,[15,22,23,24]),
                 'hands':np.isin(dominant,[20,21]) | (dominant>=25)}
        offset=np.linalg.norm(z['static_offset'][0],axis=-1)*1000
        records=[];gradients=[];examples=[]
        for i,row in enumerate(rows):
            if time.monotonic()-start>240:raise RuntimeError('Audit time budget exceeded.')
            sample=t.dataset.one(i);t.clear_cache()
            with torch.no_grad():
                v,vc,j,face,albedo=t.forward_smplx(sample['timestep_index'])
                K=torch.tensor(row['K_integer_centers'],dtype=v.dtype,device='cuda')
                q=j@K.T;uv=q[...,:2]/q[...,2:]
                ndc=t.render.world_to_ndc(j,sample['extrinsic'].cuda(),sample['intrinsic'].cuda(),t.image_size,flip_y=True)
                gpu_uv=(ndc[...,:2]+1)*torch.tensor([2704,1520],device='cuda')/2-.5
                raster=t.rasterize_smplx(sample,v,t.smplx.faces_tensor,train_mode=False)
                visible=(raster['rast_out'][0,...,-1].flip(0)>0).cpu().numpy()
                target=sample['alpha_map'][0,0].numpy()>.5
                intersection=int((visible&target).sum());union=int((visible|target).sum())
                body_ids=np.array([x for x in t.dataset.COCO if x not in (9,12)])
                btarget=sample['lmk2d_body'][0,body_ids,:2].numpy()-.5
                body_error=np.linalg.norm(uv[0,body_ids].cpu().numpy()-btarget,axis=-1)
                qf=face@K.T;face_uv=(qf[...,:2]/qf[...,2:])[0].cpu().numpy()
                ftarget=sample['lmk2d'][0,17:,:2].numpy()-.5
                face_error=np.linalg.norm(face_uv-ftarget,axis=-1)
                # Static lmk return and mapped SMPL-X face joints must agree.
                face_joint_parity=float(np.abs(face_uv-uv[0,67:118].cpu().numpy()).max())
            record=dict(slot=i,frame=row['frame'],native_projection_max_abs_px=float((uv-gpu_uv).abs().max()),
                static_face_vs_mapped_joints_max_abs_px=face_joint_parity,
                effective_body15_error_px=stats(body_error),inner_face51_error_px=stats(face_error),
                mask_IoU=intersection/max(1,union),target_uncovered_fraction=float((target&~visible).sum()/max(1,target.sum())),
                mesh_outside_target_fraction=float((visible&~target).sum()/max(1,visible.sum())))
            records.append(record)
            if i in SLOTS:
                raw=Image.open(row['image']).convert('RGB');arr=np.array(raw)
                overlay=arr.copy();overlay[visible]=(.45*overlay[visible]+.55*np.array([50,200,220])).astype(np.uint8)
                mismatch=np.full_like(arr,255);mismatch[target&visible]=[90,180,90]
                mismatch[target&~visible]=[230,90,40];mismatch[visible&~target]=[60,100,240]
                crop=crop_box(row['initial_box_xyxy'],raw.size,12)
                joined([raw.crop(crop),Image.fromarray(overlay).crop(crop),Image.fromarray(mismatch).crop(crop)]).save(OUT/f'{row["frame"]}_raw_cuda_mask.png')
                native_points=raw.copy();draw=ImageDraw.Draw(native_points)
                for p in ftarget:draw.ellipse((p[0]-1,p[1]-1,p[0]+1,p[1]+1),fill=(0,255,0))
                for p in face_uv:draw.ellipse((p[0]-1,p[1]-1,p[0]+1,p[1]+1),outline=(255,0,220))
                fc=crop_box(row['face_box_xyxy'],raw.size,12)
                joined([raw.crop(fc),native_points.crop(fc)]).save(OUT/f'{row["frame"]}_face_observed_projected.png')
                del v,vc,j,face,albedo,raster,ndc,q,qf,uv,gpu_uv
                t.clear_cache()
                loss,logs,verts,faces,lmks,albedos,result=t.compute_energy(sample,stage='rgb_global_tracking')
                gr={}
                for name in ['lmk','photo','reg_offset','reg_offset_lap','smooth_pose','smooth_joint']:
                    if name not in logs:continue
                    g=torch.autograd.grad(logs[name],[t.static_offset,t.global_orient,t.pose_embedding],retain_graph=True,allow_unused=True)
                    gr[name]=dict(loss=float(logs[name].detach()),
                        offset_L2=None if g[0] is None else float(g[0].norm()),
                        root_L2=None if g[1] is None else float(g[1][i].norm()),
                        body_L2=None if g[2] is None else float(g[2][i].norm()),
                        offset_region_L2=None if g[0] is None else {k:float(g[0][0,torch.tensor(m,device='cuda')].norm()) for k,m in regions.items()})
                pred=result['pred_rgb'];gt=result['gt_rgb'].to(pred);alpha=result['pred_alpha']
                uncovered=(sample['alpha_map'].cuda()>.5)&(alpha<1e-6)
                err=(pred-gt).abs();uncovered_err=float(err[uncovered.expand_as(err)].mean()) if uncovered.any() else None
                old_bg=t.cfg.render.background_train;t.cfg.render.background_train='white'
                with torch.no_grad():
                    rr=t.rasterize_smplx(sample,verts,t.smplx.faces_tensor,train_mode=False)
                    white_loss,white_result=t.compute_photometric_energy(sample,verts,faces,albedos,rr,stage='rgb_global_tracking')
                t.cfg.render.background_train=old_bg
                gradients.append(dict(slot=i,terms=gr,background_train=old_bg,
                    uncovered_target_pixel_RGB_error_mean=uncovered_err,photometric_with_white_diagnostic=float(white_loss),
                    photometric_with_training_background=float(logs['photo'].detach()/t.cfg.w.photo)))
                examples.append(row['frame'])
                del loss,logs,verts,faces,lmks,albedos,result,pred,gt,alpha,err,white_result,rr,g
            save(OUT/'progress.json',dict(updated=now(),state='read_only_audit',pid=os.getpid(),completed=i+1,total=59))
        assert hashes_before=={k:sha_bytes(getattr(t,k)) for k in PARAMS}
        relative=(Rotation.from_rotvec(z['global_orient'])*Rotation.from_rotvec(z['global_orient'][32]).inv()).magnitude()*180/np.pi
        result=dict(updated=now(),state='complete',elapsed_seconds=time.monotonic()-start,optimizer_updates=0,
            input_checkpoint_unchanged=sha(checkpoint)==read(OUT/'plan.json')['checkpoint_sha256'],
            parameters_unchanged=True,records=records,gradient_samples=gradients,examples=examples,
            offset_regions_mm={k:stats(offset[m]) for k,m in regions.items()},
            root_rotation_from_reference_degrees=relative.tolist(),
            limits=['Detector residuals are not independent physical truth','Mask comparison includes clothes, cap, footwear, held object and occlusion uncertainty',
                    'White comparison uses the recorded training background; no fitting occurs in this audit',
                    'Coarse skinning regions do not identify exact clothes'],quality_accepted=False)
        save(OUT/'result.json',result);save(OUT/'progress.json',dict(updated=now(),state='complete',pid=None))
        print(dict(state='complete',seconds=result['elapsed_seconds'],offset_regions_mm=result['offset_regions_mm'],
            max_projection=max(r['native_projection_max_abs_px'] for r in records),
            max_face_joint_parity=max(r['static_face_vs_mapped_joints_max_abs_px'] for r in records),
            median_mask_iou=float(np.median([r['mask_IoU'] for r in records])),root_rotation_span_max=float(relative.max())))
    except BaseException as e:
        save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()));raise

def sha_bytes(x):
    import hashlib
    return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()

if __name__=='__main__':main()
