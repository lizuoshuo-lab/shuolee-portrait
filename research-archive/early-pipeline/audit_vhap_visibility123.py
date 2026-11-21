"""Read-only mesh self-occlusion audit, separate from real visibility truth."""
import os,fcntl,time,traceback
import numpy as np
from PIL import Image,ImageDraw
from vhap_demo_adapter123 import ROOT,make_tracker
from continue_vhap_demo123 import PARAMS
from run_official_gaussians123 import read,save,sha,now

def main():
    import torch,cv2
    from review_full_sequence123 import mesh_renderer,crop_box,joined
    root=ROOT/'vhap-selected-reference';out=root/'visibility-audit'
    assert not out.exists()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir();start=time.monotonic();torch.set_num_threads(4)
    checkpoint=root/'global-epoch-30/tracked-smplx-params.npz'
    save(out/'plan.json',dict(updated=now(),optimizer_updates=0,checkpoint_sha256=sha(checkpoint),
        purpose='Are low-residual model face landmarks on the visible surface?',
        occlusion_gap_m=.005,near_surface_tolerance_m=.002,
        limits='Nearest raster pixel triangle at a continuous landmark ray; boundary/eyelid exceptions remain uncertain. Model self-occlusion is not real landmark visibility.'))
    try:
        t,rows=make_tracker(out)
        with np.load(checkpoint,allow_pickle=False) as z,torch.no_grad():
            for key in PARAMS:getattr(t,key).copy_(torch.tensor(z[key],device='cuda'))
        render,_=mesh_renderer();records=[]
        for i,row in enumerate(rows):
            with torch.no_grad():
                sample=t.dataset.one(i);v,vc,j,face,albedo=t.forward_smplx(sample['timestep_index'])
                raster=t.rasterize_smplx(sample,v,t.smplx.faces_tensor,train_mode=False)
                fid=raster['rast_out'][0,...,-1].flip(0).cpu().numpy().astype(int)-1
                v=v[0].cpu().numpy();f=face[0].cpu().numpy();K=np.array(row['K_integer_centers'])
                q=f@K.T;uv=q[:,:2]/q[:,2:]
            p=np.rint(uv).astype(int);inside=(p[:,0]>=0)&(p[:,0]<fid.shape[1])&(p[:,1]>=0)&(p[:,1]<fid.shape[0])
            tri_ids=np.full(51,-1);tri_ids[inside]=fid[p[inside,1],p[inside,0]]
            valid=tri_ids>=0;tri=v[t.smplx.faces[tri_ids.clip(0)]]
            n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            ray=np.c_[uv,np.ones(51)]@np.linalg.inv(K).T
            denom=(n*ray).sum(1);valid&=abs(denom)>1e-12
            surface_z=np.full(51,np.nan);surface_z[valid]=(n[valid]*tri[valid,0]).sum(1)/denom[valid]
            gap=f[:,2]-surface_z;hidden=valid&(gap>.005);near=valid&(abs(gap)<.002)
            rec=dict(slot=i,frame=row['frame'],valid_ray_points=int(valid.sum()),self_occluded_gt5mm=int(hidden.sum()),
                near_visible_surface_2mm=int(near.sum()),depth_gap_m=[float(x) if np.isfinite(x) else None for x in gap])
            records.append(rec)
            if i in (0,32,58):
                raw=Image.open(row['image']).convert('RGB')
                overlay=render(raw,np.array([0,0,*raw.size]),dict(vertices_camera=v[None],faces=t.smplx.faces,K_integer_centers=K))
                draw=ImageDraw.Draw(overlay)
                for k,pix in enumerate(uv):
                    col=(255,40,30) if hidden[k] else ((0,255,90) if near[k] else (255,220,0))
                    draw.ellipse((pix[0]-1,pix[1]-1,pix[0]+1,pix[1]+1),fill=col)
                crop=crop_box(row['face_box_xyxy'],raw.size,30)
                joined([raw.crop(crop),overlay.crop(crop)]).save(out/f'{row["frame"]}_face_depth_visibility.png')
            if time.monotonic()-start>120:raise RuntimeError('Read-only audit budget exceeded.')
        save(out/'result.json',dict(updated=now(),state='complete',seconds=time.monotonic()-start,records=records,
            optimizer_updates=0,checkpoint_unchanged=sha(checkpoint)==read(out/'plan.json')['checkpoint_sha256'],quality_accepted=False))
        print({r['slot']:{k:r[k] for k in ('self_occluded_gt5mm','near_visible_surface_2mm')} for r in records if r['slot'] in (0,12,23,32,46,58)})
    except BaseException as e:
        save(out/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
