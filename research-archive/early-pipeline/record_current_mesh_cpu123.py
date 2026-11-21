"""Saved-offset inspection on CPU while the single GPU trainer waits for review."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import sys
import numpy as np
from PIL import Image
import run_official_gaussians123 as io
from review_full_sequence123 import mesh_renderer,relocate
from review_shared_vertex_offset123 import join


def main(root,step):
    root=Path(root);snap=root/f'gaussian-run/step-{step:06d}'
    record_tag='20万点预算' if 'original-200k-18000' in str(root) else '原版逐三角'
    comparison=root/f'comparison-step-{step:06d}'
    out=comparison/'current-mesh';out.mkdir(exist_ok=False)
    sys.path[:0]=[str(io.PROJECT/'src'),str(io.PROJECT/'src/avatar_core')]
    import torch,smplx
    from torch.func import functional_call
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    torch.set_num_threads(4)
    with np.load(root/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    with np.load(snap/'shared-static-offset.npz',allow_pickle=False) as z:offset=z['static_offset']
    rows={r['frame']:r for r in io.read(root/'data/training-inputs.json')['records']}
    source_hashes={str(p):io.sha(p) for p in [root/'fit/fitted.npz',snap/'shared-static-offset.npz',snap/'world.ply',
        io.PROJECT/'src/avatar_core/scene/smplx_gaussian_model.py',Path(__file__)]}
    body=smplx.SMPLX(str(io.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,
        num_betas=10,num_expression_coeffs=10,use_pca=True,num_pca_comps=12,
        flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
        create_transl=False,dtype=torch.float32).cpu().eval()
    body.requires_grad_(False)
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cpu')
    params={'betas':tensor(fit['shared_betas'])}
    for key in ('global_orient','expression','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose'):
        params[key]=tensor(fit[key])
    params['body_pose']=tensor(fit['body_pose_axis_angle'])
    carrier=SimpleNamespace(offset_parameters=params,offset_camera_translation=tensor(fit['camera_translation']),
        offset_w2c=tensor(fit['w2c']),offset_body=body,static_offset=torch.zeros_like(body.v_template),_offset_forward=functional_call)
    assert np.array_equal(body.faces,fit['faces'])
    project=SMPLXGaussianModel.shared_offset_vertices
    renderer,renderer_source=mesh_renderer();source_hashes[str(renderer_source)]=io.sha(renderer_source)
    slots={str(x):i for i,x in enumerate(fit['frame_ids'])};receipts=[];parity=[];displacements=[]
    for cr in io.read(comparison/'comparison.json')['comparisons']:
        fid=cr['frame'];i=slots[fid];row=rows[fid]
        with torch.no_grad():
            carrier.static_offset.zero_()
            zero=project(carrier,i).numpy()
            err=float(abs(zero-fit['vertices_world'][i]).max());parity.append(err)
            assert err<2.e-6,err
            carrier.static_offset.copy_(tensor(offset));world=project(carrier,i).numpy().astype(float)
        E=fit['w2c'][i];K=fit['K_integer_centers'][i];camera=world@E[:3,:3].T+E[:3,3]
        raw=Image.open(relocate(row['image'])).convert('RGB');gs=Image.open(snap/(fid+'_render.png')).convert('RGB')
        assert raw.size==gs.size==(2704,1520)
        mesh=renderer(raw,np.array([0,0,*raw.size]),dict(vertices_camera=camera[None],faces=fit['faces'],K_integer_centers=K))
        q=camera@K.T;q0=fit['vertices_camera'][i]@K.T
        delta=np.linalg.norm(q[:,:2]/q[:,2:]-q0[:,:2]/q0[:,2:],axis=1)
        displacements.append(dict(frame=fid,median_px=float(np.median(delta)),p99_px=float(np.quantile(delta,.99)),max_px=float(delta.max())))
        for label,info in cr['regions'].items():
            image=join([im.crop(tuple(info['crop'])) for im in (raw,mesh,gs)])
            dest=Path('/mnt/d/00/记录/005')/('3DGS' if label=='body' else '分析')/f'20261030_{record_tag}{step}步_左原图_中当前SMPL含偏移_右高斯_{fid}_{label}.png'
            assert not dest.exists();image.save(dest);image.save(out/(fid+'_'+label+'.png'))
            receipts.append(dict(frame=fid,region=label,crop=info['crop'],path=str(dest),sha256=io.sha(dest)))
    for label,size in [('body',(440,450)),('face',(440,180)),('feet',(440,180))]:
        panel=Image.new('RGB',(880,size[1]*4),(242,242,242))
        for i,p in enumerate(sorted(out.glob('frame*_'+label+'.png'))):
            im=Image.open(p);im.thumbnail(size,Image.Resampling.LANCZOS);panel.paste(im,((i%2)*440,(i//2)*size[1]))
        panel.save(out/(label+'7.png'))
    for p,h in source_hashes.items():assert io.sha(p)==h,p
    io.save(out/'result.json',dict(updated=io.now(),state='generated_actual_review_pending',step=step,
        files=receipts,source_hashes=source_hashes,cpu_forward_reuses_existing_model_method=True,
        zero_offset_max_world_error_m=max(parity),projection_displacement_not_reconstruction_error=displacements,
        optimizer_created=False,optimizer_updates=0,GPU_used=False,image_text=False,
        source_crops_not_resampled=True,contact_thumbnail_only=True,quality_accepted=False))
    print('CPU_CURRENT_MESH_IMAGES_READY',step)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--step',type=int,required=True)
    args=p.parse_args();main(args.root,args.step)
