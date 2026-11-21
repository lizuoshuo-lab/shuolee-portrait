"""Frozen before/after mask and geometry review; no fitting or edits to model."""
import argparse,fcntl,time
import numpy as np
import torch
from PIL import Image
import fit_demo_flame_hatmask123 as h
from run_official_gaussians123 import save,sha,now
from review_full_sequence123 import mesh_renderer

OUT=h.OUT/'review'

def main():
    assert not (OUT/'geometry-mask-audit.json').exists()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=True);folder=OUT/'frames';folder.mkdir(exist_ok=True);start=time.monotonic()
    t,rows,hair=h.setup();mesh,_=mesh_renderer();saved=[];allv=[];alpha=[];normals=[]
    for label in ['before','epoch-30']:
        source=h.OUT.parent if h.OUT.name=='convergence-control' and label=='before' else h.OUT
        checkpoint='tracked-flame-geometry.npz' if h.OUT.name=='geometry-only-control' else 'tracked-flame-with-surface.npz'
        path=source/label/checkpoint;saved.append((path,sha(path)))
        with np.load(path,allow_pickle=False) as z,torch.no_grad():
            required=['shape',*h.g.f.PER_FRAME,'scene_scale','static_offset']
            for n in required:assert n in z,n
            for n in required+['tex_extra','lights']:
                if n in z:getattr(t,n).copy_(torch.as_tensor(z[n],device='cuda'))
        vs=[];aa=[]
        for i,row in enumerate(rows):
            with torch.no_grad():
                t.clear_cache();s=t.dataset.one(i);_,_,v,faces,l,tex,r=t.compute_energy(s,stage=None)
            vs.append(v[0].detach().cpu().numpy());aa.append(r['pred_alpha'][0,0].detach().cpu().numpy()>.5)
        allv.append(np.array(vs));alpha.append(aa)
    faces=faces.cpu().numpy();records=[]
    for i,row in enumerate(rows):
        xy=t.dataset.crops[i];side=t.dataset.side;b=np.array([*xy,*(xy+side)])
        raw=Image.open(row['image']).convert('RGB').crop(tuple(b));K=np.array(row['K_integer_centers'])
        target=t.dataset.samples[i]['alpha_map'][0].numpy()>.5;roi=t.dataset.samples[i]['silhouette_roi'][0].numpy()>0
        p0,p1=alpha[0][i],alpha[1][i]
        import cv2
        d=cv2.distanceTransform((~target).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        def stats(p):return dict(iou=float((p&target&roi).sum()/max(((p|target)&roi).sum(),1)),missing=int((~p&target&roi).sum()),outside=int((p&~target&roi).sum()),outside_max_distance_px=float(d[p&roi].max()) if (p&roi).any() else 0.)
        def project(v):q=v@K.T;return q[:,:2]/q[:,2:]
        shift=np.linalg.norm(project(allv[1][i])-project(allv[0][i]),axis=-1)
        records.append(dict(frame=row['frame'],before=stats(p0),after=stats(p1),vertex_projected_shift_max_px=float(shift.max())))
        panels=[raw]+[mesh(raw,b,dict(vertices_camera=v[i:i+1],faces=faces,K_integer_centers=K)) for v in allv]
        im=Image.new('RGB',(side*3,side),'white')
        for j,p in enumerate(panels):im.paste(p,(j*side,0))
        im.save(folder/(row['frame']+'_原图_mask指导前mesh_mask指导后mesh.png'))
        panels=[raw]+[Image.fromarray(p.astype('uint8')*255).convert('RGB') for p in [target,p0,p1]]
        im=Image.new('RGB',(side*4,side),'white')
        for j,p in enumerate(panels):im.paste(p,(j*side,0))
        im.save(folder/(row['frame']+'_原图_目标mask_训练前meshmask_训练后meshmask.png'))
    tri=[v[32][faces] for v in allv]
    cross=[np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]) for x in tri];area=[np.linalg.norm(x,axis=1) for x in cross]
    cosine=(cross[0]*cross[1]).sum(1)/np.maximum(area[0]*area[1],1e-20)
    for path,digest in saved:assert sha(path)==digest
    save(OUT/'geometry-mask-audit.json',dict(updated=now(),elapsed_seconds=time.monotonic()-start,optimizer_updates=0,
        checkpoints_unchanged=True,frames=59,rows=records,
        median_iou_before=float(np.median([r['before']['iou'] for r in records])),median_iou_after=float(np.median([r['after']['iou'] for r in records])),
        missing_total_before=sum(r['before']['missing'] for r in records),missing_total_after=sum(r['after']['missing'] for r in records),
        outside_total_before=sum(r['before']['outside'] for r in records),outside_total_after=sum(r['after']['outside'] for r in records),
        reference_triangle_normals_over90deg=int((cosine<0).sum()),min_area_ratio=float((area[1]/np.maximum(area[0],1e-20)).min()),
        geometry_note='Normals-over90 is a deformation indicator, not a complete self-intersection test.',quality_accepted=False))
    print('HAT_MASK_GEOMETRY_REVIEW_READY',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group();group.add_argument('--convergence',action='store_true');group.add_argument('--geometry-only',action='store_true');args=p.parse_args()
    if args.convergence:
        h.SOURCE=h.OUT/'epoch-30/tracked-flame-with-surface.npz';h.OUT=h.OUT/'convergence-control';OUT=h.OUT/'review'
    elif args.geometry_only:
        h.OUT=h.OUT/'geometry-only-control';OUT=h.OUT/'review'
    main()
