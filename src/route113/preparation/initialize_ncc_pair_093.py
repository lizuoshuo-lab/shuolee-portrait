"""Fresh neutral head and observation-supported initial point positions for093."""
import argparse,json,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import shuolee_common as c

def head(root):
    import fit_head_surface_082 as h
    h.O=root/'head-fit';h.initialize()
    a=h.load(h.O/'initialization.npz')
    assert np.isfinite(a['vertices']).all()
    c.save(h.O/'initialization-review.json',dict(allow_bounded_fit=True,fresh_calculation=True,historical_array_comparison_performed=False,gaussian_training_allowed=False))
    h.fit_arm('rigid_neutral');assert np.max(abs(h.load(h.O/'rigid_neutral.npz')['shape']))==0

def patch(root):
    torch=c.runtime()
    import nvdiffrast.torch as dr
    from fit_head_surface_082 import load,project
    from project_human_surface_083 import closest
    out=root/'clothing-init';assert not (out/'hull-before-head.npz').exists()
    initial=load(out/'points.npz');old=initial['xyz'].copy()
    reference=c.BASE/'current/person-static-human-surface-init-fresh-083'
    # Independent recalculation assertion only. Never assign from reference.
    assert len(old)==200000 and np.isfinite(old).all()
    f=load(root/'head-fit/rigid_neutral.npz');o=load(root/'head-fit/observations.npz')
    local=lambda x:(x-f['translation'])@f['rotation']/float(f['scale'])
    lm=local(f['landmarks']);lv=local(f['vertices'])
    lower=np.array([lm[17:27,0].min()-.005,lm[48:68,1].min()-.010,0])
    upper=np.array([lm[17:27,0].max()+.005,lm[17:27,1].max()+.005,.150])
    inside=lambda x:((x>=lower)&(x<=upper)).all(-1)
    replace=inside(local(old));ids=np.flatnonzero(replace)
    assert 500<=len(ids)<=20000 and not replace[:int(initial['observed_feature_count'])].any()
    faces=f['faces'];eligible=inside(lv)[faces].all(-1)
    ctx=dr.RasterizeCudaContext();g=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    tri=torch.as_tensor(faces,dtype=torch.int32,device='cuda');v=g(f['vertices']);counts=np.zeros(len(faces),int)
    for frame in np.flatnonzero(~o['held_frames']):
        e=g(o['w2c'][frame]);k=g(o['K'][frame]);w,h=o['wh'][frame]
        cam=v@e[:3,:3].T+e[:3,3];z=cam[:,2]
        clip=torch.stack((2*k[0,0]/w*cam[:,0]+(2*k[0,2]/w-1)*z,-2*k[1,1]/h*cam[:,1]+(1-2*k[1,2]/h)*z,1.00002*z-.0200002,z),-1)
        rast,_=dr.rasterize(ctx,clip[None],tri,(int(h),int(w)))
        seen=torch.unique(rast[0,:,:,3].long());seen=seen[seen>0]-1;counts[seen.cpu().numpy()]+=1
    eligible&=counts>=3;eligible_ids=np.flatnonzero(eligible);assert len(eligible_ids)>500
    new,nearest,distance=closest(old[ids].astype(np.float64),f['vertices'][faces[eligible_ids]])
    uv,z=project(new,o['K'],o['w2c']);rows=c.read(root/'data/training-inputs.json')['records'];support=[]
    for frame,row in enumerate(rows):
        assert c.sha(row['mask'])==row['mask_sha256']
        mask=np.asarray(Image.open(c.loc(row['mask'])).convert('L'));px=np.rint(uv[frame]).astype(int)
        valid=(px[:,0]>=0)&(px[:,0]<mask.shape[1])&(px[:,1]>=0)&(px[:,1]<mask.shape[0])&(z[frame]>0)
        yes=np.zeros(len(new),bool);yes[valid]=mask[px[valid,1],px[valid,0]]>127;support.append(yes)
    supported=np.array(support).all(0);assert supported.sum()>=500
    xyz=old.copy();changed=ids[supported];xyz[changed]=new[supported].astype(np.float32)
    unchanged=np.ones(len(old),bool);unchanged[changed]=False
    assert np.isfinite(xyz).all() and len(xyz)==200000 and np.array_equal(xyz[unchanged],old[unchanged])
    min_depth=min(float((xyz@e[:3,:3].T+e[:3,3])[:,2].min()) for e in o['w2c']);assert min_depth>0
    # This is an output validation after the new arrays have been calculated.
    equal083=None  # Historical learned arrays are not deployment inputs.
    for name in ['points.npz','initialization.ply','initialization-review.jpg','audit.json']:
        dest='hull-before-head.npz' if name=='points.npz' else 'hull-'+name
        assert not (out/dest).exists();(out/name).rename(out/dest)
    initial['xyz']=xyz;np.savez_compressed(out/'points.npz',**initial)
    np.savez_compressed(out/'head-sample-map.npz',candidate_ids=ids,changed_ids=changed,unchanged_ids=ids[~supported],projected_surface=new,supported=supported,triangle=eligible_ids[nearest],distance_world=distance,mask_support=np.array(support),eligible_triangles=eligible_ids)
    from scene.static_world_gaussian_model import StaticWorldGaussianModel
    from utils.graphics_utils import BasicPointCloud
    from train_smplx import prepare_views,render,BG,pil
    import train_smplx as training
    m=StaticWorldGaussianModel(3);m.load_motion(out/'motion.npz')
    m.create_from_pcd(BasicPointCloud(xyz,np.full_like(xyz,.5),np.zeros_like(xyz)),float(np.load(out/'motion.npz')['world_scale']))
    m._features_dc.data.zero_();m._features_rest.data.zero_();m.export_world_ply(out/'initialization.ply',69)
    indices=[0,33,69,99,135,174];training.RUN=out;views=prepare_views([rows[i] for i in indices],900)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=True,debug=False);sheet=Image.new('RGB',(990,1100),'white')
    for j,((cam,_,_),i) in enumerate(zip(views,indices)):
        with torch.no_grad():im=pil(render(cam,m,pipe,BG)['render'])
        im.thumbnail((320,520));tile=Image.new('RGB',(330,550),'white');tile.paste(im,((330-im.width)//2,30))
        ImageDraw.Draw(tile).text((6,6),f'raw {i+1} | fresh093 {root.name}',fill='black');sheet.paste(tile,((j%3)*330,(j//3)*550))
    sheet.save(out/'initialization-review.jpg',quality=95)
    old_uv,_=project(old[ids],o['K'],o['w2c']);font=ImageFont.load_default()
    for i in [33,69,83,135]:
        photo=Image.open(c.loc(rows[i]['image'])).convert('RGB');q=o['uv'][i];center=(np.quantile(q,.05,axis=0)+np.quantile(q,.95,axis=0))/2
        origin=np.rint(center-[220,240]).astype(int);photo=photo.crop((*origin,*(origin+[440,480])))
        board=Image.new('RGB',(1320,550),'white');board.paste(photo,(0,70))
        for j,points in enumerate([old_uv[i],uv[i]],1):
            im=photo.copy();d=ImageDraw.Draw(im)
            for k,p in enumerate(points-origin):
                if j==2 and not supported[k]:p=old_uv[i,k]-origin
                color=(0,210,210) if supported[k] else (245,160,30);d.ellipse((p[0]-1,p[1]-1,p[0]+1,p[1]+1),fill=color)
            board.paste(im,(440*j,70))
        d=ImageDraw.Draw(board);d.text((5,8),f'raw {i+1:06d} | original',font=font,fill='black')
        d.text((445,8),'Fresh hull',font=font,fill='black');d.text((885,8),'Supported neutral surface',font=font,fill='black')
        board.save(out/f'patch-review-{i+1:06d}.jpg',quality=96)
    audit=dict(created=c.stamp(),round='093',arm=root.name,from_scratch=True,initial_points=200000,face_candidates=len(ids),replaced_face_points=len(changed),unsupported_points_preserved=int((~supported).sum()),unchanged_total=int(unchanged.sum()),all_other_points_bitwise_identical=True,fresh_array_equal083=equal083,comparison_reference_not_initializer=True,min_all_point_depth=min_depth,mask_support=1.,old_subject_fit_loaded=False,old_gaussian_loaded=False,template_initializer_only=True,training_steps=0,gaussian_training_allowed=False,quality_accepted=False,point_sha256=c.sha(out/'points.npz'),head_fit_sha256=c.sha(root/'head-fit/rigid_neutral.npz'),source_fit_sha256=c.sha(root/'fit/fitted.npz'))
    c.save(out/'audit.json',audit);print(json.dumps(audit),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['head','patch']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    (head if a.stage=='head' else patch)(a.run)
