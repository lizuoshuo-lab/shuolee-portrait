"""Observation-supported nearest surface projection, before Gaussian updates.

The rejected all-patch resampling candidate is retained as a small diagnostic.
Unreliable peripheral projections keep their original freshly built hull points.
"""
import json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import shuolee_common as c
from prepare_human_surface_083 import R,REF
from fit_head_surface_082 import load,project

def closest(points,triangles):
    a=triangles[:,0];ab=triangles[:,1]-a;ac=triangles[:,2]-a
    n=np.cross(ab,ac);n2=(n*n).sum(-1);assert n2.min()>1e-20
    results=[];faceids=[];distances=[]
    for block in np.array_split(points,max(1,int(np.ceil(len(points)/64)))):
        p=block[:,None];ap=p-a;plane=p-((ap*n).sum(-1)/n2)[...,None]*n
        v=plane-a;d00=(ab*ab).sum(-1);d01=(ab*ac).sum(-1);d11=(ac*ac).sum(-1)
        d20=(v*ab).sum(-1);d21=(v*ac).sum(-1);den=d00*d11-d01*d01
        x=(d11*d20-d01*d21)/den;y=(d00*d21-d01*d20)/den
        valid=(x>=0)&(y>=0)&(x+y<=1)
        best=plane.copy();ds=np.where(valid,((p-plane)**2).sum(-1),np.inf)
        for first,last in [(0,1),(1,2),(2,0)]:
            s=triangles[:,first];e=triangles[:,last]-s
            w=np.clip(((p-s)*e).sum(-1)/(e*e).sum(-1),0,1)
            q=s+w[...,None]*e;qd=((p-q)**2).sum(-1);take=qd<ds
            best[take]=q[take];ds[take]=qd[take]
        j=np.argmin(ds,axis=1);k=np.arange(len(block));results.append(best[k,j]);faceids.append(j);distances.append(np.sqrt(ds[k,j]))
    return np.concatenate(results),np.concatenate(faceids),np.concatenate(distances)

def main():
    out=R/'clothing-init';assert not (R/'gaussian-run/started.json').exists()
    failure=c.read(out/'rejected-face-preflight.json');assert not failure['passed']
    proposal=load(out/'rejected-face-candidate.npz');f=load(R/'head-fit/rigid_neutral.npz');o=load(R/'head-fit/observations.npz')
    initial=load(out/'points.npz');old=initial['xyz'];assert np.array_equal(old,load(REF/'clothing-init/points.npz')['xyz'])
    # Exact geometry checks independent of the real input.
    test=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    q,_,_=closest(np.array([[.2,.2,1.],[-1.,-1.,0.],[.6,.6,0.]]),test)
    assert np.max(abs(q-np.array([[.2,.2,0.],[0.,0.,0.],[.5,.5,0.]])))<1e-12
    protocol=c.read(R/'analysis-before-training.json')
    shutil.copy2(R/'analysis-before-training.json',R/'initialization-protocol-v1.json')
    protocol.update(initialization_revision=2,
        previous_candidate_rejected=str(out/'rejected-face-preflight.json'),
        amendment_reason='The neutral surface projects beyond the real silhouette in some views. Preserve unsupported peripheral hull points instead of deleting them or forcing the template into the mask.',
        point_sampling='For the same fixed central-face region, compute the exact closest point on the visible neutral template patch. Replace a hull point only if its surface projection is inside every one of175 original person masks. All other hull points stay bitwise identical. Total200000points remains. No parameter sweep, no resampling and no discarded frame.',
        support_is_precondition_not_accuracy_metric=True,
        old_rejected_all_patch_resampling_not_used=True,
        preflight=['Fresh hull is bitwise identical to077 before projection.',
          'At least500points pass fixed all175mask support; report every candidate and every retained old point.',
          'Outside changed set all coordinates are exactly preserved; allpoints finite with positive depth in175cameras.',
          'Inspect native initial renders and patch overlays before any Gaussian update. Template-relative anatomy, hair/neck/clothing outside region unchanged.'],
        gaussian_updates_at_amendment=0)
    c.save(R/'analysis-before-training.json',protocol)
    ids=proposal['indices_to_replace'];faces=f['faces'][proposal['eligible_triangles']]
    new,tri,distance=closest(old[ids].astype(np.float64),f['vertices'][faces])
    uv,z=project(new,o['K'],o['w2c']);rows=c.read(R/'data/training-inputs.json')['records'];support=[]
    for frame,row in enumerate(rows):
        mask=np.asarray(Image.open(c.loc(row['mask'])).convert('L'));px=np.rint(uv[frame]).astype(int)
        valid=(px[:,0]>=0)&(px[:,0]<mask.shape[1])&(px[:,1]>=0)&(px[:,1]<mask.shape[0])&(z[frame]>0)
        yes=np.zeros(len(new),bool);yes[valid]=mask[px[valid,1],px[valid,0]]>127;support.append(yes)
    supported=np.array(support).all(0);assert int(supported.sum())>=500,int(supported.sum())
    xyz=old.copy();changed=ids[supported];xyz[changed]=new[supported].astype(np.float32)
    unchanged=np.ones(len(old),bool);unchanged[changed]=False
    assert np.array_equal(xyz[unchanged],old[unchanged]) and np.isfinite(xyz).all() and len(xyz)==200000
    min_depth=min(float((xyz@e[:3,:3].T+e[:3,3])[:,2].min()) for e in o['w2c']);assert min_depth>0
    for name in ['points.npz','initialization.ply','initialization-review.jpg','audit.json']:
        dest='hull-before-head.npz' if name=='points.npz' else 'hull-'+name
        assert not (out/dest).exists();(out/name).rename(out/dest)
    initial['xyz']=xyz;np.savez_compressed(out/'points.npz',**initial)
    np.savez_compressed(out/'head-sample-map.npz',candidate_ids=ids,changed_ids=changed,unchanged_ids=ids[~supported],
        projected_surface=new,supported=supported,triangle=proposal['eligible_triangles'][tri],distance_world=distance,mask_support=np.array(support))
    torch=c.runtime()
    from scene.static_world_gaussian_model import StaticWorldGaussianModel
    from utils.graphics_utils import BasicPointCloud
    from train_smplx import prepare_views,render,BG,pil
    import train_smplx as training
    from types import SimpleNamespace
    m=StaticWorldGaussianModel(3);m.load_motion(out/'motion.npz')
    m.create_from_pcd(BasicPointCloud(xyz,np.full_like(xyz,.5),np.zeros_like(xyz)),float(np.load(out/'motion.npz')['world_scale']))
    m._features_dc.data.zero_();m._features_rest.data.zero_();m.export_world_ply(out/'initialization.ply',69)
    indices=[0,33,69,99,135,174];training.RUN=out;views=prepare_views([rows[i] for i in indices],900)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=True,debug=False);panels=[]
    for (cam,_,_),i in zip(views,indices):
        with torch.no_grad():im=pil(render(cam,m,pipe,BG)['render'])
        im.thumbnail((320,520));tile=Image.new('RGB',(330,550),'white');tile.paste(im,((330-im.width)//2,30))
        ImageDraw.Draw(tile).text((6,6),f'raw {i+1} | supported human-surface init',fill='black');panels.append(tile)
    sheet=Image.new('RGB',(990,1100),'white')
    for i,im in enumerate(panels):sheet.paste(im,((i%3)*330,(i//3)*550))
    sheet.save(out/'initialization-review.jpg',quality=95)
    old_uv,_=project(old[ids],o['K'],o['w2c']);font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',16)
    for i in [33,69,83,135]:
        photo=Image.open(c.loc(rows[i]['image'])).convert('RGB');q=o['uv'][i];center=(np.quantile(q,.05,axis=0)+np.quantile(q,.95,axis=0))/2
        origin=np.rint(center-[220,240]).astype(int);photo=photo.crop((*origin,*(origin+[440,480])))
        board=Image.new('RGB',(1320,550),'white');board.paste(photo,(0,70))
        for j,points in enumerate([old_uv[i],uv[i]],1):
            im=photo.copy();d=ImageDraw.Draw(im)
            for k,p in enumerate(points-origin):
                if j==2 and not supported[k]:p=old_uv[i,k]-origin
                color=(0,210,210) if supported[k] else (245,160,30)
                d.ellipse((p[0]-1,p[1]-1,p[0]+1,p[1]+1),fill=color)
            board.paste(im,(440*j,70))
        d=ImageDraw.Draw(board);d.text((5,8),f'raw {i+1:06d} | original',font=font,fill='black')
        d.text((445,8),'Fresh hull positions',font=font,fill='black');d.text((885,8),'Observation-supported template projection',font=font,fill='black')
        d.text((445,36),'Cyan=move candidate;orange=kept at original hull',font=font,fill='black')
        d.text((885,36),'No color training; same point count',font=font,fill='black')
        board.save(out/f'patch-review-{i+1:06d}.jpg',quality=96)
    result=dict(created=c.stamp(),round='083',initialization_revision=2,from_scratch=True,initial_points=200000,
        face_candidates=len(ids),replaced_face_points=len(changed),unsupported_points_preserved=int((~supported).sum()),
        unchanged_total=int(unchanged.sum()),all_other_points_bitwise_identical=True,fresh_hull_equal077=True,
        selected_mask_support=1.,mask_support_is_not_quality_validation=True,all_camera_count=175,min_all_point_depth=min_depth,
        displacement_median_world=float(np.median(distance[supported])),displacement_p90_world=float(np.percentile(distance[supported],90)),
        neutral_shape=True,personalized_head_loaded=False,old_subject_fit_loaded=False,old_gaussian_loaded=False,
        template_is_initializer_only=True,no_neck_or_clothing_change=True,training_steps=0,gaussian_training_allowed=False,quality_accepted=False,
        point_sha256=c.sha(out/'points.npz'),source_fit_sha256=c.sha(R/'fit/fitted.npz'),head_fit_sha256=c.sha(R/'head-fit/rigid_neutral.npz'),source_sha256=c.sha(__file__))
    c.save(out/'audit.json',result);shutil.copy2(Path(__file__),R/'source-final'/Path(__file__).name)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
