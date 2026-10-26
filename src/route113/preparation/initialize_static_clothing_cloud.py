"""Fresh shared-world silhouette hull, without body-driven image warping."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import binary_erosion
from PIL import Image,ImageDraw
import shuolee_common as c
torch=c.runtime()
from scene.static_world_gaussian_model import StaticWorldGaussianModel
from utils.graphics_utils import BasicPointCloud
from train_smplx import prepare_views,render,BG,pil
import train_smplx as training

def main(root):
    root=Path(root);out=root/'clothing-init';out.mkdir(exist_ok=True);assert not (out/'points.npz').exists()
    fresh=np.load(root/'fit/fitted.npz');rows=c.read(root/'data/training-inputs.json')['records'];ref=69
    assert c.read(root/'source-inputs.json')['old_subject_fit_loaded'] is False
    scale=float(fresh['world_scale']);R=fresh['root_rotation'][ref];t=fresh['world_translation'][ref]
    local=(fresh['vertices_world'][ref]-t)@R/scale;spacing=.012
    axes=[np.arange(local[:,j].min()-.18,local[:,j].max()+.18,spacing) for j in range(3)]
    shape=tuple(len(a) for a in axes);assert np.prod(shape)<2500000
    grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3);world=grid@R.T*scale+t
    g=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    points=g(world);votes=torch.zeros(len(points),device='cuda');total=0.;start=time.monotonic()
    # Body geometry was used only to define this broad proposal box; no body-distance filter.
    with torch.no_grad():
        for i,row in enumerate(rows):
            assert c.sha(row['mask'])==row['mask_sha256']
            mask=g(np.asarray(Image.open(row['mask']).convert('L').resize((288,512),Image.Resampling.BILINEAR))/255.)[None,None]
            mask=torch.nn.functional.max_pool2d(mask,3,stride=1,padding=1)
            E=g(row['w2c']);K=g(row['K']);wh=g(row['wh']);cam=points@E[:3,:3].T+E[:3,3]
            q=cam@K.T;uv=q[:,:2]/q[:,2:].clamp_min(.01)-.5;uv=2*uv/(wh-1)-1
            inside=torch.nn.functional.grid_sample(mask,uv[None,:,None],align_corners=True,padding_mode='zeros')[0,0,:,0]>.5
            weight=row.get('observation_weight',1. if row['camera_kind']=='calibrated' else .5)
            votes+=weight*(inside&(cam[:,2]>0));total+=weight
            if i%25==0:print(json.dumps(dict(stage='static_silhouette_volume',done=i+1,total=len(rows),seconds=time.monotonic()-start)),flush=True)
    occupancy=(votes/total>.97).cpu().numpy().reshape(shape);boundary=occupancy&~binary_erosion(occupancy)
    surface=grid[np.flatnonzero(boundary)];assert len(surface)>1000,len(surface)
    rng=np.random.default_rng(6060);seed=surface[rng.integers(len(surface),size=200000)]
    seed=(seed+rng.uniform(-.35*spacing,.35*spacing,size=seed.shape))@R.T*scale+t
    observed=np.load(root/'clothing-support/surface.npz')['xyz'];seed[:len(observed)]=observed
    np.savez_compressed(out/'points.npz',xyz=seed.astype(np.float32),reference_frame=ref,observed_feature_count=len(observed))
    np.savez_compressed(out/'motion.npz',representation='shared_static_world',world_scale=scale,num_timesteps=len(rows),reference_frame=ref)
    model=StaticWorldGaussianModel(3);model.load_motion(out/'motion.npz')
    model.create_from_pcd(BasicPointCloud(seed,np.full_like(seed,.5),np.zeros_like(seed)),scale)
    model._features_dc.data.zero_();model._features_rest.data.zero_();model.export_world_ply(out/'initialization.ply',ref)
    ids=[0,33,69,99,135,174];training.RUN=out;views=prepare_views([rows[i] for i in ids],900)
    panels=[];pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=True,debug=False)
    with torch.no_grad():
        for (cam,_,_),i in zip(views,ids):
            im=pil(render(cam,model,pipe,BG)['render']);im.thumbnail((320,520));tile=Image.new('RGB',(330,550),'white')
            tile.paste(im,((330-im.width)//2,30));ImageDraw.Draw(tile).text((6,6),f'raw {i+1} | shared-static init',fill='black');panels.append(tile)
    sheet=Image.new('RGB',(990,1100),'white')
    for i,tile in enumerate(panels):sheet.paste(tile,((i%3)*330,(i//3)*550))
    sheet.save(out/'initialization-review.jpg',quality=94)
    result=dict(created=c.stamp(),from_scratch=True,old_gaussian_loaded=False,body_motion_loaded=False,old_subject_fit_loaded=False,
        representation='shared_static_world',body_fit_usage='fresh bounding proposal box and scale only',points=200000,frames=len(rows),
        mask_vote_threshold=.97,volume_spacing_model_m=spacing,candidates=len(grid),surface_voxels=len(surface),observed_features=len(observed),
        camera_counts={k:sum(r['camera_kind']==k for r in rows) for k in sorted({r['camera_kind'] for r in rows})},
        mask_vote_weight_policy='Original SfM frames1.0; additional frames0.5, held unchanged for the camera-input comparison.',
        source_fit_sha256=c.sha(root/'fit/fitted.npz'),point_sha256=c.sha(out/'points.npz'),source_sha256=c.sha(__file__),
        per_triangle_quota=False,body_attraction=False,training_steps=0,quality_accepted=False,
        limitation='Static hull can erode actually moving regions; not a general dynamic reconstruction.')
    c.save(out/'audit.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
