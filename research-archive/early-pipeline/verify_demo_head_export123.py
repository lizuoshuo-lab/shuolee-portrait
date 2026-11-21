"""Read-only native replay of paired +Y head exports and a bounded orbit."""
import argparse,fcntl,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
sys.path.insert(0,'/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')
from shuolee_portrait import runtime as rt
from shuolee_portrait.training import camera,image
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
import torch

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/head-gaussian-first-cycle')

def serialized_camera(out,row,rotation):
    records=rt.read(out/'cameras.json')
    if not isinstance(records,list) or len(records)!=1:raise ValueError('Expected one reference-pose camera')
    c=records[0]
    if c['img_name']!=row['frame'] or c.get('pixelConvention')!='pixel_corners':raise ValueError('Camera identity/pixel convention mismatch')
    if [c['width'],c['height']]!=row['wh']:raise ValueError('Camera image dimensions mismatch')
    K=np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1.]])
    basis=np.asarray(c['rotation'],dtype=float);position=np.asarray(c['position'],dtype=float)
    if basis.shape!=(3,3) or position.shape!=(3,) or not np.isfinite(basis).all() or not np.isfinite(position).all():raise ValueError('Invalid serialized camera pose')
    if not np.allclose(basis.T@basis,np.eye(3),atol=1e-7) or abs(np.linalg.det(basis)-1)>1e-7:raise ValueError('Nonrigid serialized camera rotation')
    E=np.eye(4);E[:3,:3]=basis.T;E[:3,3]=-basis.T@position
    expected=np.array(row['w2c']);expected[:3,:3]=expected[:3,:3]@rotation.T
    if not np.allclose(K,row['K'],atol=1e-9,rtol=0) or not np.allclose(E,expected,atol=1e-9,rtol=0):raise ValueError('Serialized camera differs from rigid export contract')
    return dict(row,K=K.tolist(),w2c=E.tolist(),wh=[c['width'],c['height']])


def main(run='head-gaussian-first-cycle',steps=None):
    root=ROOT.parent/run
    if steps is None:
        if run!='head-gaussian-first-cycle':raise ValueError('Explicit --step required for the independent long run')
        steps=[1000,18000]
    if not steps or len(set(steps))!=len(steps) or any(s<1 for s in steps):raise ValueError('Invalid steps')
    lock=open('/tmp/shuolee-native122-gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=rt.read(root/'inputs.json')['records'];row=rows[32];R=np.diag([1.,-1.,-1.])
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda')
    for step in steps:
        snap=root/'gaussian-run'/f'step-{step:06d}';out=root/'delivery'/f'step-{step:06d}'
        result=rt.read(snap/'result.json');valid=rt.read(out/'validation.json')
        assert valid['sha256']==rt.sha(out/'point_cloud.ply')
        assert valid['source_sha256']==result['world_ply_sha256']==rt.sha(snap/'world.ply')
        source=GaussianModel(3);source.load_ply(str(snap/'world.ply'));source.active_sh_degree=result['active_sh_degree']
        mapped=GaussianModel(3);mapped.load_ply(str(out/'point_cloud.ply'));mapped.active_sh_degree=result['active_sh_degree']
        moved=serialized_camera(out,row,R)
        a,_=camera(row,0);b,proj=camera(moved,0)
        with torch.no_grad():
            rgbA=render(a,source,pipe,bg)['render'];rgbB=render(b,mapped,pipe,bg)['render'];diff=abs(rgbA-rgbB)
            parity=dict(mean_abs=float(diff.mean()),max_abs=float(diff.max()))
            parity['passed']=parity['mean_abs']<2e-6 and parity['max_abs']<.005
            assert parity['passed'],parity
            image(rgbB).save(out/'reference-native.png')
            valid['native_rotated_render_parity']=parity;valid['actual_cuda_projection']=proj
            valid['serialized_camera_sha256']=rt.sha(out/'cameras.json')
            valid['native_replay_uses_serialized_camera']=True
            valid['format_passed']=True
            if step==max(steps):
                review=out/'novel';review.mkdir(exist_ok=True)
                with np.load(root/'fitted-head-meshes.npz',allow_pickle=False) as z:
                    v=z['vertices_world'][32]@R.T
                center=np.median(v,axis=0);head_size=float(np.ptp(v,axis=0).max());radius=head_size*2.2
                source_eye=np.zeros(3);direction=source_eye-center;direction/=np.linalg.norm(direction)
                novel_rows=[]
                for i,angle in enumerate(np.linspace(0,2*np.pi,36,endpoint=False)):
                    rot=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
                    eye=center+radius*(rot@direction);forward=center-eye;forward/=np.linalg.norm(forward)
                    right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=np.cross(forward,right)
                    ext=np.eye(4);ext[:3,:3]=np.stack([right,down,forward]);ext[:3,3]=-ext[:3,:3]@eye
                    novel=dict(frame=f'novel_{i:03d}',wh=[384,384],K=[[650.,0,192.],[0,650.,192.],[0,0,1]],w2c=ext.tolist(),synthetic=True)
                    cam,_=camera(novel,i);image(render(cam,mapped,pipe,bg)['render']).save(review/(novel['frame']+'.png'));novel_rows.append(novel)
                rt.save(review/'cameras.json',dict(records=novel_rows,convention='OpenCV world-to-camera; COLMAP pixel corners',synthetic=True,ground_truth_available=False))
                for start in [0,12,24]:
                    sheet=Image.new('RGB',(1536,1152),'black')
                    for j in range(12):sheet.paste(Image.open(review/f'novel_{start+j:03d}.png'),((j%4)*384,(j//4)*384))
                    sheet.save(review/f'orbit_{start:02d}-{start+11:02d}.png')
                valid['novel_synthetic_views']=36;valid['novel_quality_accepted']=False
            rt.save(out/'validation.json',valid)
            print(step,parity,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',choices=['head-gaussian-first-cycle','head-gaussian-long300k'],default='head-gaussian-first-cycle')
    parser.add_argument('--step',action='append',type=int)
    args=parser.parse_args();main(args.run,args.step)
