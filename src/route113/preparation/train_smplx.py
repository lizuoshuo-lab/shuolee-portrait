"""Unchanged image/camera preview helpers extracted from the historical adapter.
No historical training entry or dynamic model is exposed here.
"""
from types import SimpleNamespace
import numpy as np
from PIL import Image
import shuolee_common as c
torch=c.runtime()
from scene.cameras import CameraProj2
from gaussian_renderer import render
RUN=c.ROOT/'gaussian-run'
BG=torch.ones(3,device='cuda')
def prepare_views(rows, max_height):
    views=[]; records=[]; max_error=0.
    for i,row in enumerate(rows):
        assert c.sha(row['image'])==row['source_sha256']
        assert c.sha(row['mask'])==row['mask_sha256']
        image=Image.open(row['image']).convert('RGB')
        mask=Image.open(row['mask']).convert('L')
        assert image.size==mask.size==tuple(row['wh'])
        a=np.asarray(mask); yy,xx=np.where(a>127); assert len(xx)>1000
        pad=round(.1*(yy.max()-yy.min()+1))
        x0=max(0,int(xx.min())-pad); x1=min(image.width,int(xx.max())+pad+1)
        y0=max(0,int(yy.min())-pad); y1=min(image.height,int(yy.max())+pad+1)
        scale=min(1.,max_height/(y1-y0))
        W=round((x1-x0)*scale);H=round((y1-y0)*scale)
        image=image.crop((x0,y0,x1,y1)).resize((W,H),Image.Resampling.LANCZOS)
        mask=mask.crop((x0,y0,x1,y1)).resize((W,H),Image.Resampling.BILINEAR)
        K=np.array(row['K']); A=np.array([[W/(x1-x0),0,-x0*W/(x1-x0)],
                                       [0,H/(y1-y0),-y0*H/(y1-y0)],[0,0,1.]])
        K=A@K;E=np.array(row['w2c'])
        camera=CameraProj2(i,E[:3,:3].T,E[:3,3],K[0,0],K[1,1],'white',W,None,H,row['image'],row['frame'],i,
                           cx=K[0,2],cy=K[1,2],timestep=i)
        # Compare the exact original CUDA projection convention, including the
        # half-pixel center, to the independently cropped pinhole coordinates.
        points=np.array([[.1,.2,.3],[-.3,.3,.5],[.5,.5,.4]],dtype=np.float32)
        cp=points@E[:3,:3].T+E[:3,3]; q=cp@K.T; uv=q[:,:2]/q[:,2:]-.5
        ndc=np.c_[points,np.ones(len(points))]@camera.full_proj_transform.numpy()
        actual=((ndc[:,:2]/ndc[:,3:]+1)*[W,H]-1)/2
        error=float(np.abs(actual-uv).max());max_error=max(error,max_error)
        assert error<.01,(row['frame'],error)
        rgb=np.asarray(image).copy();alpha=np.asarray(mask).copy()
        views.append((camera,torch.from_numpy(rgb).permute(2,0,1),torch.from_numpy(alpha)[None]))
        records.append(dict(frame=row['frame'],crop=[x0,y0,x1,y1],wh=[W,H],K=K.tolist(),w2c=E.tolist()))
    c.save(RUN/'training-cameras.json',dict(records=records,max_projection_difference_px=max_error))
    return views

def pil(tensor):
    return Image.fromarray((tensor.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8))
