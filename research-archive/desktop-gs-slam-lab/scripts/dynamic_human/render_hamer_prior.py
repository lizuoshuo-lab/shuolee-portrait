"""Official mesh renderer on native160px context; CPU OSMesa preview only."""
from pathlib import Path
import json,hashlib,contextlib,io
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from hamer.configs import get_config
from hamer.models.mano_wrapper import MANO
from hamer.utils.renderer import Renderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');O=L/'outputs/dynamic-human-hand-prior-096';R=Path('/mnt/d/GS-SLAM-Data/models/hamer/_DATA');V=O/'mesh-panels';V.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=get_config(str(R/'hamer_ckpts/model_config.yaml'),update_cachedir=False);m=MANO(model_path='/mnt/d/GS-SLAM-Data/models/mano_v1_2/numpy',is_rhand=True,use_pca=False,flat_hand_mean=True)
rows=[r for r in read(O/'input-manifest.json')['records'] if r['frame']==732 or (r['camera'] in ['03','04','13','17'] and r['frame'] in [720,726])];font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12);records=[];colors=[(235,70,20),(210,40,160),(30,150,210),(65,180,55),(165,75,210)]
for r in rows:
 c,t=r['camera'],r['frame'];stem=f'camera{c}-frame{t:06d}';a=np.load(O/'predictions'/f'{stem}.npz');z=np.load(L/r['source']);x0,y0,x1,y1=map(int,r['context_bbox']);raw=z['rgb'][y0:y1,x0:x1];assert raw.shape==(160,160,3)
 rendercfg=cfg.clone();rendercfg.defrost();rendercfg.MODEL.IMAGE_SIZE=160;rendercfg.EXTRA.FOCAL_LENGTH=float(5000*a['box_sizes'][1]/256);rendercfg.freeze();tensor=torch.from_numpy(raw.copy()).permute(2,0,1).float()/255;tensor=(tensor-torch.tensor(cfg.MODEL.IMAGE_MEAN)[:,None,None])/torch.tensor(cfg.MODEL.IMAGE_STD)[:,None,None]
 renderer=Renderer(rendercfg,m.faces);rgba=renderer(a['pred_vertices'][1].copy(),a['pred_cam_t'][1].copy(),tensor,return_rgba=True,mesh_base_color=(.5,.75,1.),scene_bg_color=(1,1,1));assert rgba.shape==(160,160,4) and np.isfinite(rgba).all(),(rgba.shape,bool(np.isfinite(rgba).all()));overlay=rgba[:,:,:3]*rgba[:,:,3:]+raw/255*(1-rgba[:,:,3:]);points=Image.fromarray(raw.copy());draw=ImageDraw.Draw(points);xy=a['source_keypoints_2d'][1]-[x0,y0]
 for f in range(5):
  chain=[0]+list(range(1+4*f,5+4*f))
  for i,j in zip(chain[:-1],chain[1:]):draw.line([tuple(xy[i]),tuple(xy[j])],fill=colors[f],width=1)
 im=Image.new('RGB',(480,206),'white');draw=ImageDraw.Draw(im);draw.text((4,4),f'{c}/{t} | Predicted prior, NOT Gaussian reconstruction',font=font,fill='black');draw.text((4,24),'Native photograph       Predicted joints          Predicted MANO mesh',font=font,fill='black');im.paste(Image.fromarray(raw),(0,46));im.paste(points,(160,46));im.paste(Image.fromarray(np.rint(np.clip(overlay,0,1)*255).astype(np.uint8)),(320,46));path=V/f'{stem}.png';im.save(path);np.savez(V/f'{stem}-rgba.npz',rgba=rgba,render_focal_length=rendercfg.EXTRA.FOCAL_LENGTH,camera_t=a['pred_cam_t'][1],source_bbox=np.array([x0,y0,x1,y1]));records.append(dict(camera=c,frame=t,file=str(path.relative_to(O)),sha256=sha(path),source_photo_pixels_exact=np.array_equal(np.array(im)[46:,0:160],raw),renderer='official HaMeR pyrender using native-context intrinsics; OSMesa CPU; mesh is monocular prediction, not photometric reconstruction'));print('MESH',c,t,flush=True)
save(O/'mesh-preview.json',dict(status='23native_context_mesh_overlays_complete',records=records,view_scale='native160px; focal scaled by box_size/256, unchanged camera translation and vertices',Gaussian_parameters_changed=False,goal_complete=False))
