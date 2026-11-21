"""Prepare train-only native crops; no predicted joints become training targets."""
from pathlib import Path
import contextlib,io,json,hashlib,shutil
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from yacs.config import CfgNode as CN
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.datasets.utils import generate_image_patch_cv2

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
S=L/'outputs/dynamic-human-enerf-second-interval-079'
O=L/'outputs/dynamic-human-hand-prior-096';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'input-manifest.json').exists()
source=L/'outputs/dynamic-human-enerf-hand-motion-082/observations.json'
rs=[r for r in read(source)['records'] if r['split']=='fit']
assert len(rs)==105 and len({r['camera'] for r in rs})==15
assert not {r['camera'] for r in rs}&{'00','06','12'}
assert all(r['frame']%2==0 and r['hand']=='right' for r in rs)
# Official ViT settings for preprocessing verification; checkpoint settings must
# be checked against these before inference, never silently assumed compatible.
cfg=CN({'MODEL':{'IMAGE_SIZE':256,'BBOX_SHAPE':[192,256],'IMAGE_MEAN':[.485,.456,.406],'IMAGE_STD':[.229,.224,.225]}})
records=[];checks=[];panels=[];font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',13)
for r in rs:
 c,t=r['camera'],r['frame'];p=S/'native-input'/f'camera{c}-frame{t:06d}.npz'
 assert sha(p)==r['input_sha256']
 z=np.load(p);rgb=z['rgb'];assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['rectified_sha256']
 assert rgb.shape==(1080,1920,3) and np.array_equal(z['K'],r['K']) and np.array_equal(z['w2c'],r['w2c'])
 box=np.array(r['crop'],np.float32);bgr=np.ascontiguousarray(rgb[:,:,::-1])
 ds=ViTDetDataset(cfg,bgr,box[None],np.ones(1),rescale_factor=1.)
 with contextlib.redirect_stdout(io.StringIO()):item=ds[0]
 center=(box[:2]+box[2:])/2
 # Square 160px context expanded to 192:256 aspect -> square crop 213.33px;
 # backbone then removes 32px each side, retaining the 160px native width.
 side=(box[2]-box[0])*256/192
 warp,trans=generate_image_patch_cv2(bgr,*center,side,side,256,256,False,1.,0,border_mode=cv2.BORDER_CONSTANT)
 expected=(warp[:,:,::-1].astype(np.float32).transpose(2,0,1)-255*np.array(cfg.MODEL.IMAGE_MEAN,dtype=np.float32)[:,None,None])/(255*np.array(cfg.MODEL.IMAGE_STD,dtype=np.float32)[:,None,None])
 error=float(np.max(abs(item['img']-expected)));assert error<2e-6
 inv=cv2.invertAffineTransform(trans)
 corners=np.array([[32,0],[224,0],[224,256],[32,256]],np.float32)
 native=corners@inv[:,:2].T+inv[:,2]
 assert np.allclose(native[:,0].min(),box[0],atol=2e-4) and np.allclose(native[:,0].max(),box[2],atol=2e-4)
 rec=dict(camera=c,frame=t,source=str(p.relative_to(L)),source_sha256=r['input_sha256'],rgb_sha256=r['rectified_sha256'],right=1,context_bbox=box.tolist(),rescale_factor=1.,box_center=item['box_center'].tolist(),box_size=float(item['box_size']),affine_native_to_patch=trans.tolist(),affine_patch_to_native=inv.tolist(),backbone_native_corners=native.tolist(),K=z['K'].tolist(),w2c=z['w2c'].tolist(),split='train_only_prior_qualification')
 records.append(rec);checks.append(dict(camera=c,frame=t,max_tensor_error=error))
 if c in ['03','04','13','17'] and t in [720,726,732]:
  raw=rgb[int(box[1]):int(box[3]),int(box[0]):int(box[2])];patch=warp[:,:,::-1]
  panel=Image.new('RGB',(416,286),'white');d=ImageDraw.Draw(panel);d.text((4,4),f'{c}/{t}: native context | official network crop',font=font,fill='black');panel.paste(Image.fromarray(raw),(0,30));panel.paste(Image.fromarray(patch),(160,30));out=O/f'input-camera{c}-{t}.png';panel.save(out);panels.append(dict(file=out.name,sha256=sha(out)))
save(O/'input-manifest.json',dict(status='105training_images_prepared_no_predictions',records=records,source_observations_sha256=sha(source),expected_model_cfg=dict(cfg.MODEL),scope='Existing right-hand context proposals only. Old RTMHand joints/scores are NOT labels or copied. No heldout-camera or odd-time images used. Source context was previously reviewed; this is not per-finger annotation.',no_predictions=True,goal_complete=False))
save(O/'preprocessing-check.json',dict(status='105_native_RGB_BRG_roundtrip_affine_normalization_checks_pass',checks=checks,panels=panels,normalization='Native RGB -> BGR -> official crop -> RGB -> ImageNet normalization. No image enhancement.',crop='Existing160pxcontext,rescale1.0;official192:256aspectexpansion/backbonecrop retained. It is not a hand-tight detector box with default2.5padding.',checkpoint_cfg_verified=False,no_MANO_loaded=True,no_hand_predictions=True,goal_complete=False))
shutil.copy2(__file__,O/Path(__file__).name);print('105 train-only native crops and official tensors checked',flush=True)
