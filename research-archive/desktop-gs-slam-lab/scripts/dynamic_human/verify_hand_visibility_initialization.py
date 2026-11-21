from pathlib import Path
import json,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_mano_gaussian_surface import ManoGaussianSurface,render_hand_surface
from train_enerf_camera10_time import ss
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-hand-consensus-097';O=L/'outputs/dynamic-human-enerf-hand-visibility-100';D=O/'initialization';D.mkdir(exist_ok=True);read=lambda p:json.loads(p.read_text());t=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda');u8=lambda a:np.rint(np.clip(a,0,1)*255).astype(np.uint8)
torch.set_num_threads(6);cp=torch.load(S/'local_bspline/resume.pt',map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};active=torch.from_numpy(np.load(O/'context-reset-mask.npy')).cuda();g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);old=HandJointRenderer(S/'head/capture-affines.npz',g,g,active);boxes={(r['camera'],r['frame']):r['crop'] for r in read(L/'outputs/dynamic-human-enerf-hand-motion-082/observations.json')['records']};font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12);records=[]
for c,f in [('03',726),('04',732),('13',720)]:
 z=np.load(S/'native-input'/f'camera{c}-frame{f:06d}.npz');x0,y0,x1,y1=boxes[c,f];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;K=t(K);W=t(z['w2c']);arrays=[z['rgb'][y0:y1,x0:x1]];target=t(z['rgb'][y0:y1,x0:x1]/255)*t(z['matte'][y0:y1,x0:x1])[...,None]+.5*(1-t(z['matte'][y0:y1,x0:x1])[...,None])
 for b in ['narrow_reset','context_reset']:
  hand=ManoGaussianSurface(H/'all_views/parameters.npz',S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],read(O/'protocol.json')['initial_color'],True);reset=torch.from_numpy(np.load(O/('reset-mask.npy' if b=='narrow_reset' else 'context-reset-mask.npy'))).cuda();pp=dict(p);pp['opacities']=p['opacities'].clone()
  with torch.no_grad():hand.opacity_logits.fill_(float(np.log(.2/.8)));pp['opacities'][reset]=torch.minimum(pp['opacities'][reset],torch.tensor(float(np.log(.02/.98)),device='cuda'))
  rr,al,_=render_hand_surface(pp,old,hand,c,f,W,K,160,160);pred=rr[0]+.5*(1-al[0]);loss=.8*(pred-target).abs().mean()+.2*(1-ss(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]))+hand.regularization();loss.backward();grads={k:float(v.grad.abs().max()) for k,v in hand.named_parameters() if v.requires_grad};assert all(np.isfinite(x) and x>0 for x in grads.values());arrays.append(u8(pred.detach().cpu().numpy()));records.append(dict(branch=b,camera=c,frame=f,loss=float(loss),gradients=grads,reset_count=int(reset.sum())))
 im=Image.new('RGB',(480,202),'white');d=ImageDraw.Draw(im)
 for i,(a,label) in enumerate(zip(arrays,['Original','1937 reset initial','6330 reset initial'])):im.paste(Image.fromarray(a),(i*160,42));d.text((i*160+3,4),label,font=font,fill='black');d.text((i*160+3,22),f'{c}/{f} untrained',font=font,fill='black')
 im.save(D/f'camera{c}-frame{f:06d}.png')
(D/'summary.json').write_text(json.dumps(dict(status='both_initializations_actualCUDAforward_backward_finite',records=records,no_optimizer_steps=True,not_a_reconstruction_result=True),indent=2)+'\n')
