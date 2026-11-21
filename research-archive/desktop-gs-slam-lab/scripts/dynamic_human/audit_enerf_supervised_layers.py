"""Independent CPU audit of saved per-actor compositing diagnostics."""
from pathlib import Path
import json,hashlib,shutil,numpy as np,cv2
from PIL import Image
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-actor-supervision-070/actor-layers';I=L/'outputs/dynamic-human-enerf-heldout-cameras-068'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
report=read(R/'render-summary.json');inputs={(r['camera'],r['frame']):r for r in read(I/'input-audit.json')['records']};checks=[];totals={}
assert sha(L/'outputs/dynamic-human-enerf-actor-supervision-070/residual_graph/resume.pt')==report['checkpoint_sha256']
for row in report['records']:
 c,t=row['camera'],row['frame'];name=f'camera{c}-frame{t:06d}';source=I/'native-input'/f'{name}.npz';assert sha(source)==inputs[c,t]['input_sha256']
 z=np.load(source);d=np.load(R/f'{name}-layers.npz');prev=np.load(L/'outputs/dynamic-human-enerf-actor-supervision-070/heldout/native-evaluation'/f'{name}-prediction.npz')
 assert all(np.isfinite(d[k]).all() for k in d.files)
 assert np.array_equal(d['rgb'],prev['rgb']) and np.array_equal(d['alpha'],prev['alpha'])
 err_rgb=float(np.abs(d['actor_rgb'].sum(2)-d['rgb']).max());err_alpha=float(np.abs(d['actor_alpha'].sum(-1)-d['alpha']).max())
 assert err_rgb<5e-6 and err_alpha<5e-6
 a=d['alpha'];pred=np.clip(d['rgb']+(1-a[...,None])*.5,0,1);ref=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None])
 core=cv2.erode((z['matte']>.95).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&z['valid'];low=core&(a<.9);bad=core&(np.abs(pred-ref).mean(-1)>.08)
 overlap=(d['isolated_alpha']>.5).all(-1)&z['valid'];ratio=d['actor_alpha'][...,0]/np.maximum(a,1e-8);mixed=overlap&(ratio>.1)&(ratio<.9)
 counts=dict(core_pixels=int(core.sum()),core_alpha_below_point9=int(low.sum()),core_mae_above_point08=int(bad.sum()),bad_core_pixels_with_alpha_atleast_point9=int((bad&~low).sum()),isolated_overlap_pixels=int(overlap.sum()),mixed_actor_pixels=int(mixed.sum()))
 for k,v in counts.items():assert row['statistics'][k]==v;(totals.setdefault(k,0));totals[k]+=v
 x0,y0,x1,y1=map(int,z['bbox']);w=x1-x0;h=y1-y0
 images=[z['rgb'],u8(pred)]+[u8(d['isolated_rgb'][:,:,j]+.5*(1-d['isolated_alpha'][...,j:j+1])) for j in [0,1]]
 panel=np.array(Image.open(R/f'{name}-actors.png'));assert panel.shape==(h+36,4*w,3)
 for i,img in enumerate(images):assert np.array_equal(panel[36:,i*w:(i+1)*w],img[y0:y1,x0:x1])
 identity=d['actor_alpha'][...,0:1]*np.array([1.,.45,.05])+d['actor_alpha'][...,1:2]*np.array([.1,.55,1.])+.5*(1-a[...,None])
 dz=d['isolated_depth'][...,0]-d['isolated_depth'][...,1];depthmap=np.full((1080,1920,3),.5,np.float32);scale=np.clip(np.abs(dz)/.25,0,1)
 color=np.where((dz<0)[...,None],np.array([1.,.45,.05]),np.array([.1,.55,1.]));depthmap[overlap]=(.5*(1-scale[...,None])+color*scale[...,None])[overlap]
 panel=np.array(Image.open(R/f'{name}-layers.png'));assert panel.shape==(h+36,3*w,3)
 for i,img in enumerate([np.repeat(u8(a)[...,None],3,-1),u8(identity),u8(depthmap)]):assert np.array_equal(panel[36:,i*w:(i+1)*w],img[y0:y1,x0:x1])
 for region in row['projected_region_statistics']:
  b=region['bbox'];sl=(slice(b[1],b[3]),slice(b[0],b[2]));s=core[sl];frac=(d['actor_alpha'][sl][s]/np.maximum(a[sl][s,None],1e-8)).mean(0)
  assert np.allclose(frac,region['mean_joint_actor_fraction'],atol=1e-7)
 checks.append(dict(camera=c,frame=t,rgb_sum_error=err_rgb,alpha_sum_error=err_alpha,layers_sha256=sha(R/f'{name}-layers.npz')))
totals['core_low_alpha_fraction']=totals['core_alpha_below_point9']/totals['core_pixels']
totals['high_error_core_with_high_alpha_fraction']=totals['bad_core_pixels_with_alpha_atleast_point9']/totals['core_mae_above_point08']
save(R/'independent-audit.json',dict(status='9native_layer_arrays18panels_pixel_verified_and_statistics_recomputed',records=checks,aggregate=totals,
 qualification='Independent recomputation checks saved pixel consistency and model-contribution accounting only; does not certify true actor identity, depth, complete hidden surfaces or causal diagnosis.',goal_complete=False))
print(json.dumps(totals))