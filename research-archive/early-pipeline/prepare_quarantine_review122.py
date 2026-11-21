"""Native unchanged-image before/after evidence for the single family ablation."""
from audit_offplane_support122 import B, read, save, sha, now
from PIL import Image,ImageDraw
import numpy as np
O=B/'material-quarantine';new=read(O/'result.json');old=read(B/'material-region-comparison/observations.json')
assert not (O/'figure-index.json').exists()
frames=[230,246,262,294,310,326];ids=[5,249,9]
groups=['face_candidate','clothing_candidate','ground_candidate'];size=160
fig=Image.new('RGB',(960,480));idx=[]
for col,f in enumerate(frames):
    im=Image.open(B/f'frame_{f:06d}.png').convert('RGB')
    for row,(tid,g) in enumerate(zip(ids,groups)):
        a=next(v for v in old if v['track']==tid and v['frame']==f)
        b=next(v for v in new['validation'][g]['observations'] if v['track']==tid and v['target']==f)
        assert a['evaluated'] and b['valid_projection'] and np.allclose(a['observed'],b['observed'],rtol=0,atol=0)
        u,v=a['observed'];x,y=round(u)-80,round(v)-80;crop=im.crop((x,y,x+160,y+160));d=ImageDraw.Draw(crop)
        d.ellipse((u-x-3,v-y-3,u-x+3,v-y+3),outline=(0,255,80),width=1)
        for pred,color,rad in [(a['predicted'],(255,40,180),5),(b['predicted'],(0,180,255),3)]:
            p,q=np.array(pred)-[x,y];d.line((p-rad,q,p+rad,q),fill=color,width=1);d.line((p,q-rad,p,q+rad),fill=color,width=1)
        fig.paste(crop,(160*col,160*row));idx.append(dict(frame=f,track=tid,group=g,crop=[x,y,x+160,y+160],before_px=a['error_px'],after_px=b['error_px']))
fig.save(O/'same_native_before_after.png');save(O/'figure-index.json',dict(updated=now(),rows=idx,legend='green observed;magenta before;cyan after. Same nativepixels, no warp.'))
first=next(p for p in read(B/'geometry-result.json')['pairs'] if p['frames']==[230,246])
init=new['initialization'];audit=dict(first_pair_R_max_difference=float(np.max(abs(np.array(init['first_pair_R'])-first['E_R']))),
    first_pair_t_max_difference=float(np.max(abs(np.array(init['first_pair_t'])-np.array(first['E_t_direction'])/np.linalg.norm(first['E_t_direction'])))),
    all934_nonquarantined_tracks_initialized=init['initialized_tracks']==934,
    all190_heldout_tracks_retained=new['heldout_tracks']==190,
    all681_background_targets_retained=new['validation']['background']['summary']['n_total']==681,
    K_exactly_unchanged=np.array_equal(new['K_integer_centers'],read(B/'shared-path/solver-repair/result.json')['K_integer_centers']),
    training=False)
save(O/'comparison-controls.json',audit);print(audit)
