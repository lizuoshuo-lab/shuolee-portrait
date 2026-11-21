"""Frozen-state attribution of face-adjacent Gaussian noise; no optimizer updates."""
from pathlib import Path
from types import SimpleNamespace
import argparse, copy, importlib.util, json, math, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path('/opt/gs-slam-lab/msv2-demo-transparent-full-20261104')
OUT = Path('/mnt/d/00/记录/2026-11-04_MSV2_原始_2倍_4倍对比/face-noise-analysis')


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def setup():
    ex = load('noise_experiment', BASE / 'experiment.py')
    OUT.mkdir(parents=True, exist_ok=True)
    return ex


def cpu():
    from plyfile import PlyData
    from scipy.ndimage import maximum_filter, map_coordinates
    ex = setup(); rows = ex.read(BASE / 'native-inputs.json')['records']
    path = BASE / 'model/point_cloud/iteration_18000/point_cloud.ply'
    v = PlyData.read(path)['vertex'].data
    xyz = np.stack([v[n] for n in ['x','y','z']], axis=1).astype(np.float64)
    total = np.zeros(len(v), np.int32); supported = total.copy(); exact = total.copy()
    face_seen = total.copy(); face_outside = total.copy()
    labels = ['frame_000001', 'frame_000012', 'frame_000028', 'frame_000047', 'frame_000059']
    for row in rows:
        with Image.open(row['mask']) as im: mask = np.array(im.convert('L'))
        extended = maximum_filter(mask > 127, size=7)
        w = np.array(row['w2c']); k = np.array(row['K']); cp = xyz @ w[:3,:3].T + w[:3,3]
        q = cp @ k.T; uv = q[:,:2] / np.maximum(q[:,2:], 1e-9) - .5
        on = (cp[:,2] > .01) & (uv[:,0] >= 0) & (uv[:,0] < row['wh'][0]-1) & (uv[:,1] >= 0) & (uv[:,1] < row['wh'][1]-1)
        value = map_coordinates(mask.astype(np.float32)/255, uv[:,::-1].T, order=1, mode='constant', cval=0)
        near = map_coordinates(extended.astype(np.float32), uv[:,::-1].T, order=1, mode='constant', cval=0)
        total += on; exact += on & (value > .5); supported += on & (near > .5)
        box = np.array(row['face_box_xyxy']); pad = np.maximum((box[2:]-box[:2])*.7, 35)
        head = on & (uv[:,0] >= box[0]-pad[0]) & (uv[:,0] <= box[2]+pad[0]) & (uv[:,1] >= box[1]-pad[1]) & (uv[:,1] <= box[3]+pad[1])
        face_seen += head; face_outside += head & (value < .01)
        if row['frame'] in labels:
            with Image.open(row['image']) as im: rgb = np.array(im.convert('RGB'))
            overlay = rgb.astype(float); overlay[:,:,1] = overlay[:,:,1]*.6 + mask*.4
            overlay = np.uint8(overlay.clip(0,255))
            box = np.r_[np.maximum(np.floor(box[:2]-pad),0), np.minimum(np.ceil(box[2:]+pad),row['wh'])].astype(int).tolist()
            a = Image.fromarray(rgb).crop(box).resize((400,400),Image.Resampling.LANCZOS)
            b = Image.fromarray(overlay).crop(box).resize((400,400),Image.Resampling.LANCZOS)
            c = Image.fromarray(mask).convert('RGB').crop(box).resize((400,400),Image.Resampling.NEAREST)
            canvas = Image.new('RGB',(1200,440),'#eee');d=ImageDraw.Draw(canvas)
            font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
            for j,(im,label) in enumerate([(a,'原图'),(b,'原始遮罩覆盖'),(c,'原始遮罩')]):
                d.text((j*400+4,4),row['frame']+' '+label,font=font,fill='black');canvas.paste(im,(j*400,40))
            canvas.save(OUT/(row['frame']+'_input_mask.png'))
    ratio = supported / np.maximum(total,1)
    opacity = 1/(1+np.exp(-v['opacity'].astype(np.float64)))
    scales = np.stack([np.exp(v[f'scale_{i}'].astype(np.float64)) for i in range(3)],axis=1)
    anisotropy = scales.max(1)/np.maximum(scales.min(1),1e-12)
    np.savez_compressed(OUT/'frozen-point-support.npz', support=ratio, in_frame=total,
        foreground_count=exact, dilated_foreground_count=supported, face_seen=face_seen,
        face_outside=face_outside, opacity=opacity, max_scale=scales.max(1), anisotropy=anisotropy)
    def stats(sel):
        return dict(points=int(sel.sum()), support_quantiles=np.quantile(ratio[sel],[0,.1,.5,.9,1]).tolist(),
            opacity_quantiles=np.quantile(opacity[sel],[.1,.5,.9,.99]).tolist(),
            scale_quantiles=np.quantile(scales.max(1)[sel],[.1,.5,.9,.99]).tolist(),
            anisotropy_quantiles=np.quantile(anisotropy[sel],[.1,.5,.9,.99]).tolist())
    masks = {'all':np.ones(len(v),bool),'face_adjacent':face_seen>0,'face_exterior_many_views':face_outside>=10,
        'majority_outside_dilated_foreground':(ratio<.5)&(total>=10)}
    result={k:stats(s) for k,s in masks.items() if s.any()}
    ex.save(OUT/'frozen-support.json',dict(source=str(path),sha256=ex.sha(path),optimizer_updates=0,
        model_unchanged=True,mask_margin_native_pixels=3,stats=result,
        caveat='Projected-center support is a diagnostic, not visibility-aware ground truth. Occlusion, silhouette uncertainty and subject movement can reduce support.'))
    print('FROZEN_SUPPORT',json.dumps(result),flush=True)


def penalty():
    ex=setup();torch=ex.runtime()
    from scene import GaussianModel
    from scipy.ndimage import maximum_filter
    sys.path.insert(0,str(BASE))
    import transparent_hooks as rgba
    path=BASE/'model/point_cloud/iteration_18000/point_cloud.ply';source_hash=ex.sha(path)
    g=GaussianModel(3);g.load_ply(str(path));g.active_sh_degree=3;g.init_culling(59)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    rows=ex.read(BASE/'training-inputs.json')['records'];checks=[]
    for i in [0,27,58]:
        r=rows[i];cam=ex.Camera(r,i);mask=torch.as_tensor(cam.mask.copy(),device='cuda',dtype=torch.float32)/255
        scale=r['upsampling_scale'];radius=8*scale
        ring_np=maximum_filter(cam.mask>2,size=2*radius+1)&(cam.mask<255)
        ring=torch.as_tensor(ring_np,device='cuda')
        box=np.array(r['face_box_xyxy']);pad=np.maximum((box[2:]-box[:2])*.6,24*scale)
        x0,y0=np.maximum(np.floor(box[:2]-pad),0).astype(int)
        x1,y1=np.minimum(np.ceil(box[2:]+pad),r['wh']).astype(int)
        face_region=torch.zeros_like(mask,dtype=torch.bool);face_region[y0:y1,x0:x1]=True
        outside=(mask<.01)&face_region;inside=(mask>.99)&face_region
        alpha,_=rgba.render_alpha(cam,g,pipe,culling=g._culling[:,i]);a=alpha[0]
        positive=-(mask*a.clamp_min(1e-6).log()).sum()/mask.sum()
        negative_map=-(1-mask)*(1-a).clamp_min(1e-6).log()
        negative=negative_map.sum()/(1-mask).sum()
        old=.2*.5*(positive+negative)
        ring_weight=(1-mask)*ring
        boundary=negative_map[ring].sum()/ring_weight.sum()
        new=old+.1*boundary
        values={}
        for label,loss in [('existing',old),('existing_plus_boundary_mask',new)]:
            da,dop,dxyz=torch.autograd.grad(loss,(alpha,g._opacity,g._xyz),retain_graph=True)
            values[label]=dict(loss=float(loss),face_exterior_alpha_gradient_mean=float(da[0][outside].mean()),
                face_exterior_opacity_gradient_mean=float(da[0][outside&ring].mean()),
                interior_alpha_gradient_mean=float(da[0][inside].mean()),
                opacity_logit_gradient_l1=float(dop.abs().sum()),position_gradient_l1=float(dxyz.abs().sum()),
                finite=bool(torch.isfinite(dop).all() and torch.isfinite(dxyz).all()))
            assert values[label]['finite'] and values[label]['opacity_logit_gradient_l1']>0
        changed_mask=torch.ones_like(mask)
        wrong=-.1*(changed_mask*a.clamp_min(1e-6).log()).mean()
        wrong_da=torch.autograd.grad(wrong,alpha,retain_graph=True)[0]
        checks.append(dict(frame=r['frame'],training_wh=r['wh'],mask_sha256=ex.sha(r['mask']),
            mask_has_zero_background_and_one_foreground=bool((mask==0).any() and (mask==1).any()),
            face_exterior_pixels=int(outside.sum()),global_background_equivalent_pixels=float((1-mask).sum()),
            boundary_background_equivalent_pixels=float(ring_weight.sum()),
            face_exterior_alpha_mean=float(a[outside].mean()),face_exterior_gt_01=int((a[outside]>.1).sum()),
            face_exterior_penalty_share=float(negative_map[outside].sum()/negative_map.sum()),
            face_exterior_on_boundary_fraction=float(negative_map[outside&ring].sum()/negative_map[outside].sum()),
            foreground_alpha_mean=float(a[inside].mean()),gradients=values,
            all_foreground_counterfactual_face_gradient_mean=float(wrong_da[0][outside].mean())))
        with torch.no_grad():
            pixels=np.rint(a.clamp(0,1).cpu().numpy()*255).astype('uint8')
            Image.fromarray(pixels).crop([x0,y0,x1,y1]).save(OUT/(r['frame']+'_rendered_alpha.png'))
            Image.fromarray(cam.mask).crop([x0,y0,x1,y1]).save(OUT/(r['frame']+'_target_mask.png'))
            Image.fromarray(np.uint8(ring_np)*255).crop([x0,y0,x1,y1]).save(OUT/(r['frame']+'_boundary_region.png'))
        del cam,mask,alpha,a,old,new,positive,negative,negative_map,wrong,da,dop,dxyz
        torch.cuda.empty_cache()
    assert ex.sha(path)==source_hash
    ex.save(OUT/'mask-penalty-diagnostic.json',dict(optimizer_updates=0,source_model_unchanged=True,source_sha256=source_hash,
        no_gaussians_removed=True,no_model_parameter_edits=True,checks=checks,
        proposed_single_variable='Existing mask BCE retained, plus a separately normalized negative BCE in the 8-native-pixel exterior boundary band. Model, RGB loss, source masks and MSV2 schedule unchanged.',
        purpose='Check mask connectivity and local penalty dilution. Gradient checks alone do not prove trained quality.'))
    print('MASK_PENALTY_DIAGNOSTIC',json.dumps(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['cpu','penalty']);args=p.parse_args();globals()[args.action]()
