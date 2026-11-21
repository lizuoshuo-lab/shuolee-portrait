"""Bounded R6 same-camera contribution, hide/restore and shape/center controls."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_smpl_delivery_round6_world_scale import array, face_frame, mesh_front_depth, vertices
from measure_smpl_delivery_round5_ear016 import pin, read, require, sha


ROOT = Path(__file__).resolve().parents[1]/'outputs'
DELIVERY = ROOT/'smpl-delivery-round6/receipt.json'
DELIVERY_SHA = 'a728539832d89b4f4d11b644d1d2398f1eedb157f13e021eaf2234e3b5658c1c'
ASSET = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-scale-20261111')
LEASE = Path('/tmp/shuolee-native122-gpu.lock')
XYZ = ('x','y','z')
SCALE = ('scale_0','scale_1','scale_2')
QUAT = ('rot_1','rot_2','rot_3','rot_0')
SPECS = {
    '016': dict(view='static_rotated_oblique', top=8, hotspot=[[(84,145),(135,139),(139,145),(85,152)],
        [(270,124),(354,113),(355,121),(270,134)]], valid=[[(172,168),(225,156),(240,187),(188,211)]], crop=(70,100,370,225)),
    '006': dict(view='static_rotated_side', top=16, hotspot=[[(620,389),(632,384),(661,441),(694,533),
        (681,548),(653,476),(637,439)]], valid=[[(555,340),(604,354),(619,393),(578,423),(549,394)]], crop=(530,335,710,563)),
}


def pixels(polygons, wh, origin):
    image = Image.new('1', tuple(wh))
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon([tuple(np.asarray(p)+origin) for p in polygon], fill=1)
    y,x = np.nonzero(np.asarray(image))
    return np.column_stack((x,y))


def ranked(xyz, rotation, sigma, opacity, camera, xy):
    E,K = (np.asarray(camera[k]) for k in ('w2c','K'))
    width,height = camera['wh']
    camera_xyz = xyz@E[:3,:3].T+E[:3,3]
    q = camera_xyz@K.T
    z = q[:,2]
    uv = q[:,:2]/z[:,None]-.5
    require(z.min()>.2 and K[0,1]==0 and K[1,0]==0, 'Unexpected stock static camera.')
    focus = K.diagonal()[:2]
    tanfov = np.asarray((width,height))/(2*focus)
    ratio = np.clip(camera_xyz[:,:2]/z[:,None], -1.3*tanfov, 1.3*tanfov)
    J = np.zeros((len(z),2,3))
    J[:,[0,1],[0,1]] = focus/z[:,None]
    J[:,:,2] = -focus*ratio/z[:,None]
    projected = J@E[:3,:3]@(rotation*sigma[:,None,:])
    cov = projected@projected.transpose(0,2,1)+.3*np.eye(2)
    a,b,c = cov[:,0,0],cov[:,0,1],cov[:,1,1]
    det,mid = a*c-b*b,.5*(a+c)
    radius = np.ceil(3*np.sqrt(mid+np.sqrt(np.maximum(.1,mid*mid-det))))
    block = np.asarray([16,16]); grid = (np.asarray([width,height])+block-1)//block
    tile_lo = np.clip(np.trunc((uv-radius[:,None])/block).astype(int),0,grid)
    tile_hi = np.clip(np.trunc((uv+radius[:,None]+block-1)/block).astype(int),0,grid)
    pixel_tiles = xy//block
    select = np.flatnonzero(((tile_lo <= pixel_tiles.max(0)) & (tile_hi > pixel_tiles.min(0))).all(1) & (opacity >= 1/255))
    ids = select[np.argsort(z[select], kind='stable')]
    d = uv[ids,None]-xy
    x,y = d.transpose(2,0,1)
    exponent = -.5*(c[ids,None]*x*x+a[ids,None]*y*y-2*b[ids,None]*x*y)/det[ids,None]
    alpha = np.minimum(.99,opacity[ids,None]*np.exp(exponent))
    in_tile = ((pixel_tiles >= tile_lo[ids,None]) & (pixel_tiles < tile_hi[ids,None])).all(2)
    alpha[~in_tile | (alpha < 1/255)] = 0.
    trans_after = np.cumprod(1-alpha, axis=0)
    trans_before = np.vstack((np.ones((1,len(xy))),trans_after[:-1]))
    weights = np.where(trans_after >= .0001, alpha*trans_before, 0.)
    contribution = weights.sum(1)
    order = np.argsort(-contribution, kind='stable')
    exact_shape = np.sqrt(np.linalg.eigvalsh(cov)[:,-1])
    return ids[order], contribution[order], weights.sum(0), uv, exact_shape


def prepare(subject, delivery):
    spec, entry = SPECS[subject], delivery['subjects'][subject]
    review_path = pin(entry['review'],entry['review_sha256']); review=read(review_path)
    run_path = pin(entry['result'],entry['result_sha256']); run=run_path.parent; result=read(run_path)
    plan=read(pin(review_path.parent/'plan.json',review['plan_sha256']))
    view=plan['static_views'][spec['view']]
    arm=plan['static_arms']['round6']
    record=review['static_render_receipts']['round6'][spec['view']]
    require(result['final_snapshot']=='step-010000' and record['static_ply_sha256']==arm['ply_sha256']==result['sha256']
            and all(view['camera'][k]==record['camera'][k] for k in ('frame','wh','K','w2c')),'Frozen static endpoint differs.')
    png_path=pin(review_path.parent/record['file'],record['sha256'])
    with Image.open(png_path) as image:
        archived=np.asarray(image.convert('RGB'))
    crop=np.asarray(view['crop_xyxy']); w,h=crop[2:]-crop[:2]
    require(archived.shape==(h,3*w,3),'Frozen static panel crop differs.')
    ply=vertices(pin(Path(arm['ply']),arm['ply_sha256']))
    snap=run/'gaussian-run'/result['final_snapshot']
    replay_path=pin(run/result['bound_replay'],result['bound_replay_sha256']); replay=read(replay_path)
    bound=read(pin(snap/'bound-replay.json',replay['bound_replay_sha256']))
    local=vertices(pin(snap/'bound-local.ply',bound['bound_ply_sha256']))
    require(len(ply)==len(local)==bound['points'] and all(ply[k].tobytes()==local[k].tobytes() for k in ('opacity','f_dc_0','f_dc_1','f_dc_2')), 'Bound and static point identity differs.')
    with np.load(pin(replay_path.parent/replay['mesh_file'],replay['mesh_sha256']),allow_pickle=False) as archive:
        meshes=dict(archive)
    scale_report=read(pin(run/'gaussian-run/world-scale.json',result['unified_main_training']['world_scale']['sha256']))
    owner=local['binding_0'].astype(int); ref=replay['reference_pose_index']; faces=meshes['faces']
    center,axes,s=face_frame(meshes['vertices_world'][ref],faces)
    posed_scales=np.stack([face_frame(v,faces)[2] for v in meshes['vertices_world']])
    smax=posed_scales.max(0); extent=scale_report['extent_m']; low,high=.016*extent,.025*extent
    local_xyz=array(local,XYZ); xyz=array(ply,XYZ)
    rotation=Rotation.from_quat(array(ply,QUAT)).as_matrix(); sigma=np.exp(array(ply,SCALE)); opacity=expit(ply['opacity'].astype(float))
    M=np.asarray(arm['world_to_viewer']['rotation']); origin=np.asarray(arm['world_to_viewer']['origin_world'])
    anchor=(center[owner]-origin)@M.T
    mapped=anchor+(np.einsum('nij,nj->ni',axes[owner],local_xyz)*s[owner,None])@M.T
    require(abs(mapped-xyz).max()<2e-6,'Bound coordinates do not map to portable row order.')
    # The diagnostic center-only control removes the signed normal displacement.
    tangent_local=local_xyz.copy(); tangent_local[:,1]=0.
    tangent_centers=anchor+(np.einsum('nij,nj->ni',axes[owner],tangent_local)*s[owner,None])@M.T
    normal_local=local_xyz.copy(); normal_local[:,[0,2]]=0.
    normal_centers=anchor+(np.einsum('nij,nj->ni',axes[owner],normal_local)*s[owner,None])@M.T
    xy=pixels(spec['hotspot'],view['camera']['wh'],crop[:2]); valid=pixels(spec['valid'],view['camera']['wh'],crop[:2]);
    ids,score,approx,uv,ellipse=ranked(xyz,rotation,sigma,opacity,view['camera'],xy)
    lx,ly=(xy-crop[:2]).T; observed=archived[ly,lx+2*w,0]/255.
    rank_error=np.abs(approx-observed)
    require(rank_error.mean()<.008 and np.quantile(rank_error,.99)<.035, 'CPU shortlist does not reproduce native alpha reasonably.')
    top=ids[:spec['top']]; neutral=ids[score<=1e-9][:len(top)]
    sorted_sigma=np.sort(sigma,axis=1); stats=[]
    for i,weight in zip(top,score[:len(top)],strict=True):
        face=owner[i]; q=sorted_sigma[i]; ss=smax[face];
        stats.append(dict(index=int(i),triangle=int(face),sigma1_mm=float(q[2]*1000),sigma2_mm=float(q[1]*1000),
            sigma1_over_sigma2=float(q[2]/q[1]),sigma1_over_triangle_smax=float(q[2]/ss),local_major_activation=float(q[2]/s[face]),
            triangle_ref_mm=float(s[face]*1000),triangle_smax_mm=float(ss*1000),
            cap_floor_applied=bool(3*ss<low), cap_mm=float(np.clip(3*ss,low,high)*1000),
            normal_offset_mm=float(local_xyz[i,1]*s[face]*1000),tangent_offset_mm=float(np.linalg.norm(local_xyz[i,[0,2]])*s[face]*1000),
            opacity=float(opacity[i]),pixel_center_native=uv[i].tolist(),screen_ellipse_major_sigma_px=float(ellipse[i]),
            ranked_transmittance_weight=float(weight)))
    state=dict(subject=subject, view=spec['view'], camera=view['camera'], crop=crop.tolist(), review_sha256=entry['review_sha256'],
        archived_png_sha256=sha(png_path), ply=arm['ply'],ply_sha256=arm['ply_sha256'], bound_ply_sha256=bound['bound_ply_sha256'],
        mesh_sha256=replay['mesh_sha256'],hotspot_poly_crop_local=spec['hotspot'],control_poly_crop_local=spec['valid'],
        hotspot_pixels=len(xy),control_pixels=len(valid),cpu_alpha_mae=float(rank_error.mean()),cpu_alpha_p99_abs=float(np.quantile(rank_error,.99)),
        total_predicted_alpha=float(approx.sum()), shortlisted_predicted_contribution=float(score[:len(top)].sum()),top=stats,neutral=neutral.tolist())
    return SimpleNamespace(receipt=state,xy=xy,valid=valid,archived=archived,top=top,neutral=neutral,
        tangent_centers=tangent_centers,normal_centers=normal_centers,
        log_scales=array(ply,SCALE),spec=spec,run=run,meshes=meshes,owner=owner,local_xyz=local_xyz,replay=replay,faces=faces)


def native_support(data):
    replay=data.replay; run=data.run
    rows=read(pin(run/'data/training-inputs.json',replay['training_inputs_sha256']))['records']
    cams=read(pin(run/'gaussian-run/training-cameras.json',replay['training_camera_sha256']))['records']
    ids=data.top; owner=data.owner[ids]; support=[dict(depth_tested_frames=[], matte_foreground_frames=[],matte_background_frames=[]) for _ in ids]
    require(not set(replay['retained_holdouts']) & {r['frame'] for r in rows},'Heldout frame entered support probe.')
    for slot,(row,cam) in enumerate(zip(rows,cams,strict=True)):
        require(row['frame']==cam['frame']==data.meshes['frame_ids'][slot] and cam['pose_index']==slot,'Native support roster changed.')
        v=data.meshes['vertices_world'][slot]; centers,axes,s=face_frame(v,data.faces)
        p=centers[owner]+np.einsum('nij,nj->ni',axes[owner],data.local_xyz[ids])*s[owner,None]
        K,E=(np.asarray(cam[key]) for key in ('K','w2c'))
        q=(p@E[:3,:3].T+E[:3,3])@K.T; vq=(v@E[:3,:3].T+E[:3,3])@K.T
        uv=q[:,:2]/q[:,2:3]; valid=(q[:,2]>.2)&(uv>=0).all(1)&(uv<cam['wh']).all(1); pos=np.flatnonzero(valid); ij=np.floor(uv[pos]).astype(int)
        depth=mesh_front_depth(vq[:,:2]/vq[:,2:3],vq[:,2],data.faces,ij+.5); visible=q[pos,2]<=depth+.025
        with Image.open(pin(row['mask'],row['mask_sha256'])) as image:
            matte=np.asarray(image.convert('L'))[ij[:,1],ij[:,0]]
        for i,a in zip(pos[visible],matte[visible],strict=True):
            support[i]['depth_tested_frames'].append(row['frame'])
            if a>=253: support[i]['matte_foreground_frames'].append(row['frame'])
            if a<=2: support[i]['matte_background_frames'].append(row['frame'])
    for point,info in zip(data.receipt['top'],support,strict=True): point['native_support']=info


def render_trials(data,output):
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from shuolee_portrait.training import camera,render_pose
    from review_bound_scale_coverage import scale_pixels

    state=data.receipt; model=GaussianModel(0);model.load_ply(state['ply']);cam,actual=camera(state['camera'],0,pose_index=0)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    black,white=torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
    require(model.binding is None and model.optimizer is None,'Use the static PLY and no optimizer.')
    def capture():
        images,error=scale_pixels(render,render_pose,cam,model,pipe,black,white,0,1.)
        return images,error
    base,error=capture(); x0,y0,x1,y1=state['crop']; h,w=y1-y0,x1-x0
    tiles=[t[y0:y1,x0:x1] for t in base]
    canvas=np.concatenate([np.repeat(t[...,None],3,axis=2) if t.ndim==2 else t for t in tiles],axis=1)
    require(np.array_equal(np.rint(np.clip(canvas,0,1)*255).astype('uint8'),data.archived),'Stock GPU baseline differs from frozen PNG.')
    saved=(model._opacity.clone(),model._scaling.clone(),model._xyz.clone())
    top=torch.as_tensor(data.top,device='cuda'); neutral=torch.as_tensor(data.neutral,device='cuda')
    direct=torch.zeros_like(model.get_xyz);direct[top]=1
    contrib=render(cam,model,pipe,black,override_color=direct)['render'][0].cpu().numpy()
    out=dict(stock_composition_max_abs=error,black_white_alpha_exact_archive=True, same_camera=actual)
    c=np.asarray(data.spec['crop'])+np.tile([x0,y0],2); l,t,r,b=c; width,height=r-l,b-t
    def measure(images):
        def region(xy):
            x,y=xy.T; a,bb=base[2][y,x],images[2][y,x]; rgb_delta=np.abs(images[0][y,x]-base[0][y,x])
            return dict(alpha_sum_before=float(a.sum()),alpha_sum_after=float(bb.sum()),
                alpha_fraction_removed=float((a.sum()-bb.sum())/max(a.sum(),1e-8)),black_rgb_abs_mae_vs_baseline=float(rgb_delta.mean()),
                white_rgb_abs_mae_vs_baseline=float(np.abs(images[1][y,x]-base[1][y,x]).mean()),
                alpha_p50_before_after=[float(np.median(a)),float(np.median(bb))],
                direct_top_contribution_alpha_sum=float(contrib[y,x].sum()))
        return dict(hotspot=region(data.xy), nearby_content=region(data.valid))
    def crop_trial(images):
        return [np.rint(np.clip(img[t:b,l:r],0,1)*255).astype('uint8') for img in images]

    trials={'baseline':crop_trial(base)}
    for name in ('hide_top1','hide_top','limit_major_to_4x_second','reduce_two_widest_sigmas_to_75pct',
                 'remove_normal_displacement','remove_tangent_displacement','hide_noncontributing'):
        try:
            if name.startswith('hide'):
                ids={'hide_top1':top[:1],'hide_top':top,'hide_noncontributing':neutral}[name]
                model._opacity[ids]=-100.
            elif name=='limit_major_to_4x_second':
                current=model._scaling[top]; sorted_log=current.sort(dim=1).values
                model._scaling[top]=torch.minimum(current,sorted_log[:,1:2]+np.log(4.))
            elif name=='reduce_two_widest_sigmas_to_75pct':
                current=model._scaling[top]
                widest=current.topk(2,dim=1).indices
                model._scaling[top]=current.scatter(1,widest,current.gather(1,widest)+np.log(.75))
            else:
                centers=data.tangent_centers if name=='remove_normal_displacement' else data.normal_centers
                model._xyz[top]=torch.as_tensor(centers[data.top],device='cuda',dtype=model._xyz.dtype)
            images,_=capture();trials[name]=crop_trial(images);out[name]=measure(images)
        finally:
            for field,old in zip((model._opacity,model._scaling,model._xyz),saved,strict=True):field.copy_(old)
    restored,_=capture(); require(all(np.array_equal(a,b) for a,b in zip(base,restored,strict=True)), 'Restoration did not reproduce GPU baseline.')
    out['restored_bitwise']=True
    trials['restored']=crop_trial(restored)
    sheet=Image.new('RGB',(3*width,len(trials)*(height+18)), '#202020'); draw=ImageDraw.Draw(sheet)
    out['render_crop_sha256']={}
    for i,(label,tiles) in enumerate(trials.items()):
        draw.text((4,i*(height+18)+2),label,fill='white')
        for j,tile in enumerate(tiles):
            sheet.paste(Image.fromarray(tile).convert('RGB'),(j*width,i*(height+18)+18))
        out['render_crop_sha256'][label]=hashlib.sha256(b''.join(tile.tobytes() for tile in tiles)).hexdigest()
    path=output/f'{state["subject"]}-causal.png'; sheet.save(path);out['panel']=dict(path=str(path),sha256=sha(path))
    state['interventions']=out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    delivery=read(pin(DELIVERY,DELIVERY_SHA)); data=[prepare(s,delivery) for s in SPECS]
    for item in data:
        native_support(item)
        print(json.dumps(dict(subject=item.receipt['subject'],state='ranked_cpu',top=item.receipt['top'][:3],
            cpu_alpha_mae=item.receipt['cpu_alpha_mae'],predicted_alpha_fraction=item.receipt['shortlisted_predicted_contribution']/item.receipt['total_predicted_alpha'])),flush=True)
    if args.check:return
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(PROJECT/'src'))
    summaries={}
    with LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from shuolee_portrait import runtime as rt
        from shuolee_portrait.cli import gpu_lock
        rt.configure(ASSET)
        with gpu_lock(),torch.no_grad():
            require(torch.cuda.device_count()==1,'Expose exactly one CUDA device.')
            for item in data:
                render_trials(item,args.output)
                report=dict(schema='shuolee-round6-static-artifact-causal-v1',delivery_sha256=DELIVERY_SHA,
                    optimizer_updates=0,scope='synthetic static view; no native RGB truth; no source model mutation',**item.receipt)
                p=args.output/f'{item.receipt["subject"]}-result.json'
                with p.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
                summaries[item.receipt['subject']]=dict(path=str(p),sha256=sha(p),panel=item.receipt['interventions']['panel'],
                    view=item.receipt['view'],point_count=len(item.top))
                print('CAUSAL_COMPLETE',p,sha(p),json.dumps({k:v['hotspot']['alpha_fraction_removed'] for k,v in item.receipt['interventions'].items() if isinstance(v,dict) and 'hotspot' in v}),flush=True)
    receipt=args.output/'receipt.json'
    with receipt.open('x') as stream:
        json.dump(dict(schema='shuolee-round6-static-artifact-causal-v1',state='completed',delivery_sha256=DELIVERY_SHA,
            script_sha256=sha(Path(__file__)), optimizer_updates=0, subjects=summaries,
            scope='R6 static 45 and 90 degree causal diagnosis; no matching source RGB or quality sign-off'),stream,indent=2);stream.write('\n')
    print('RECEIPT',receipt,sha(receipt),flush=True)


if __name__=='__main__':main()
