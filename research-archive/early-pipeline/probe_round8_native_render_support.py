"""Diagnose the exact R6 causal groups through native SH0 color derivatives."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_smpl_delivery_round6_world_scale import mesh_front_depth
from measure_smpl_delivery_round5_ear016 import pin, read, require, sha


ROOT = Path(__file__).resolve().parents[1]/'outputs'
DELIVERY = ROOT/'smpl-delivery-round6/receipt.json'
DELIVERY_SHA = 'a728539832d89b4f4d11b644d1d2398f1eedb157f13e021eaf2234e3b5658c1c'
R7_RECEIPT = ROOT/'smpl-delivery-round7/receipt.json'
R7_SHA = '465196daa6a388809b8c57b2f050a6ea97f5f59141cedfd7c53e0a9a400f7f4a'
CAUSAL = R7_RECEIPT.parent/'causal-diagnostic/receipt.json'
CAUSAL_SHA = '8d44e129dad6bfdacdb38dcb1c794723c930d4739b26e450df10f6f314a73939'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-scale-20261111')
CANDIDATE = Path('/mnt/d/Projects/Shuolee-smpl-support-20261111/src/shuolee_portrait/render_support.py')
ASSET = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
LEASE = Path('/tmp/shuolee-native122-gpu.lock')
REGIONS = dict(reliable_foreground=.99,reliable_background=.01,boundary_px=2,exterior_boundary_px=3)
FIELDS = ('reliable_fg','reliable_bg','total')
PANEL_FRAMES = {'016':'frame_000051','006':'frame_000044'}


def color_weights(draw, inside, outside, count):
    import torch

    with torch.enable_grad():
        color = torch.zeros((count,3),device=inside.device,dtype=torch.float32,requires_grad=True)
        rgb = draw(color)
        require(rgb.shape == (3,*inside.shape) and inside.shape == outside.shape, 'Unexpected RGB override or mask shape.')
        loss = rgb[0][inside].sum()+rgb[1][outside].sum()+rgb[2].sum()
        (weights,) = torch.autograd.grad(loss,color)
    require(torch.isfinite(weights).all() and (weights >= 0).all(), 'Invalid raster contribution weights.')
    return weights.detach()


def synthetic(render, model_type, camera, pipe, black, candidate):
    import torch

    model = model_type(0)
    cam,_ = camera(dict(frame='synthetic_overlap',wh=[33,33],K=[[28,0,16.5],[0,28,16.5],[0,0,1]],
                        w2c=np.eye(4).tolist()),0,pose_index=0)
    model._xyz = torch.tensor([[0.,0.,2.],[0.,0.,2.5]],device='cuda')
    model._rotation = torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]],device='cuda')
    model._scaling = torch.full((2,3),.10,device='cuda').log()
    opacity = torch.tensor([[.6],[.8]],device='cuda')
    model._opacity = opacity.logit()
    inside = torch.zeros((33,33),device='cuda',dtype=torch.bool); inside[16,16] = True
    outside = torch.zeros_like(inside); outside[16,17] = True

    def draw(color):
        return render(cam,model,pipe,black,override_color=color)['render']

    def objective(rgb):
        return torch.stack((rgb[0][inside].sum(),rgb[1][outside].sum(),rgb[2].sum()))

    gradient = color_weights(draw,inside,outside,2)
    with torch.no_grad():
        shared = candidate(draw,inside,outside,2)
    torch.testing.assert_close(shared,gradient,rtol=2e-6,atol=2e-6)
    expected = torch.tensor([.6,.8*(1-.6)],device='cuda')
    torch.testing.assert_close(gradient[:,0],expected,rtol=0,atol=2e-6)
    epsilon = .02
    with torch.no_grad():
        center = torch.tensor([[.24,.38,.32],[.51,.43,.56]],device='cuda')
        derivatives = []
        for row in range(2):
            delta = torch.zeros_like(center); delta[row] = epsilon
            derivatives.append((objective(draw(center+delta))-objective(draw(center-delta)))/(2*epsilon))
        finite = torch.stack(derivatives)
        torch.testing.assert_close(gradient,finite,rtol=8e-5,atol=2e-5)
        model._opacity[0] = -100.
    occluder_hidden = color_weights(draw,inside,outside,2)
    torch.testing.assert_close(occluder_hidden[:,0],torch.tensor([0.,.8],device='cuda'),rtol=0,atol=2e-6)
    with torch.no_grad():
        model._opacity.copy_(opacity.logit())
    restored = color_weights(draw,inside,outside,2)
    torch.testing.assert_close(restored,gradient,rtol=2e-6,atol=2e-6)
    return dict(state='validated',fixture='two concentric visible Gaussian centers at z=2 m and z=2.5 m',
        expected_center_front_back=expected.tolist(),measured_center_front_back=gradient[:,0].tolist(),
        occluder_hidden_center_front_back=occluder_hidden[:,0].tolist(),
        channel_fields=list(FIELDS),override_color_gradient=gradient.tolist(),central_finite_difference=finite.tolist(),
        finite_difference_epsilon=epsilon,finite_difference_max_abs=float((gradient-finite).abs().max()),
        candidate_helper_under_no_grad_max_abs=float((shared-gradient).abs().max()),
        restored_max_abs=float((restored-gradient).abs().max()))


def prepare(subject, delivery, causal):
    published = delivery['subjects'][subject]
    prior = causal['subjects'][subject]
    source = read(pin(prior['path'],prior['sha256']))
    path = pin(published['result'],published['result_sha256']); root=path.parent; result=read(path)
    replay_path = pin(root/result['bound_replay'],result['bound_replay_sha256']); replay=read(replay_path)
    snapshot = root/'gaussian-run'/result['final_snapshot']
    manifest = read(pin(snapshot/'bound-replay.json',replay['bound_replay_sha256']))
    require(result['state']=='completed' and snapshot.name=='step-010000' and replay['state']=='completed'
            and source['bound_ply_sha256']==manifest['bound_ply_sha256'] and source['mesh_sha256']==replay['mesh_sha256']
            and source['review_sha256']==published['review_sha256'] and source['ply_sha256']==result['sha256'], 'Frozen R6 causal/model identity differs.')
    rows_path = pin(root/'data/training-inputs.json',replay['training_inputs_sha256'])
    rows = read(rows_path)['records']
    cameras_path = pin(root/'gaussian-run/training-cameras.json',replay['training_camera_sha256'])
    cameras = read(cameras_path)['records']
    with np.load(pin(replay_path.parent/replay['mesh_file'],replay['mesh_sha256']),allow_pickle=False) as archive:
        mesh = dict(archive)
    ids = [row['frame'] for row in rows]
    require(len(rows)==len(cameras)==len(replay['records'])==len(set(ids))==24 and set(ids).isdisjoint(replay['retained_holdouts'])
            and ids==mesh['frame_ids'].tolist() and ids==[row['frame'] for row in replay['records']], 'The original registered source roster differs.')
    for index,(row,camera,pose) in enumerate(zip(rows,cameras,replay['records'],strict=True)):
        require(camera==pose['camera'] and row['frame']==camera['frame'] and row['source_frame_index']==pose['source_frame_index']
                and row['source_sha256']==pose['source_sha256']==mesh['image_sha256'][index]
                and row['mask_sha256']==pose['mask_sha256']==mesh['mask_sha256'][index]
                and camera['pose_index']==pose['pose_index']==int(mesh['pose_indices'][index])==index
                and all(row[k]==camera[k] for k in ('K','w2c','wh')), 'Source RGB, mask, camera or pose identity changed.')
        pin(row['image'],row['source_sha256']); pin(row['mask'],row['mask_sha256'])
    selected = np.asarray([point['index'] for point in source['top']],dtype=np.int64)
    require(len(selected)==prior['point_count']==len(np.unique(selected)) and selected.min()>=0 and selected.max()<manifest['points'],
            'Invalid R6 causal row indices.')
    return SimpleNamespace(subject=subject,root=root,snapshot=snapshot,source=source,causal_sha=prior['sha256'],
        result_sha=published['result_sha256'],result=result,manifest=manifest,replay=replay,rows=rows,cameras=cameras,mesh=mesh,selected=selected)


def center_proxy(data, slot, xyz, inside, outside):
    rec=data.cameras[slot]; E,K=(np.asarray(rec[k]) for k in ('w2c','K')); v=data.mesh['vertices_world'][slot]
    p=(xyz@E[:3,:3].T+E[:3,3])@K.T; z=p[:,2]; uv=p[:,:2]/z[:,None].clip(.01)
    selected=np.flatnonzero((z>.2)&(uv>=0).all(1)&(uv<rec['wh']).all(1)); pixel=np.floor(uv[selected]).astype(int)
    vq=(v@E[:3,:3].T+E[:3,3])@K.T
    depth=mesh_front_depth(vq[:,:2]/vq[:,2:3].clip(.01),vq[:,2],data.mesh['faces'],pixel+.5)
    visible=z[selected]<=depth+.025
    x,y=pixel.T
    flags=np.zeros((len(xyz),3),dtype=bool)
    flags[selected,0]=visible; flags[selected,1]=visible&inside[y,x]; flags[selected,2]=visible&outside[y,x]
    return flags


def panel(path, row, total, selected, outside):
    with Image.open(row['image']) as image:
        rgb=np.asarray(image.convert('RGB'))
    visible=total>.01; y,x=np.nonzero(visible)
    require(len(x)>0,'No native subject alpha in diagnostic panel.')
    h,w=visible.shape; box=(max(0,int(x.min())-12),max(0,int(y.min())-12),min(w,int(x.max())+13),min(h,int(y.max())+13))
    l,t,r,b=box; width,height=r-l,b-t
    result=Image.new('RGB',(4*width,height+20),'#202020');draw=ImageDraw.Draw(result)
    for index,(label,values) in enumerate((('Source RGB',rgb),('All Gaussian alpha',total),
          ('Causal group visible alpha',selected),('Causal group reliable BG alpha',np.where(outside,selected,0)))):
        view=values[t:b,l:r]
        if view.dtype!=np.uint8: view=np.rint(np.clip(view,0.,1.)*255).astype('uint8')
        result.paste(Image.fromarray(view).convert('RGB'),(index*width,20));draw.text((index*width+3,3),label,fill='white')
    result.save(path)
    return dict(path=str(path),sha256=sha(path),frame=row['frame'],native_crop_xyxy=list(box),
                source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],contribution_scale='linear, no normalization')


def audit(data, output, render, camera, pipe, black, weigh):
    import torch
    from shuolee_portrait.bound_replay import RAW_PARAMETERS, load_bound_snapshot
    from shuolee_portrait.coverage_refinement import _reliable_regions

    model,manifest=load_bound_snapshot(data.root,data.snapshot,asset_path=ASSET/'smplx/SMPLX_NEUTRAL.npz')
    require(manifest==data.manifest and model.optimizer is None,'Unexpected bound model or optimizer.')
    ids=torch.as_tensor(data.selected,device='cuda',dtype=torch.long)
    require(model.binding[ids].tolist()==[point['triangle'] for point in data.source['top']], 'Exact causal triangle IDs changed.')
    def state_hash():
        return {name:hashlib.sha256(getattr(model,name).detach().cpu().numpy().tobytes()).hexdigest() for name in RAW_PARAMETERS}
    frozen=state_hash(); frames=[]; contributions=[]; proxies=[]; panel_record=None
    for slot,row in enumerate(data.rows):
        cam,rec=camera(row,slot,pose_index=slot)
        require(rec==data.cameras[slot], 'Replayed camera differs from pinned original.')
        with Image.open(row['mask']) as image:
            require(image.size==tuple(row['wh']), 'Native matte dimensions differ.')
            matte=torch.as_tensor(np.array(image.convert('L')),device='cuda')[None].float()/255.
        inside,outside=_reliable_regions(matte,REGIONS)
        require(inside.any() and outside.any() and not (inside&outside).any(),'Reliable native masks invalid.')
        model.select_mesh_by_timestep(slot)
        def draw(color):
            return render(cam,model,pipe,black,override_color=color)['render']
        with torch.no_grad():
            weights=weigh(draw,inside,outside,len(model.binding))
            require(torch.isfinite(weights).all() and (weights>=0).all(), 'Invalid raster contribution weights.')
            colors=torch.zeros_like(model.get_xyz); colors[:,0]=1;colors[ids,1]=1
            actual=draw(colors)
            def region_sums(alpha):
                return torch.stack((alpha[inside].sum(dtype=torch.float64),alpha[outside].sum(dtype=torch.float64),alpha.sum(dtype=torch.float64)))
            selected=weights[ids]; group=selected.sum(dim=0,dtype=torch.float64); whole=weights.sum(dim=0,dtype=torch.float64)
            direct_all,direct_group=region_sums(actual[0]),region_sums(actual[1])
            torch.testing.assert_close(whole,direct_all,rtol=2e-4,atol=2e-3)
            torch.testing.assert_close(group,direct_group,rtol=2e-4,atol=2e-3)
            centers=model.get_xyz[ids].cpu().numpy()
            a,b=inside.cpu().numpy(),outside.cpu().numpy()
            flags=center_proxy(data,slot,centers,a,b)
            contributions.append(selected.cpu().numpy().astype(float));proxies.append(flags)
            frames.append(dict(frame=row['frame'],pose_index=slot,source_frame_index=row['source_frame_index'],
                source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],camera=rec,reliable_pixels=[int(a.sum()),int(b.sum())],
                all_weight_sums=dict(zip(FIELDS,whole.tolist(),strict=True)),causal_group_weight_sums=dict(zip(FIELDS,group.tolist(),strict=True)),
                override_group_absolute_errors=dict(zip(FIELDS,(group-direct_group).abs().tolist(),strict=True)),
                override_all_absolute_errors=dict(zip(FIELDS,(whole-direct_all).abs().tolist(),strict=True)),
                causal_group_center_proxy_counts=dict(zip(('depth_tested','reliable_fg','reliable_bg'),flags.sum(0).tolist(),strict=True))))
            if row['frame']==PANEL_FRAMES[data.subject]:
                panel_record=panel(output/f'{data.subject}-native-support.png',row,actual[0].cpu().numpy(),actual[1].cpu().numpy(),b)
    require(panel_record and state_hash()==frozen and all(getattr(model,k).grad is None for k in RAW_PARAMETERS),
            'Frozen geometry, opacity or color state changed during diagnosis.')
    values,flags=np.stack(contributions),np.stack(proxies)
    summary=values.sum((0,1)); by_point=values.sum(0); missed=(~flags[...,0])[...,None]*values
    points=[]
    for i,(point,weights,old) in enumerate(zip(data.selected,by_point,data.source['top'],strict=True)):
        points.append(dict(index=int(point),triangle=old['triangle'],original_r7_raw_matte_proxy=old['native_support'],
            center_proxy_counts=dict(zip(('depth_tested','reliable_fg','reliable_bg'),flags[:,i].sum(0).tolist(),strict=True)),
            render_weight_sums=dict(zip(FIELDS,weights.tolist(),strict=True)),
            render_weights_in_center_not_depth_visible_views=dict(zip(FIELDS,missed[:,i].sum(0).tolist(),strict=True)),
            positive_weight_frame_counts=dict(zip(FIELDS,(values[:,i]>1e-6).sum(0).tolist(),strict=True)),
            per_frame=[dict(frame=frame['frame'],weights=dict(zip(FIELDS,w.tolist(),strict=True)),
                       center_proxy=dict(zip(('depth_tested','reliable_fg','reliable_bg'),f.tolist(),strict=True)))
                       for frame,w,f in zip(frames,values[:,i],flags[:,i],strict=True)]))
    no_visible=~flags[...,0].any(0)
    return dict(schema='shuolee-round8-native-render-support-diagnostic-v1',state='completed',subject=data.subject,
        causal_sha256=data.causal_sha,r6_result_sha256=data.result_sha,bound_ply_sha256=manifest['bound_ply_sha256'],
        posed_mesh_sha256=data.replay['mesh_sha256'],training_inputs_sha256=data.replay['training_inputs_sha256'],
        training_camera_sha256=data.replay['training_camera_sha256'],retained_holdouts=data.replay['retained_holdouts'],
        frozen_gaussian_raw_sha256=frozen,no_gaussian_state_change=True,optimizer_updates=0,frame_count=len(frames),
        masks=REGIONS,center_proxy='pixel-center nearest CPU mesh ray, 25 mm occlusion slack, same reliable masks; distinct from raster contribution',
        contribution_units='unnormalized effective alpha-pixels, sum(T_i*alpha_i) from stock color derivative',
        totals=dict(causal_group_weights=dict(zip(FIELDS,summary.tolist(),strict=True)),
          causal_group_weights_in_center_not_depth_visible_views=dict(zip(FIELDS,missed.sum((0,1)).tolist(),strict=True)),
          no_depth_visible_center_rows=int(no_visible.sum()),no_depth_visible_center_rows_with_positive_fg=int((no_visible&(by_point[:,0]>1e-6)).sum()),
          no_depth_visible_center_row_weights=dict(zip(FIELDS,by_point[no_visible].sum(0).tolist(),strict=True))),
        frames=frames,points=points,panel=panel_record,quality_accepted=False)


def write(path, report):
    with path.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,allow_nan=False);f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    delivery=read(pin(DELIVERY,DELIVERY_SHA)); causal=read(pin(CAUSAL,CAUSAL_SHA)); r7=read(pin(R7_RECEIPT,R7_SHA))
    require(causal['state']=='completed' and causal['delivery_sha256']==DELIVERY_SHA and r7['baseline_commit']==delivery['commit'],
            'R6/R7 diagnostic lineage differs.')
    data=[prepare(subject,delivery,causal) for subject in ('016','006')]
    if args.check:
        print(json.dumps({item.subject:dict(poses=len(item.rows),causal_rows=item.selected.tolist()) for item in data}),flush=True);return
    start=time.monotonic();sys.path.insert(0,str(PROJECT/'src'))
    with LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from shuolee_portrait import runtime as rt
        from shuolee_portrait.cli import gpu_lock
        from shuolee_portrait.training import camera
        rt.configure(ASSET)
        from gaussian_renderer import render
        from scene.gaussian_model import GaussianModel
        with gpu_lock():
            require(torch.cuda.device_count()==1,'Expose exactly one CUDA device.')
            helper_sha=sha(CANDIDATE)
            spec=importlib.util.spec_from_file_location('r8_candidate_render_support',CANDIDATE)
            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
            pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
            black=torch.zeros(3,device='cuda'); control=synthetic(render,GaussianModel,camera,pipe,black,helper.color_weights)
            print('SYNTHETIC_VALIDATED',json.dumps(control),flush=True)
            args.output.mkdir(parents=True,exist_ok=False)
            subjects={}
            for item in data:
                report=audit(item,args.output,render,camera,pipe,black,helper.color_weights)
                path=args.output/f'{item.subject}-result.json';write(path,report)
                subjects[item.subject]=dict(path=str(path),sha256=sha(path),panel=report['panel'],frame_count=report['frame_count'],totals=report['totals'])
                print('SUBJECT_COMPLETE',item.subject,json.dumps(report['totals']),sha(path),flush=True)
    path=args.output/'receipt.json'
    write(path,dict(schema='shuolee-round8-native-render-support-diagnostic-v1',state='completed',
        r6_delivery_sha256=DELIVERY_SHA,r7_delivery_sha256=R7_SHA,r7_causal_receipt_sha256=CAUSAL_SHA,
        candidate_color_helper_path=str(CANDIDATE),candidate_color_helper_sha256=helper_sha,
        script_sha256=sha(Path(__file__)),scope='Frozen R6 causal groups on 24 selected native source cameras per subject, no optimization or data mutation',
        synthetic=control,subjects=subjects,optimizer_updates=0,quality_accepted=False,wall_seconds=time.monotonic()-start))
    print('RECEIPT',path,sha(path),flush=True)


if __name__=='__main__':main()
