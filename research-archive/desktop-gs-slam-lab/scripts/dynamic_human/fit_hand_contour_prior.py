"""Paired continuous MANO refinement with qualified weak photograph contours.

This fits a geometry prior, not the final GS appearance reconstruction.
"""
from pathlib import Path
import argparse,json,hashlib,time,shutil
import numpy as np,torch,cv2
from scipy.interpolate import BSpline
from PIL import Image,ImageDraw,ImageFont
from gsplat.rendering import rasterization
from fit_hamer_consensus import SharedHand,KNOTS,NCONTROL,project
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');H=L/'outputs/dynamic-human-hand-consensus-097';O=L/'outputs/dynamic-human-hand-contour-prior-102'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tt=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')

def coverage(vertices,faces,W,K):
    tri=vertices[faces];e=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];x=torch.nn.functional.normalize(e,dim=-1);normal=torch.nn.functional.normalize(torch.cross(e,e2,dim=-1),dim=-1);y=torch.cross(normal,x,dim=-1);F=torch.stack([x,y,normal],-1);means=tri.mean(1);edge=torch.sqrt((e.square().sum(-1)+e2.square().sum(-1)+(tri[:,2]-tri[:,1]).square().sum(-1))/3).clamp_min(1e-5);scales=torch.stack([.55*edge,.55*edge,.0825*edge],-1);cov=(F*scales.square()[:,None,:])@F.transpose(-1,-2)
    _,alpha,_=rasterization(means,None,None,torch.full((len(faces),),.8,device='cuda'),torch.ones((len(faces),1),device='cuda'),W[None],K[None],160,160,covars=cov,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
    q=means@W[:3,:3].T+W[:3,3];proj=q@K.T;uv=proj[:,:2]/proj[:,2:]
    return alpha[0,:,:,0],uv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['prepare','keypoint_control','contour_candidate'],required=True);args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(10201)
    protocol=read(O/'protocol.json');assert read(O/'constraint-visual-review.json')['approved_for_bounded_prior_fit']
    assert sha(H/'all_views/parameters.npz')==protocol['initial_parameters_sha256'];z=np.load(H/'all_views/initialization.npz');base=dict(np.load(H/'all_views/parameters.npz'));model=SharedHand(base);model.betas.requires_grad_(False);model.log_scale.requires_grad_(False);faces=torch.tensor(model.mano.faces,dtype=torch.long,device='cuda');B=tt(z['B']);W=tt(z['w2c']);K=tt(z['K']);xy=tt(z['xy']);weight=tt(z['weight']);pose0=model.pose.detach().clone();tr0=model.translation.detach().clone();data=[]
    for r in protocol['records']:
        p=O/r['input'].replace('\\','/');assert sha(p)==r['input_sha256'];a=np.load(p);outside=(1-cv2.dilate(a['mask'].astype(np.uint8),np.ones((7,7),np.uint8))).astype(np.uint8);distance=cv2.distanceTransform(outside,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        data.append(dict(record=r,rgb=a['rgb'],K=tt(a['K']),W=tt(a['w2c']),positive=torch.tensor(a['positive'],device='cuda'),negative=torch.tensor(a['negative'],device='cuda'),distal=tt(a['distal'])[None,None],distance=tt(distance)[None,None]))
    fit=[i for i,a in enumerate(data) if a['record']['fit']];assert len(fit)==18
    schedule=np.random.default_rng(10201).permutation(np.tile(np.array(fit),84))[:1500]
    scene_scale=1.664387822151184
    def evaluate():
        J,V,_,_=model(B);q=J@W[:,:3,:3].transpose(-1,-2)+W[:,None,:3,3];v=q@K.transpose(-1,-2);uv=v[:,:,:2]/v[:,:,2:].clamp_min(.1);return J,V,uv,q[:,:,2]
    def contour(V,d):
        a,uv=coverage(V[d['record']['source_index']],faces,d['W'],d['K']);a=a.clamp(1e-5,1-1e-5)
        pos=-a[d['positive']].log().mean();neg=-(1-a[d['negative']]).log().mean()
        grid=(uv/159*2-1)[None,None];dist=torch.nn.functional.grid_sample(d['distance'],grid,align_corners=True,padding_mode='border').flatten();valid=torch.nn.functional.grid_sample(d['distal'],grid,align_corners=True,padding_mode='zeros').flatten().detach()
        boundary=(dist.clamp_max(30).square()/25*valid).sum()/valid.sum().clamp_min(1)
        return pos+.25*neg+boundary,a,dict(positive_BCE=float(pos.detach()),negative_BCE=float(neg.detach()),outside_distance_penalty=float(boundary.detach()))
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12)
    if args.branch=='prepare':
        assert not (O/'smoke.json').exists();J,V,uv,depth=evaluate();loss,a,parts=contour(V,data[next(i for i,d in enumerate(data) if d['record']['camera']=='03' and d['record']['frame']==732)])
        gg=torch.autograd.grad(loss,[model.pose,model.translation]);assert all(torch.isfinite(g).all() and g.abs().max()>0 for g in gg)
        save(O/'smoke.json',dict(status='surrogate_coverage_and_pose_translation_gradients_finite',gradient_norms=[float(g.norm()) for g in gg],parts=parts,no_optimizer_steps=True,source_models_unchanged=True))
        with torch.no_grad():
            _,V,_,_=evaluate()
            for c in ['03','04','13']:
                d=next(d for d in data if d['record']['camera']==c and d['record']['frame']==732);a,_=coverage(V[d['record']['source_index']],faces,d['W'],d['K']);alpha=a.cpu().numpy();im=Image.new('RGB',(480,194),'white');draw=ImageDraw.Draw(im)
                for i,(arr,label) in enumerate([(d['rgb'],'Source'),(np.repeat(np.rint(alpha*255).astype(np.uint8)[...,None],3,-1),'Initial model coverage'),(np.where(d['positive'].cpu().numpy()[...,None],d['rgb'],128).astype(np.uint8),'Weak positive interior')]):im.paste(Image.fromarray(arr),(i*160,34));draw.text((i*160+3,4),label,font=font,fill='black');draw.text((i*160+3,18),c+'/732',font=font,fill='black')
                im.save(O/f'initial-{c}.png')
        print('Contour surrogate preflight passed; no optimizer steps.',flush=True);return
    assert read(O/'smoke.json')['no_optimizer_steps'];assert read(O/'smoke-visual-review.json')['ready_for_paired_fit']
    R=O/args.branch;R.mkdir(exist_ok=True);assert not (R/'summary.json').exists()
    grid=np.linspace(719,734,61);b1=tt(BSpline(KNOTS,np.eye(NCONTROL),3).derivative(1)(grid));b2=tt(BSpline(KNOTS,np.eye(NCONTROL),3).derivative(2)(grid));opt=torch.optim.Adam([dict(params=[model.pose],lr=.002),dict(params=[model.translation],lr=.0003)]);history=[];start=time.monotonic();np.save(R/'schedule.npy',schedule);steps=protocol['steps_each'];save(R/'protocol.json',dict(**protocol,branch=args.branch,pose_lr=.002,translation_lr=.0003,shape_scale_fixed=True,additional_distance_loss='Mean squared distance outside3pxdilatedmask at projectedtrianglecenters, /25px^2, distalweighted. Provides attraction when surrogatealpha has no overlap. Samecoefficient8withBCE in candidate only.',pose_regularizer_reference='starting097learnedpose',optimization_state='newAdam forbothbranches'))
    for step in range(steps):
        opt.zero_grad(set_to_none=True);J,V,uv,depth=evaluate();err2=(uv-xy).square().sum(-1);reproj=(weight*16*torch.log1p(err2/16)).sum()/weight.sum();cp=model.pose.reshape(NCONTROL,-1);reg=2*(b2@cp).square().mean()+.05*(b1@cp).square().mean()+2000*(b2@model.translation).square().mean()+.3*(model.pose[:,1:]-pose0[:,1:]).square().mean()+100*torch.relu(.2-depth).square().mean()
        loss=reproj+reg;cl=torch.zeros((),device='cuda')
        if args.branch=='contour_candidate':cl,_,_=contour(V,data[int(schedule[step])]);loss=loss+8*cl
        assert torch.isfinite(loss);loss.backward();assert torch.isfinite(model.pose.grad).all() and torch.isfinite(model.translation.grad).all();torch.nn.utils.clip_grad_norm_([model.pose,model.translation],100);opt.step()
        with torch.no_grad():model.pose.copy_(torch.maximum(torch.minimum(model.pose,pose0+.35),pose0-.35));model.translation.copy_(torch.maximum(torch.minimum(model.translation,tr0+.04*scene_scale),tr0-.04*scene_scale))
        factor=.2+.8*(1-(step+1)/steps)
        for group,lr in zip(opt.param_groups,[.002,.0003]):group['lr']=lr*factor
        if step==0 or (step+1)%100==0:
            row=dict(step=step+1,loss=float(loss),robust_reprojection=float(reproj),regularizer=float(reg),contour_loss=float(cl),candidate_keypoint_median_px=float(torch.sqrt(err2).median()),elapsed_seconds=time.monotonic()-start);history.append(row);save(R/'progress.json',row);print(args.branch,json.dumps(row),flush=True)
    state={k:getattr(model,k).detach().cpu().numpy() for k in ['root_base','pose','translation','betas','log_scale']};assert np.array_equal(state['betas'],base['betas'].astype(np.float32)) and np.array_equal(state['log_scale'],base['log_scale'].astype(np.float32));np.savez(R/'parameters.npz',**state,knots=KNOTS)
    with torch.no_grad():J,V,uv,depth=[a.cpu().numpy() for a in evaluate()]
    old=np.load(H/'all_views/predictions.npz');np.savez_compressed(R/'predictions.npz',camera=old['camera'],frame=old['frame'],capture_time=old['capture_time'],joints_world=J,vertices_world=V,projected_xy=uv,depth=depth)
    save(R/'summary.json',dict(status='1500continuous_prior_updates_complete_native_qualification_pending',branch=args.branch,steps=steps,elapsed_seconds=time.monotonic()-start,parameters_sha256=sha(R/'parameters.npz'),max_pose_coefficient_change_rad=float(abs(state['pose']-base['pose']).max()),max_translation_coefficient_change_world=float(np.linalg.norm(state['translation']-base['translation'],axis=-1).max()),keypoint_candidate_median_px=float(np.median(np.linalg.norm(uv-z['xy'],axis=-1))),shape_scale_exact=True,Gaussian_updates=False,goal_complete=False));save(R/'history.json',history);shutil.copy2(__file__,R/Path(__file__).name);print('PAIRED_PRIOR_COMPLETE',args.branch,flush=True)
if __name__=='__main__':main()
