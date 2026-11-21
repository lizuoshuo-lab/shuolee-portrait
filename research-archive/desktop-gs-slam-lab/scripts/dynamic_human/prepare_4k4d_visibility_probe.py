"""Derive the frozen visibility control from the already audited color probe."""
from pathlib import Path
LAB=Path(__file__).resolve().parents[2];S=LAB/'scripts/dynamic_human'
t=(S/'diagnose_4k4d_source_support.py').read_text(encoding='utf8')
t=t.replace('dynamic-human-4k4d-projection-control-040/original_projection','dynamic-human-4k4d-source-gate-training-044/source_support_blending')
t=t.replace('dynamic-human-4k4d-source-support-042','dynamic-human-4k4d-visibility-047').replace("==2000","==3000").replace('2000,2000','3000,3000')
t=t.replace("from easyvolcap.utils.sh_utils import eval_sh", "from easyvolcap.utils.sh_utils import eval_sh\nfrom easyvolcap.utils.base_utils import dotdict\nfrom audit_enerf_input import read_cameras\nfrom fourkd_source_support import enable_source_foreground_support\nimport torch.nn.functional as F")
t=t.replace("'source_foreground_gate_unpremultiply'", "'source_visibility2'").replace("'source_foreground_gate'", "'source_visibility1'")
t=t.replace("Reweight original learned blending weights by source-mask support>=0.5; second variant also unpremultiplies sampled sourceRGB by sampled mask. Original SH residual retained.",
'''Reweight existing mask-supported blend weights by model-derived source visibility. Native zbuffer first point with radial alpha>0.25. Point visible if z<=frontdepth+tau*worldradius, tau1 or2. Nearest pixel sampling. No visible source falls back to original supported blend. Original SH residual retained.''')
t=t.replace('dataset=DATASETS.build(cfg.val_dataloader_cfg.dataset_cfg);captured={}', '''enable_source_foreground_support(model.sampler)
dataset=DATASETS.build(cfg.val_dataloader_cfg.dataset_cfg);captured={}
cameras=read_cameras(Path('/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6/optimized'))
geo=np.load(LAB/'outputs/dynamic-human-4k4d-point-footprints-046/geometry.npz')
X=torch.tensor(geo['xyz'],device='cuda')[None];radius=torch.tensor(geo['radius'],device='cuda')[None,:,None];opacity=torch.tensor(geo['opacity'],device='cuda')[None,:,None]
visibility={};zbuffer_audit=[]
with torch.no_grad():
    for c in range(18):
        if c in [0,6,12]:continue
        cam=cameras[f'{c:02d}'];K=torch.tensor(cam['K'],dtype=torch.float32,device='cuda')[None];w2c=torch.tensor(cam['w2c'],dtype=torch.float32,device='cuda')
        b=dotdict(K=K,R=w2c[:3,:3][None],T=w2c[:3,3][None,:,None],meta=dotdict(H=torch.tensor([1080]),W=torch.tensor([1920])))
        idx,alpha=model.sampler.render_pytorch(X,torch.zeros_like(X),radius,opacity,b,return_frags=True)
        valid=alpha>.25;first=valid.to(torch.int32).argmax(-1);has=valid.any(-1)
        pidx=idx.gather(-1,first[...,None])[...,0]
        pc=X@b.R.mT+b.T.mT;depth=pc[...,2]
        front=depth[0,pidx];front=torch.where(has,front,0)
        uv=pc@K.mT;uv=uv[...,:2]/uv[...,2:]
        grid=(uv+.5)/torch.tensor([1920,1080],device='cuda')*2-1
        front_at=F.grid_sample(front[:,None],grid[:,None],align_corners=False,mode='nearest',padding_mode='zeros')[0,0,0]
        inside=(uv[0,:,0]>=0)&(uv[0,:,0]<1920)&(uv[0,:,1]>=0)&(uv[0,:,1]<1080)&(depth[0]>0)&(front_at>0)
        visibility[c]=torch.stack([inside&(depth[0]<=front_at+tau*radius[0,:,0]) for tau in [1.,2.]])
        np.savez_compressed(OUT/f'source{c:02d}-visibility.npz',front_depth=front[0].cpu().numpy(),point_depth=depth[0].cpu().numpy(),point_uv=uv[0].cpu().numpy(),sampled_front=front_at.cpu().numpy(),inside=inside.cpu().numpy(),visible=visibility[c].cpu().numpy())
        zbuffer_audit.append(dict(camera=c,occupied_pixels=int(has.sum()),visible_points=[int(v.sum()) for v in visibility[c]]))
        print('SOURCE_DEPTH',c,zbuffer_audit[-1],flush=True)
        del idx,alpha,valid,first,has,pidx,front,pc,depth,uv,grid,front_at,b
(OUT/'source-depth-summary.json').write_text(json.dumps(zbuffer_audit,indent=2)+'\\n')
del X,radius,opacity;torch.cuda.empty_cache()
''')
start=t.index('    original=((weights*colors)')
end=t.index('    full=self.render_pytorch', start)
t=t[:start]+'''    masks=compute_src_inps(batch,'src_msks')
    masks=pad_image(masks,size=batch.persistent.src_feat_rgb.shape[-2:])
    support=sample_geometry_feature_image(pcd,masks,batch.src_exts,batch.src_ixts,batch.persistent.src_scale,padding_mode='zeros')
    supported=support>=.5;gated=weights*supported;denom=gated.sum(-3,keepdim=True);usable=denom>1e-7
    norm=torch.where(usable,gated/denom.clamp_min(1e-7),weights)
    original=((norm*colors).sum(-3)+residual).clamp(0,1)
    assert (original-rgb).abs().max()<2e-6
    assert np.allclose(pcd[0].cpu().numpy(),geo['xyz'],atol=1e-7)
    variants=[];stats={};captured['visibility_masks']={}
    for n,key in enumerate(['source_visibility1','source_visibility2']):
        vis=torch.stack([visibility[c][n] for c in sources])[None,...,None]
        candidate=norm*vis;total=candidate.sum(-3,keepdim=True)
        vweights=torch.where(total>1e-7,candidate/total.clamp_min(1e-7),norm)
        variants.append((key,((vweights*colors).sum(-3)+residual).clamp(0,1)))
        stats[key]=dict(no_visible_source=int((total<=1e-7).sum()),mean_rejected_weight=float((norm*(~vis)).sum(-3).mean()))
        captured['visibility_masks']['sample_'+key]=vis[0,:,::257,0].cpu().numpy()
''' +t[end:]
t=t.replace("captured['sample_weights']=weights", "captured['statistics']['visibility']=stats\n    captured['sample_weights']=norm")
t=t.replace('sample_support=captured[\'sample_support\'])',"sample_support=captured['sample_support'],**captured['visibility_masks'])")
t=t.replace('Source-mask support','Source visibility 1r').replace('Support + unpremultiply','Source visibility 2r')
t=t.replace('Original rendering','Support blend 3000')
t=t.replace('six_readonly_source_support_controls_complete','six_readonly_source_visibility_controls_complete')
(S/'diagnose_4k4d_visibility.py').write_text(t,encoding='utf8')
compile(t,'diagnose_4k4d_visibility.py','exec')
print('047 frozen visibility control prepared')
