"""Decompose the actual photo objective into hand-context and remaining pixels.

No optimizer steps. SSIM windows are assigned by center; their original11px
support and full-image denominators are preserved. This is a gradient analysis,
not an Adam update simulation or a causal explanation of reconstruction errors.
"""
from pathlib import Path
import json,hashlib,time,shutil
import numpy as np,torch
import torch.nn.functional as F
from train_enerf_camera10_time import ss
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_fine_model import HandFineGraph,HandFineRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';J=L/'outputs/dynamic-human-enerf-hand-joint-084';P=L/'outputs/dynamic-human-enerf-hand-fine-085';H=L/'outputs/dynamic-human-enerf-hand-motion-082';O=L/'outputs/dynamic-human-enerf-hand-gradient-087';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
assert not (O/'summary.json').exists();torch.set_num_threads(8);torch.manual_seed(8701)
save(O/'protocol.json',dict(status='fixed_before_gradient_analysis',checkpoints=['079parentwithzerofine','085fine_photo'],contexts='105originaleven-timeknowncameraobservations',source_ROI='Unchanged082source-defined160x160right-handcontextrectangle; includesotherobjects,notsemanticGT.',background='Fixedgray.5forbothcheckpoints;diagnostic differsfromrandomtrainingbackground.',loss='Original.8RGBL1+.2(1-SSIM)+.1alphaL1. Decompose RGB/alpha pixels andSSIMwindowcenters intorectangle andcomplement, preservingglobaldenominators. Regularizerseparate.',gradient_groups='6selectedGaussian families,4activecoarsefamilies,4finefamilies; eachfamilyseparately; no mixed-unitgradientnorm.',qualification='Nooptimizersteps;gradientcosinesareEuclideanparametercoordinates,notAdamupdateorcausalproof. Noheldoutpixels.',goal_complete=False))
by={(r['camera'],r['frame']):r for r in read(H/'observations.json')['records']};contexts=[r for r in read(S/'input-audit.json')['records'] if r['split']=='train'];assert len(contexts)==105
obs=[]
for r in contexts:
    c,t=r['camera'],r['frame'];path=S/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(path)==r['input_sha256']==by[c,t]['input_sha256'];z=np.load(path);x0,y0,x1,y1=map(int,z['bbox']);hx0,hy0,hx1,hy1=by[c,t]['crop'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;roi=np.zeros((y1-y0,x1-x0),np.float32);roi[max(hy0-y0,0):min(hy1-y0,y1-y0),max(hx0-x0,0):min(hx1-x0,x1-x0)]=1;assert roi.sum()>20000
    obs.append(dict(camera=c,frame=t,roi=tensor(roi)[...,None],rgb=tensor(z['rgb'][y0:y1,x0:x1]/255),matte=tensor(z['matte'][y0:y1,x0:x1])[...,None],valid=tensor(z['valid'][y0:y1,x0:x1])[...,None],K=tensor(K),W=tensor(z['w2c']),width=x1-x0,height=y1-y0,source_roi=by[c,t]['crop'],input_sha256=r['input_sha256']))
def ssim_map(a,b):
    v=torch.arange(11,device=a.device,dtype=a.dtype)-5;v=torch.exp(-v*v/(2*1.5**2));v/=v.sum();w=(v[:,None]*v[None,:]).expand(3,1,11,11);mu=F.conv2d(a,w,groups=3);mv=F.conv2d(b,w,groups=3);va=F.conv2d(a*a,w,groups=3)-mu*mu;vb=F.conv2d(b*b,w,groups=3)-mv*mv;cov=F.conv2d(a*b,w,groups=3)-mu*mv
    return ((2*mu*mv+.01**2)*(2*cov+.03**2))/((mu*mu+mv*mv+.01**2)*(va+vb+.03**2))
def metrics(a,b,c):
    a=a.double().flatten();b=b.double().flatten();c=c.double().flatten();full=a+b+c;norm=lambda x:float(x.norm());cos=lambda x,y:float(torch.dot(x,y)/(x.norm()*y.norm()).clamp_min(1e-30)) if x.norm()>1e-20 and y.norm()>1e-20 else None
    return dict(roi_norm=norm(a),rest_norm=norm(b),regularizer_norm=norm(c),roi_rest_cosine=cos(a,b),roi_regularizer_cosine=cos(a,c),roi_total_cosine=cos(a,full),roi_rest_norm_ratio=norm(a)/max(norm(b),1e-30),regularizer_photo_norm_ratio=norm(c)/max(norm(a+b),1e-30))
results={};started=time.monotonic()
for branch in ['parent','fine_photo']:
    root=S/'local_bspline' if branch=='parent' else P/'fine_photo';path=root/'resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);names=['means','scales','quats','opacities','sh0','shN'];p={k:torch.nn.Parameter(v.cuda(),requires_grad=k in names) for k,v in cp['params'].items()};active=torch.from_numpy(np.load(J/'support/active-gaussians.npy')).cuda();nodes=torch.from_numpy(np.load(J/'support/active-nodes.npy')).cuda();g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);ref=GraphMotion(S/'graph.npz',cp['scene_scale']);ref.load_state_dict(cp['graph'] if branch=='parent' else cp['reference_graph']);ref.requires_grad_(False);fine=HandFineGraph(P/'fine-graph.npz',cp['scene_scale'])
    if branch!='parent':fine.load_state_dict(cp['fine_graph'])
    model=HandFineRenderer(S/'head/capture-affines.npz',g,ref,active,fine);families=[('gaussian.'+k,p[k],active) for k in names]+[('coarse.'+k,v,nodes) for k,v in g.named_parameters()]+[('fine.'+k,v,torch.ones(32,device='cuda',dtype=torch.bool)) for k,v in fine.named_parameters()];parameters=[v for k,v,m in families];sums={k:[torch.zeros_like(v[m],dtype=torch.float64) for _ in range(3)] for k,v,m in families};rows=[];maxloss=0.;maxgrad=0.
    for n,o in enumerate(obs):
        rgb,alpha,info=model.render(p,o['camera'],o['frame'],o['W'],o['K'],o['width'],o['height'],2);pred=(rgb[0]+.5*(1-alpha[0]))*o['valid'];target=(o['rgb']*o['matte']+.5*(1-o['matte']))*o['valid'];err=(pred-target).abs();ae=(alpha[0]-o['matte']).abs()*o['valid'];sm=ssim_map(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]);roi=o['roi'];sr=roi[5:-5,5:-5].permute(2,0,1)[None]
        local=.8*(err*roi).mean()+.2*((1-sm)*sr).mean()+.1*(ae*roi).mean();rest=.8*(err*(1-roi)).mean()+.2*((1-sm)*(1-sr)).mean()+.1*(ae*(1-roi)).mean();reg=10*(info['graph_rigidity']+info['fine_graph_rigidity'])+.001*(info['graph_rotation_size']+info['fine_graph_rotation_size'])+.1*(info['graph_translation_size']+info['fine_graph_translation_size']);orig=.8*err.mean()+.2*(1-ss(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]))+.1*ae.mean();delta=float(abs(orig-local-rest));assert delta<2e-7;maxloss=max(maxloss,delta)
        grads=[]
        for loss in [local,rest,reg]:
            raw=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True);grads.append([torch.zeros_like(v[m]) if x is None else x[m].detach() for x,(_,v,m) in zip(raw,families)])
        if n in [0,50,104]:
            allg=torch.autograd.grad(orig+reg,parameters,retain_graph=True,allow_unused=True)
            for j,(x,(_,v,m)) in enumerate(zip(allg,families)):
                actual=torch.zeros_like(v[m]) if x is None else x[m];expected=grads[0][j]+grads[1][j]+grads[2][j];rel=float((actual-expected).norm()/actual.norm().clamp_min(1e-7));assert rel<2e-4,(branch,n,j,rel);maxgrad=max(maxgrad,rel)
        values={}
        for j,(key,v,m) in enumerate(families):
            aa=[grads[k][j] for k in range(3)];assert all(torch.isfinite(a).all() for a in aa);values[key]=metrics(*aa)
            for k,a in enumerate(aa):sums[key][k]+=a.double()
        rows.append(dict(camera=o['camera'],frame=o['frame'],source_roi=o['source_roi'],input_sha256=o['input_sha256'],roi_area_fraction=float(roi.mean()),roi_loss=float(local),rest_loss=float(rest),regularizer=float(reg),families=values));del rgb,alpha,info,pred,target,err,ae,sm,local,rest,reg,orig,grads
        if (n+1)%15==0:save(O/'progress.json',dict(branch=branch,contexts=n+1,elapsed_seconds=time.monotonic()-started));print('HAND_GRADIENT',branch,n+1,flush=True)
    aggregate={k:metrics(*[a/105 for a in values]) for k,values in sums.items()};np.savez_compressed(O/f'{branch}-aggregate-gradients.npz',**{k.replace('.','_')+'_'+part:values[i].cpu().numpy()/105 for k,values in sums.items() for i,part in enumerate(['roi','rest','regularizer'])});assert sha(path)==before
    results[branch]=dict(checkpoint_sha256=before,contexts=105,loss_decomposition_max_error=maxloss,full_gradient_decomposition_relative_error_max=maxgrad,records=rows,aggregate=aggregate);save(O/f'{branch}-records.json',results[branch]);del model,p,g,ref,fine,cp,parameters,sums;torch.cuda.empty_cache()
save(O/'summary.json',dict(status='210native_context_gradient_decompositions_complete_no_updates',branches={b:{k:v for k,v in r.items() if k!='records'} for b,r in results.items()},elapsed_seconds=time.monotonic()-started,qualification='ROIincludescontextobjects. Euclideanperfamilygradients,notAdamstep. Fixedgraydiagnostic. Trainingonly,notheldoutqualification.',goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name);print('HAND_GRADIENT_COMPLETE',flush=True)
