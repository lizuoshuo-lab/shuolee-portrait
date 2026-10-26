"""Preregistered native3DGS controls and zero-step175-camera support audit."""
import json,math,time,traceback
import numpy as np
from types import SimpleNamespace
import shuolee_static as s
import shuolee_pixel_center as pc
from native_person_alignment_113 import initialize_surface,native_weights,pixel_error,volume_pixel_error,NativePersonAlignment
torch=s.torch
R=s.B/'current/person-static-native-align-fresh-113';O=R/'native-preflight'
T=lambda a:torch.tensor(a,dtype=torch.float32,device='cuda')

def camera(E,W=64,H=64,fx=70.,fy=70.):
    cam=s.RawCamera(0,E[:3,:3].T,E[:3,3],2*np.arctan(W/(2*fx)),2*np.arctan(H/(2*fy)),W,H,'','synthetic',0,preload_img=False)
    return pc.corrected_camera(cam)

def model(points):
    m=s.GaussianModel(3);m._xyz=torch.nn.Parameter(T(points));n=len(points)
    m._rotation=torch.nn.Parameter(T([[1,0,0,0]]*n));m._scaling=torch.nn.Parameter(torch.log(T([[.16,.16,.04]]*n)))
    m._opacity=torch.nn.Parameter(T([[0.]]*n));m._features_dc=torch.nn.Parameter(T([[[0,0,0]]]*n));m._features_rest=torch.nn.Parameter(torch.zeros(n,15,3,device='cuda'))
    m.use_app=False;return m

def synthetic(volume=False):
    error=(lambda m,w,cam,px,target:volume_pixel_error(m._xyz,m._rotation,m.get_scaling,w,cam,px,target,T(.03))) if volume else (lambda m,w,cam,px,target:pixel_error(m._xyz,m.get_smallest_axis(),w,cam,px,target,T(.03)))
    E=np.eye(4);cam=camera(E);ray=T([(32-cam.Cx)/cam.Fx,(32-cam.Cy)/cam.Fy,1]);target=ray*2
    m=model(torch.stack((ray*1.97,ray*2.03)).cpu().numpy())
    with torch.no_grad():m._opacity.copy_(torch.logit(T([[.4],[2/3]])))
    corners,parity=native_weights(m,cam,s.PIPE,T([[32.,32.]]));px,w,_=corners[0]
    value,detail=error(m,w[:,0],cam,px[0],target);assert value is not None
    loss=2*(torch.sqrt(1+value)-1);gm,gq,ga,gs=torch.autograd.grad(loss,(m._xyz,m._rotation,m._opacity,m._scaling),allow_unused=True)
    duplicate=dict(**detail,loss=float(loss),z_gradient=gm[:,2].tolist(),opacity_no_gradient=ga is None,scale_no_gradient=gs is None,
        both_toward=bool(gm[0,2]<0 and gm[1,2]>0),weight_alpha_error=parity)
    # Slanted anisotropic Gaussian; freeze native assignments during FD.
    m=model([[.01,-.01,2.]]);m._rotation.data.copy_(T([[math.cos(.1),0,math.sin(.1),0]]));target=T([.015,.018,2.04])
    corner,_=native_weights(m,cam,s.PIPE,T([[32.,32.]]));px,w,_=corner[0]
    def value():
        v,_=error(m,w[:,0],cam,px[0],target);return 2*(torch.sqrt(1+v)-1)
    v=value();gm,gq=torch.autograd.grad(v,(m._xyz,m._rotation));checks=[]
    for name,param,grad,idx in [('x',m._xyz,gm,(0,0)),('z',m._xyz,gm,(0,2)),('qw',m._rotation,gq,(0,0)),('qy',m._rotation,gq,(0,2))]:
        old=float(param[idx]);eps=1e-4
        with torch.no_grad():
            param[idx]=old+eps;a=value();param[idx]=old-eps;b=value();param[idx]=old
        fd=float((a-b)/(2*eps));ad=float(grad[idx]);limit=.002+.002*max(abs(ad),abs(fd));checks.append(dict(name=name,ad=ad,fd=fd,limit=limit,passed=abs(ad-fd)<limit))
    passed=duplicate['both_toward'] and duplicate['opacity_no_gradient'] and duplicate['scale_no_gradient'] and duplicate['mean_squared_error']<1e-9 and duplicate['variance']>.0008 and all(q['passed'] for q in checks)
    extra=[]
    if volume:
        # Physically identical spheres may have arbitrary quaternion frames.
        m=model([[.01,-.01,2.]]);m._scaling.data.fill_(math.log(.16));w=T([.8]);p=T([32,32]);target=T([.015,.018,2.04])
        values=[]
        for angle in [0.,math.pi/3,math.pi/2]:
            m._rotation.data.copy_(T([[math.cos(angle/2),0,math.sin(angle/2),0]]));v,_=error(m,w,cam,p,target);values.append(float(v))
        extra.append(dict(test='isotropic_frame_invariance',range=max(values)-min(values),passed=max(values)-min(values)<1e-6))
        m._scaling.data.copy_(torch.log(T([[.16,.16,.04]])))
        m._rotation.data.copy_(T([[math.sqrt(.5),0,math.sqrt(.5),0]]));v,detail=error(m,w,cam,p,target)
        extra.append(dict(test='grazing_finite_covariance',value=float(v),passed=bool(torch.isfinite(v)) and detail['reason']=='valid'))
        passed=passed and all(x['passed'] for x in extra)
    return dict(passed=passed,duplicate_layers=duplicate,gradients=checks,extra=extra)

def main(volume=False):
    global O
    if volume:O=R/'native-volume-preflight'
    O.mkdir(exist_ok=True);assert not (O/'result.json').exists()
    s.save(O/'protocol.json',dict(created=time.time(),height=1536,initial_points=200000,numerical='Alpha weight parity<2e-4, frozen visibility FD .002+.2percent, unchanged ordinary RGB max<2e-6',
        query_selection='Up to2 observations in each of4image regions percamera, seeded uniformly withinregion, split into batchesof3. No loss/residual selection.',
        admission='Controls pass, all175frames have >=1 valid query, >=90percent all queried observations valid, >=80percent in each represented image region. Any empty or grazing-dominated corner invalidates its observation. Failed ones retained in report.',
        training_safeguard='At each500stepwindow invalid-query fraction must stay<=10percent; otherwise stop with checkpoint. Query sampling independent of existing training RNG. Training uses up to3uniformfeaturequeries perviewstep.',
        source_sha256={str(p):s.sha(p) for p in [s.Path(__file__),s.Path(__file__).with_name('native_person_alignment_113.py')]},training_started=False))
    start=time.time();out=dict(passed=False,optimizer_updates=0,quality_accepted=False)
    try:
        if volume:
            # The portable entry runs the final recorded 113 geometry directly.
            # Its controls/support gates remain unchanged; an old failed trial
            # is provenance, not an input required on a new server.
            s.save(O/'amendment-before-check.json',dict(reason='Frozen113 conditional-volume geometry; historical failed plane trial is preserved in source provenance and is not rerun or fabricated here.',
                formula='z=d^T Q(mu-o)/(d^T Q d);Q=R diag(1/s^2) R^T. Positive finite covariance guarantees positive denominator, including tangential views. Native screenweights unchanged, squared contributor residuals precede averaging.',
                unchanged='Same fresh200kcenters/normals,targets,cameras,probe selection and >=90percent support gate; no threshold lowering or training has occurred.',
                conditional_gradient='Detach scales and visibility only for auxiliary, optimize centers/rotation. Full RGB/mask keep all parametergradients.',
                caveat='3DGS projected alpha is still the native screen approximation, not exact ray volume integration. Geometric target is uncertain, notGT.',training_updates=0))
        control=synthetic(volume);s.save(O/'synthetic-result.json',control);assert control['passed'],'Native controls failed'
        doc=s.read(R/'data/training-inputs.json');s.require_registered_cameras(doc);rows=doc['records']
        points=np.load(R/'clothing-init/points.npz')['xyz'];scale=float(np.load(R/'clothing-init/motion.npz')['world_scale'])
        m=s.GaussianModel(3);torch.manual_seed(6060);m.create_from_pcd(s.BasicPointCloud(points,np.full_like(points,.5),np.zeros_like(points)),scale)
        with torch.no_grad():m._scaling.clamp_(max=np.log(.012*scale));m._features_dc.zero_();m._features_rest.zero_()
        initialization=initialize_surface(m,R/'surface-init/normals.npz',s.sha(R/'clothing-init/points.npz'))
        cams=[]
        for i,row in enumerate(rows):
            H=1536;W=round(row['wh'][0]*H/row['wh'][1]);K=np.diag([W/row['wh'][0],H/row['wh'][1],1.])@np.asarray(row['K'])
            cams.append(camera(np.asarray(row['w2c']),W,H,K[0,0],K[1,1]))
        target=NativePersonAlignment(R/'shared-targets/native-targets.npz',cams,rows,geometry='conditional_volume' if volume else 'plane');records=[];rgb_parity=[]
        for i,cam in enumerate(cams):
            regions=target.observations[i]['regions'];rng=np.random.default_rng(6060+i);selection=[]
            for region in range(4):
                ids=np.flatnonzero(regions==region)
                if len(ids):selection.extend(rng.choice(ids,min(2,len(ids)),replace=False).tolist())
            selected=[]
            for j in range(0,len(selection),3):
                value,detail=target.loss(m,i,s.PIPE,selection=selection[j:j+3]);assert torch.isfinite(value)
                gm,gq=torch.autograd.grad(value,(m._xyz,m._rotation),allow_unused=True)
                finite=(gm is not None and bool(torch.isfinite(gm).all())) and (gq is None or bool(torch.isfinite(gq).all()))
                assert finite;selected.extend(detail['queries'])
                del value,gm,gq
            records.append(dict(raw=i+1,queries=selected,valid=sum(q['valid'] for q in selected),count=len(selected)))
            if i in [0,33,69,99,135,174]:
                with torch.no_grad():
                    before=s.render(cam,m,s.PIPE,s.BG,return_plane=False,return_depth_normal=False)['render']
                    after=pc.render(cam,m,s.PIPE,s.BG,return_plane=False,return_depth_normal=False)['render']
                    rgb_parity.append(float((before-after).abs().max()))
            if i%25==0:s.save(O/'progress.json',dict(done=i+1,total=175,seconds=time.time()-start));print(json.dumps(dict(done=i+1,total=175,seconds=time.time()-start)),flush=True)
        queries=[q for r in records for q in r['queries']];ratio=sum(q['valid'] for q in queries)/len(queries);regional={}
        for region in range(4):
            q=[q for q in queries if q['region']==region];regional[str(region)]=dict(count=len(q),valid=sum(a['valid'] for a in q),fraction=sum(a['valid'] for a in q)/len(q) if q else None)
        passed=all(r['valid']>0 for r in records) and ratio>=.9 and all(q['fraction'] is None or q['fraction']>=.8 for q in regional.values()) and max(rgb_parity)<2e-6
        out.update(passed=passed,controls=control,initialization=initialization,records=records,valid_fraction=ratio,regions=regional,rgb_max_deltas=rgb_parity,
            frames_with_support=sum(r['valid']>0 for r in records),target_sha256=s.sha(R/'shared-targets/native-targets.npz'),normals_sha256=s.sha(R/'surface-init/normals.npz'))
    except Exception as e:out.update(error=str(e),traceback=traceback.format_exc())
    out['seconds']=time.time()-start;s.save(O/'result.json',out);print(json.dumps({k:v for k,v in out.items() if k!='records'}),flush=True)

if __name__=='__main__':
    import sys
    main('--volume' in sys.argv)
