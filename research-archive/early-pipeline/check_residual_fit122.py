"""One registered R6 component/entry run; no person model or coefficient fit."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,os,sys,time,traceback,signal,subprocess,ast
import numpy as np
import cv2
from PIL import Image

H=Path(__file__).resolve().parent
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/robust-portrait-122/native-residual-fit'
R=O.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def linux(p):return '/mnt/'+p[0].lower()+p[2:].replace('\\','/') if len(p)>2 and p[1]==':' else p
now=lambda:datetime.now(timezone.utc).isoformat()
if (O/'execution-registration.json').exists():raise FileExistsError('R6 has executed; inspect evidence, do not rerun')
plan=read(O/'plan.json');impl=read(O/'implementation.json')
assert all(sha(linux(p))==h for p,h in plan['frozen'].items())
assert all(sha(linux(p))==h for p,h in impl['files'].items())
save(O/'execution-registration.json',dict(updated=now(),pid=os.getpid(),command=sys.argv,plan_sha256=sha(O/'plan.json'),sources=impl['files'],script_sha256=sha(__file__)))
start=time.monotonic();checks=[]
def check(name,condition,**data):
    row=dict(name=name,passed=bool(condition),**data);checks.append(row)
    save(O/'checks-progress.json',checks)
    save(O/'progress.json',dict(state='running',pid=os.getpid(),last_check=name,checks=len(checks),seconds=time.monotonic()-start,updated=now()))
    if not condition:raise AssertionError(name)
def reject(name,fn):
    try:fn()
    except (ValueError,RuntimeError) as e:check(name,True,error=str(e));return
    raise AssertionError('Missing rejection:'+name)
def timeout(*_):raise TimeoutError('Registered600CPU/60GPU budget exceeded')
signal.signal(signal.SIGALRM,timeout);signal.alarm(600)

from native_residual_camera122 import projection,with_projection,project_world
from native_residual_fit122 import FixedOutputProjector,TriangleBVH,point_visibility,diagnostic_depth,shared_anatomical_visibility,cameras_from_fit_rows
from native_surface122_v2 import evaluate_support
from fit_gate_output_residual122 import mesh_mask

def camera(index=0,width=320,height=240):
    k=[[180.,0,160.2],[0,170.,118.7],[0,0,1.]]
    base=dict(id=str(index),image_name=f'{index:06d}.png',width=width,height=height,K=k,w2c=np.eye(4).tolist())
    p=projection([.12,-.03,.002,.01,-.002,.0001],[.004,-.003],[.0008,-.0001,-.0005,.0001],input_domain='synthetic_known_camera')
    return with_projection(base,p)
def cvD(c):
    p=c['residual_projection'];k=p['radial'];t=p['tangential'];s=p['thin_prism']
    return np.array([k[0],k[1],t[0],t[1],k[2],k[3],k[4],k[5],*s],float)
def cvproj(points,c):
    E=np.asarray(c['w2c']);rv=cv2.Rodrigues(E[:3,:3])[0]
    return cv2.projectPoints(np.asarray(points,float),rv,E[:3,3],np.asarray(c['K']),cvD(c))[0].reshape(-1,2)-.5
def zeroD(c):
    c=copy.deepcopy(c)
    for n in ['radial','tangential','thin_prism']:c['residual_projection'][n]=[0.]*len(c['residual_projection'][n])
    return c

try:
    import torch
    torch.set_num_threads(2)
    cams=[row['camera'] for row in read(R/'native-residual-admission/synthetic/cameras-residual.json')['cameras']]
    rng=np.random.default_rng(6060)
    xyz=np.column_stack([rng.uniform(-.7,.7,32),rng.uniform(-.65,.65,32),rng.uniform(2.4,3.8,32)])
    points=np.repeat(xyz[None],4,axis=0)
    oracle=np.stack([cvproj(p,c) for p,c in zip(points,cams)])
    cpu=FixedOutputProjector(cams,device='cpu',dtype=torch.float64)
    x=torch.tensor(points,dtype=torch.float64,requires_grad=True)
    uv,pc=cpu.project(x);cpu_projection_error=float(np.max(np.abs(uv.detach().numpy()-oracle)))
    check('Four varyingKED Torch projections match independent OpenCV',cpu_projection_error<2e-10,max_coordinate_error_px=cpu_projection_error)
    weights=torch.tensor(rng.uniform(-1,1,uv.shape),dtype=torch.float64)
    (uv*weights).sum().backward();grad=x.grad.numpy().copy();gd=[]
    for view,point,axis in [(0,0,0),(0,1,1),(0,2,2),(1,3,0),(1,4,1),(1,5,2),(2,6,0),(2,7,1),(2,8,2),(3,9,0),(3,10,1),(3,11,2)]:
        a=points[view].copy();b=a.copy();a[point,axis]+=1e-6;b[point,axis]-=1e-6
        fd=float(((cvproj(a,cams[view])-cvproj(b,cams[view]))*weights[view].numpy()).sum()/2e-6)
        gd.append(abs(fd-grad[view,point,axis]))
    check('Twelve XYZ gradient entries match independent finite differences',max(gd)<2e-6,max_gradient_error=max(gd))
    check('Camera constants receive no autograd',not any(t.requires_grad for t in [cpu.E,cpu.K,cpu.D]))
    zcams=[zeroD(c) for c in cams];zz=FixedOutputProjector(zcams,device='cpu',dtype=torch.float64)
    zuv,zpc=zz.project(x.detach());q=zpc@zz.K.transpose(1,2);legacy=q[...,:2]/q[...,2:]-.5
    check('D0 keeps legacy positive-depth projection',float((zuv-legacy).abs().max())<1e-10)
    behind=x.detach().clone();behind[0,0,2]=-100
    reject('Behind-camera fit candidate fails explicitly',lambda:cpu.project(behind))
    missing=copy.deepcopy(cams);del missing[0]['residual_projection']
    reject('Fit projector rejects missingD',lambda:FixedOutputProjector(missing,device='cpu',dtype=torch.float64))
    rows=[dict(frame=i,wh=[c['width'],c['height']],K=c['K'],w2c=c['w2c'],camera=c) for i,c in enumerate(cams)]
    check('Fit row transports unchanged complete cameras',cameras_from_fit_rows(rows)==cams)
    badrows=copy.deepcopy(rows);badrows[1]['K'][0][2]+=1
    # deepcopy preserves aliasing in rows; replace the legacy copy independently.
    badrows=copy.deepcopy(rows);badrows[1]['K']=np.asarray(rows[1]['K']).copy().tolist();badrows[1]['K'][0][2]+=1
    reject('Prior K copy cannot disagree with authoritative camera',lambda:cameras_from_fit_rows(badrows))

    v=np.array([[-1,-.8,3],[-1,.8,3],[1,.8,3],[1,-.8,3],[-.25,-.3,2],[-.25,.3,2],[.25,.3,2],[.25,-.3,2]],float)
    f=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
    bvh=TriangleBVH(v,f);c=camera()
    query=np.array([[0,0,3],[.7,0,3],[0,0,2],[.7,.3,3],[10,0,3],[0,0,-2],[.9,.7,3]])
    normals=np.tile([0,0,-1.],(len(query),1));normals[3]*=-1
    vis,vd=point_visibility(bvh,c,query,normals,tolerance=1e-8)
    expected=np.array([False,True,True,False,False,False,True])
    check('Layered mesh exact visible occluded facing andsensor decisions',np.array_equal(vis,expected),visible=vis.tolist())
    dist,tri=bvh.trace(np.array([[0,0,0],[1,0,0],[3,0,0],[0,0,4]]),np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,-1]]))
    check('Parallel slab andtriangle-edge rays retain correct hits',np.allclose(dist[[0,1,3]],[2,3,1],atol=1e-10) and tri[2]==-1 and np.isinf(dist[2]))
    Rw=cv2.Rodrigues(np.array([.23,-.17,.08]))[0];s=2.3;trans=np.array([4.,-2.,7.])
    changed=copy.deepcopy(c);E=np.eye(4);E[:3,:3]=Rw.T;E[:3,3]=-Rw.T@trans;changed['w2c']=E.tolist()
    vv=v@Rw.T*s+trans;qq=query@Rw.T*s+trans;nn=normals@Rw.T
    vis2,vd2=point_visibility(TriangleBVH(vv,f),changed,qq,nn,tolerance=1e-8*s)
    finite=np.isfinite(vd['projected_uv']).all(1)
    simerr=float(np.max(np.abs(vd['projected_uv'][finite]-vd2['projected_uv'][finite])))
    check('World similarity does not alter residual visibility',np.array_equal(vis,vis2) and simerr<1e-8,max_projection_error_px=simerr)
    edge=camera();edge['K']=[[120.,0,160.5],[0,120.,120.5],[0,0,1.]];edge['residual_projection']=projection([.3,0,0,0,0,0],[0,0],[0,0,0,0],input_domain='synthetic_known_camera')
    ev=np.array([[2.,-.2,2.],[2.,.2,2.],[2.6,.2,2.],[2.6,-.2,2.]])
    eb=TriangleBVH(ev,f[:2]);ep=np.array([[2.3,0,2.]]);en=np.array([[0.,0,-1.]])
    vin,_=point_visibility(eb,zeroD(edge),ep,en,tolerance=1e-9);vout,_=point_visibility(eb,edge,ep,en,tolerance=1e-9)
    check('NonzeroD changes actual sensor membership near edge',bool(vin[0]) and not bool(vout[0]))
    vz,dz=point_visibility(bvh,zeroD(c),query,normals,tolerance=1e-8)
    common=vd['in_image']&dz['in_image']
    check('Lens does not alter geometry of shared visible rays',np.array_equal(vis[common],vz[common]))
    reg=np.ones((len(v),2));face_ids=f[[0,2]];bary=np.array([[.2,.4,.4],[.2,.4,.4]])
    regional,facevis,footvis=shared_anatomical_visibility(np.repeat(v[None],2,axis=0),f,[c,dict(copy.deepcopy(c),id='1')],reg,face_ids,bary,np.array([0,2]),np.array([0,1]))
    check('Shared fitter visibility returns every declared region faceandfoot',regional.shape==(2,2) and facevis.shape==(2,2) and footvis.shape==(2,2) and np.isfinite(regional).all())
    dynamic=np.repeat(v[None],2,axis=0);dynamic[1,0,0]+=.1
    reject('Visibility refuses per-frame deformed meshes',lambda:shared_anatomical_visibility(dynamic,f,[c,c],reg,face_ids,bary,np.array([0,2]),np.array([0,1])))

    raster=copy.deepcopy(c);raster.update(width=192,height=128);raster['K']=[[115.,0,94.2],[0,104.,63.7],[0,0,1.]]
    depth,triimg=diagnostic_depth(bvh,raster);pixelmask=mesh_mask(bvh,raster)
    xx,yy=np.meshgrid(np.arange(192)+.5,np.arange(128)+.5);pix=np.column_stack([xx.ravel(),yy.ravel()])
    ideal=cv2.undistortPointsIter(pix[:,None],np.asarray(raster['K']),cvD(raster),None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,80,1e-14)).reshape(-1,2)
    exact=np.zeros(len(pix));back=np.abs(ideal[:,0]*3)<=1;back&=np.abs(ideal[:,1]*3)<=.8;exact[back]=3
    front=(np.abs(ideal[:,0]*2)<=.25)&(np.abs(ideal[:,1]*2)<=.3);exact[front]=2
    deptherr=float(np.max(np.abs(depth.ravel()-exact)))
    check('Native nonlinear depth matches independent analytic planes',deptherr<1e-9 and np.array_equal(pixelmask,exact.reshape(128,192)>0),max_depth_error=deptherr,pixels=len(pix))
    depth0,_=diagnostic_depth(bvh,zeroD(raster))
    def rgb_depth(d):
        out=np.full(d.shape+(3,),13,np.uint8);out[(d>0)&(d<2.5)]=[60,170,145];out[d>=2.5]=[75,115,185];return out
    diff=np.zeros(depth.shape+(3,),np.uint8);diff[abs(depth-depth0)>1e-6]=[238,116,76]
    Image.fromarray(np.concatenate([rgb_depth(depth),rgb_depth(depth0),diff],axis=1)).save(O/'visibility-control.png')
    np.savez_compressed(O/'visibility-control.npz',fullD=depth,zeroD=depth0,independent_depth=exact.reshape(128,192),mask=pixelmask)
    check('Visibility diagnostic shows nonlinear sensor difference',np.count_nonzero(abs(depth-depth0)>1e-6)>0,different_pixels=int(np.count_nonzero(abs(depth-depth0)>1e-6)))

    # Independent sphere silhouettes in native output pixels, and observed-valid masks.
    data=O/'mask-control';data.mkdir();mc=[];views=[];maskarrays=[]
    for i in range(4):
        cc=copy.deepcopy(c);cc.update(id=str(i),image_name=f'mask{i}.png',width=96,height=72)
        cc['K']=[[55.+i,0,47.2+i*.1],[0,54.+i,35.7],[0,0,1.]];cc['w2c'][0][3]=(.3-i*.2)
        cc['residual_projection']['radial'][0]=.3
        mc.append(cc);xg,yg=np.meshgrid(np.arange(96)+.5,np.arange(72)+.5)
        rays2=cv2.undistortPointsIter(np.column_stack([xg.ravel(),yg.ravel()])[:,None],np.asarray(cc['K']),cvD(cc),None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,80,1e-14)).reshape(-1,2)
        dirs=np.column_stack([rays2,np.ones(len(rays2))]);dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
        center=np.array([-cc['w2c'][0][3],0.,0.]);oc=center-np.array([0.,0.,3.]);b=np.sum(dirs*oc,axis=1);dis=b*b-(oc@oc-.6**2)
        mask=(dis>=0).reshape(72,96);valid=np.full((72,96),255,np.uint8)
        if i==0:valid[15:55,30:65]=0
        mp=data/f'{i}-alpha.png';vp=data/f'{i}-valid.png';Image.fromarray(mask.astype(np.uint8)*255).save(mp);Image.fromarray(valid).save(vp)
        views.append(dict(frame_id=i,alpha=dict(path=str(mp),sha256=sha(mp)),observation_valid=dict(path=str(vp),sha256=sha(vp))))
        maskarrays.append((mask,valid))
    pts=np.column_stack([rng.uniform(-.9,.9,600),rng.uniform(-.8,.8,600),rng.uniform(2.2,3.8,600)])
    got=evaluate_support(pts,views,dict(enumerate(mc)),data,tolerance_px=0,min_views=3,vote_fraction=.97)
    positive=np.zeros(len(pts),int);observed=np.zeros(len(pts),int)
    for cc,(mask,valid) in zip(mc,maskarrays):
        u=cvproj(pts,cc);ok=(u[:,0]>=-.5)&(u[:,0]<95.5)&(u[:,1]>=-.5)&(u[:,1]<71.5);ids=np.flatnonzero(ok);ij=np.floor(u[ids]+.5).astype(int)
        keep=valid[ij[:,1],ij[:,0]]==255;ids=ids[keep];ij=ij[keep]
        observed[ids]+=1;positive[ids]+=mask[ij[:,1],ij[:,0]]
    check('Clothing support uses same per-view KED and native masks',np.array_equal(got['observed'],observed) and np.array_equal(got['positive'],positive))
    check('Unknown pixels abstain from background votes',np.any(observed==3) and np.any((observed==3)&(positive==3)&got['inside']))
    from native_surface122 import evaluate_support as old_support
    zc=[zeroD(cc) for cc in mc]
    for cc in zc:cc['K']=copy.deepcopy(zc[0]['K'])
    oldviews=[dict(w2c=cc['w2c'],alpha=v['alpha'],observation_valid=v['observation_valid']) for cc,v in zip(zc,views)]
    ki=np.asarray(zc[0]['K']).copy();ki[:2,2]-=.5
    old=old_support(pts,oldviews,ki,data,tolerance_px=0,min_views=3,vote_fraction=.97)
    new=evaluate_support(pts,views,dict(enumerate(zc)),data,tolerance_px=0,min_views=3,vote_fraction=.97)
    check('D0 silhouette sampling preserves old integerpixel behavior',all(np.array_equal(old[k],new[k]) for k in old))
    wrong=copy.deepcopy(dict(enumerate(mc)));wrong[2]['width']+=1
    reject('Clothing mask refuses implicit raster resize',lambda:evaluate_support(pts,views,wrong,data,tolerance_px=0,min_views=3,vote_fraction=.97))
    check('Native mask files unchanged',all(sha(v[k]['path'])==v[k]['sha256'] for v in views for k in ['alpha','observation_valid']))

    # Mesh-size performance check, deterministic and never a person fit.
    nl,nc=64,160;sv=[[0,0,4.],[0,0,2.]]
    for j in range(1,nl):
        th=np.pi*j/nl
        for i in range(nc):
            ph=2*np.pi*i/nc;sv.append([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),3+np.cos(th)])
    sf=[]
    for i in range(nc):sf.append([0,2+i,2+(i+1)%nc])
    for j in range(nl-2):
        for i in range(nc):
            a=2+j*nc+i;b=2+j*nc+(i+1)%nc;d=a+nc;e=b+nc;sf.extend([[a,d,b],[b,d,e]])
    for i in range(nc):sf.append([1,2+(nl-2)*nc+(i+1)%nc,2+(nl-2)*nc+i])
    sv=np.asarray(sv,float);sf=np.asarray(sf);bt=time.monotonic();sb=TriangleBVH(sv,sf);sd,st=sb.trace(np.zeros(3),sv);bs=time.monotonic()-bt
    check('10kmesh BVH allvertex rays finish bounded and do not pass ownsurface',np.all(st>=0) and np.all(sd<=np.linalg.norm(sv,axis=1)+1e-8) and bs<60,vertices=len(sv),triangles=len(sf),seconds=bs)

    # Execute the exact nested functions from the versioned fitter, with known
    # geometric tensors only. The SMPLprior/main optimizer is not invoked.
    tree=ast.parse((H/'fitting_output_residual122.py').read_text(encoding='utf-8'));main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    functions=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in ['project','visibility']]
    from types import SimpleNamespace
    env=dict(torch=torch,fixed_projector=cpu,shared_anatomical_visibility=shared_anatomical_visibility,
        faces=torch.tensor(f),cameras=[c,dict(copy.deepcopy(c),id='1')],regional=torch.tensor(reg),face_ids=torch.tensor(face_ids),face_bary=torch.tensor(bary),
        model=SimpleNamespace(lmk_faces_idx=torch.tensor([0,2])),feet_vertices=torch.tensor([0,1]),g=lambda a:torch.tensor(a,dtype=torch.float64))
    exec(compile(ast.Module(body=functions,type_ignores=[]),'fitting_output_residual122.py nested registered functions','exec'),env)
    wired=env['project'](x.detach())[0];vresult=env['visibility'](dict(v=torch.tensor(np.repeat(v[None],2,axis=0))))
    check('Actual versioned fitter project andvisibility bodies execute sharedadapters',torch.equal(wired,uv.detach()) and all(np.array_equal(a.numpy(),b) for a,b in zip(vresult,[regional,facevis,footvis])))

    # Reject actual person at new preparation entry before any upstream GPU work.
    cmd=[sys.executable,str(H/'prepare_native_person122_v2.py'),'fit','--input',str(R/'native-residual-admission/real-diagnostic-input.json'),'--output',str(O/'actual-preparation')]
    child=subprocess.run(cmd,capture_output=True,text=True,timeout=150)
    (O/'actual-preparation.log').write_text(child.stdout+child.stderr,encoding='utf-8')
    state=read(O/'actual-preparation/preparation-status.json');audit=read(O/'actual-preparation/admission.json')
    check('New preparation actually refuses failedperson before GPUpriorfit',child.returncode==3 and state['state']=='input_rejected' and all(state[k] is False for k in ['GPU_imported','prior_started','fit_started','surface_started','training_started']))
    check('Actual102targets and13failedface retained at13.049px',audit['material_geometry']['targets']==102 and audit['heldout_face']['observations']==13 and audit['heldout_face']['accepted']==0 and abs(audit['heldout_face']['maximum_px']-13.048757643629694)<1e-8,face=audit['heldout_face'])

    signal.alarm(60)
    import fcntl
    with open('/tmp/shuolee-native122-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if active:raise RuntimeError('Another GPU process active; no concurrent projection test')
        gpustart=time.monotonic();gx=torch.tensor(points,dtype=torch.float32,device='cuda',requires_grad=True)
        gp=FixedOutputProjector(cams,device='cuda',dtype=torch.float32);guv,_=gp.project(gx)
        (guv*weights.to(device='cuda',dtype=torch.float32)).sum().backward();torch.cuda.synchronize()
        gpuerror=float(np.max(np.abs(guv.detach().cpu().numpy()-oracle)));gradienterror=float(np.max(np.abs(gx.grad.cpu().numpy()-grad)))
        check('Single CUDA projection/backward matches independentcamera andCPUgradient',gpuerror<1e-4 and gradienterror<1e-4,max_coordinate_error_px=gpuerror,max_gradient_difference=gradienterror,seconds=time.monotonic()-gpustart)
        del gx,gp,guv;torch.cuda.empty_cache()
    signal.alarm(0)
    check('Historical and implemented source hashes unchanged duringexecution',all(sha(linux(p))==h for p,h in plan['frozen'].items()) and all(sha(linux(p))==h for p,h in impl['files'].items()))
    check('No person fit orGaussian arrays created',not list((O/'actual-preparation').rglob('*.npz')))
    result=dict(stage='R6',state='completed_pending_review',updated=now(),seconds=time.monotonic()-start,checks=checks,
        CPU_projection_max_px=cpu_projection_error,FD_gradient_max=max(gd),GPU_projection_max_px=gpuerror,GPU_gradient_max=gradienterror,
        visibility_depth_max=deptherr,world_similarity_max_px=simerr,mesh_benchmark=dict(vertices=len(sv),triangles=len(sf),seconds=bs),
        actual_person_admission=audit['heldout_face'],actual_person_exit=child.returncode,
        person_fit=False,person_training=False,physical_quality_accepted=False,positive_person_preparation_executed=False,optimizer_entry_connected=False,
        scope='Versioned projector,continuous mesh selfvisibility,silhouette sampling,fitter function bodies,overlay mask and new preparation refusal. GPU only small pointprojection/backward. No SMPL-X optimizer/model inference executed.',
        visibility_policy_changed=True,coefficient_fitted=False,images_warped=False)
    save(O/'result.json',result);save(O/'progress.json',dict(state='completed_pending_review',updated=now(),checks=len(checks),seconds=result['seconds']))
    print(json.dumps(dict(completed=True,checks=len(checks),seconds=result['seconds'],person_training=False)))
except Exception as error:
    signal.alarm(0)
    save(O/'result.json',dict(state='failed',updated=now(),error=repr(error),traceback=traceback.format_exc(),checks=checks,person_training=False))
    save(O/'progress.json',dict(state='failed',updated=now(),error=repr(error),checks=len(checks)))
    raise
