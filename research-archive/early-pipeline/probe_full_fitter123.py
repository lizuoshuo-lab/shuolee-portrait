"""One full fitter integration on known geometry, not a real-person fit.

The real entry continues to reject synthetic data. This fixture deliberately
calls the internal optimizer after proving that public-entry rejection; it
cannot turn a synthetic receipt into production admission.
"""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,sys,time,traceback
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

P=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/generic-2d3d-research/adapter-a0c')
ROOT=Path('/opt/gs-slam-lab/shuolee-adapter-a0c-synthetic')
ASSETS=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
sys.path.insert(0,str(P/'src'))
from shuolee_portrait.alignment_inputs import read,sha,COCO133_NAMES
from shuolee_portrait.fit_backend import inspect_inputs,require_inputs
from shuolee_portrait.initialization_inputs import write_initialization_receipt
from shuolee_portrait import runtime as rt

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def link(p,**kw):return dict(path=str(p),sha256=sha(p),**kw)
plan=read(O/'plan.json');spec=plan['synthetic'];start=time.monotonic()
lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
needlock=P/'.gpu-task.lock'
fd=os.open(needlock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps(dict(pid=os.getpid(),command=sys.argv,scope='known_synthetic_full_fitter')).encode());os.close(fd)
try:
    assert not ROOT.exists(),'Inspect existing synthetic fixture; never resume.'
    ROOT.mkdir();(ROOT/'data').mkdir();(ROOT/'inference').mkdir();(ROOT/'raw').mkdir();(ROOT/'data/masks').mkdir()
    save(O/'full-fit-progress.json',dict(state='running',pid=os.getpid(),root=str(ROOT),command=sys.argv,stage='prepare_known_geometry'))
    rt.configure(ASSETS,ROOT)
    import torch,smplx,nvdiffrast.torch as dr
    from smplx.joint_names import JOINT_NAMES
    from shuolee_portrait.smplx_correspondence import face_world,project_integer_centres,camera_clip_coordinates
    torch.set_num_threads(4);torch.manual_seed(spec['seed']);torch.backends.cuda.matmul.allow_tf32=False
    model=smplx.SMPLX(str(rt.c.MODEL),num_betas=10,num_expression_coeffs=10,use_pca=False,flat_hand_mean=False,batch_size=1,use_face_contour=False).cuda()
    with torch.no_grad():b=model();v=b.vertices-b.joints[:,:1];j=b.joints-b.joints[:,:1];face=face_world(v,model)
    n=spec['views'];W,H=spec['camera_wh'];K=np.array([[spec['focal'],0.,(W-1)/2],[0.,spec['focal'],(H-1)/2],[0.,0.,1.]])
    Ks=np.tile(K,(n,1,1));Kr=Ks.copy();Kr[:,:2,2]+=.5;Es=[]
    for angle in np.linspace(-.14,.14,n):
        C=np.array([2.7*np.sin(angle),-.05,2.7*np.cos(angle)])
        z=-C/np.linalg.norm(C);x=np.cross(z,[0.,1.,0.]);x/=np.linalg.norm(x);y=np.cross(z,x)
        R=np.stack([x,y,z]);E=np.eye(4);E[:3,:3]=R;E[:3,3]=-R@C;Es.append(E)
    Es=np.stack(Es);g=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    eg,kg,kr=g(Es),g(Ks),g(Kr)
    mapping=torch.load(ASSETS/'gvhmr/hmr4d/utils/body_model/smplx2smpl_sparse.pt',weights_only=True,map_location='cpu').to_dense()
    reg=(torch.load(ASSETS/'gvhmr/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt',weights_only=True,map_location='cpu')@mapping).cuda()
    with torch.no_grad():
        co=torch.einsum('jv,nvc->njc',reg,v).expand(n,-1,-1)
        bodyuv,_=project_integer_centres(co,eg,kg);faceuv,_=project_integer_centres(face.expand(n,-1,-1),eg,kg)
        jointsuv,_=project_integer_centres(j.expand(n,-1,-1),eg,kg)
    feet=[JOINT_NAMES.index(x) for x in ['left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']]
    handnames=['wrist','thumb1','thumb2','thumb3','thumb','index1','index2','index3','index','middle1','middle2','middle3','middle','ring1','ring2','ring3','ring','pinky1','pinky2','pinky3','pinky']
    hands=[JOINT_NAMES.index(side+'_'+name) for side in ['left','right'] for name in handnames]
    points=np.zeros((n,133,2));points[:,:17]=bodyuv.cpu().numpy();points[:,17:23]=jointsuv[:,feet].cpu().numpy();points[:,40:91]=faceuv.cpu().numpy();points[:,91:]=jointsuv[:,hands].cpu().numpy()
    # Unused face-contour entries remain a named, inactive part of COCO133.
    points[:,23:40]=faceuv[:,[13]].cpu().numpy()
    ctx=dr.RasterizeCudaContext();rows=[];boxes=[];crop_records=[]
    for i in range(n):
        name=f'synthetic_{i:02d}';cam=v[0]@eg[i,:3,:3].T+eg[i,:3,3]
        clip=camera_clip_coordinates(cam,kr[i],W,H)
        raster,_=dr.rasterize(ctx,clip[None],model.faces_tensor.int(),(H,W))
        mask=(raster[0,:,:,3]>0).flip(0).cpu().numpy();yy,xx=np.nonzero(mask)
        assert len(xx)>0
        box=[float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)];boxes.append(box)
        img=ROOT/'raw'/f'{name}.png';mp=ROOT/'data/masks'/f'{name}.png'
        Image.fromarray((mask*255).astype(np.uint8)).save(mp)
        Image.fromarray(np.repeat((mask*150).astype(np.uint8)[...,None],3,axis=2)).save(img)
        row=dict(frame=name,name=img.name,source_frame_index=i,time_seconds=float(i),wh=[W,H],image=str(img),source_sha256=sha(img),
            K=Kr[i].tolist(),K_integer_centers=Ks[i].tolist(),w2c=Es[i].tolist(),mask=str(mp),mask_sha256=sha(mp),
            official_prediction_slot=i,camera_kind='synthetic_known',observation_weight=1.,initial_box_xyxy=box)
        rows.append(row);crop_records.append(dict(frame=name,source_sha256=row['source_sha256'],source_wh=[W,H],
            crop_kind='none_synthetic_oracle',source_to_crop=np.eye(3).tolist(),crop_to_source=np.eye(3).tolist()))
    for name in ['inputs.json','training-inputs.json']:save(ROOT/'data'/name,dict(records=rows,synthetic=True))
    save(ROOT/'inference/crop-mapping.json',dict(records=crop_records))
    ids=np.array([r['frame'] for r in rows]);np.savez_compressed(ROOT/'data/observations.npz',frame_ids=ids,keypoints=points,
        scores=np.ones((n,133)),flip_disagreement=np.zeros((n,133)),boxes=boxes,joint_names=COCO133_NAMES)
    vit=np.concatenate([bodyuv.cpu().numpy(),np.ones((n,17,1))],axis=-1)
    delta=Rotation.from_rotvec([0.,spec['root_yaw_radians'],0.]).as_matrix();offset=np.array(spec['root_translation_offset'])
    init_j=(j[0].cpu().numpy()@delta.T)+offset
    incam_j=np.einsum('nij,vj->nvi',Es[:,:3,:3],init_j)+Es[:,:3,3][:,None]
    pred=dict(incam_body_pose=np.zeros((n,63)),incam_betas=np.zeros((n,10)),
        incam_global_orient=Rotation.from_matrix(Es[:,:3,:3]@delta).as_rotvec(),
        incam_transl=incam_j[:,0].copy(),incam_joints=incam_j)
    np.savez_compressed(ROOT/'inference/prediction-postprocessed.npz',**pred)
    receipt=write_initialization_receipt(ROOT,vit,purpose='synthetic_contract_test',producer_sha256=sha(Path(__file__)),inputs_sha256=sha(ROOT/'data/inputs.json'))
    review=dict(synthetic_control=True,coordinate_domain='native_integer_pixel_centres',observations_sha256=sha(ROOT/'data/observations.npz'),inputs_sha256=sha(ROOT/'data/inputs.json'),
        limitation='Known mathematical projection oracle, not detection or photometric visibility ground truth',records=[])
    for i,row in enumerate(rows):
        for slot in range(51):
            ibug=slot+17
            review['records'].append(dict(frame=row['frame'],ibug68_index=ibug,anchor_id=f'ibug68:{ibug}',
                uv_native=points[i,slot+40].tolist(),visibility='observed',semantic_identity_verified=True,
                split='HELD' if ibug in [17,24,31,38,45,52,59,66] else 'FIT',physical_family_id=f'synthetic-fixed-{ibug}',
                review_evidence='known_geometry_oracle',visibility_source='synthetic_known',localization_sigma_px=1.,reliability=1.))
    save(ROOT/'data/face-review.json',review)
    desc=link(ROOT/'data/observations.npz',convention='coco_wholebody_133',coordinate_domain='native_integer_pixel_centres')
    manifest=dict(schema='shuolee-alignment-input/1',purpose='synthetic_contract_test',camera_model='rectified_pinhole',
        frames=link(ROOT/'data/inputs.json'),wholebody=desc,face=dict(**desc,coordinates_key='keypoints'),
        reviews=link(ROOT/'data/face-review.json',format='reviewed_anatomical_v1'))
    save(ROOT/'alignment-inputs.json',manifest)
    cfg=dict(fit_backend='smplx_direct_v1',fit_steps=spec['steps'],alignment_inputs='alignment-inputs.json',direct_face_binding_policy='reviewed_anatomical_v1')
    save(ROOT/'config.json',cfg)
    bundle,preflight=inspect_inputs(ROOT);save(O/'synthetic-public-preflight.json',preflight)
    assert preflight['initialization']['ready'] and not preflight['fitting_allowed']
    try:require_inputs(ROOT)
    except RuntimeError:pass
    else:raise AssertionError('Synthetic data was admitted by production entry.')
    save(O/'full-fit-progress.json',dict(state='running',pid=os.getpid(),root=str(ROOT),command=sys.argv,stage='internal_known_geometry_fit',production_entry_denied=True))
    from shuolee_portrait.fitting_direct import _fit_prepared
    _fit_prepared(ROOT,spec['steps'],bundle)
    with np.load(ROOT/'fit/initialization.npz') as z:initial=z['vertices_world']
    with np.load(ROOT/'fit/fitted.npz') as z:fitted={k:z[k] for k in z.files}
    with torch.no_grad():
        inituv,_=project_integer_centres(face_world(g(initial),model),eg,kg)
        finaluv,_=project_integer_centres(face_world(g(fitted['vertices_world']),model),eg,kg)
        cj,_=project_integer_centres(g(fitted['coco17_world']),eg,kg)
        jj,_=project_integer_centres(g(fitted['joints_world']),eg,kg)
    initial_error=np.linalg.norm(inituv.cpu().numpy()-faceuv.cpu().numpy(),axis=-1)
    final_error=np.linalg.norm(finaluv.cpu().numpy()-faceuv.cpu().numpy(),axis=-1)
    parity=dict(face=float(np.max(np.abs(finaluv.cpu().numpy()-fitted['projected_face51']))),
        body=float(np.max(np.abs(cj.cpu().numpy()-fitted['projected_coco17']))),
        hand_foot_joint=float(np.max(np.abs(jj.cpu().numpy()-fitted['projected_joints']))))
    hist=read(ROOT/'fit/history.json');finite=all(np.isfinite([x[k] for x in hist]).all() for k in ['loss','body','feet','hands','face'])
    checks=dict(finite_actual_losses=bool(finite),actual_export_matches_saved_3D=max(parity.values())<.001,
        face_mean_error_halved=float(final_error.mean())<.5*float(initial_error.mean()),
        production_synthetic_entry_denied=True)
    output=dict(updated=datetime.now(timezone.utc).isoformat(),state='synthetic_full_fit_passed' if all(checks.values()) else 'synthetic_full_fit_failed',
        checks=checks,source_sha256=sha(P/'src/shuolee_portrait/fitting_direct.py'),initializer_module_sha256=sha(P/'src/shuolee_portrait/initialization_inputs.py'),
        root=str(ROOT),steps=spec['steps'],views=n,initial_face_mean_px=float(initial_error.mean()),final_face_mean_px=float(final_error.mean()),
        final_face_max_px=float(final_error.max()),projection_parity_max_px=parity,all_losses=hist[-1],
        total_seconds=time.monotonic()-start,real_person_fit=False,physical_calibration_accepted=False,quality_accepted=False,
        scope='Full actual optimizer on a known mathematical SMPL-X projection oracle; not image detection, real accuracy or generalization proof.')
    save(O/'full-fit-result.json',output);print(json.dumps(output,ensure_ascii=False),flush=True)
except Exception as error:
    save(O/'full-fit-result.json',dict(state='synthetic_full_fit_implementation_failed',error=f'{type(error).__name__}: {error}',root=str(ROOT),real_person_fit=False,quality_accepted=False))
    (O/'full-fit-error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    save(O/'full-fit-progress.json',dict(state='terminal',pid=None,root=str(ROOT),seconds=time.monotonic()-start))
    if needlock.exists() and read(needlock).get('pid')==os.getpid():needlock.unlink()
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
