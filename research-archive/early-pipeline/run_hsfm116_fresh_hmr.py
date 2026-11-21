"""Fresh13-image HMR2.0a SMPL initialization; no previous subject geometry."""
from pathlib import Path
import datetime as dt, hashlib, json, os, subprocess, sys, time, traceback
import numpy as np
from PIL import Image,ImageDraw
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
B=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee';H=B/'current/portrait-alignment-open-source-116/hsfm'
J=H/'joint-diagnostic-reserved';O=J/'fresh-hmr'
PRIVATE=B/'current/novel-view-face/head-structure/private-runtime/python-deps'
REPO=LAB/'repos/GVHMR-shuolee'
sys.path[:0]=[str(REPO),str(PRIVATE),str(PRIVATE/'smplx-official')]
os.chdir(REPO)
from hsfm116_adapter import colmap_to_raster_K,estimate_root_translation
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,v):
    p=O/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
    assert not O.exists(),'Fresh initializer already exists; do not repeat or overwrite'
    assert read(J/'frozen-validation.json')['coverage_passed'],'Frozen validation has insufficient coverage'
    O.mkdir();t0=time.monotonic()
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True)
    procs=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:s=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        if any(x in s for x in ('python','train','supervis')):procs.append({'pid':int(p.parent.name),'command':s})
    save('process-check.json',dict(updated=now(),self_pid=os.getpid(),processes=procs,gpu=gpu.stdout))
    assert gpu.returncode==0 and not gpu.stdout.strip(),'Existing GPU workload'
    rec=read(H/'input-pilot/inputs.json')['records'];obs=np.load(H/'input-pilot/observations.npz')
    vis=np.load(J/'visibility.npz');assert np.array_equal(obs['raw_frame_ids'],vis['raw_frame_ids'])
    save('protocol.json',dict(created=now(),script_sha256=sha(__file__),preregistration_sha256=sha(J/'preregistration.json'),
        raw_frame_ids=[r['raw_frame'] for r in rec],initializer='Existing official HMR2.0a checkpoint, full head in GVHMR adapter; not HSfM default2.0b',
        bbox='Mask xyxy, square side1.2*max(height,width/0.75), center=(min+max-1)/2 integer-center convention',
        crop='Native get_batch RGB preprocessing/antialias/normalization, 256square then backbone central192width; only network input crop',
        root='SMPL45 named body23, weighted perspective LS in original full-image K. Not HMR weak-perspective translation.',
        fresh_inference=True,previous_subject_fit_loaded=False,training=False,gaussian=False,budget_seconds=300))
    import torch
    from hmr4d.network.hmr2 import HMR2
    from hmr4d.network.hmr2.configs import get_config
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch
    from pytorch3d.transforms import axis_angle_to_matrix
    import smplx
    torch.set_num_threads(2);torch.manual_seed(116)
    cfg=get_config(str(REPO/'hmr4d/network/hmr2/configs/model_config.yaml'));cfg.defrost();cfg.MODEL.BBOX_SHAPE=[192,256];cfg.freeze()
    model=HMR2(cfg)
    ckpt=torch.load('/mnt/d/GS-SLAM-Data/models/gvhmr/hmr2/epoch=10-step=25000.ckpt',map_location='cpu',weights_only=True,mmap=True)
    state={k:v for k,v in ckpt['state_dict'].items() if k.split('.')[0] in ['backbone','smpl_head']}
    model.load_state_dict(state,strict=True,assign=True);del ckpt,state
    model=model.cuda().eval();torch.cuda.reset_peak_memory_stats()
    params=[]
    for i,r in enumerate(rec):
        assert sha(r['image'])==r['image_sha256'];rgb=np.asarray(Image.open(r['image']).convert('RGB'))
        b=np.array(r['bbox_xyxy']);ctr=(b[:2]+b[2:]-1)/2;wh=b[2:]-b[:2];side=1.2*max(wh[1],wh[0]/.75)
        bx=torch.tensor([[*ctr,side]],dtype=torch.float32)
        img,bbox=get_batch(rgb[None],bx,img_ds=1.,path_type='np')
        K=torch.tensor(colmap_to_raster_K(r['K_from_existing_manifest']),dtype=torch.float32)[None]
        with torch.no_grad():pr=model({'img':img.cuda(),'bbx_xys':bbox.cuda(),'K_fullimg':K.cuda()},feat_mode=False)['smpl_params']
        pr={k:v.detach().cpu() for k,v in pr.items()};assert all(torch.isfinite(v).all() for v in pr.values())
        params.append(pr);save('progress.json',dict(updated=now(),state='fresh_forward',pid=os.getpid(),done=i+1,total=len(rec)))
        assert time.monotonic()-t0<300
    del model;torch.cuda.empty_cache()
    # User-provided neutral SMPL template, loaded by its public loader; no old fitted parameters.
    layer=smplx.SMPL(str(LAB/'repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl'),gender='neutral',num_betas=10,batch_size=1).cuda().eval()
    mapping=np.array([24,26,25,28,27,16,17,18,19,20,21,1,2,4,5,7,8,29,30,31,32,33,34])
    body=[];beta=[];glob=[];roots=[];joints=[];root_checks=[];boards=[]
    for i,(r,p) in enumerate(zip(rec,params)):
        with torch.no_grad():pred=layer(betas=p['betas'].cuda(),body_pose=p['body_pose'].cuda(),global_orient=p['global_orient'].cuda())
        X=pred.joints[0].cpu().numpy();X-=X[:1].copy()
        K=colmap_to_raster_K(r['K_from_existing_manifest'])
        # Side roots are initializer diagnostics only. Optimization uses frozen visible-front subset.
        conf=np.where(vis['visible'][i],obs['scores'][i,:23],0) if vis['visible'][i].sum()>=4 else obs['scores'][i,:23]
        root,check=estimate_root_translation(X[mapping],obs['keypoints'][i,:23],K,conf,.5)
        proj=(X[mapping]+root)@K.T;proj=proj[:,:2]/proj[:,2:]
        error=np.linalg.norm(proj-obs['keypoints'][i,:23],axis=-1)
        check.update(raw_frame=r['raw_frame'],used_for_joint_constraint=bool(vis['visible'][i].any()),same_name_reprojection_median=float(np.median(error[conf>=.5])))
        root_checks.append(check);roots.append(root);joints.append(X)
        body.append(p['body_pose'].numpy()[0]);glob.append(p['global_orient'].numpy()[0]);beta.append(p['betas'].numpy()[0])
        rgb=Image.open(r['image']).convert('RGB');s=600/rgb.height;w=round(rgb.width*s)
        panel=Image.new('RGB',(w,630),'white');panel.paste(rgb.resize((w,600)),(0,30));d=ImageDraw.Draw(panel)
        d.text((4,5),f"raw{r['raw_frame']:06d} | observed green / HMR red",fill='black')
        for k in range(23):
            if conf[k]<.5:continue
            a=obs['keypoints'][i,k]*s+[0,30];b=proj[k]*s+[0,30]
            d.line([tuple(a),tuple(b)],fill='yellow',width=1);d.ellipse((*tuple(a-2),*tuple(a+2)),fill='lime');d.ellipse((*tuple(b-2),*tuple(b+2)),fill='red')
        boards.append(panel)
    np.savez_compressed(O/'fresh-priors.npz',raw_frame_ids=[r['raw_frame'] for r in rec],body_pose=np.array(body),global_orient=np.array(glob),betas=np.array(beta),root_transl=np.array(roots),root_relative_joints=np.array(joints),smpl45_to_body23=mapping)
    for b in range(0,len(boards),4):
        ps=boards[b:b+4];im=Image.new('RGB',(sum(p.width for p in ps),630),'white');x=0
        for p in ps:im.paste(p,(x,0));x+=p.width
        im.save(O/f'fresh-prior-review-{b//4+1}.jpg',quality=95)
    save('result.json',dict(updated=now(),state='fresh_hmr_initializer_complete_not_alignment_acceptance',frames=len(rec),seconds=time.monotonic()-t0,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,root_checks=root_checks,output_sha256=sha(O/'fresh-priors.npz'),
        reused_subject_fit=False,physical_camera_accuracy_verified=False,subject_fit_accepted=False,gaussian_training=False))
    save('progress.json',dict(updated=now(),state='completed',pid=os.getpid()));print('Fresh HMR inference completed',flush=True)
try:main()
except Exception as e:
    if O.exists():save('failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
    raise
