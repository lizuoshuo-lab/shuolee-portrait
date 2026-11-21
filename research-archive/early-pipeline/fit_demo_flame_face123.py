"""Original VHAP FLAME landmark-only prestage; no regressed 3D face prior.

Adapts frozen native images/K/2D labels and selects the reviewed reference.
All losses, trainable parameter groups and Adam schedules are upstream.
The fitted 3D landmarks are model estimates, never independent observations.
"""
from pathlib import Path
import argparse, copy, dataclasses, fcntl, json, logging, os, subprocess, sys, time, traceback, types
import numpy as np
from PIL import Image, ImageDraw
from run_official_gaussians123 import read, save, sha, now
from intake_demo_vhap123 import ROOT
from preprocess_demo_fps3_123 import DATA

REPO = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/VHAP_smplx')
PRIVATE = ROOT.parents[1]/'novel-view-face/head-structure/private-runtime'
OUT = ROOT/'flame-face-prestage'
WORK = DATA/'flame-face-prestage'
COMMIT = '8980d4948d91b05ff96ae91bae29e3f15a238f5e'
# Declared before fitting. Never selected by residuals. IBUG68 numbering.
HELD = np.array([19,24,29,31,35,37,40,43,46,50,52,56,58,62,66])
PER_FRAME = ('rotation','translation','neck_pose','jaw_pose','eyes_pose','expr')


def assets():
    source = PRIVATE/'code/VHAP/asset/flame'
    result = {n:source/n for n in ['head_template_mesh.obj','landmark_embedding_with_eyes.npy',
        'tex_mean_painted.png','uv_masks.npz']}
    result.update({'flame2023.pkl':PRIVATE/'models/flame2023/flame2023.pkl',
                   'FLAME_masks.pkl':PRIVATE/'models/flame-masks/FLAME_masks.pkl'})
    return result


def register():
    assert not OUT.exists(), 'Do not duplicate this registered face candidate.'
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    files=[Path(__file__), DATA/'data/inputs.json',DATA/'data/observations.npz',
           ROOT/'reference-selection/selection.json']
    files += [REPO/p for p in ['vhap/model/tracker.py','vhap/model/flame.py','vhap/model/lbs.py',
                              'vhap/config/base.py','vhap/util/render_nvdiffrast.py','vhap/util/camera.py']]
    files += list(assets().values())
    sources={str(p):sha(p) for p in files}
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),state='registered_face_only_original_VHAP',
        user_direction='VHAP自带面部拟合优先；没有再SMPLify；先拟合3D面部关键点再给全身VHAP；不用网络三维模板。',
        previous_turn_classification='progress: white control and original input-contract audit completed',
        original_repository='https://github.com/jmaraval/VHAP_smplx',commit=COMMIT,
        stage='Existing FLAME GlobalTracker landmark-only pipeline; photometric=False',
        inputs='59 native-frame RTMW 2D facial labels and conditional full K; no new detector or 3D-regression network',
        identity='one shared 300-dimensional FLAME shape; framewise rigid pose/neck/jaw/eyes/expression',
        initialization='Reviewed frame33/source256; neutral original FLAME + OpenCV solvePnP for its rigid seed; no previous fitted mesh',
        native_crop='One integer square size covering all original face boxes plus40px, per-frame integer translation only; K and2D points translate together. No image resize/warp.',
        observation_policy='IBUG inner51 only; contour17 excluded. Device-overlap slots25..31 nose/mouth confidence zero based on prior raw-image review, before errors. Other visibility remains unverified.',
        heldout_IBUG68=HELD.tolist(),heldout_kind='Different semantic labels excluded at every frame; same detector, not independent measured truth',
        budget=dict(reference_updates=1000,sequential_updates=2950,global_updates=1770,
                    max_total_updates=5720,max_seconds=420,GPU_tasks=1,new_downloads=0),
        scope='Head-only sparse geometry. No clothing or full-body fit and no Gaussian launch.',
        pass_condition='Actual all-frame original/2D/fitted3D projections; positive depth, fullK parity, held labels and side-view orientation; no automatic admission from low FIT error.',
        stop='One fresh run; stop on implementation/identity/finite/positive-depth failure or budget; no threshold sweep.',
        output_units='FLAME template units. Monocular absolute metric scale is not independently observed.',
        downstream='Export indexed camera-relative 3D face points, canonical reference and same K/frame identities. Do not copy FLAME vertex ids to SMPLX or ingest before review.',
        source_hashes=sources,network_template_used=False,quality_accepted=False))
    print(json.dumps({'plan':str(OUT/'plan.json')}),flush=True)


class Samples:
    def __init__(self,rows,obs,confidence,side):
        import torch
        from vhap.util.camera import convert_camera_convention
        self.rows=rows; self.obs=obs; self.confidence=confidence; self.side=side
        self.num_timesteps=len(rows); self.batchify_all_views=False; self.samples=[]; self.crops=[]
        for i,r in enumerate(rows):
            p=obs['keypoints'][i,23:91].copy(); center=(p.min(0)+p.max(0))/2
            xy=np.floor(center-side/2).astype(int); self.crops.append(xy)
            rgb=np.array(Image.open(r['image']).convert('RGB').crop((*xy,*(xy+side))),copy=True)
            K=np.asarray(r['K'],dtype=np.float64).copy(); K[:2,2]-=xy
            c2w,glK=convert_camera_convention('opencv->opengl',np.eye(4)[None],K[None],side,side)
            weights=confidence[i,23:91].copy(); weights[:17]=0; weights[HELD]=0
            if 25<=i<=31:weights[27:36]=0;weights[48:68]=0
            lmk=np.c_[p-xy+.5,weights].astype(np.float32)
            self.samples.append(dict(rgb=torch.from_numpy(rgb).permute(2,0,1).float()/255,
                intrinsic=torch.tensor(glK[0],dtype=torch.float32),
                extrinsic=torch.tensor(np.linalg.inv(c2w[0]),dtype=torch.float32),
                lmk2d=torch.tensor(lmk),timestep_index=i))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.samples[i]
    def one(self,i):
        import torch
        return {k:(torch.tensor([v]) if k=='timestep_index' else v[None]) for k,v in self[i].items()}


def make_tracker():
    import torch
    # Existing licensed FLAME pickle contains legacy chumpy objects. Match the
    # previously verified process-local compatibility, without editing packages.
    for name,value in dict(bool=np.bool_,int=np.int_,float=np.float64,complex=np.complex128,
                           object=np.object_,unicode=np.str_,str=np.str_).items():
        if name not in np.__dict__:setattr(np,name,value)
    sys.path.insert(0,str(REPO))
    from vhap.config import base as c
    from vhap.model.tracker import FlameTracker,GlobalTracker
    from smplify_score_contract123 import relative_confidence
    rows=read(DATA/'data/inputs.json')['records']
    with np.load(DATA/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    assert len(rows)==59 and list(obs['frame_ids'])==[r['frame'] for r in rows]
    confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    p=obs['keypoints'][:,23:91];side=int(np.ceil(np.max(p.max(1)-p.min(1))+40))
    pipeline=c.PipelineConfig(c.StageLmkInitRigidConfig(),c.StageLmkInitAllConfig(),c.StageLmkSequentialTrackingConfig(),
        c.StageLmkGlobalTrackingConfig(),c.StageRgbInitTextureConfig(),c.StageRgbInitAllConfig(),c.StageRgbInitOffsetConfig(),
        c.StageRgbSequentialTrackingConfig(),c.StageRgbGlobalTrackingConfig())
    cfg=c.BaseTrackingConfig(data=c.DataConfig(DATA,'demo',calibrated=True,scaled=True),
        model=c.ModelConfig(use_static_offset=False,use_dynamic_offset=False,tex_extra=False,tex_resolution=256),
        render=c.RenderConfig(),log=c.LogConfig(interval_scalar=100,interval_media=1000000),
        exp=c.ExperimentConfig(output_folder=OUT,photometric=False),lr=c.LearningRateConfig(),w=c.LossWeightConfig(),
        pipeline=pipeline,batch_size=1,async_func=False)
    class Adapter(GlobalTracker):
        def __init__(self):
            FlameTracker.__init__(self,cfg)
            self.calibrated=True;self.scaled=True;self.n_timesteps=len(rows);self.global_step=0
            self.out_dir=OUT;self.started=time.monotonic();self.stage_counts={}
            self.logger=logging.getLogger('vhap-face');self.log_interval_scalar=100;self.log_interval_media=1000000
            self.dataset=Samples(rows,obs,confidence,side);self.image_size=(side,side)
            self.init_params()
        def optimize_iter(self,sample,optimizer,stage):
            if time.monotonic()-self.started>420:raise RuntimeError('Face-stage budget420s reached.')
            super().optimize_iter(sample,optimizer,stage)
            self.stage_counts[stage]=self.stage_counts.get(stage,0)+1
        def log_scalars(self,log_dict,timestep,**kw):
            save(OUT/'progress.json',dict(updated=now(),state='face_only_fitting',pid=os.getpid(),step=self.global_step+1,
                frame=int(timestep),stage=kw.get('stage'),seconds=time.monotonic()-self.started,
                losses={k:float(v.detach()) for k,v in log_dict.items()},downstream_VHAP_started=False))
        def log_media(self,*a,**kw):pass
        def evaluate(self,make_visualization=True,epoch=0,evaluate=True):
            # Original schedule callback; write checkpoint, not a new solver.
            if epoch in [0,30]:export(self,rows,f'epoch-{epoch:02d}')
    return Adapter(),rows


def initialize_and_check(t,rows):
    import cv2,torch
    from vhap.model.lbs import vertices2landmarks
    slot=int(read(ROOT/'reference-selection/selection.json')['slot']);assert slot==32
    s=t.dataset.one(slot)
    with torch.no_grad():verts,can,lmks,albedo=t.forward_flame(s['timestep_index'])
    points=lmks[0].cpu().numpy();source=t.dataset.obs['keypoints'][slot,23:91]
    fit=np.arange(17,68)[s['lmk2d'][0,17:68,2].numpy()>0]
    K=np.asarray(rows[slot]['K_integer_centers'])
    ok,rv,tv=cv2.solvePnP(points[fit].astype(np.float64),source[fit].astype(np.float64),K,None,flags=cv2.SOLVEPNP_EPNP)
    assert ok
    with torch.no_grad():
        # FLAME applies root rotation around its regressed joint, whereas PnP
        # rotates the supplied point coordinates around the origin.
        t.rotation[:]=torch.tensor(rv.ravel(),device='cuda');t.translation.zero_()
        _,_,rotated,_=t.forward_flame(s['timestep_index'])
        desired=points@cv2.Rodrigues(rv)[0].T+tv.ravel()
        shift=desired-rotated[0].cpu().numpy()
        assert float(np.max(np.ptp(shift,axis=0)))<1e-5
        t.translation[:]=torch.tensor(shift.mean(0),device='cuda')
        v,c,l,a=t.forward_flame(s['timestep_index'])
        assert float(l[...,2].min())>.1
        native=l@torch.tensor(K,dtype=torch.float32,device='cuda').T;native=native[...,:2]/native[...,2:]
        ndc=t.render.world_to_ndc(l,s['extrinsic'].cuda(),s['intrinsic'].cuda(),t.image_size,flip_y=True)
        replay=(ndc[...,:2]+1)*(t.dataset.side/2)-.5+torch.tensor(t.dataset.crops[slot],device='cuda')
        delta=float(abs(replay-native).max());assert delta<.001,delta
        binding=vertices2landmarks(v,t.flame.faces,t.flame.full_lmk_faces_idx.repeat(len(v),1),t.flame.full_lmk_bary_coords.repeat(len(v),1,1))
        bdelta=float(abs(binding-l).max());assert bdelta<1e-7
    t.get_train_parameters('lmk_init_all')
    loss,*_=t.compute_energy(s,stage='lmk_init_all')
    gg=torch.autograd.grad(loss,[t.shape,t.expr,t.rotation,t.translation,t.jaw_pose],allow_unused=False)
    gstats={name:dict(finite=bool(torch.isfinite(g).all()),max_abs=float(g.abs().max())) for name,g in zip(['shape','expr','rotation','translation','jaw'],gg)}
    assert all(g['finite'] and g['max_abs']>0 for g in gstats.values())
    save(OUT/'preflight.json',dict(updated=now(),state='passed',optimizer_updates=0,anchor_slot=slot,
        native_fullK_reprojection_max_abs_px=delta,landmark_barycentric_replay_max_abs=bdelta,
        gradients=gstats,FLAME_landmarks_shape=list(l.shape),vertices=len(t.flame.v_template),
        template_seed_PnP_median_px=float(np.median(np.linalg.norm(native[0,fit].cpu().numpy()-source[fit],axis=1))),
        regressed_3D_model_loaded=False,conditional_K_unchanged=True,quality_accepted=False))
    return slot


def export(t,rows,label):
    import torch
    out=OUT/label;out.mkdir(exist_ok=False)
    arrays={k:getattr(t,k).detach().cpu().numpy() for k in ['shape','expr',*PER_FRAME,'scene_scale']}
    arrays['frame_ids']=np.array([r['frame'] for r in rows]);all3=[];all2=[];stats=[]
    from review_full_sequence123 import mesh_renderer
    render_mesh,_=mesh_renderer()
    for i,r in enumerate(rows):
        with torch.no_grad():v,c,l,_=t.forward_flame(torch.tensor([i]))
        l=l[0].cpu().numpy();v=v[0].cpu().numpy();K=np.array(r['K_integer_centers'])
        proj=l@K.T;proj=proj[:,:2]/proj[:,2:];assert l[:,2].min()>.1 and np.isfinite(proj).all()
        all3.append(l);all2.append(proj)
        raw=Image.open(r['image']).convert('RGB');xy=t.dataset.crops[i];box=tuple([*xy,*(xy+t.dataset.side)])
        target=t.dataset.obs['keypoints'][i,23:91];fit=t.dataset.samples[i]['lmk2d'][:,2].numpy()>0
        active=np.arange(17,68);error=np.linalg.norm(proj[:68]-target,axis=1)
        stats.append(dict(frame=r['frame'],fit_median_px=float(np.median(error[fit])),fit_max_px=float(error[fit].max()),
            held_median_px=float(np.median(error[HELD])),held_max_px=float(error[HELD].max()),
            device_overlap_frame=25<=i<=31,held_is_same_detector_not_3D_truth=True))
        if label=='epoch-30':
            orig=raw.crop(box);panels=[orig.copy(),orig.copy(),orig.copy()]
            for panel,points,color in [(panels[1],target[17:68],'#00b958'),(panels[2],proj[17:68],'#b632db')]:
                d=ImageDraw.Draw(panel)
                for x,y in points-xy:d.ellipse((x-1,y-1,x+1,y+1),outline=color,width=1)
            joined=Image.new('RGB',(3*t.dataset.side,t.dataset.side),'white')
            for j,p in enumerate(panels):joined.paste(p,(j*t.dataset.side,0))
            joined.save(out/f'{r["frame"]}_原图_二维点_拟合三维点投影.png')
            over=render_mesh(raw,np.array([0,0,*raw.size]),dict(vertices_camera=v[None],faces=t.flame.faces.cpu().numpy(),K_integer_centers=K))
            im=Image.new('RGB',(2*t.dataset.side,t.dataset.side),'white');im.paste(orig);im.paste(over.crop(box),(t.dataset.side,0))
            im.save(out/f'{r["frame"]}_原图_FLAME投影.png')
    arrays.update(landmarks3d_camera=np.array(all3),landmarks2d_native=np.array(all2),
        K_integer_centers=np.array([r['K_integer_centers'] for r in rows]),w2c=np.array([r['w2c'] for r in rows]))
    np.savez_compressed(out/'face-parameters-and-points.npz',**arrays)
    save(out/'result.json',dict(updated=now(),frames=len(rows),step=t.global_step,rows=stats,
        fit_median_of_frame_medians_px=float(np.median([s['fit_median_px'] for s in stats])),
        held_median_of_frame_medians_px=float(np.median([s['held_median_px'] for s in stats])),
        held_max_px=max(s['held_max_px'] for s in stats),barycentric_id_domain='FLAME; NOT SMPLX vertex indices',
        IBUG68_inner51=list(range(17,68)),independent_3D_ground_truth=False,quality_accepted=False))


def run(preflight_only=False,repair=None):
    import torch
    from torch.utils.data import DataLoader
    plan=read(OUT/'plan.json')
    marker=OUT/(f'{repair}-started.json' if repair else ('preflight-started.json' if preflight_only else 'fit-started.json'))
    assert not marker.exists(), 'Do not repeat or resume this stage.'
    fixes=read(OUT/'preflight-repair2.json') if (OUT/'preflight-repair2.json').exists() else (read(OUT/'preflight-repair.json') if (OUT/'preflight-repair.json').exists() else None)
    for p,h in plan['source_hashes'].items():
        expected=fixes['new_script_sha256'] if fixes and Path(p)==Path(__file__) else h
        assert sha(p)==expected,p
    assert subprocess.check_output(['git','-C',str(REPO),'-c','core.autocrlf=true','-c','core.filemode=false','status','--porcelain'],text=True)==''
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assetdir=WORK/'asset/flame';assetdir.mkdir(parents=True,exist_ok=True)
    for name,source in assets().items():
        dest=assetdir/name
        if dest.exists():assert dest.resolve()==source.resolve()
        else:dest.symlink_to(source)
    os.chdir(WORK);torch.set_num_threads(4);torch.manual_seed(6060);torch.backends.cuda.matmul.allow_tf32=False
    save(marker,dict(updated=now(),pid=os.getpid(),command=sys.argv));started=time.monotonic()
    try:
        t,rows=make_tracker();anchor=initialize_and_check(t,rows)
        save(OUT/'config.json',json.loads(json.dumps(dataclasses.asdict(t.cfg),default=str)))
        if preflight_only:return
        for stage in ['lmk_init_rigid','lmk_init_all']:t.optimize_stage(stage,sample=t.dataset.one(anchor))
        # Same bidirectional staging already used in the body adapter.
        t.backward_pass=False
        for name in ('compute_pose_smooth_energy','compute_joint_smooth_energy','compute_expr_smooth_energy'):
            original=getattr(t,name)
            def smooth(self,idx,_original=original):
                if not self.backward_pass:return _original(idx)
                proxy=copy.copy(self)
                for key in PER_FRAME:setattr(proxy,key,getattr(self,key).flip(0))
                return _original.__func__(proxy,self.n_timesteps-1-idx)
            setattr(t,name,types.MethodType(smooth,t))
        def cp(src,dst):
            with torch.no_grad():
                for key in PER_FRAME:getattr(t,key)[dst].copy_(getattr(t,key)[src])
        cp(anchor,anchor-1);t.optimize_stage('lmk_sequential_tracking',sample=t.dataset.one(anchor))
        for i in range(anchor+1,59):cp(i-1,i);t.optimize_stage('lmk_sequential_tracking',sample=t.dataset.one(i))
        t.backward_pass=True
        for i in range(anchor-1,-1,-1):cp(i+1,i);t.optimize_stage('lmk_sequential_tracking',sample=t.dataset.one(i))
        t.backward_pass=False;t.evaluate(epoch=0)
        loader=DataLoader(t.dataset,batch_size=1,shuffle=True,num_workers=0)
        t.optimize_stage('lmk_global_tracking',dataloader=loader,lr_scale=.1)
        assert t.global_step==5720
        save(OUT/'result.json',dict(updated=now(),state='complete_pending_face_review',frames=59,
            steps=t.global_step,stage_counts=t.stage_counts,elapsed_seconds=time.monotonic()-started,
            regressed_3D_prior_used=False,original_landmark_optimizer=True,downstream_VHAP_started=False,
            Gaussian_started=False,quality_accepted=False))
        save(OUT/'progress.json',dict(updated=now(),state='complete_pending_face_review',pid=None,step=t.global_step))
    except BaseException as e:
        save(OUT/(f'{repair}-failure.json' if repair else ('preflight-failure.json' if preflight_only else 'failure.json')),dict(updated=now(),error=repr(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','preflight','preflight-repaired','preflight-barycentric','fit']);arg=p.parse_args()
    if arg.mode=='register':register()
    else:run(arg.mode.startswith('preflight'),arg.mode if arg.mode in ['preflight-repaired','preflight-barycentric'] else None)
