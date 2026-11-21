"""Replay one next-frame tracking step from exact saved maps; compare pose selection only.

No map update, trajectory correction, GT fitting, or official source modification.
The two candidate policies share the identical optimizer path.
"""
from pathlib import Path
import csv, hashlib, importlib.util, json, sys, time, traceback
from datetime import datetime, timezone
import numpy as np

LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
ROOT=Path('/opt/gs-slam-lab/splatam')
REPO=ROOT/'repos/SplaTAM'
NAME='freiburg1_room_full_320x240_rgbd_s0_20261010T125931Z_0d3dc918'
RUN=ROOT/'outputs/runs'/NAME
TRAJ=LAB/'outputs/splatam-repeat-evaluations-001'/NAME/'trajectory'
OUT=LAB/'outputs/splatam-candidate-selection-001'
CHECKPOINTS=(100,800,1100)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=False)
    config=read(RUN/'effective-config.json'); execution=read(RUN/'execution.json')
    files=[Path(__file__),RUN/'effective-config.json',RUN/'execution.json',RUN/'params.npz',RUN/'sensor-config.yaml',TRAJ/'metrics.json',TRAJ/'reference-independent.tum']+[RUN/f'params{k}.npz' for k in CHECKPOINTS]
    hashes={str(p):sha(p) for p in files}
    for rel,h in execution['source_sha256_before'].items():assert sha(REPO/rel)==h,rel
    for key in ('extension_file','module_file'):
        p=Path(execution['renderer_provenance'][key]);hashes[str(p)]=sha(p)
    assert hashes[execution['renderer_provenance']['extension_file']]==execution['renderer_provenance']['extension_sha256_before']
    assert config['tracking']['num_iters']==200 and not config['tracking']['use_gt_poses']
    assert not config['tracking']['use_depth_loss_thres']
    plan={'status':'registered_before_new_GPU_work','checkpoints':CHECKPOINTS,'query_frames':[n+1 for n in CHECKPOINTS],
          'input_hashes':hashes,'official_sources':execution['source_sha256_before'],
          'control':'Same post-mapping checkpoint, exact next RGB-D frame, same official constant-velocity initialization and 200 Adam steps. Only candidate selection differs.',
          'policies':['official: compare loss before step, store pose after step','consistent: compare loss and store pose before the same step'],
          'primary_measure':'Re-render selected poses with original loss; compare support and fixed original full-trajectory alignment APE.',
          'selection':'Frames chosen in advance: checkpoint100 as earlier control,800 and1100 matching previous mechanism probes. No perturbations or GT optimization.',
          'limits':['Sparse one-frame replay, not a new full SLAM run or drift cause attribution','Historical loss/map at jumps838/1133 not reconstructed','Consistent loss selection need not improve independent pose accuracy']}
    save(OUT/'plan.json',plan);(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0,str(REPO))
    spec=importlib.util.spec_from_file_location('candidate_official',REPO/'scripts/splatam.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    torch.set_num_threads(4)
    data=ROOT/'data/tum/rgbd_dataset_freiburg1_room'
    dataset=official.get_dataset(config_dict=official.load_dataset_config(str(RUN/'sensor-config.yaml')),basedir=str(data.parent),sequence=data.name,start=0,end=-1,stride=1,desired_height=240,desired_width=320,device='cuda',relative_pose=True,ignore_bad=config['data'].get('ignore_bad',False),use_train_split=config['data'].get('use_train_split',True))
    with np.load(RUN/'params.npz',allow_pickle=False) as z:
        k=z['intrinsics'].copy();w2c=z['w2c'].copy();hist_rots=z['cam_unnorm_rots'].copy();hist_trans=z['cam_trans'].copy()
    cam=official.setup_camera(320,240,k,w2c)
    captured=[];real_renderer=official.Renderer
    def factory(*args,**kwargs):
        renderer=real_renderer(*args,**kwargs)
        def forward(**kw):
            out=renderer(**kw);captured.append(out[0]);return out
        return forward
    official.Renderer=factory
    all_rows=[];summary=[]
    start=time.perf_counter()
    for checkpoint in CHECKPOINTS:
        f=checkpoint+1;raw=dict(np.load(RUN/f'params{checkpoint}.npz',allow_pickle=False))
        params={n:torch.tensor(v,device='cuda',dtype=torch.float32,requires_grad=n.startswith('cam_')) for n,v in raw.items()}
        gaussians=len(raw['means3D']);variables={'max_2D_radius':torch.zeros(gaussians,device='cuda')}
        color,depth,intrinsics,_unused_gt=dataset[f]
        assert np.array_equal(k,intrinsics[:3,:3].cpu().numpy())
        curr={'im':color.permute(2,0,1)/255,'depth':depth.permute(2,0,1),'cam':cam,'w2c':torch.tensor(w2c,device='cuda')}
        source_image_hashes={str(dataset.color_paths[f]):sha(dataset.color_paths[f]),str(dataset.depth_paths[f]):sha(dataset.depth_paths[f])}
        save(OUT/f'frame{f}-inputs.json',source_image_hashes)
        params=official.initialize_camera_pose(params,f,config['tracking']['forward_prop'])
        opt=official.initialize_optimizer(params,config['tracking']['lrs'],tracking=True)
        def posecopy():return tuple(params[n][...,f].detach().clone() for n in ['cam_unnorm_rots','cam_trans'])
        def assign(pose):
            with torch.no_grad():
                for n,v in zip(['cam_unnorm_rots','cam_trans'],pose):params[n][...,f].copy_(v)
        def loss_and_obs():
            captured.clear()
            loss,_,weighted=official.get_loss(params,curr,variables,f,config['tracking']['loss_weights'],True,.99,True,False,tracking=True)
            assert len(captured)==2
            im,ds=captured;d=ds[0:1];var=ds[2:3]-d*d
            mask=((curr['depth']>0)&~torch.isnan(d)&~torch.isnan(var)&(ds[1]>.99)).detach()
            count=int(mask.sum().item());assert count>0
            independent=(.5*torch.abs(curr['im']-im).sum(dim=0,keepdim=True)+torch.abs(curr['depth']-d))[mask].sum()
            assert torch.isfinite(loss) and torch.isclose(loss,independent,rtol=2e-5,atol=.01)
            return loss,{'loss':float(loss.detach()),'support_pixels':count,'loss_per_support':float(loss.detach())/count}
        initial=posecopy();best_loss=float('inf');official_pick=initial;consistent_pick=initial;best_iter=None
        for iteration in range(200):
            before=posecopy();loss,obs=loss_and_obs()
            loss.backward();opt.step();opt.zero_grad(set_to_none=True)
            if obs['loss']<best_loss:
                best_loss=obs['loss'];best_iter=iteration;consistent_pick=before;official_pick=posecopy()
            all_rows.append({'checkpoint':checkpoint,'query_frame':f,'iteration':iteration,**obs})
            del loss;captured.clear()
        candidates={'initial':initial,'official_selection':official_pick,'consistent_selection':consistent_pick,
                    'historical_saved_next_pose':(torch.tensor(hist_rots[...,f],device='cuda'),torch.tensor(hist_trans[...,f],device='cuda'))}
        result={'checkpoint':checkpoint,'frame':f,'gaussians':gaussians,'best_pre_step_iteration':best_iter,'candidate_selection_reference_loss':best_loss,'candidates':{}}
        for name,pose in candidates.items():
            assign(pose);loss,obs=loss_and_obs();im=captured[0].detach().permute(1,2,0).cpu().numpy()
            Image.fromarray((np.clip(im,0,1)*255).round().astype(np.uint8)).save(OUT/f'f{f}-{name}.png')
            R=official.build_rotation(F.normalize(params['cam_unnorm_rots'][...,f])).detach().cpu().numpy()
            T=np.eye(4);T[:3,:3]=R;T[:3,3]=params['cam_trans'][...,f].detach().cpu().numpy().reshape(3)
            result['candidates'][name]={**obs,'w2c':T.tolist()}
            del loss;captured.clear()
        Image.fromarray(color.cpu().numpy().astype(np.uint8)).save(OUT/f'f{f}-input.png')
        for n in raw:
            if not n.startswith('cam_'):assert np.array_equal(params[n].detach().cpu().numpy(),raw[n]),n
        assert all(sha(Path(p))==h for p,h in source_image_hashes.items())
        summary.append(result)
        save(OUT/'partial.json',{'status':'running','completed_frames':summary})
        print(json.dumps({'frame':f,'best_iteration':best_iter,'selected_losses':{n:v['loss'] for n,v in result['candidates'].items()}}),flush=True)
        del params,raw,opt,variables,curr,candidates,initial,official_pick,consistent_pick,before,color,depth,im
        torch.cuda.empty_cache()
    # Independent reference is read only after all candidate policies finish.
    ref=np.loadtxt(TRAJ/'reference-independent.tum');align=np.array(read(TRAJ/'metrics.json')['alignment']['transform_estimated_to_reference'])
    for row in summary:
        for c in row['candidates'].values():
            c2w=np.linalg.inv(c['w2c']);aligned=align@c2w
            c['frozen_alignment_position_error_m']=float(np.linalg.norm(aligned[:3,3]-ref[row['frame'],1:4]))
        a=row['candidates']['official_selection'];b=row['candidates']['consistent_selection']
        row['consistent_minus_official_loss']=b['loss']-a['loss']
        row['consistent_minus_official_position_error_m']=b['frozen_alignment_position_error_m']-a['frozen_alignment_position_error_m']
    with (OUT/'iterations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=all_rows[0]);writer.writeheader();writer.writerows(all_rows)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    assert all(sha(REPO/p)==h for p,h in execution['source_sha256_before'].items())
    result={'status':'completed_three_next_frame_tracking_replays','rows':summary,'optimizer_steps':600,'wall_seconds':time.perf_counter()-start,
            'official_sources_and_maps_unchanged':True,'shared_optimizer_path':True,'trajectory_accuracy_improvement_not_claimed':True,
            'reference_alignment':'Existing full-sequence SE3 transform held fixed, used after optimization only','plan_sha256':sha(OUT/'plan.json')}
    save(OUT/'metrics.json',result);print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds'],'output':str(OUT)}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():(OUT/'failure.txt').write_text(traceback.format_exc())
        raise
