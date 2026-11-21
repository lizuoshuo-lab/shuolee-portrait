"""Bounded checkpoint continuations with observable tracking support.

Saved maps/poses are exact inputs, but RNG/history are not complete checkpoints:
these are new continuations, never claimed to reproduce historical maps.
"""
from pathlib import Path
import ast,copy,csv,hashlib,importlib.util,json,shutil,sys,time,traceback
import numpy as np
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
ROOT=Path('/opt/gs-slam-lab/splatam');REPO=ROOT/'repos/SplaTAM'
NAME='freiburg1_room_full_320x240_rgbd_s0_20261010T125931Z_0d3dc918'
RUN=ROOT/'outputs/runs'/NAME
OUT=LAB/'outputs/splatam-room-window-continuation-001'
WINDOWS=((800,842),(1100,1137))
FOCAL=set(range(835,841))|set(range(1130,1136))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
class WindowFinished(Exception):pass

def main():
    OUT.mkdir(exist_ok=False)
    config=read(RUN/'effective-config.json');old=read(RUN/'execution.json')
    for rel,h in old['source_sha256_before'].items():assert sha(REPO/rel)==h,rel
    assert not config['tracking']['use_gt_poses'] and config['tracking']['num_iters']==200
    assert not config['mapping']['use_gaussian_splatting_densification']
    save(OUT/'plan.json',dict(status='registered_before_GPU',windows=WINDOWS,seed_per_window=0,
        source_sha256=sha(REPO/'scripts/splatam.py'),script_sha256=sha(__file__),
        changes=['Resume at checkpoint + 1, because saved checkpoint already contains that frame mapping','Rebuild keyframe/GT list through checkpoint inclusive','Read-only after-tracking and end-frame observation hooks','Suppress progress evaluation/display; original training/mapping losses and candidate policy unchanged'],
        boundaries=['RNG is newly seeded, not restored historical state','Upstream resume resets auxiliary arrays; keyframe poses reconstructed from saved poses','This is a diagnostic continuation, not exact historical replay or full-sequence evaluation','GT cannot select online candidates; independent evaluation after both windows'],
        checkpoints={str(k):sha(RUN/f'params{k}.npz') for k,_ in WINDOWS}))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4);sys.path.insert(0,str(REPO))
    module_spec=importlib.util.spec_from_file_location('window_official',REPO/'scripts/splatam.py')
    official=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(official)
    src=(REPO/'scripts/splatam.py').read_text();tree=ast.parse(src)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rgbd_slam')
    edits=[]
    for n in ast.walk(fn):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='range' and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id=='checkpoint_time_idx':
            n.args[0]=ast.BinOp(n.args[0],ast.Add(),ast.Constant(1));edits.append(n.lineno)
    assert len(edits)==2,edits
    class Observe(ast.NodeTransformer):
        def visit_Assign(self,n):
            if any(isinstance(t,ast.Name) and t.id=='tracking_end_time' for t in n.targets):
                return [ast.parse('_observe_after_tracking(params, variables, tracking_curr_data, time_idx)').body[0],n]
            return self.generic_visit(n)
        def visit_Expr(self,n):
            if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.cuda.empty_cache':
                return [ast.parse('_observe_frame_end(params, time_idx)').body[0],n]
            return self.generic_visit(n)
    fn=Observe().visit(fn);module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    (OUT/'derived-rgbd-slam.py').write_text(ast.unparse(module),encoding='utf-8')
    exec(compile(module,str(OUT/'derived-rgbd-slam.py'),'exec'),official.__dict__)
    real_loss=official.get_loss;real_renderer=official.Renderer
    captured=[];capture=False;rows=[];iterations=[];state={};window_root=None
    def factory(*args,**kw):
        renderer=real_renderer(*args,**kw)
        def forward(**kwargs):
            result=renderer(**kwargs)
            if capture:captured.append(result[0])
            return result
        return forward
    official.Renderer=factory
    def statistics(data,loss):
        assert len(captured)==2
        im,ds=[x.detach() for x in captured];d=ds[0:1];var=ds[2:3]-d*d
        valid=(data['depth']>0)&torch.isfinite(d)&torch.isfinite(var)
        mask=valid&(ds[1]>.99)
        color=.5*(data['im']-im).abs().sum(0,keepdim=True);depth=(data['depth']-d).abs()
        count=int(mask.sum().item());assert count>0
        total=float((color+depth)[mask].sum())
        assert np.isclose(total,float(loss.detach()),rtol=2e-5,atol=.02)
        fixed=state.get('initial_mask',mask)&valid
        return dict(loss=total,support_pixels=count,fixed_support_pixels=int(fixed.sum()),
            depth_mean_m=float(depth[mask].mean()),weighted_color_mean=float(color[mask].mean()),
            fixed_support_depth_mean_m=float(depth[fixed].mean()),fixed_support_weighted_color_mean=float(color[fixed].mean()),
            fixed_support_loss=float((color+depth)[fixed].sum())),mask
    def w2c(params,f):
        t=torch.eye(4,device='cuda');t[:3,:3]=official.build_rotation(F.normalize(params['cam_unnorm_rots'][...,f].detach()));t[:3,3]=params['cam_trans'][...,f].detach()
        return t.cpu().numpy().tolist()
    def observed_loss(*args,**kw):
        nonlocal capture
        tracking=kw.get('tracking',False);f=args[3];it=kw.get('tracking_iteration',-1)
        capture=tracking and (f in FOCAL or it==0);captured.clear()
        if tracking and it==0:
            state.clear();state['initial_pose']=w2c(args[0],f)
            if f in (838,1133):
                np.savez(window_root/f'pretracking-map-{f}.npz',**{k:v.detach().cpu().numpy() for k,v in args[0].items()})
                save(window_root/f'pretracking-binding-{f}.json',dict(frame=f,map_gaussians=len(args[0]['means3D']),new_continuation=True))
        result=real_loss(*args,**kw)
        if capture:
            obs,mask=statistics(args[1],result[0])
            if it==0:state['initial_mask']=mask.detach().clone();state['initial']=obs
            if f in FOCAL:iterations.append(dict(frame=f,iteration=it,**obs))
        capture=False;captured.clear();return result
    official.get_loss=observed_loss
    def after(params,variables,data,f):
        nonlocal capture
        capture=True;captured.clear()
        private=dict(variables);private['max_2D_radius']=variables['max_2D_radius'].clone()
        result=real_loss(params,data,private,f,config['tracking']['loss_weights'],True,.99,True,False,tracking=True)
        obs,mask=statistics(data,result[0]);capture=False
        rows.append(dict(frame=f,gaussians_before_mapping=len(params['means3D']),initial=state['initial'],selected=obs,initial_w2c=state['initial_pose'],selected_w2c=w2c(params,f)))
        save(OUT/'frames.json',rows)
        del result,private;captured.clear()
    def end(params,f):
        save(OUT/'progress.json',dict(status='running',completed_frame=f,completed_windows=len(summaries),tracking_frames=len(rows)))
        if f%5==0:print(json.dumps({'frame':f,'gaussians':len(params['means3D'])}),flush=True)
        if f==stop:
            np.savez(window_root/'final-params.npz',**{k:v.detach().cpu().numpy() for k,v in params.items()})
            raise WindowFinished()
    official._observe_after_tracking=after;official._observe_frame_end=end
    summaries=[];start=time.perf_counter()
    for checkpoint,stop in WINDOWS:
        window_root=OUT/f'from-{checkpoint}';window_root.mkdir()
        for filename in (f'params{checkpoint}.npz',f'keyframe_time_indices{checkpoint}.npy'):
            shutil.copy2(RUN/filename,window_root/filename)
        cfg=copy.deepcopy(config);cfg.update(load_checkpoint=True,checkpoint_time_idx=checkpoint,workdir=str(OUT),run_name=window_root.name,
            report_iter_progress=False,report_global_progress_every=100000,save_checkpoints=False)
        save(window_root/'config.json',cfg)
        official.seed_everything(0)
        try:official.rgbd_slam(cfg)
        except WindowFinished:pass
        assert rows[-1]['frame']==stop
        summaries.append(dict(checkpoint=checkpoint,last_frame=stop,tracking_frames=stop-checkpoint))
        torch.cuda.empty_cache()
    with (OUT/'iterations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=iterations[0]);w.writeheader();w.writerows(iterations)
    traj=LAB/'outputs/splatam-repeat-evaluations-001'/NAME/'trajectory'
    ref=np.loadtxt(traj/'reference-independent.tum');align=np.array(read(traj/'metrics.json')['alignment']['transform_estimated_to_reference'])
    with np.load(RUN/'params.npz') as oldparams:
        for row in rows:
            f=row['frame'];c=np.linalg.inv(row['selected_w2c']);row['fixed_alignment_position_error_m']=float(np.linalg.norm((align@c)[:3,3]-ref[f,1:4]))
            histq=torch.tensor(oldparams['cam_unnorm_rots'][...,f],device='cuda');histt=oldparams['cam_trans'][...,f]
            hist=np.eye(4);hist[:3,:3]=official.build_rotation(F.normalize(histq)).cpu().numpy();hist[:3,3]=histt.reshape(3)
            row['historical_position_error_m']=float(np.linalg.norm((align@np.linalg.inv(hist))[:3,3]-ref[f,1:4]))
            row['continuation_vs_historical_position_m']=float(np.linalg.norm(c[:3,3]-np.linalg.inv(hist)[:3,3]))
    save(OUT/'frames.json',rows)
    for rel,h in old['source_sha256_before'].items():assert sha(REPO/rel)==h
    result=dict(status='completed_new_checkpoint_continuations',windows=summaries,tracking_frames=len(rows),tracking_optimizer_steps=200*len(rows),mapping_iterations=30*len(rows),
        wall_seconds=time.perf_counter()-start,GT_used_for_candidate_choice=False,official_source_unchanged=True,exact_historical_replay=False,
        focal_frames=[r for r in rows if r['frame'] in (837,838,839,1132,1133,1134)])
    save(OUT/'metrics.json',result);save(OUT/'progress.json',dict(status='completed',tracking_frames=len(rows)))
    print(json.dumps({k:v for k,v in result.items() if k!='focal_frames'}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
