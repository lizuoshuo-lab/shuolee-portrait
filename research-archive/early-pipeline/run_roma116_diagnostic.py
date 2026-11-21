"""Single preregistered RoMa material-correspondence diagnostic; requires official local weights."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
sys.path[:0]=['/opt/gs-slam-lab/shuolee-roma116/python','/opt/gs-slam-lab/shuolee-roma116/RoMa']
from roma116_coordinates import sample_direction
from evaluate_pixsfm116_candidate import point_fit,quant

B=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/portrait-alignment-open-source-116';R=O/'roma';S=B/'current/full-scene-localization-074'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(8388608),b''):h.update(data)
    return h.hexdigest()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def no_download(*a,**kw):raise RuntimeError('Use explicitly verified local official weights; network fetching disabled')
def timed_out(*_):raise TimeoutError('Registered RoMa smoke budget exhausted')
def epipolar_distance(xy_a,xy_b,ea,eb,k):
    """Symmetric epipolar distance in the undistorted f-pixel plane, not raw pixels."""
    def rays(xy):
        d=(xy+.5-k[1:3])/k[0];u=d.copy()
        for _ in range(12):u=d/(1+k[3]*(u*u).sum(-1,keepdims=True))
        return np.c_[u,np.ones(len(u))]
    a,b=rays(xy_a),rays(xy_b);rot=eb[:3,:3]@ea[:3,:3].T;t=eb[:3,3]-rot@ea[:3,3]
    tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    essential=tx@rot;lb=a@essential.T;la=b@essential
    residual=np.abs((b*lb).sum(-1));da=np.linalg.norm(la[:,:2],axis=-1);db=np.linalg.norm(lb[:,:2],axis=-1)
    with np.errstate(divide='ignore',invalid='ignore'):
        result=.5*k[0]*residual*(1/da+1/db)
    return np.where((da>1e-12)&(db>1e-12),result,np.nan)
def process_guard():
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    jobs=[line for line in query.splitlines() if line.strip() and line.split(',')[0].strip()!=str(os.getpid())]
    if jobs:raise RuntimeError('GPU already has a compute task: '+repr(jobs))
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:a=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (OSError,UnicodeDecodeError):continue
        if any(x in a for x in ('run_pixsfm116_ba.py','train_smplx.py','run_roma116_diagnostic.py')) and ('python' in a or 'bash' in a):
            # The parent launcher names this script too; only competing Python interpreters count here.
            if 'python' in a and int(p.name)!=os.getppid():raise RuntimeError(f'Competing job PID{p.name}: {a}')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weights-dir',required=True);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    reg=read(R/'smoke-preregistration.json');freeze=read(R/'runner-manifest.json')
    assert sha(__file__)==freeze['runner_sha256'] and sha(Path(__file__).parent/'roma116_coordinates.py')==freeze['coordinate_adapter_sha256']
    assert sha(R/'smoke-preregistration.json')==freeze['preregistration_sha256']
    assert sha(R/'weight-manifest.json')==freeze['weight_manifest_sha256']
    assert sha(R/'probes.npz')==reg['probes']['sha256'];assert sha(O/'pixsfm/independent-validation/tracks.npz')==reg['frozen_validation_sha256']
    for path,digest in reg['input_hashes'].items():assert sha(path)==digest,path
    for path,digest in freeze['source_camera_hashes'].items():assert sha(path)==digest,path
    manifest=read(R/'weight-manifest.json');files={}
    for m in manifest['files']:
        p=Path(args.weights_dir)/m['filename'];assert p.is_file(),f'Missing official weight {p}'
        assert p.stat().st_size==m['expected_bytes'],f'Unexpected length {p}'
        files[m['filename']]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
    if args.verify_only:
        print(json.dumps(dict(state='weight_files_present_structure_not_yet_loaded',files=files)));return
    assert not (R/'smoke').exists(),'Never repeat an active or completed diagnostic'
    process_guard();out=R/'smoke';out.mkdir();start=time.monotonic()
    signal.signal(signal.SIGALRM,timed_out);signal.alarm(reg['budget_seconds'])
    def status(stage,**kw):save(out/'progress.json',dict(updated=now(),pid=os.getpid(),state='running',stage=stage,**kw))
    try:
        status('load_official_weights');save(out/'input-weights.json',files)
        torch.manual_seed(reg['seed']);np.random.seed(reg['seed']);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.hub.load_state_dict_from_url=no_download;torch.hub.load=no_download
        from romatch import roma_outdoor
        state=torch.load(files['roma_outdoor.pth']['path'],map_location='cpu',weights_only=True)
        dino=torch.load(files['dinov2_vitl14_pretrain.pth']['path'],map_location='cpu',weights_only=True)
        model=roma_outdoor(device='cuda',weights=state,dinov2_weights=dino,coarse_res=tuple(reg['official_model']['coarse_resolution_hw']),
            upsample_res=tuple(reg['official_model']['upsample_resolution_hw']),amp_dtype=torch.float16,symmetric=True,use_custom_corr=False,do_compile=False).eval()
        del state,dino
        with np.load(R/'probes.npz') as z:probes=z['probes'];triples=z['triples']
        cameras=read(S/'cameras.json');E=np.array([r['w2c'] for r in cameras['records']]);intr=np.array(cameras['intrinsics'])
        assert len(E)==175 and cameras['model']=='SIMPLE_RADIAL'
        assert [r['source_frame_index'] for r in cameras['records']]==list(range(1,176))
        assert [r['name'] for r in cameras['records']]==[f'raw_{i:06d}.png' for i in range(1,176)]
        results=[];timings=[]
        for ti,triple in enumerate(triples):
            a,b,c=map(int,triple);ids=np.flatnonzero(probes[:,0]==a);q=torch.tensor(probes[ids,2:],device='cuda',dtype=torch.float32)
            point_targets={};confs={};backs={};validity={}
            for label,src,dst,queries in [('AB',a,b,q),('AC',a,c,q)]:
                status('match_original_images',triple=ti,pair=label,raw_frames=[src,dst]);t=time.monotonic()
                with torch.inference_mode():
                    warp,certainty=model.match(str(S/'images'/f'raw_{src:06d}.png'),str(S/'images'/f'raw_{dst:06d}.png'),device='cuda')
                    pred,conf,ok=sample_direction(warp,certainty,queries,(2704,1520),(2704,1520),'AtoB')
                    back,rc,rok=sample_direction(warp,certainty,pred,(2704,1520),(2704,1520),'BtoA')
                assert torch.isfinite(warp).all() and torch.isfinite(certainty).all()
                torch.cuda.synchronize();timings.append(dict(triple=ti,pair=label,seconds=time.monotonic()-t))
                point_targets[label]=pred;confs[label]=torch.minimum(conf,rc);backs[label]=torch.linalg.norm(back-queries,dim=-1);validity[label]=ok&rok
                del warp,certainty
            status('match_original_images',triple=ti,pair='BC',raw_frames=[b,c]);t=time.monotonic()
            with torch.inference_mode():
                warp,certainty=model.match(str(S/'images'/f'raw_{b:06d}.png'),str(S/'images'/f'raw_{c:06d}.png'),device='cuda')
                via,cf,ok=sample_direction(warp,certainty,point_targets['AB'],(2704,1520),(2704,1520),'AtoB')
                back,cb,rok=sample_direction(warp,certainty,via,(2704,1520),(2704,1520),'BtoA')
            assert torch.isfinite(warp).all() and torch.isfinite(certainty).all()
            torch.cuda.synchronize();timings.append(dict(triple=ti,pair='BC',seconds=time.monotonic()-t))
            minimum=torch.minimum(torch.minimum(confs['AB'],confs['AC']),torch.minimum(cf,cb))
            fb=torch.stack([backs['AB'],backs['AC'],torch.linalg.norm(back-point_targets['AB'],dim=-1)],-1)
            cycle=torch.linalg.norm(via-point_targets['AC'],dim=-1)
            valid=validity['AB']&validity['AC']&ok&rok
            eligible=valid&(minimum>=reg['confidence_threshold'])&(fb.max(-1).values<=reg['forward_backward_max_raw_px'])&(cycle<=reg['three_view_cycle_max_raw_px'])
            pts=torch.stack([q,point_targets['AB'],point_targets['AC']],dim=1).cpu().numpy()
            epi=np.stack([epipolar_distance(pts[:,i],pts[:,j],E[int(triple[i])-1],E[int(triple[j])-1],intr) for i,j in [(0,1),(0,2),(1,2)]],axis=-1)
            held_projection=[];depths=[];xyz=[];solver=[]
            for obs in pts:
                if not np.isfinite(obs).all():
                    xyz.append(np.full(3,np.nan));held_projection.append(np.full((3,2),np.nan));depths.append(np.full(3,np.nan));solver.append(False);continue
                measured=np.c_[np.asarray(triple)-1,obs]
                x,p,z,success=point_fit(measured,np.array([False,True,False]),E,intr)
                xyz.append(x);held_projection.append(p);depths.append(z);solver.append(success)
            item=dict(triple_index=ti,raw_frames=triple.tolist(),probe_ids=ids,groups=probes[ids,1].astype(int),points=pts,
                via_c=via.cpu().numpy(),confidence=minimum.cpu().numpy(),fb=fb.cpu().numpy(),cycle=cycle.cpu().numpy(),valid=valid.cpu().numpy(),eligible=eligible.cpu().numpy(),
                projection=np.asarray(held_projection),xyz=np.asarray(xyz),depth=np.asarray(depths),point_solver_ok=np.asarray(solver),epipolar_undistorted_px=epi)
            np.savez_compressed(out/f'triple-{ti:02d}.npz',**item);results.append(item)
            del warp,certainty
            print(json.dumps(dict(triple=ti,anchors=len(ids),image_consistent=int(eligible.sum()))),flush=True)
        labels=['background','face_lower','face_upper','torso','legs_shoes'];summary={}
        for group,label in enumerate(labels):
            total=accepted=0;blocks=[];cycle_all=[];held_all=[];held_eligible=[];positive=[];epi_all=[];epi_eligible=[];fit_failures=0
            for r in results:
                m=r['groups']==group;good=m&r['eligible'];total+=int(m.sum());accepted+=int(good.sum())
                if good.sum():blocks.append(r['triple_index'])
                cycle_all.extend(r['cycle'][m]);errors=np.linalg.norm(r['projection'][:,1]-r['points'][:,1],axis=-1)
                held_all.extend(errors[m]);held_eligible.extend(errors[good]);positive.extend((r['depth'][good]>0).all(-1))
                epi_all.extend(r['epipolar_undistorted_px'][m].ravel());epi_eligible.extend(r['epipolar_undistorted_px'][good].ravel());fit_failures+=int((~r['point_solver_ok'][m]).sum())
            summary[label]=dict(probes=total,image_consistent=accepted,triples_with_support=blocks,cycle_all_raw_px=quant(cycle_all),
                held_projection_all_raw_px=quant(held_all),held_projection_image_consistent_raw_px=quant(held_eligible),positive_depth_all_consistent=bool(positive and all(positive)),
                epipolar_all_undistorted_px=quant(epi_all),epipolar_image_consistent_undistorted_px=quant(epi_eligible),point_solver_failures=fit_failures)
        need=reg['required_support'];support_gate=(summary['face_lower']['image_consistent']>=need['lower_face_total'] and len(summary['face_lower']['triples_with_support'])>=need['lower_face_distinct_triples'] and
            summary['face_upper']['image_consistent']>=need['upper_face_total'] and summary['torso']['image_consistent']>=need['torso_total'] and summary['legs_shoes']['image_consistent']>=need['legs_total'] and summary['background']['image_consistent']>=need['background_total'])
        for path,digest in freeze['source_camera_hashes'].items():assert sha(path)==digest
        assert sha(O/'pixsfm/independent-validation/tracks.npz')==reg['frozen_validation_sha256']
        save(out/'result.json',dict(updated=now(),state='completed_waiting_original_image_review',elapsed_seconds=time.monotonic()-start,summary=summary,numeric_correspondence_coverage_passed=support_gate,
            pair_timings=timings,max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),visual_review='pending',source074_modified=False,weights_loaded_strictly=True,
            new_camera_candidate=False,gaussian_training_started=False,root_cause_resolved=False,quality_accepted=False))
        save(out/'progress.json',dict(updated=now(),state='completed',pid=os.getpid(),visual_review='pending'))
    except BaseException as e:
        save(out/'failure.json',dict(updated=now(),error=str(e),traceback=traceback.format_exc()));save(out/'progress.json',dict(updated=now(),state='failed',pid=os.getpid(),error=str(e)));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
