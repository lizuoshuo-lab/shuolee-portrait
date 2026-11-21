"""N2: frozen RoMa confidence on N1 visibility counterexamples, no calibration."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,signal,subprocess,sys,time,traceback
import numpy as np
import torch
sys.path[:0]=['/opt/gs-slam-lab/shuolee-roma116/python','/opt/gs-slam-lab/shuolee-roma116/RoMa']
from roma116_coordinates import sample_direction

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I=R/'video-update-20261027';N=I/'native-temporal-material'
P=I/'material-visibility-plan.json';O=I/'material-visibility'
W=Path('/mnt/c/Users/Administrator/Downloads/shuolee-roma116')

def read(p):return json.loads(p.read_text())
def now():return datetime.now(timezone.utc).isoformat()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def no_download(*a,**k):raise RuntimeError('Only existing pinned local weights permitted')
def expired(*a):raise TimeoutError('N2 fixed GPU budget exhausted')

def register():
    assert not P.exists() and not O.exists()
    assert read(N/'result.json')['total_seeds']==307
    acquisition=read(R.parent/'portrait-alignment-open-source-116/roma/weight-acquisition.json')
    model=read(R.parent/'portrait-alignment-open-source-116/roma/smoke-preregistration.json')['official_model']
    paths=[N/'seeds.json',N/'observations.npz']+[N/f'frame{f:06d}.png' for f in [278,0,70,266,556]]
    save(P,dict(stage='N2',registered=now(),source_sha256=sha(Path(__file__)),coordinate_adapter_sha256=sha(Path(__file__).parent/'roma116_coordinates.py'),
        hypothesis='Image-pair learned confidence might reject semantically wrong but temporally self-consistent tracks. Test once against pre-reviewed occlusion counterexamples and a nearby visible view.',
        fixed_inputs={str(p):sha(p) for p in paths},anchor_frame=278,target_frames=[0,70,266,556],all_seed_ids=list(range(307)),
        pre_reviewed_negative_cases=[dict(track=2,frame=70,reason='Eye candidate is occluded; N1 reaches hat'),
            dict(track=3,frame=70,reason='Eye candidate is occluded; N1 reaches hat'),
            dict(track=11,frame=0,reason='Ground candidate covered by skirt'),
            dict(track=287,frame=0,reason='Boot candidate covered by skirt'),
            dict(track=287,frame=556,reason='Boot detail covered by skirt'),
            dict(track=291,frame=556,reason='Boot detail covered by skirt')],
        nearby_visible_view=266,weights=acquisition['files'],model=model,
        fixed_gate=dict(min_bidirectional_confidence=.9,max_forward_backward_native_px=1.0,valid_map=True),
        gate_scope='Correspondence rejection only; not physical face reprojection admission. All raw predictions and scores saved; no threshold search, residual selection or use of camera geometry.',
        budget=dict(attempts=1,pairs=4,wall_seconds=300,GPU_tasks=1),
        no_camera_or_geometry=True,no_download=True,old_116_not_rerun=True,
        interpretation='Any pre-reviewed invisible point retained by confidence gate falsifies using that gate alone as physical identity/visibility proof. Even zero false positives does not establish general visibility or subpixel accuracy.'))
    print('N2_VISIBILITY_REGISTERED',flush=True)

def run():
    plan=read(P);assert sha(Path(__file__))==plan['source_sha256']
    assert sha(Path(__file__).parent/'roma116_coordinates.py')==plan['coordinate_adapter_sha256']
    assert not O.exists(),'Do not repeat N2'
    for p,h in plan['fixed_inputs'].items():assert sha(Path(p))==h
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if gpu:raise RuntimeError('Other GPU task exists:'+gpu)
    for m in plan['weights']:
        p=W/m['filename'];assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256']
    O.mkdir();t0=time.monotonic()
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['budget']['wall_seconds'])
    def progress(stage,**kw):save(O/'progress.json',dict(updated=now(),pid=os.getpid(),stage=stage,**kw))
    try:
        torch.manual_seed(122);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.hub.load_state_dict_from_url=no_download;torch.hub.load=no_download
        progress('load_local_official_weights')
        from romatch import roma_outdoor
        state=torch.load(W/'roma_outdoor.pth',map_location='cpu',weights_only=True)
        backbone=torch.load(W/'dinov2_vitl14_pretrain.pth',map_location='cpu',weights_only=True)
        spec=plan['model']
        model=roma_outdoor(device='cuda',weights=state,dinov2_weights=backbone,
            coarse_res=tuple(spec['coarse_resolution_hw']),upsample_res=tuple(spec['upsample_resolution_hw']),
            amp_dtype=torch.float16,symmetric=True,use_custom_corr=False,do_compile=False).eval()
        del state,backbone
        seeds=read(N/'seeds.json')['seeds'];queries=torch.tensor([s['uv'] for s in seeds],device='cuda',dtype=torch.float32)
        n1=np.load(N/'observations.npz');summaries=[];negative=[]
        for frame in plan['target_frames']:
            progress('match',target_frame=frame);begin=time.monotonic()
            with torch.inference_mode():
                warp,confidence=model.match(str(N/'frame000278.png'),str(N/f'frame{frame:06d}.png'),device='cuda')
                q,c,v=sample_direction(warp,confidence,queries,(2704,1520),(2704,1520),'AtoB')
                back,cb,vb=sample_direction(warp,confidence,q,(2704,1520),(2704,1520),'BtoA')
                fb=torch.linalg.norm(back-queries,dim=-1);conf=torch.minimum(c,cb)
                keep=v&vb&(conf>=.9)&(fb<1.)
            torch.cuda.synchronize()
            vals=dict(uv=q.cpu().numpy(),confidence=conf.cpu().numpy(),fb=fb.cpu().numpy(),eligible=keep.cpu().numpy(),valid=(v&vb).cpu().numpy())
            vals['n1_uv']=n1['uv'][frame];vals['n1_valid']=n1['valid'][frame]
            vals['difference_from_n1']=np.linalg.norm(vals['uv']-vals['n1_uv'],axis=-1)
            np.savez_compressed(O/f'frame{frame:06d}.npz',**vals)
            group=[]
            for label in sorted(set(s['group'] for s in seeds)):
                ids=np.array([s['id'] for s in seeds if s['group']==label]);mask=vals['eligible'][ids]
                group.append(dict(group=label,seeds=len(ids),eligible=int(mask.sum()),
                    fb_median_native_px=float(np.median(vals['fb'][ids])),fb_max_native_px=float(vals['fb'][ids].max()),
                    confidence_median=float(np.median(vals['confidence'][ids])),
                    note='No absolute image-space ground truth; fb is self-consistency only'))
            for case in plan['pre_reviewed_negative_cases']:
                if case['frame']!=frame:continue
                track=case['track'];negative.append(dict(**case,predicted_uv=vals['uv'][track].tolist(),confidence=float(vals['confidence'][track]),
                    fb_native_px=float(vals['fb'][track]),eligible=bool(vals['eligible'][track])))
            summaries.append(dict(frame=frame,seconds=time.monotonic()-begin,groups=group))
            print(json.dumps(dict(frame=frame,eligible=int(keep.sum()),face_eligible=int(keep[:9].sum())),ensure_ascii=False),flush=True)
            del warp,confidence
        result=dict(stage='N2',state='completed_pending_visual_review',updated=now(),pairs=summaries,negative_cases=negative,
            known_occlusion_false_positives=sum(c['eligible'] for c in negative),negative_case_count=len(negative),
            elapsed_seconds=time.monotonic()-t0,max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
            physical_face_accepted=False,training_started=False,K_or_poses_estimated=False,thresholds_tuned=False)
        save(O/'result.json',result);save(O/'progress.json',dict(updated=now(),stage='completed_pending_visual_review',pid=None))
        print(json.dumps({k:v for k,v in result.items() if k not in ['pairs','negative_cases']}),flush=True)
    except BaseException as e:
        save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        save(O/'progress.json',dict(updated=now(),stage='failed',pid=None));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');a=p.parse_args()
    register() if a.register else run()
