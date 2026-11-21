"""One preregistered native-crop RoMa contrast; never changes training images/cameras."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,signal,subprocess,sys,time,traceback,itertools
import numpy as np
import torch
from scipy.spatial import cKDTree
sys.path[:0]=['/opt/gs-slam-lab/shuolee-roma116/python','/opt/gs-slam-lab/shuolee-roma116/RoMa']
from roma116_coordinates import sample_direction
from run_video10_matcher_contrast123 import family_indices,conflict
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
Q=R/'video10-correspondence-contrast';D=Q/'head-roi-contrast';OUT=D/'run'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def local(s):return Path('/mnt/'+s[0].lower()+s[2:].replace('\\','/')) if len(s)>1 and s[1]==':' else Path(s)
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def nodownload(*a,**kw):raise RuntimeError('No downloads permitted')
def timeout(*a):raise TimeoutError('One420second inference budget exhausted')

def main():
    assert not OUT.exists(),'No repeated inference'
    p=read(D/'plan.json');qs=read(D/'queries.json');rows=read(D/'review-allocation.json')['rows']
    assert sha(D/'queries.json')==p['queries_sha256'] and sha(D/'review-allocation.json')==p['review_allocation_sha256']
    for s,h in p['source_sha256'].items():assert sha(local(s))==h,s
    for w in p['weights']:
        f=local(p['weight_root'])/w['filename'];assert f.stat().st_size==w['bytes'] and sha(f)==w['sha256']
    for c in p['crops'].values():assert sha(local(c['crop_file']))==c['sha256']
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    assert not busy,'ExistingGPUtask:'+busy
    offsets={int(f):torch.tensor(c['xyxy'][:2],dtype=torch.float32) for f,c in p['crops'].items()}
    # Exact translated identity, independently constructed normalized maps.
    xx,yy=torch.meshgrid(torch.arange(768,dtype=torch.float32),torch.arange(768,dtype=torch.float32),indexing='xy')
    grid=torch.stack([xx,yy],-1);a,b=p['frames'][:2];ga=(grid+.5)*2/768-1
    ab=(grid+offsets[a]-offsets[b]+.5)*2/768-1;ba=(grid+offsets[b]-offsets[a]+.5)*2/768-1
    warp=torch.cat([torch.cat([ga,ab],-1),torch.cat([ba,ga],-1)],1);certainty=torch.ones(warp.shape[:2])
    xy=torch.tensor([[400.25,300.75],[200.125,500.625],[500.5,200.5]])+offsets[a]
    got,_,valid=sample_direction(warp,certainty,xy-offsets[a],(768,768),(768,768),'AtoB');got+=offsets[b]
    err=float((got-xy).abs().max());assert valid.all() and err<1e-3
    save(D/'execution-admission.json',dict(updated=now(),source_sha256=sha(Path(__file__)),plan_sha256=sha(D/'plan.json'),native_translation_control_max_abs_px=err,model_loaded=False))
    OUT.mkdir();started=time.monotonic();signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_seconds'])
    def state(s,**kw):save(D/'progress.json',dict(updated=now(),state=s,pid=os.getpid(),elapsed=time.monotonic()-started,training_started=False,**kw))
    try:
        state('loading_existing_weights');torch.manual_seed(12310);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.hub.load_state_dict_from_url=nodownload;torch.hub.load=nodownload
        from romatch import roma_outdoor
        w=local(p['weight_root']);model_state=torch.load(w/'roma_outdoor.pth',map_location='cpu',weights_only=True);backbone=torch.load(w/'dinov2_vitl14_pretrain.pth',map_location='cpu',weights_only=True)
        model=roma_outdoor(device='cuda',weights=model_state,dinov2_weights=backbone,coarse_res=(448,448),upsample_res=(768,768),amp_dtype=torch.float16,symmetric=True,use_custom_corr=False,do_compile=False).eval()
        del model_state,backbone
        maps={};oldmaps={}
        for a,b in p['pairs']:
            state('matching',pair=[a,b])
            with torch.inference_mode():wa,ce=model.match(str(local(p['crops'][str(a)]['crop_file'])),str(local(p['crops'][str(b)]['crop_file'])),device='cuda')
            torch.cuda.synchronize();maps[a,b]=(wa.detach().float().cpu(),ce.detach().float().cpu());del wa,ce
            np.savez_compressed(OUT/f'map_{a:06d}_{b:06d}.npz',warp=maps[a,b][0].numpy(),certainty=maps[a,b][1].numpy())
            with np.load(Q/'run'/f'map_{a:06d}_{b:06d}.npz') as z:oldmaps[a,b]=(torch.from_numpy(z['warp'].copy()),torch.from_numpy(z['certainty'].copy()))
        del model;torch.cuda.empty_cache();nodes,frozen,trees=family_indices(p['frames']);gate=p['frozen_gate']
        def sample(which,a,b,xy):
            wa,ce=(maps if which=='roi' else oldmaps)[min(a,b),max(a,b)];hw=(768,768) if which=='roi' else (1520,2704)
            offa=offsets[a] if which=='roi' else torch.zeros(2);offb=offsets[b] if which=='roi' else torch.zeros(2)
            v,c,ok=sample_direction(wa,ce,xy-offa,hw,hw,'AtoB' if a<b else 'BtoA')
            return v+offb,c,ok
        def pair(which,a,b,xy):
            uv,c,v=sample(which,a,b,xy);back,cb,vb=sample(which,b,a,uv);fb=torch.linalg.norm(back-xy,dim=-1);conf=torch.minimum(c,cb)
            valid=v&vb&torch.isfinite(uv).all(-1)&torch.isfinite(back).all(-1)&torch.isfinite(conf)
            eligible=valid&(conf>=gate['min_bidirectional_confidence'])&(fb<gate['max_fb_native_px'])
            return dict(uv=uv,back_uv=back,fb=fb,confidence=conf,valid=valid,eligible=eligible)
        def lineage(src,dst,uv):
            tree,labels=trees[dst];out=[]
            for q,xy in zip(qs[str(src)],uv):
                nearby=tree.query_ball_point(xy,r=8) if np.isfinite(xy).all() else []
                out.append(conflict(q['split'],{frozen[int(labels[j])] for j in nearby}))
            return np.array(out,bool)
        pairs=[];cycles=[];memory={}
        for a,b in itertools.permutations(p['frames'],2):
            xy=torch.tensor([x['uv'] for x in qs[str(a)]],dtype=torch.float32)
            evs={}
            for kind in ['baseline','roi']:
                evs[kind]={k:v.numpy() for k,v in pair(kind,a,b,xy).items()};evs[kind]['split_conflict']=lineage(a,b,evs[kind]['uv'])
                np.savez_compressed(OUT/f'{kind}_queries_{a:06d}_{b:06d}.npz',query_uv=xy.numpy(),feature=np.array([q['feature'] for q in qs[str(a)]]),**evs[kind])
            old,new=evs['baseline'],evs['roi'];memory[a,b]=evs
            pairs.append(dict(source=a,target=b,n=len(xy),old_gate=int(old['eligible'].sum()),new_gate=int(new['eligible'].sum()),new_conflict=int((new['eligible']&new['split_conflict']).sum()),
                gained=int((~old['eligible']&new['eligible']).sum()),lost=int((old['eligible']&~new['eligible']).sum()),median_displacement_px=float(np.median(np.linalg.norm(new['uv']-old['uv'],axis=-1)))))
        for a,b,c in itertools.permutations(p['frames']):
            xy=torch.tensor([x['uv'] for x in qs[str(a)]],dtype=torch.float32);counts={}
            for kind in ['baseline','roi']:
                ab=pair(kind,a,b,xy);bc=pair(kind,b,c,ab['uv']);ca=pair(kind,c,a,bc['uv']);closure=torch.linalg.norm(ca['uv']-xy,dim=-1)
                passed=ab['eligible']&bc['eligible']&ca['eligible']&(closure<gate['max_threeview_cycle_native_px'])
                split_bad=lineage(a,b,ab['uv'].numpy())|lineage(a,c,bc['uv'].numpy())
                np.savez_compressed(OUT/f'{kind}_cycle_{a:06d}_{b:06d}_{c:06d}.npz',query_uv=xy.numpy(),b_uv=ab['uv'].numpy(),c_uv=bc['uv'].numpy(),closure=closure.numpy(),eligible=passed.numpy(),split_conflict=split_bad)
                counts[kind]=dict(passed=int(passed.sum()),no_split_conflict=int((passed.numpy()&~split_bad).sum()))
            cycles.append(dict(cycle=[a,b,c,a],counts=counts))
        result=dict(updated=now(),state='completed_pending_native_review',seconds=time.monotonic()-started,pairs=pairs,cycles=cycles,
            frozen_queries=1800,review_queries=len(rows),new_features=0,K_changed=False,cameras_changed=False,training_started=False,physical_identity_certified=False,
            max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),limitation='Crop context and effective resolution changed together. Gates are self-consistency, not physical accuracy. No labels reassigned or current camera refitted.')
        save(D/'result.json',result);state('complete');v=read(D/'progress.json');v['pid']=None;save(D/'progress.json',v)
        print(json.dumps(result),flush=True)
    except BaseException as e:
        save(D/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()));state('failed',error=repr(e));v=read(D/'progress.json');v['pid']=None;save(D/'progress.json',v);raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
