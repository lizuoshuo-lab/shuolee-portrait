"""One registered RoMa contrast at frozen native SIFT query locations."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,itertools,json,os,signal,subprocess,sys,time,traceback
import numpy as np
import torch
from scipy.spatial import cKDTree
sys.path[:0]=['/opt/gs-slam-lab/shuolee-roma116/python','/opt/gs-slam-lab/shuolee-roma116/RoMa']
from roma116_coordinates import sample_direction

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
Q=R/'video10-correspondence-contrast';O=R/'video10-temporal-bridge';N=R/'video10-semantic-role'
OUT=Q/'run';HW=(1520,2704)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def local(s):return Path('/mnt/'+s[0].lower()+s[2:].replace('\\','/')) if len(s)>1 and s[1]==':' else Path(s)
def no_download(*a,**k):raise RuntimeError('Only preregistered local weights permitted')
def expired(*a):raise TimeoutError('Registered900second budget exhausted; no repeated inference')

def inputs():
    p=read(Q/'plan.json');pre=read(Q/'preflight.json')
    assert sha(Q/'plan.json')==pre['plan_sha256']
    assert sha(Q/'review-allocation.json')==p['review_allocation_sha256']
    assert sha(Path(__file__).parent/'roma116_coordinates.py')==p['coordinate_adapter_sha256']
    files=[local(s) for s in p['input_sha256']]
    for s,h in p['input_sha256'].items():assert sha(local(s))==h,s
    for w in p['weights']:
        f=local(p['weight_root'])/w['filename'];assert f.stat().st_size==w['bytes'] and sha(f)==w['sha256']
    frames=p['frames'];features={};images={};baseline={};bp=read(O/'plan.json')['frames']
    for f in frames:
        fp=next(x for x in files if x.name==f'features_{f:06d}.npz')
        with np.load(fp) as d:features[f]=d['uv'].copy()
        assert features[f].shape==(p['queries']['per_frame_counts'][str(f)],2)
        images[f]=next(x for x in files if x.name==f'native_frame_{f:06d}.png')
    for a,b in p['pairs']:
        path=O/'run'/f'pair_{bp.index(a)}_{bp.index(b)}.npz'
        with np.load(path) as d:matches=d['matches'].copy()
        assert matches.ndim==2 and matches.shape[1]==2
        baseline[a,b]=(path,matches)
    for x in read(Q/'review-allocation.json')['rows']:
        assert np.array_equal(features[x['source']][x['query_feature']],np.array(x['source_uv'],np.float32))
    return p,features,images,baseline

def family_indices(frames):
    tracks=read(N/'secondary-observations.json')['tracks']
    bynode={};frozen={};points={f:[] for f in frames};labels={f:[] for f in frames}
    for t in tracks:
        frozen[t['family']]=t['split']
        for o in t['observations']:
            f=o['frame'];key=(f,o['feature'])
            if key in bynode:assert bynode[key]==t['family']
            bynode[key]=t['family']
            if f in points:points[f].append(o['uv']);labels[f].append(t['family'])
    trees={f:(cKDTree(np.array(points[f])),np.array(labels[f],np.int64)) for f in frames}
    return bynode,frozen,trees

def conflict(source_split,target_splits):
    return source_split=='quarantine' or 'quarantine' in target_splits or (source_split=='fit' and 'heldout' in target_splits) or (source_split=='heldout' and 'fit' in target_splits)

def lineage(frame,target,uv,bynode,frozen,trees):
    tree,labels=trees[target];finite=np.isfinite(uv).all(-1)
    neighbors=[[] for _ in range(len(uv))]
    for i,near in zip(np.flatnonzero(finite),tree.query_ball_point(uv[finite],r=8.0)):neighbors[i]=near
    src=np.full(len(uv),-1,np.int64);bad=np.zeros(len(uv),bool);target_count=np.zeros(len(uv),np.int32)
    for i,near in enumerate(neighbors):
        fam=bynode.get((frame,i),-1);src[i]=fam
        targets={int(labels[j]) for j in near};target_count[i]=len(targets)
        bad[i]=conflict(frozen.get(fam,'unassigned'),{frozen[t] for t in targets})
    return dict(source_family=src,split_conflict=bad,target_family_count=target_count)

def check():
    assert not OUT.exists() and not (Q/'execution-admission.json').exists()
    p,feat,imgs,base=inputs();nodes,frozen,trees=family_indices(p['frames'])
    assert conflict('fit',{'heldout'}) and conflict('heldout',{'fit'})
    assert conflict('quarantine',set()) and not conflict('fit',{'fit'})
    assert not conflict('unassigned',set())
    admission=dict(updated=now(),source_sha256=sha(Path(__file__)),plan_sha256=sha(Q/'plan.json'),preflight_sha256=sha(Q/'preflight.json'),
        baseline_pair_sha256={str(path):sha(path) for path,_ in base.values()},pairs=len(base),review_queries=72,
        lineage_radius_native_px=8,source='Same8px near-duplicate radius as frozen old family graph; proximity flags do not certify identity.',
        lineage_policy='Retain all raw predictions. Flag links reaching an opposite split or quarantine family; do not reassign either family. Unassigned source points remain unassigned.',
        cycle_policy='For both directions of each three-frame cycle at every frozen source query, require valid maps, bidirectional confidence and FB gate on all3edges, and<1pxclosure. Self-consistency only.',
        source_query_coordinates_unchanged=True,model_loaded=False,inference_started=False,training_started=False)
    save(Q/'execution-admission.json',admission);print(json.dumps(admission),flush=True)

def sample(maps,a,b,xy):
    warp,certainty=maps[min(a,b),max(a,b)]
    direction='AtoB' if a<b else 'BtoA'
    return sample_direction(warp,certainty,xy,HW,HW,direction)

def pair_eval(maps,a,b,xy,g):
    q,c,v=sample(maps,a,b,xy);back,cb,vb=sample(maps,b,a,q)
    fb=torch.linalg.norm(back-xy,dim=-1);confidence=torch.minimum(c,cb)
    valid=v&vb&torch.isfinite(q).all(-1)&torch.isfinite(back).all(-1)&torch.isfinite(confidence)
    keep=valid&(confidence>=g['min_bidirectional_confidence'])&(fb<g['max_fb_native_px'])
    return dict(uv=q,back_uv=back,confidence=confidence,fb=fb,valid=valid,eligible=keep)

def run():
    assert not OUT.exists(),'Registered attempt already exists. Never rerun.'
    p,features,images,baseline=inputs();ad=read(Q/'execution-admission.json')
    assert sha(Path(__file__))==ad['source_sha256'] and sha(Q/'plan.json')==ad['plan_sha256']
    for path,h in ad['baseline_pair_sha256'].items():assert sha(Path(path))==h
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if gpu:raise RuntimeError('AnotherGPUtask exists:'+gpu)
    OUT.mkdir();t0=time.monotonic();signal.signal(signal.SIGALRM,expired);signal.alarm(p['budget']['wall_seconds'])
    def state(s,**kw):save(Q/'progress.json',dict(updated=now(),state=s,pid=os.getpid(),elapsed=time.monotonic()-t0,training_started=False,**kw))
    try:
        state('loading_existing_official_weights');torch.manual_seed(12310);torch.set_num_threads(4)
        torch.set_float32_matmul_precision(p['model']['matmul_precision']);torch.hub.load_state_dict_from_url=no_download;torch.hub.load=no_download
        from romatch import roma_outdoor
        w=local(p['weight_root']);model_state=torch.load(w/'roma_outdoor.pth',map_location='cpu',weights_only=True)
        backbone=torch.load(w/'dinov2_vitl14_pretrain.pth',map_location='cpu',weights_only=True)
        spec=p['model'];model=roma_outdoor(device='cuda',weights=model_state,dinov2_weights=backbone,
            coarse_res=tuple(spec['coarse_resolution_hw']),upsample_res=tuple(spec['upsample_resolution_hw']),
            amp_dtype=torch.float16,symmetric=True,use_custom_corr=False,do_compile=False).eval()
        del model_state,backbone
        maps={};pair_summaries=[];g=p['frozen_gate'];nodes,frozen,trees=family_indices(p['frames'])
        for pair_index,(a,b) in enumerate(p['pairs']):
            state('matching',pair_index=pair_index,pair=[a,b]);begin=time.monotonic()
            with torch.inference_mode():warp,certainty=model.match(str(images[a]),str(images[b]),device='cuda')
            torch.cuda.synchronize();maps[a,b]=(warp.detach().float().cpu(),certainty.detach().float().cpu());del warp,certainty
            np.savez_compressed(OUT/f'map_{a:06d}_{b:06d}.npz',warp=maps[a,b][0].numpy(),certainty=maps[a,b][1].numpy())
            _,old=baseline[a,b];summaries=[]
            for src,dst,oldpairs in [(a,b,old),(b,a,old[:,::-1])]:
                xy=torch.from_numpy(features[src]);ev=pair_eval(maps,src,dst,xy,g)
                vals={k:v.numpy() for k,v in ev.items()};vals['query_uv']=features[src]
                vals['baseline_target_index']=np.full(len(xy),-1,np.int64)
                vals['baseline_target_index'][oldpairs[:,0]]=oldpairs[:,1]
                vals['baseline_uv']=np.full((len(xy),2),np.nan,np.float32)
                vals['baseline_uv'][oldpairs[:,0]]=features[dst][oldpairs[:,1]]
                vals['method_displacement_px']=np.linalg.norm(vals['uv']-vals['baseline_uv'],axis=-1)
                vals.update(lineage(src,dst,vals['uv'],nodes,frozen,trees))
                np.savez_compressed(OUT/f'queries_{src:06d}_{dst:06d}.npz',**vals)
                summ=dict(source=src,target=dst,queries=len(xy),old_pair_matches=len(oldpairs),roma_gate_pass=int(vals['eligible'].sum()),
                    opposite_or_quarantined_family_reached=int(vals['split_conflict'].sum()),
                    gated_split_conflicts=int((vals['split_conflict']&vals['eligible']).sum()),
                    valid=int(vals['valid'].sum()),physical_identity_certified=False)
                summaries.append(summ)
            pair_summaries.append(dict(pair=[a,b],seconds=time.monotonic()-begin,directions=summaries))
            print(json.dumps(pair_summaries[-1]),flush=True)
        del model;torch.cuda.empty_cache();cycles=[]
        for tri in p['triplets']:
            for a,b,c in itertools.permutations(tri):
                state('cycle_consistency',cycle=[a,b,c,a]);xy=torch.from_numpy(features[a])
                ab=pair_eval(maps,a,b,xy,g);bc=pair_eval(maps,b,c,ab['uv'],g);ca=pair_eval(maps,c,a,bc['uv'],g)
                closure=torch.linalg.norm(ca['uv']-xy,dim=-1);edge_ok=ab['eligible']&bc['eligible']&ca['eligible']
                passed=edge_ok&(closure<g['max_threeview_cycle_native_px'])
                np.savez_compressed(OUT/f'cycle_{a:06d}_{b:06d}_{c:06d}.npz',query_uv=features[a],b_uv=ab['uv'].numpy(),c_uv=bc['uv'].numpy(),
                    returned_uv=ca['uv'].numpy(),closure=closure.numpy(),edge_gates=edge_ok.numpy(),eligible=passed.numpy())
                cycles.append(dict(cycle=[a,b,c,a],queries=len(xy),all_edges_pass=int(edge_ok.sum()),cycle_pass=int(passed.sum())))
        result=dict(updated=now(),state='complete_pending_physical_review',pairs=pair_summaries,cycles=cycles,
            wall_seconds=time.monotonic()-t0,max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),source_sha256=sha(Path(__file__)),
            input_plan_sha256=sha(Q/'plan.json'),review_allocation_sha256=sha(Q/'review-allocation.json'),
            queries_unchanged=True,new_features=0,K_changed=False,cameras_estimated=False,smplx_fitted=False,
            training_started=False,physical_identity_certified=False,quality_accepted=False,
            limits='Gate and cycle counts are algorithm self-consistency, not physical truth or camera registration. Opposite-split proximity is diagnostic quarantine only; no old family split changes.')
        save(OUT/'result.json',result);state('complete_pending_physical_review',pairs_done=7)
        prog=read(Q/'progress.json');prog['pid']=None;save(Q/'progress.json',prog)
        print(json.dumps({k:v for k,v in result.items() if k not in ['pairs','cycles']}),flush=True)
    except BaseException as e:
        save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        state('failed',error=repr(e));prog=read(Q/'progress.json');prog['pid']=None;save(Q/'progress.json',prog);raise
    finally:signal.alarm(0)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    check() if args.check else run()
