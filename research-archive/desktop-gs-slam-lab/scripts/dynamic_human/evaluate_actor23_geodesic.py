"""Uncropped native evaluation; target images are absent from render inputs."""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np,torch
from PIL import Image
from actor23_graph_model import ActorGraph,render
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');I=L/'outputs/dynamic-human-actor23-independent-106';O=L/'outputs/dynamic-human-actor23-geodesic-107';S=L/'scripts/dynamic_human'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tt=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['path_rigidity','path_rigidity_binding'],required=True);ap.add_argument('--scope',choices=['main','continuity'],default='main');ap.add_argument('--preview',action='store_true');a=ap.parse_args();assert not a.preview or a.scope=='main';B=O/a.branch;R=B/('native-evaluation' if a.scope=='main' else 'continuity-extra');R.mkdir(exist_ok=True);assert not (R/'metrics.json').exists();contract=read(O/'training-source-contract.json')
for name,digest in contract['source_sha256'].items():assert sha(S/name)==digest,name
checkpoint=B/'resume.pt';before=sha(checkpoint);assert before==read(B/'completion.json')['checkpoint_sha256'];cp=torch.load(checkpoint,map_location='cpu',weights_only=False);assert cp['step']==15000 and cp['protocol']['branch']==a.branch
p={k:v.cuda() for k,v in cp['params'].items()};assert sha(B/'graph.npz')==read(O/'protocol.json')['variants'][a.branch]['graph_sha256'];g=ActorGraph(B/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);torch.set_num_threads(6)
plan=read(I/('evaluation-protocol.json' if a.scope=='main' else 'continuity-protocol.json'));decision=read(I/'evaluation-clock-decision.json');assert sha(I/'evaluation-clock-decision.json')==plan['evaluation_clock_sha256'];offsets=decision['evaluation_offsets'];rows=[]
if (R/'progress.json').exists():
    previous=read(R/'progress.json');assert previous['checkpoint_sha256']==before;rows=previous['records']
    for r in rows:
        stem=f"camera{r['camera']}-frame{r['frame']:06d}"
        for name,digest in r['output_sha256'].items():assert sha(R/name)==digest
done={(r['camera'],r['frame']) for r in rows}
priority={(r['camera'],r['frame']) for r in read(I/'review-plan/protocol.json')['full_person']}
ordered=sorted(plan['records'],key=lambda r:((r['camera'],r['frame']) not in priority,r['camera'],r['frame']))
with torch.no_grad():
    for r in ordered:
        if a.preview and len(rows)>=18:break
        c,f=r['camera'],r['frame']
        if (c,f) in done:continue
        path=I/r['input'];assert sha(path)==r['input_sha256'];z=np.load(path);time_index=f+offsets[c]
        raw,alpha,_=render(p,g,time_index,tt(z['w2c']),tt(z['K']),1920,1080,2);rgb=raw[0].cpu().numpy();aa=alpha[0,:,:,0].cpu().numpy();assert np.isfinite(rgb).all() and np.isfinite(aa).all();pred=np.clip(rgb+.5*(1-aa[...,None]),0,1);target=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);x0,y0,x1,y1=z['bbox'];box=np.zeros_like(z['valid']);box[y0:y1,x0:x1]=True;metrics={}
        for name,mask in {'full_valid':z['valid'],'foreground':z['valid']&(z['matte']>.5),'foreground_bbox':z['valid']&box}.items():
            error=pred.astype(np.float64)[mask]-target.astype(np.float64)[mask];metrics[name]=dict(pixels=int(mask.sum()),psnr_db=float(-10*np.log10(np.mean(error*error))),mae=float(abs(error).mean()))
        stem=f'camera{c}-frame{f:06d}';np.savez_compressed(R/f'{stem}-prediction.npz',rgb=rgb,alpha=aa);Image.fromarray(u8(pred)).save(R/f'{stem}-trained.png');Image.fromarray(u8(aa)).save(R/f'{stem}-alpha.png');files=[f'{stem}-prediction.npz',f'{stem}-trained.png',f'{stem}-alpha.png'];rows.append(dict(camera=c,frame=f,split=r['split'],render_time_index=time_index,metrics=metrics,input_sha256=r['input_sha256'],output_sha256={name:sha(R/name) for name in files}));save(R/'progress.json',dict(images=len(rows),checkpoint_sha256=before,records=rows));print('ACTOR23_EVAL',a.branch,len(rows),c,f,flush=True)
if a.preview:
    assert len(rows)==18 and {(r['camera'],r['frame']) for r in rows}==priority and sha(checkpoint)==before
    shutil.copy2(__file__,R/'preview-evaluator.py');print('ACTOR23_PREVIEW_COMPLETE',a.branch,flush=True);raise SystemExit(0)
assert len(rows)==len(plan['records']) and len(rows)==(264 if a.scope=='main' else 10) and sha(checkpoint)==before
save(R/'metrics.json',dict(status='full_native_renders_complete_independent_comparison_and_visual_review_required',branch=a.branch,scope=a.scope,records=rows,checkpoint_sha256=before,frame_rendering='Full1920x1080 without target RGB/matte/bbox in render inputs; mask only in metrics and display target composition.',goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name);print('ACTOR23_EVALUATION_COMPLETE',a.branch,a.scope,flush=True)
