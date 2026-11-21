"""Paired fixed-topology 3D refinement: real-only vs real+generated pitch views.

Both start from the same native MiniGS2 baked Mip PLY. This is a separate gsplat
refinement experiment, not a rerun of native MiniGS2 density control.
"""
import hashlib
import json
from pathlib import Path
import random
import time
import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData,PlyElement
import torch
from local_refinement import BASE,ROOT,read_model,render,ssim

OUT=ROOT/'difix-distill-pilot-001';TEACHER=ROOT/'difix-novel-paths-001'
STEPS=1200;SEED=17


def dump(path,value):
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(value,indent=2));p.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def camera(row,width=576,height=1024):
    c=dict(row);V=np.eye(4)
    if 'viewmat' in row:V=np.asarray(row['viewmat'])
    else:
        R=np.array(row['rotation']);P=np.array(row['position']);V[:3,:3]=R.T;V[:3,3]=-R.T@P
    c.update(fx=row['fx']*width/row['width'],fy=row['fy']*height/row['height'],width=width,height=height,
             viewmat=torch.tensor(V,device='cuda',dtype=torch.float32)[None])
    return c


def load(path,mask=False):
    a=np.asarray(Image.open(path).convert('L' if mask else 'RGB')).copy()
    a=cv2.resize(a,(576,1024),interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_AREA)
    return torch.from_numpy(a).float()/255


def export(model,source,target):
    vertex=PlyData.read(str(source))['vertex'].data.copy();assert len(vertex)==len(model['means'])
    for i,k in enumerate(['x','y','z']):vertex[k]=model['means'][:,i].detach().cpu().numpy()
    for prefix,values in [('rot',model['quats']),('scale',model['scales']),('f_dc',model['sh0'][:,0]),('f_rest',model['shN'].transpose(1,2).flatten(1))]:
        values=values.detach().cpu().numpy()
        for i in range(values.shape[1]):vertex[f'{prefix}_{i}']=values[:,i]
    vertex['opacity']=model['opacities'].detach().cpu().numpy()
    PlyData([PlyElement.describe(vertex,'vertex')]).write(str(target))


def loss_rgb(pred,reference,valid):
    a=pred.permute(2,0,1);b=reference.permute(2,0,1);v=valid[None]
    l1=((a-b).abs()*v).sum()/(3*v.sum().clamp_min(1))
    return .8*l1+.2*(1-ssim(a*v+b*(1-v),b))


@torch.no_grad()
def evaluate(model,arm,cams,photos):
    dest=OUT/arm/'evaluation';dest.mkdir();rows=[]
    for row in cams:
        if int(row['img_name'])%8:continue
        c=camera(row);gt=load(photos/(row['img_name']+'.png')).cuda()
        image,_,_=render(model,c,mode='mip');image=image.clamp(0,1)
        mae=float((image-gt).abs().mean());mse=float((image-gt).square().mean())
        rows.append(dict(frame=row['img_name'],psnr=float(-10*np.log10(max(mse,1e-12))),mae=mae))
        Image.fromarray((image*255).round().byte().cpu().numpy()).save(dest/(row['img_name']+'.png'))
    pathdir=OUT/arm/'paths';pathdir.mkdir()
    pathrows=json.loads((TEACHER/'cameras.json').read_text())
    for row in pathrows:
        image,_,_=render(model,camera(row),mode='mip')
        Image.fromarray((image.clamp(0,1)*255).round().byte().cpu().numpy()).save(pathdir/(row['name']+'.png'))
    result=dict(status='complete',rows=rows,mean_psnr=float(np.mean([r['psnr'] for r in rows])),
        mean_mae=float(np.mean([r['mae'] for r in rows])),novel_paths=120,no_novel_gt=True,
        primary_decision='Visual detail, appearance preservation, anatomy and temporal continuity; real metrics reported separately')
    dump(OUT/arm/'evaluation.json',result);return result


def main():
    assert json.loads((TEACHER/'status.json').read_text())['status']=='generation_complete'
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(4)
    base=json.loads((ROOT/'native-resolution-cache-full-18000-s0/manifest.json').read_text());source=Path(base['baked_ply'])
    assert sha(source)==base['baked_ply_sha256']
    cameras=sorted(json.loads((BASE/'viewer/cameras.json').read_text()),key=lambda c:c['img_name'])
    photos=ROOT/'source-resolution-rectified-002/dataset/images'
    training=[]
    for i,row in enumerate(cameras):
        if i%8==0:continue
        training.append((camera(row),load(photos/(row['img_name']+'.png')),1-load(ROOT/'motion-masks-001'/(row['img_name']+'.png'),True)))
    # Fixed angular samples selected from the predefined path, never by generated quality.
    indices=[0,3,6,9,12,15,30,33,36,39,42,45]
    selected=[c for c in json.loads((TEACHER/'cameras.json').read_text()) if c['kind']=='elevation' and c['index'] in indices]
    pseudo=[]
    for row in selected:
        with np.load(TEACHER/'depth'/(row['name']+'.npz')) as geometry:valid=torch.from_numpy((geometry['alpha']>.5).astype('float32'))
        pseudo.append((camera(row),load(TEACHER/'output'/(row['name']+'.png')),valid))
    rates={'means':5e-6,'scales':.00025,'quats':.00005,'opacities':.001,'sh0':.001,'shN':.00005}
    protocol=dict(status='frozen_before_training',created=time.time(),source_sha256=sha(source),seed=SEED,steps=STEPS,
        paired_real_order=True,real_frames=51,pseudo_frames=[r['name'] for r in selected],pseudo_coefficient=.25,
        pseudo_ramp_steps=200,pseudo_steps='Every even step, in addition to the same real-camera loss',
        pseudo_validity='Original rendered alpha>0.5; no generated depth used',learning_rates=rates,
        renderer='gsplat; 3D Mip smoothing already folded in initial parameters, no extra3Dfilter;2Dantialiasing retained',
        topology='fixed; all Gaussian parameters can update; no clone/split/prune',resolution=[576,1024],
        depth_loss='No new depth term in either arm; starting model trained with depth. Real images anchor updates.',
        scope='Exploratory3Ddistillation pilot; outside-visibility geometry not recovered',
        evaluation='All8 real development images plus120fixedpathframes; generated teacher pixels are not GT',
        independent_novel_pose_indices=[i for i in range(60) if i not in indices],accepted=False)
    dump(OUT/'protocol.json',protocol)
    generator=random.Random(SEED);order=[]
    while len(order)<STEPS:order.extend(generator.sample(range(len(training)),len(training)))
    pseudo_rng=random.Random(SEED+1);pseudo_order=[pseudo_rng.randrange(len(pseudo)) for _ in range(STEPS)]
    results={}
    for arm in ['control','candidate']:
        folder=OUT/arm;folder.mkdir();torch.manual_seed(SEED);np.random.seed(SEED);model=read_model(source)
        optimizer=torch.optim.Adam([dict(params=[p],lr=rates[k],name=k) for k,p in model.items()],eps=1e-15)
        start=time.time()
        with (folder/'train.jsonl').open('x',buffering=1) as stream:
            for step in range(1,STEPS+1):
                c,im,valid=training[order[step-1]];im=im.cuda();valid=valid.cuda()
                image,_,_=render(model,c,mode='mip');real=loss_rgb(image,im,valid);total=real
                generated=torch.zeros((),device='cuda');weight=0.
                if arm=='candidate' and step%2==0:
                    pc,target,mask=pseudo[pseudo_order[step-1]]
                    pred,_,_=render(model,pc,mode='mip');generated=loss_rgb(pred,target.cuda(),mask.cuda())
                    weight=.25*min(1,step/200);total=real+weight*generated
                assert torch.isfinite(total)
                optimizer.zero_grad(set_to_none=True);total.backward()
                if step==1 or step%100==0:assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.values())
                optimizer.step()
                if step==1 or step%100==0:
                    record=dict(step=step,real_loss=float(real),generated_loss=float(generated),pseudo_weight=weight,
                        elapsed_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                    stream.write(json.dumps(record)+'\n');dump(OUT/'status.json',dict(status='training',arm=arm,**record));print(json.dumps(dict(arm=arm,**record)),flush=True)
        export(model,source,folder/'model.ply')
        scores=evaluate(model,arm,cameras,photos)
        results[arm]=dict(status='complete',steps=STEPS,gaussians=len(model['means']),elapsed_seconds=time.time()-start,
            model_sha256=sha(folder/'model.ply'),evaluation=scores)
        dump(folder/'manifest.json',results[arm]);del optimizer,model;torch.cuda.empty_cache()
    assert sha(source)==protocol['source_sha256']
    dump(OUT/'summary.json',dict(status='complete_pending_visual_review',protocol=protocol,models=results,accepted=False,
        geometry_updated=True,source_preserved=True,human_ratings_collected=0))
    dump(OUT/'status.json',dict(status='complete',finished=time.time()))


if __name__=='__main__':main()
