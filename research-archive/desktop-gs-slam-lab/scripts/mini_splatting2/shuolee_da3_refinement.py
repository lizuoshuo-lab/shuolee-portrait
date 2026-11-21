"""Native-renderer, paired fixed-topology refinement using validated DA3 depths.

An exploratory intervention on the retained full-resolution model. This does
not rerun Mini-Splatting2 densification, and does not claim novel-view GT.
"""
import ast
from collections import OrderedDict
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from shuolee_native_depth_audit import ROOT, BASE, REPO, RUNTIME, Camera, GaussianModel, render_imp, expected_depth
from shuolee_geometry_diagnosis import sha, write_json
from native_depth_guidance import inverse_depth_loss
from fused_ssim import fused_ssim
from evaluate_head_detail import metric
import diff_gaussian_rasterization_ms as native

OUT=ROOT/'shuolee/current/da3-refinement'
PRIOR=ROOT/'shuolee/current/geometry-diagnosis/da3-priors'
SOURCE=ROOT/'native-resolution-cache-full-18000-s0/model/point_cloud/iteration_18000'
PHOTOS=ROOT/'source-resolution-rectified-002/dataset/images'
STEPS=510
SEED=29
RATES={'_xyz':5e-6,'_scaling':.00025,'_rotation':.00005,'_opacity':.001,'_features_dc':.001,'_features_rest':.00005}
PIPE=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False)


def camera(c,scale=2):
    R=np.asarray(c['rotation']);C=np.asarray(c['position']);t=-R.T@C
    name=c.get('img_name',c.get('name','path'))
    if 'img_name' in c:
        a=np.asarray(Image.open(PHOTOS/(name+'.png')).convert('RGB')).copy()
        if scale==1:a=cv2.resize(a,(c['width'],c['height']),interpolation=cv2.INTER_AREA)
        assert a.shape[:2]==(c['height']*scale,c['width']*scale)
        rgb=torch.from_numpy(a).permute(2,0,1).float()/255
    else:rgb=torch.zeros(3,c['height'],c['width'])
    return Camera(c.get('id',0),R,t,2*math.atan(c['width']/(2*c['fx'])),
        2*math.atan(c['height']/(2*c['fy'])),rgb,None,name,c.get('id',0),data_device='cpu')


def load_model():
    pc=GaussianModel(3);pc.load_ply(str(SOURCE/'point_cloud.ply'))
    pc._mip_iteration=18000;pc._mip_cameras=[]
    pc._mip_filter=torch.load(SOURCE/'filter_3d.pt',map_location='cuda',weights_only=True)
    pc._mip_cache_key=(id(pc._xyz),len(pc.get_xyz),180);pc._mip_refreshes=0
    return pc


def refs(c,cache):
    if c.image_name in cache:
        item=cache.pop(c.image_name);cache[c.image_name]=item;return item
    size=(c.image_height,c.image_width)
    gt=c.original_image.cuda()
    motion=np.asarray(Image.open(ROOT/f'motion-masks-001/{c.image_name}.png').convert('L'))
    valid=torch.from_numpy((motion==0).astype('float32'))[None,None].cuda()
    valid=F.interpolate(valid,size=size,mode='nearest')[0]
    with np.load(PRIOR/(c.image_name+'.npz')) as n:
        inverse=torch.tensor(n['inverse_depth'],device='cuda')[None,None]
        confidence=torch.tensor(n['weight'],device='cuda')[None,None]
    # Conservative support follows the low-resolution pixel footprint. Do not
    # interpolate valid targets across rejected silhouettes.
    inverse=F.interpolate(inverse,size=size,mode='nearest')[0,0]
    confidence=F.interpolate(confidence,size=size,mode='nearest')[0,0]
    normalizer=inverse[confidence>.25].median()
    item=(gt,valid,inverse,confidence,normalizer)
    cache[c.image_name]=item
    while len(cache)>2:cache.popitem(last=False)
    return item


@torch.no_grad()
def evaluate(pc,folder,test_cams,boxes):
    folder.mkdir(parents=True,exist_ok=True);bg=torch.zeros(3,device='cuda');culled=torch.zeros(len(pc.get_xyz),dtype=torch.bool,device='cuda')
    rows=[]
    for c in test_cams:
        pred=render_imp(c,pc,PIPE,bg,culling=culled)['render'].clamp(0,1)
        encoded=(pred*255).round().byte().permute(1,2,0).cpu().numpy()
        gt=c.original_image.permute(1,2,0).numpy();p=encoded.astype('float32')/255
        Image.fromarray(encoded).save(folder/(c.image_name+'.png'))
        h,w=encoded.shape[:2];b=boxes[int(c.image_name)]
        x0,y0,x1,y1=[round(v*s) for v,s in zip(b,[w,h,w,h])];hy1=y0+round((y1-y0)/3)
        regions={'full':(0,0,w,h),'person':(x0,y0,x1,y1),'head':(x0,y0,x1,hy1)}
        measured={}
        for name,(a,b,d,e) in regions.items():
            measured[name]=metric(p[b:e,a:d],gt[b:e,a:d])
            pt=torch.from_numpy(p[b:e,a:d].copy()).permute(2,0,1).cuda()[None]
            rt=torch.from_numpy(gt[b:e,a:d].copy()).permute(2,0,1).cuda()[None]
            measured[name]['ssim']=float(fused_ssim(pt,rt))
            if name=='head':Image.fromarray(encoded[b:e,a:d]).save(folder/(c.image_name+'-head.png'))
        mask=np.ones((h,w),bool);mask[y0:y1,x0:x1]=False
        mse=((p-gt)**2)[mask].mean()
        measured['outside_person_box']={'psnr':float(-10*np.log10(max(float(mse),1e-12))),'mae':float(abs(p-gt)[mask].mean())}
        rows.append(dict(frame=c.image_name,regions=regions,metrics=measured))
    result=dict(status='complete',rows=rows,aggregate={region:{key:float(np.mean([r['metrics'][region][key] for r in rows]))
        for key in rows[0]['metrics'][region]} for region in rows[0]['metrics']},
        independent_test=False,scope='Eight repeatedly used real development frames; top-third person box includes head, shoulders and background.')
    write_json(folder/'summary.json',result)
    return result


def main():
    if OUT.exists():
        state=json.loads((OUT/'status.json').read_text())
        assert state['status']=='loading_inputs' and not list(OUT.rglob('*.ply')) and not list(OUT.rglob('train.jsonl')),'Cannot restart an existing training run'
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.cuda.set_per_process_memory_fraction(.82)
    prior=json.loads((PRIOR/'manifest.json').read_text())
    assert prior['status']=='complete' and prior['appearance_holdout_excluded']
    assert str(RUNTIME/'deps') in native._C.__file__
    raw=SOURCE/'point_cloud.ply';digest=sha(raw)
    camera_rows=sorted(json.loads((BASE/'viewer/cameras.json').read_text()),key=lambda c:c['img_name'])
    train_rows=[c for c in camera_rows if int(c['img_name'])%8!=0];test_rows=[c for c in camera_rows if int(c['img_name'])%8==0]
    assert len(train_rows)==51 and len(test_rows)==8
    assert {c['img_name'] for c in train_rows}=={r['frame'] for r in prior['records']}
    generator=random.Random(SEED);order=[]
    while len(order)<STEPS:order.extend(generator.sample(range(len(train_rows)),len(train_rows)))
    protocol=dict(owner='shuolee',status='frozen_before_training',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        source_ply=str(raw),source_sha256=digest,source_script_sha256=sha(Path(__file__)),
        native_library_sha256=sha(Path(native._C.__file__)),prior_manifest_sha256=sha(PRIOR/'manifest.json'),
        steps=STEPS,seed=SEED,training_order=[train_rows[i]['img_name'] for i in order[:STEPS]],
        learning_rates=RATES,optimizer='torch Adam eps1e-15, new state in both arms',resolution=[1514,2694],
        reference_cache_bound=2,arms=['real_only','da3_guided'],source_role='Retained full-resolution model previously trained with DA2 depth0.003',
        rgb='0.8 motion-masked L1 +0.2 masked fused SSIM',depth_coefficient=.003,depth_ramp=100,
        depth='Alpha-normalized expected Gaussian-center z, inverse-depth residual normalized by accepted prior median; alpha>0.95',
        topology='Fixed: no cloning, densification, pruning or reset; identical all-false rasterizer culling',
        mip='Original full-resolution3Dfilter frozen in both arms; native2Dcompensation retained. No rasterizer replacement.',
        control='Same initial PLY, seed, camera order, optimizer and budget. real_only removes new depth supervision; da3_guided adds validated prior.',
        gate='Bounded pilot only; inspect8real development frames and fixed continuous paths. No automatic best promotion from metrics.',
        theoretical_limit='Single seed and a short fixed-topology intervention cannot validate full training or unseen geometry.')
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'status.json',dict(status='loading_inputs',training_started=False))
    train=[camera(c) for c in train_rows];test=[camera(c) for c in test_rows]
    assignment=next(n for n in ast.parse(Path(__file__).with_name('local_refinement.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in n.targets))
    boxes=ast.literal_eval(assignment.value)
    pc=load_model();evaluate(pc,OUT/'initial/evaluation',test,boxes);del pc;torch.cuda.empty_cache()
    results={}
    for arm in ['real_only','da3_guided']:
        dest=OUT/arm;dest.mkdir();torch.manual_seed(SEED);np.random.seed(SEED)
        pc=load_model();start=time.time();cache=OrderedDict();torch.cuda.reset_peak_memory_stats()
        optimizer=torch.optim.Adam([dict(params=[getattr(pc,key)],name=key,lr=value) for key,value in RATES.items()],eps=1e-15)
        bg=torch.zeros(3,device='cuda');culled=torch.zeros(len(pc.get_xyz),device='cuda',dtype=torch.bool)
        with (dest/'train.jsonl').open('x',buffering=1) as log:
            for step,index in enumerate(order[:STEPS],1):
                c=train[index];gt,valid,inverse,confidence,normalizer=refs(c,cache)
                image=render_imp(c,pc,PIPE,bg,culling=culled)['render']
                l1=((image-gt).abs()*valid).sum()/(3*valid.sum().clamp_min(1))
                rgb=.8*l1+.2*(1-fused_ssim((image*valid+gt*(1-valid))[None],gt[None]))
                depth=rgb.new_zeros(());active_fraction=0.
                if arm=='da3_guided':
                    z,alpha=expected_depth(c,pc,culled)
                    raw_depth,weight=inverse_depth_loss(z,alpha,inverse,confidence,valid,normalizer)
                    depth=.003*min(1.,step/100)*raw_depth
                    if step==1 or step%50==0:active_fraction=float((weight>0).float().mean())
                total=rgb+depth;assert torch.isfinite(total)
                optimizer.zero_grad(set_to_none=True);total.backward()
                if step==1 or step%50==0:
                    assert all(getattr(pc,k).grad is not None and torch.isfinite(getattr(pc,k).grad).all() for k in RATES)
                optimizer.step()
                if step==1 or step%50==0 or step==STEPS:
                    row=dict(step=step,frame=c.image_name,rgb_loss=float(rgb.detach()),depth_loss=float(depth.detach()),
                        active_depth_fraction=active_fraction,elapsed_seconds=time.time()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                    log.write(json.dumps(row)+'\n');write_json(OUT/'status.json',dict(status='training',arm=arm,**row));print(json.dumps(dict(arm=arm,**row)),flush=True)
        pc.save_ply(str(dest/'model.ply'));torch.save(pc._mip_filter.cpu(),dest/'filter_3d.pt')
        torch.save(dict(step=STEPS,optimizer=optimizer.state_dict(),source_sha256=digest),dest/'optimizer.pt')
        del optimizer;cache.clear();torch.cuda.empty_cache()
        result=evaluate(pc,dest/'evaluation',test,boxes)
        results[arm]=dict(status='complete',steps=STEPS,gaussians=len(pc.get_xyz),elapsed_seconds=time.time()-start,
            model_sha256=sha(dest/'model.ply'),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,evaluation=result['aggregate'])
        write_json(dest/'manifest.json',results[arm]);del pc;torch.cuda.empty_cache()
    assert sha(raw)==digest
    write_json(OUT/'summary.json',dict(status='complete_pending_visual_review',protocol=protocol,arms=results,
        source_unchanged=True,accepted_as_best=False,human_reviews=0,training_changed_3d=True))
    write_json(OUT/'status.json',dict(status='complete_pending_visual_review',training_running=False,accepted_as_best=False))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
