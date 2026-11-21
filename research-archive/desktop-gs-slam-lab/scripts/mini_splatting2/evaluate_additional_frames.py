"""One frozen comparison on24 previously unused frames, after candidate training."""
import hashlib
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from gsplat import rasterization
from local_refinement import ROOT,read_model,scores,ssim,save_image,SUBJECT_BOX

DATA=ROOT/'additional-frames-001';OUT=ROOT/'additional-frame-evaluation-001'
NAMES={'base':'native-baseline-18000-s0','motion':'native-motion-rgb-18000-s0',
       'mip':'native-mip-18000-s0','combined':'native-motion-mip-18000-s0',
       'depthmask':'native-motion-mip-depthmask-18000-s0'}

@torch.no_grad()
def main():
    global OUT,NAMES
    p=argparse.ArgumentParser();p.add_argument('--output-name',default='additional-frame-evaluation-001')
    p.add_argument('--variant',action='append',default=[],help='key=completed_run_directory')
    a=p.parse_args();OUT=ROOT/a.output_name
    if a.variant:
        NAMES=dict(v.split('=',1) for v in a.variant)
        assert len(NAMES)==len(a.variant),'Duplicate variant keys'
    OUT.mkdir(exist_ok=False);state=dict(status='waiting_for_candidates',started=time.time(),models=NAMES)
    mp=OUT/'manifest.json'
    def save():mp.write_text(json.dumps(state,indent=2))
    save()
    while True:
        q=json.loads((ROOT/'native-depthmask-queue.json').read_text())
        if q['status']=='complete':break
        if 'failed' in q['status']:
            state['status']='upstream_failed';save();raise RuntimeError(q)
        time.sleep(15)
    data_state=json.loads((DATA/'manifest.json').read_text());assert data_state['frozen_reconstruction_verified'] and data_state['registered_queries']==24
    cams=json.loads((DATA/'cameras.json').read_text());original=json.loads((DATA/'selection.json').read_text())['original']
    image_dir=Path('/mnt/c')/Path(data_state['images_path'].replace('C:\\','').replace('\\','/'))
    paths={};frozen={}
    for key,name in NAMES.items():
        m=json.loads((ROOT/name/'manifest.json').read_text());assert m['status']=='complete'
        path=Path(m['baked_ply'] if 'baked_ply' in m else m['ply']);paths[key]=path
        frozen[key]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    protocol=dict(frozen_before_any_evaluation=time.time(),models=frozen,
        fresh_evaluation_queries=not bool(a.variant),
        repeated_evaluation_note='These24 frames were previously inspected; subsequent variants are development comparisons, not a fresh final test' if a.variant else 'First evaluation of preselected24 frames',
        selection_sha256=hashlib.sha256((DATA/'selection.json').read_bytes()).hexdigest(),
        cameras_sha256=hashlib.sha256((DATA/'cameras.json').read_bytes()).hexdigest(),
        query_images={c['name']:hashlib.sha256((image_dir/c['name']).read_bytes()).hexdigest() for c in cams},
        rules=['no model updates after this freeze for this evaluation','all24 predetermined frames reported',
            'fixed-map camera pose localization only; no reference geometry or intrinsic refinement',
            'same-video temporal interpolation, not independent scene or extrapolation evidence'],
        roi='linear interpolation of preexisting development subject rectangles by original video index; includes background')
    (OUT/'frozen-protocol.json').write_text(json.dumps(protocol,indent=2))
    state.update(status='evaluating',protocol='frozen-protocol.json');save()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    import lpips
    perceptual=lpips.LPIPS(net='alex').cuda().eval()
    box_keys=sorted(SUBJECT_BOX);box_times=np.array([original[f'{k:06d}.jpg'] for k in box_keys]);boxes=np.array([SUBJECT_BOX[k] for k in box_keys])
    data=[];gt_dir=OUT/'gt';gt_dir.mkdir()
    for c in cams:
        arr=cv2.cvtColor(cv2.imread(str(image_dir/c['name'])),cv2.COLOR_BGR2RGB)
        h,w=arr.shape[:2];assert [w,h]==[c['width'],c['height']]
        R=np.array(c['rotation']);P=np.array(c['position']);V=np.eye(4);V[:3,:3]=R.T;V[:3,3]=-R.T@P
        v=torch.tensor(V,dtype=torch.float32,device='cuda')[None]
        box=[float(np.interp(c['source_index'],box_times,boxes[:,i])) for i in range(4)]
        data.append((c,arr,v,box));save_image(gt_dir/c['name'],torch.tensor(arr,device='cuda')/255.)
    models={}
    for key,path in paths.items():
        state['current']=key;save();model=read_model(path);dest=OUT/key;dest.mkdir();rows=[]
        for c,arr,view,box in data:
            for div in [1,2,4,8]:
                w,h=round(c['width']/div),round(c['height']/div);sx=w/c['width'];sy=h/c['height']
                K=torch.tensor([[c['fx']*sx,0,c['cx']*sx],[0,c['fy']*sy,c['cy']*sy],[0,0,1]],device='cuda',dtype=torch.float32)[None]
                filtered='baked' in path.name
                rendered,_,_=rasterization(model['means'],model['quats'],model['scales'].exp(),model['opacities'].sigmoid(),
                    torch.cat([model['sh0'],model['shN']],1),view,K,w,h,sh_degree=3,packed=False,near_plane=.2,
                    rasterize_mode='antialiased' if filtered else 'classic',eps2d=.1 if filtered else .3)
                pred=rendered[0].clamp(0,1)
                target=arr if div==1 else cv2.resize(arr,(w,h),interpolation=cv2.INTER_AREA)
                gt=torch.tensor(target,device='cuda',dtype=torch.float32)/255
                x0,y0,x1,y1=[round(v*s) for v,s in zip(box,[w,h,w,h])]
                row=dict(frame=c['name'],source_index=c['source_index'],divisor=div,full=scores(pred,gt),
                    subject_crop=scores(pred[y0:y1,x0:x1],gt[y0:y1,x0:x1]))
                if div==1:
                    row['ssim']=float(ssim(pred.permute(2,0,1),gt.permute(2,0,1)))
                    row['lpips_alex']=float(perceptual(pred.permute(2,0,1)[None]*2-1,gt.permute(2,0,1)[None]*2-1))
                rows.append(row)
                if div in [1,8]:save_image(dest/f'{Path(c["name"]).stem}-s{div}.png',pred)
        aggregate={str(d):dict(full_psnr=float(np.mean([r['full']['psnr'] for r in rows if r['divisor']==d])),
            crop_psnr=float(np.mean([r['subject_crop']['psnr'] for r in rows if r['divisor']==d]))) for d in [1,2,4,8]}
        aggregate['1'].update(ssim=float(np.mean([r['ssim'] for r in rows if r['divisor']==1])),lpips_alex=float(np.mean([r['lpips_alex'] for r in rows if r['divisor']==1])))
        models[key]=dict(aggregate=aggregate,rows=rows,gaussians=len(model['means']))
        (dest/'metrics.json').write_text(json.dumps(models[key],indent=2));print(key,json.dumps(aggregate),flush=True)
        del model;torch.cuda.empty_cache()
    assert all(hashlib.sha256(paths[k].read_bytes()).hexdigest()==frozen[k]['sha256'] for k in paths)
    summary=dict(status='complete',frames=24,models=models,protocol=protocol,
        interpretation='additional appearance validation within the original trajectory; not final independent scene or wide-pitch validation')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));state.update(status='complete',finished=time.time());save()

if __name__=='__main__':main()
