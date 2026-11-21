"""Training-view fit diagnostic for the frozen topology pair; never a heldout score."""
import ast
import hashlib
import json
from pathlib import Path
import statistics
import time
import argparse

import cv2
import numpy as np
from PIL import Image
import torch
from evaluate_resolution_stage import ROOT,LIBRARIES,ModelParams,PipelineParams,GaussianModel,Scene,refresh_filter,sceneLoadTypeCallbacks,cameraList_from_camInfos,render,metric

RUNS={'base':'native-resolution-cache-full-18000-s0','late':'native-late-full-18000-s0'}

@torch.no_grad()
def main():
    torch.set_num_threads(4);cv2.setNumThreads(4)
    out=ROOT/'native-training-fit-diagnostic-001';out.mkdir(exist_ok=False)
    canonical=ROOT/'source-resolution-rectified-002/dataset'
    info=sceneLoadTypeCallbacks['Colmap'](str(canonical),'images',True)
    assignment=next(n for n in ast.parse(Path(__file__).with_name('local_refinement.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in n.targets))
    boxes=ast.literal_eval(assignment.value);keys=sorted(boxes);values=np.array([boxes[k] for k in keys])
    names=sorted(c.image_name for c in info.train_cameras);assert len(names)==51 and all(int(n)%8!=0 for n in names)
    frozen=dict(created=time.time(),runs=RUNS,frames=names,training_data=True,heldout=False,
        camera_poses='original fixed SfM poses',resolution=[1514,2694],
        head_box='same top-third subject definition as primary; subject rectangles interpolated by frame index, including background',
        libraries=LIBRARIES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Measures fitting of already-used photos. Compare model changes within this diagnostic; not evidence of novel-view generalization.')
    (out/'protocol.json').write_text(json.dumps(frozen,indent=2))
    results={};background=torch.zeros(3,device='cuda')
    for key,name in RUNS.items():
        run=ROOT/name;manifest=json.loads((run/'manifest.json').read_text());assert manifest['status']=='complete'
        assert manifest['arm']=='full' and manifest['resolution']==[1514,2694]
        p=argparse.ArgumentParser();mp=ModelParams(p);pp=PipelineParams(p)
        args=p.parse_args(['-s',manifest['dataset'],'-m',str(run/'model'),'-r','1','--eval','--data_device','cpu'])
        pc=GaussianModel(3);scene=Scene(mp.extract(args),pc,load_iteration=18000,shuffle=False)
        pc._mip_cameras=scene.getTrainCameras();filt=refresh_filter(pc,force=True)
        stored=torch.load(run/'model/point_cloud/iteration_18000/filter_3d.pt',weights_only=True,map_location='cuda')
        diff=float((filt-stored).abs().max());assert diff<1e-6
        cams=cameraList_from_camInfos(info.train_cameras,1,args);dest=out/key;dest.mkdir();rows=[]
        for c in cams:
            raw=np.asarray(Image.open(canonical/'images'/(c.image_name+'.png')).convert('RGB'))
            image=render(c,pc,pp.extract(args),background)['render'].clamp(0,1)
            encoded=(image.permute(1,2,0).cpu().numpy()*255).round().astype('uint8');assert raw.shape==encoded.shape
            pred=encoded.astype('float32')/255;gt=raw.astype('float32')/255
            b=[np.interp(int(c.image_name),keys,values[:,axis]) for axis in range(4)]
            lower=[round(v*s) for v,s in zip(b,[757,1347,757,1347])]
            x0,y0,x1,y1=[2*v for v in lower];hy1=2*(lower[1]+round((lower[3]-lower[1])/3))
            measured=dict(full=metric(pred,gt),head=metric(pred[y0:hy1,x0:x1],gt[y0:hy1,x0:x1]))
            Image.fromarray(encoded[y0:hy1,x0:x1]).save(dest/(c.image_name+'-head.png'))
            if key=='base':Image.fromarray(raw[y0:hy1,x0:x1]).save(dest/(c.image_name+'-gt.png'))
            rows.append(dict(frame=c.image_name,head_box=[x0,y0,x1,hy1],metrics=measured))
            del image
        results[key]=dict(run=name,gaussians=len(pc.get_xyz),trained_filter_max_difference=diff,rows=rows,
            aggregate={region:{m:statistics.mean(r['metrics'][region][m] for r in rows) for m in rows[0]['metrics'][region]} for region in ['full','head']})
        (dest/'summary.json').write_text(json.dumps(results[key],indent=2))
        del pc,scene,cams,filt,stored;torch.cuda.empty_cache()
    assert [r['frame'] for r in results['base']['rows']]==[r['frame'] for r in results['late']['rows']]
    delta={region:{m:results['late']['aggregate'][region][m]-results['base']['aggregate'][region][m]
            for m in results['base']['aggregate'][region]} for region in ['full','head']}
    per=[dict(frame=a['frame'],head_psnr=b['metrics']['head']['psnr']-a['metrics']['head']['psnr'],
        head_gradient=b['metrics']['head']['normalized_gradient_error']-a['metrics']['head']['normalized_gradient_error'])
        for a,b in zip(results['base']['rows'],results['late']['rows'])]
    summary=dict(status='complete',finished=time.time(),models=results,late_minus_base=delta,per_frame=per,
        training_only=True,accepted=False,protocol=frozen)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(delta),flush=True)

if __name__=='__main__':main()
