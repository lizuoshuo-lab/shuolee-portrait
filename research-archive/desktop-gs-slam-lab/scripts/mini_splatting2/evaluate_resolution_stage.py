"""Native rendering at common target resolutions, retaining each model's trained Mip3D filter."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001');REPO=RUNTIME/'repo-depthguide'
sys.path[:0]=[str(RUNTIME/'deps'),str(REPO),'/opt/gs-slam-lab/mini-splatting2-native-001/deps']
import diff_gaussian_rasterization as native_rgb
import diff_gaussian_rasterization_ms as native_ms
LIBRARIES={name:dict(path=module._C.__file__,sha256=hashlib.sha256(Path(module._C.__file__).read_bytes()).hexdigest())
    for name,module in [('rgb',native_rgb),('ms',native_ms)]}
assert all(str(RUNTIME/'deps') in v['path'] for v in LIBRARIES.values()),'Legacy rasterizer must not be used for Mip evaluation'
from arguments import ModelParams,PipelineParams
from scene import Scene,GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from native_mip_filter import refresh_filter
from gaussian_renderer import render
from evaluate_head_detail import metric


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    name=sys.argv[1];run=ROOT/name;m=json.loads((run/'manifest.json').read_text());assert m['status']=='complete'
    out=run/'resolution-evaluation-v2';out.mkdir(exist_ok=False)
    p=argparse.ArgumentParser();mp=ModelParams(p);pp=PipelineParams(p)
    args=p.parse_args(['-s',m['dataset'],'-m',str(run/'model'),'-r','1','--eval','--data_device','cpu'])
    pc=GaussianModel(3);scene=Scene(mp.extract(args),pc,load_iteration=18000,shuffle=False)
    runtime_audit=json.loads((ROOT/'native-resolution-evaluator-correction-001/training-runtime.json').read_text())
    actual_ms=next(x for x in runtime_audit['processes'][0]['libraries'] if '/diff_gaussian_rasterization_ms/' in x['path'])
    assert LIBRARIES['ms']['sha256']==actual_ms['sha256']
    pc._mip_cameras=scene.getTrainCameras();filt=refresh_filter(pc,force=True)
    stored=torch.load(run/'model/point_cloud/iteration_18000/filter_3d.pt',map_location='cuda',weights_only=True)
    difference=float((filt-stored).abs().max());assert difference<1e-6
    canonical=ROOT/'source-resolution-rectified-002/dataset'
    info=sceneLoadTypeCallbacks['Colmap'](str(canonical),'images',True)
    assignment=next(n for n in ast.parse(Path(__file__).with_name('local_refinement.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in n.targets))
    boxes=ast.literal_eval(assignment.value)
    rows=[];fingerprints={};background=torch.zeros(3,device='cuda');camera_probe=None
    for scale in [1,2]:
        cams=cameraList_from_camInfos(info.test_cameras,scale,args)
        target=out/('full' if scale==1 else 'low');target.mkdir()
        for c in cams:
            assert int(c.image_name)%8==0
            gtpath=canonical/'images'/(c.image_name+'.png')
            raw=np.asarray(Image.open(gtpath).convert('RGB'),dtype='uint8')
            if scale==2:
                raw=cv2.resize(raw,(c.image_width,c.image_height),interpolation=cv2.INTER_AREA)
                matched=np.asarray(Image.open(canonical.parent/'matched-low-resolution-control/images'/(c.image_name+'.png')).convert('RGB'))
                assert np.array_equal(raw,matched)
            assert raw.shape[:2]==(c.image_height,c.image_width)
            gt=raw.astype('float32')/255
            image=render(c,pc,pp.extract(args),background)['render'].clamp(0,1)
            if camera_probe is None and ((m['arm']=='low' and scale==2) or (m['arm']=='full' and scale==1)):
                original=next(x for x in scene.getTestCameras() if x.image_name==c.image_name)
                assert (original.image_width,original.image_height)==(c.image_width,c.image_height)
                reference=render(original,pc,pp.extract(args),background)['render'].clamp(0,1)
                camera_probe=float((reference-image).abs().max());assert camera_probe<1e-5
                del reference
            pred=image.permute(1,2,0).cpu().numpy();assert pred.shape==gt.shape
            # Evaluate the actual saved 8-bit rendering, consistent with preceding reports.
            encoded=(pred*255).round().astype('uint8');pred=encoded.astype('float32')/255
            prefix=target/c.image_name
            Image.fromarray(encoded).save(str(prefix)+'-render.png');Image.fromarray(raw).save(str(prefix)+'-gt.png')
            b=boxes[int(c.image_name)];low=[round(v*s) for v,s in zip(b,[757,1347,757,1347])]
            factor=2 if scale==1 else 1;x0,y0,x1,y1=[factor*v for v in low]
            hy1=factor*(low[1]+round((low[3]-low[1])/3))
            regions={'full':(0,0,c.image_width,c.image_height),'person':(x0,y0,x1,y1),'head':(x0,y0,x1,hy1)}
            measured={}
            for region,(a,b,d,e) in regions.items():
                measured[region]=metric(pred[b:e,a:d],gt[b:e,a:d])
                if region!='full':
                    Image.fromarray(encoded[b:e,a:d]).save(str(prefix)+f'-{region}.png')
                    Image.fromarray(raw[b:e,a:d]).save(str(prefix)+f'-{region}-gt.png')
            fingerprints[c.image_name]=hashlib.sha256(gtpath.read_bytes()).hexdigest()
            rows.append(dict(frame=c.image_name,target='full' if scale==1 else 'low',size=[c.image_width,c.image_height],
                regions={k:list(v) for k,v in regions.items()},metrics=measured))
            del image
        del cams;torch.cuda.empty_cache()
    result=dict(status='complete',created=time.time(),run=name,arm=m['arm'],seed=m['seed'],gaussians=len(pc.get_xyz),rows=rows,
        aggregate={scale:{region:{key:statistics.mean(r['metrics'][region][key] for r in rows if r['target']==scale)
            for key in rows[0]['metrics'][region]} for region in ['full','person','head']} for scale in ['full','low']},
        references=fingerprints,trained_filter_max_absolute_difference=difference,libraries=LIBRARIES,
        canonical_vs_original_camera_max_rgb_error=camera_probe,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        correction='Version2 explicitly prioritizes validated Mip CUDA libraries; version1 legacy-kernel output is excluded',
        protocol='Native CUDA RGB; raw model with original training-camera3Dfilter plus view-dependent2Dcompensation exactly once; common target images and frozen regions',
        accepted=False,scope='one video and one seed; development comparison, no independent high-pitch GT')
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['aggregate']),flush=True)


if __name__=='__main__':main()
