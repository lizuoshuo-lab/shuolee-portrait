"""Training-only spatial coverage of available depth priors, before renderer alpha gating."""
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def main():
    torch.set_num_threads(2);out=ROOT/'head-prior-coverage-001';out.mkdir(exist_ok=False)
    regions=json.loads((ROOT/'training-head-regions-003/manifest.json').read_text())
    review=json.loads((ROOT/'training-head-regions-003/visual-review.json').read_text())
    assert review['approved_for_training_attention_experiment'] and regions['training_only']
    rows=[]
    for row in regions['rows']:
        name=row['frame'];assert int(name)%8!=0
        prior=ROOT/'depth-priors'/(name+'.npz')
        with np.load(prior) as raw:
            inverse=torch.from_numpy(raw['inverse_depth'].astype('float32').copy())[None,None]
            confidence=torch.from_numpy(raw['weight'].astype('float32').copy())[None,None]
        weighted=F.interpolate(inverse*confidence,size=(1347,757),mode='area')
        confidence=F.interpolate(confidence,size=(1347,757),mode='area')
        inverse=weighted/confidence.clamp_min(1e-8)
        confidence=confidence[0,0];inverse=inverse[0,0]
        confidence=confidence*((inverse>0)&(confidence>.25))
        rawmask=np.asarray(Image.open(ROOT/'motion-masks-001'/(name+'.png')).convert('L')).copy()
        motion=F.interpolate(torch.from_numpy((rawmask>0).astype('float32'))[None,None],size=(1347,757),mode='nearest')[0,0]
        valid=1-motion;eligible=(confidence>0)&(valid>0)
        head=torch.zeros_like(eligible);x0,y0,x1,y1=row['box'];head[y0:y1,x0:x1]=True
        weight=confidence*valid
        values={}
        for key,region in [('head',head),('outside',~head)]:
            values[key]=dict(eligible_fraction=float(eligible[region].float().mean()),
                mean_confidence_including_zero=float(weight[region].mean()),
                motion_masked_fraction=float(motion[region].mean()))
        rows.append(dict(frame=name,box=row['box'],head_area_fraction=float(head.float().mean()),
            head_fraction_of_available_depth_weight=float(weight[head].sum()/weight.sum().clamp_min(1)),
            prior_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),**values))
    result=dict(status='complete',created=time.time(),frames=51,rows=rows,
        aggregate=dict(head_area_fraction=statistics.mean(row['head_area_fraction'] for row in rows),
            head_fraction_of_available_depth_weight=statistics.mean(row['head_fraction_of_available_depth_weight'] for row in rows),
            **{key:{metric:statistics.mean(row[key][metric] for row in rows) for metric in rows[0][key]} for key in ['head','outside']}),
        scope='Available prior-confidence and motion validity only; actual depth loss also requires renderedalpha>0.95.',
        no_rendered_alpha_or_model_geometry_checked=True,depth_is_prediction_not_gt=True,
        interpretation='Spatial coverage diagnostic, not causal evidence of blur or depth accuracy.',
        training_changed=False,device='CPU')
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['aggregate']))


if __name__=='__main__':main()
