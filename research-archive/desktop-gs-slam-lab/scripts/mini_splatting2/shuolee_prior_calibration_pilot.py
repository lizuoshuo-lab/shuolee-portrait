"""Training-free local depth calibration pilot on reserved SfM point IDs.

Two fixed diagnostic hypotheses, no parameter sweep or Gaussian updates.
The evaluation set is development-only and still shares SfM initialization.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from shuolee_geometry_diagnosis import OUT, stats, write_json, sha


def main():
    out=OUT/'prior-calibration-pilot.json'
    assert not out.exists()
    protocol=dict(owner='shuolee',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_sha256=sha(Path(__file__)),
        fitting='Non-reserved training-view SfM points with >=3deg triangulation and prior confidence>0.85',
        development_validation='point_id%5==0, excluded from fitting across all views; still shared SfM, not independent GT',
        neighborhood=dict(max_neighbors=4,max_radius_px=48,min_neighbors=3,min_separation_px=5,weight='1/(distance+4)^2'),
        gate='estimated relative depth error<=0.15; unsupported observations retain original weight',
        correction='local weighted log(SfM depth / prior depth), clipped to log(1.5) in either direction',
        no_head_specific_fitting=True,no_parameter_sweep=True,models_modified=False)
    write_json(OUT/'prior-calibration-protocol.json',protocol)
    obs=json.loads((OUT/'sparse-observations.json').read_text())
    frames=sorted({r['frame'] for r in obs});rows=[]
    for frame in frames:
        source=[r for r in obs if r['frame']==frame and not r['prior_fit_reserved'] and r['confidence']>.85]
        query=[r for r in obs if r['frame']==frame and r['prior_fit_reserved'] and r['confidence']>.85]
        assert not ({r['point_id'] for r in source}&{r['point_id'] for r in query})
        if len(source)<3:continue
        xy=np.asarray([r['xy'] for r in source]);tree=cKDTree(xy)
        dist,index=tree.query([r['xy'] for r in query],k=min(12,len(source)))
        for i,q in enumerate(query):
            candidates=[(float(d),int(j)) for d,j in zip(np.atleast_1d(dist[i]),np.atleast_1d(index[i])) if 5<=d<=48][:4]
            supported=len(candidates)>=3
            correction=1.;estimated=None
            if supported:
                ww=np.array([1/(d+4)**2 for d,_ in candidates]);ww/=ww.sum()
                ratios=np.array([source[j]['signed_depth_ratio'] for _,j in candidates])
                correction=float(np.exp(np.clip(np.sum(ww*(-np.log(ratios))),-np.log(1.5),np.log(1.5))))
                estimated=float(np.sum(ww*abs(ratios-1)))
            corrected=q['signed_depth_ratio']*correction
            rows.append(dict(frame=frame,point_id=q['point_id'],head_crop=q['head_crop'],supported=supported,
                gate_retained=not supported or estimated<=.15,original_relative_error=q['relative_depth_error'],
                corrected_relative_error=abs(corrected-1),correction=correction,estimated_error=estimated))
    aggregate={}
    for region in ['head_crop','outside_head_crop']:
        rr=[r for r in rows if r['head_crop']==(region=='head_crop')]
        kept=[r for r in rr if r['gate_retained']]
        aggregate[region]=dict(observations=len(rr),unique_points=len({r['point_id'] for r in rr}),
            neighborhood_supported_fraction=float(np.mean([r['supported'] for r in rr])),
            original_relative_error=stats([r['original_relative_error'] for r in rr]),
            gate_retained_fraction=len(kept)/len(rr),gate_retained_relative_error=stats([r['original_relative_error'] for r in kept]),
            local_correction_all_relative_error=stats([r['corrected_relative_error'] for r in rr]),
            correction_over5pct_worsening_fraction=float(np.mean([r['corrected_relative_error']-r['original_relative_error']>.05 for r in rr])))
    write_json(out,dict(status='complete',protocol=protocol,aggregate=aggregate,rows=rows,
        valid_for_dense_training=False,decision='Pointwise development diagnostic only; dense correction still needs edge/occlusion and multi-view checks.'))
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
