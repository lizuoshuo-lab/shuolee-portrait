"""Separate fixed ray/target inconsistency from learned depth error.

Read-only post-evaluation analysis. This does not turn estimated shared targets
into geometric ground truth or identify the cause of observational residuals.
"""
from pathlib import Path
import json,hashlib,math
import numpy as np
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-native-align-fresh-113';O=R/'evaluation'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def stats(v):
    a=np.asarray(v,dtype=float)
    return dict(count=len(a),median=float(np.median(a)),mean=float(np.mean(a)),p90=float(np.quantile(a,.9)),max=float(a.max())) if len(a) else dict(count=0)

def main():
    e=read(O/'held-native-queries.json')['records'];a=np.load(R/'shared-targets/native-targets.npz')
    xyz=a['xyz'];obs=a['observations'];lookup={int(pid):i for i,pid in enumerate(a['point_ids'])}
    observation_lookup={}
    for row in obs:
        key=(int(row[0]),int(row[1]));assert key not in observation_lookup,'Duplicate point/frame observation';observation_lookup[key]=row
    cache={};out=[]
    for q in e:
        if not q['valid']:continue
        i=q['raw']-1;k=lookup[q['point_id']];key=(i,k)
        if key not in cache:
            ob=observation_lookup[key];wh=a['wh'][i];H=1536;W=round(wh[0]*H/wh[1]);ratio=np.array([W,H])/wh
            uv=(ob[2:4]+.5)*ratio-.5;base=np.floor(uv);fraction=uv-base
            E=a['w2c'][i];rot=E[:3,:3];origin=-rot.T@E[:3,3];v=xyz[k]-origin;floors=[]
            for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
                coef=(fraction[0] if dx else 1-fraction[0])*(fraction[1] if dy else 1-fraction[1])
                if coef==0:continue
                pixel=base+[dx,dy];ray=np.r_[(pixel-(np.array([W,H])-1)/2)/(np.diag(a['K'])[:2]*ratio),1.]
                d=rot.T@ray;closest=origin+d*(d@v)/(d@d);floors.append(float(np.sum((closest-xyz[k])**2)))
            cache[key]=np.asarray(floors)
        actual=np.asarray([c['expected_squared_error'] for c in q['corners']]);floor=cache[key]
        assert len(floor)==len(actual)
        excess=actual-floor
        out.append(dict(raw=q['raw'],point_id=q['point_id'],region=q['region'],model=q['model'],
            fixed_ray_target_squared_mean=float(floor.mean()),total_squared_mean=float(actual.mean()),
            learned_along_ray_squared_mean=float(excess.mean()),minimum_excess_squared=float(excess.min())))
    minimum=min(q['minimum_excess_squared'] for q in out)
    # Float32 camera origin/matrices differ slightly from the source double E.
    assert minimum>=-1e-6,('Ray decomposition inconsistent',minimum)
    records={label:{(q['raw'],q['point_id']):q for q in out if q['model']==label} for label in ['098','113']}
    common=set(records['098'])&set(records['113']);summary={}
    for region in range(4):
        keys=[k for k in common if records['113'][k]['region']==region];summary[str(region)]={}
        for label in ['098','113']:
            rows=[records[label][k] for k in keys]
            summary[str(region)][label]={field:stats([math.sqrt(max(q[field],0)) for q in rows]) for field in ['total_squared_mean','fixed_ray_target_squared_mean','learned_along_ray_squared_mean']}
    result=dict(passed=True,summary=summary,common_valid_observations=len(common),minimum_excess_squared=minimum,
        units='training world units',aggregation='Equal mean of active bilinear-corner squared errors then square root per observation; common valid queries; descriptive statistic, not the exact training loss.',
        explanation='For every contributing ray point Y: ||Y-X||^2 = distance(X,ray)^2 + ||Y-projection_ray(X)||^2. The first term is constant for fixed cameras and observed pixels. Current auxiliary updates can change only the second term.',
        limitation='Uncertainty and wrong matches may remain. Source doubles approximate camera float32 transforms. No real3DGT or unseen-image claims.',
        sources={str(p):sha(p) for p in [O/'held-native-queries.json',R/'shared-targets/native-targets.npz',Path(__file__)]},optimizer_updates=0,quality_accepted=False)
    (O/'ray-error-decomposition.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},ensure_ascii=False))

if __name__=='__main__':main()
