"""Compare recorded pre/post-visual positions under one frozen reference alignment.
No replay, calibration, timing shift, scale fit, or new alignment is performed.
"""
from pathlib import Path
from decimal import Decimal
from collections import OrderedDict
import csv, hashlib, json, traceback
import numpy as np

LAB=Path(__file__).resolve().parents[1]
BASE=LAB/'outputs/fast-livo2-hilti-gt-001'
NAME='exp14_basement_2_livo_20261010T142554Z_ef89e887'
RUN=BASE/'runs'/NAME
LOG=RUN/'official-log-after'
DIAG=LAB/'outputs/fast-livo2-hilti-vio-diagnostic-001/20261010T160003Z_aa0ce023'
GT=Path(r'\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab\fast-livo2-data\hilti2022-recovery-003-exp14\ground_truth\exp14_basement_2_imu.txt')
OUT=LAB/'outputs/fast-livo2-phase-comparison-001'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def pose_file(p):
    rows=[l.split() for l in Path(p).read_text().splitlines() if l.strip() and not l.startswith('#')]
    t=np.array([int(Decimal(r[0])*1000000000) for r in rows],np.int64)
    assert np.all(np.diff(t)>0)
    return t,np.array([[float(x) for x in r[1:4]] for r in rows])
def state_file(p,width):
    rows=[l.split() for l in Path(p).read_text().splitlines() if l.strip()]
    assert all(len(r)==width for r in rows)
    groups=OrderedDict()
    for i,r in enumerate(rows):groups.setdefault(Decimal(r[0]),[]).append(i)
    return np.array(rows,float),groups
def interp(t,p,q):
    values=np.full((len(q),3),np.nan);valid=np.zeros(len(q),bool);brackets=[]
    for j,stamp in enumerate(q):
        k=int(np.searchsorted(t,stamp))
        if k<len(t) and t[k]==stamp:
            values[j]=p[k];valid[j]=True;brackets.append((k,k,0.));continue
        if k==0 or k==len(t) or t[k]-t[k-1]>200000000:
            brackets.append((-1,-1,0.));continue
        a=float((int(stamp)-int(t[k-1]))/(int(t[k])-int(t[k-1])))
        values[j]=(1-a)*p[k-1]+a*p[k];valid[j]=True;brackets.append((k-1,k,a))
    return values,valid,brackets

def main():
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),BASE/'evaluation/metrics.json',BASE/'evaluation/paired-errors.csv',LOG/'mat_pre.txt',LOG/'mat_out.txt',LOG/'result'/f'{NAME}.txt',DIAG/'metrics.json',DIAG/'vio-transitions.csv',GT]
    hashes={str(p):sha(p) for p in files}
    old=read(BASE/'evaluation/metrics.json');assert sha(GT)==old['dense_GT_sha256']
    plan={'status':'registered_before_comparison','inputs':hashes,
          'population':'All original common reference timestamps having both recorded pre-visual and post-visual interpolation support; no extrapolation or final-row imputation.',
          'alignment':'Original LIVO SE3 alignment held fixed for both phases; no new GT or time fitting.',
          'phase':'Each paired timestamp has LIO-out == VIO-pre, followed by VIO-out in the same state-log group. Last unpaired LIO remains excluded.',
          'primary':'Within-run position-error change at common reference samples. This is not an independently aligned new official ATE or causal attribution of drift.',
          'precision':'Six-significant-digit state log positions; check reconstruction against original primary trajectory with source precision bounds.'}
    save(OUT/'plan.json',plan);(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    pre,pg=state_file(LOG/'mat_pre.txt',19);post,og=state_file(LOG/'mat_out.txt',20)
    timestamps,primary=pose_file(LOG/'result'/f'{NAME}.txt')
    assert len(og)==len(timestamps)==738
    trs=list(csv.DictReader((DIAG/'vio-transitions.csv').open()))
    assert len(trs)==737
    before=[];after=[];times=[];max_rounding_ratio=0.
    for i,(stamp,ids) in enumerate(og.items()):
        assert len(ids)==len(pg[stamp])
        if len(ids)==1:
            assert i==737;continue
        a=post[ids[0]];b=post[ids[1]]
        assert np.array_equal(a[:19],pre[pg[stamp][1]]) and b[19]==0
        assert int(trs[i]['sensor_timestamp_ns'])==timestamps[i]
        vals=np.abs(a[4:7]);bound=np.where(vals>0,.5*10.**(np.floor(np.log10(np.maximum(vals,1e-300)))-5),0)+.75e-6
        ratio=np.abs(a[4:7]-primary[i])/bound;assert np.all(ratio<=1)
        max_rounding_ratio=max(max_rounding_ratio,float(ratio.max()))
        before.append(a[4:7]);after.append(b[4:7]);times.append(timestamps[i])
    times=np.array(times,np.int64);before=np.array(before);after=np.array(after)
    rt,rp=pose_file(GT);indices=np.array(old['common_reference_indices']);queries=rt[indices];reference=rp[indices]
    bi,bv,br=interp(times,before,queries);ai,av,ar=interp(times,after,queries)
    assert np.array_equal(bv,av) and br==ar
    pri,pv,_=interp(timestamps,primary,queries);assert pv.all()
    T=np.array(old['metrics']['livo']['SE3_alignment_estimated_to_reference'])
    err=lambda x:np.linalg.norm(x@T[:3,:3].T+T[:3,3]-reference,axis=1)
    primary_error=err(pri)
    frozen=list(csv.DictReader((BASE/'evaluation/paired-errors.csv').open()))
    frozen_error=np.array([float(r['livo_aligned_position_error_m']) for r in frozen])
    assert np.allclose(primary_error,frozen_error,rtol=0,atol=2e-12)
    eb,ea=err(bi)[bv],err(ai)[bv];delta=ea-eb
    points=[]
    for n,j in enumerate(np.flatnonzero(bv)):
        points.append(dict(original_GT_index=int(indices[j]),timestamp_ns=int(queries[j]),pre_position_error_m=float(eb[n]),post_position_error_m=float(ea[n]),post_minus_pre_error_m=float(delta[n]),interpolation_left=int(br[j][0]),interpolation_right=int(br[j][1]),interpolation_fraction=br[j][2]))
    with (OUT/'paired-phase-errors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=points[0]);w.writeheader();w.writerows(points)
    result={'status':'completed_frozen_alignment_pre_post_visual_position_comparison','paired_visual_timestamps':len(times),'original_reference_points':len(queries),'paired_reference_points':int(bv.sum()),'excluded_reference_indices':indices[~bv].tolist(),
            'pre_phase_position_error_rmse_m':float(np.sqrt(np.mean(eb**2))),'post_phase_position_error_rmse_m':float(np.sqrt(np.mean(ea**2))),
            'error_change_mean_m':float(delta.mean()),'error_change_median_m':float(np.median(delta)),'error_change_min_m':float(delta.min()),'error_change_max_m':float(delta.max()),
            'error_decreased_points':int((delta<0).sum()),'error_increased_points':int((delta>0).sum()),'unchanged_points':int((delta==0).sum()),
            'primary_error_reproduced_max_abs_m':float(np.max(np.abs(primary_error-frozen_error))),
            'logged_pre_vs_primary_interpolated_position_max_m':float(np.max(np.linalg.norm(bi[bv]-pri[bv],axis=1))),
            'state_log_precision_bound_max_ratio':max_rounding_ratio,'new_alignment_fitted':False,'no_replay_or_original_modification':True,
            'limitations':['Logged coordinates are rounded; no new photometric residual or accepted-iteration evidence','Interpolate estimates at fixed reference times; reference itself is not resampled','Development reference is not surveyed millimetre GT','Same-run comparison includes accumulated previous visual updates in both phases','One frozen global alignment removes refit confounding but is not an absolute-pose oracle']}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=(queries[bv]-queries[bv][0])/1e9
    fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    ax[0].plot(t,100*eb,label='Before visual update',lw=1.2);ax[0].plot(t,100*ea,label='After visual update',lw=1.2)
    ax[0].set_ylabel('Position error / cm');ax[0].legend();ax[0].grid(alpha=.2)
    ax[1].plot(t,1000*delta,lw=1,color='#007f88');ax[1].axhline(0,color='gray',lw=.7)
    ax[1].set(xlabel='Reference elapsed time / s',ylabel='After - before / mm');ax[1].grid(alpha=.2)
    fig.suptitle('Actual pre/post-visual states; identical frozen alignment and reference support')
    fig.savefig(OUT/'phase-position-errors.png',dpi=180);plt.close(fig)
    assert all(sha(p)==h for p,h in hashes.items())
    save(OUT/'metrics.json',result)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():(OUT/'failure.txt').write_text(traceback.format_exc())
        raise
