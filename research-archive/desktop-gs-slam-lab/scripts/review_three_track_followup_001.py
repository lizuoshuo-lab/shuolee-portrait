"""Independent CPU checks and plots for the recorded follow-up experiments."""
from pathlib import Path
from decimal import Decimal
import csv, json, hashlib
import numpy as np
LAB=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=LAB/'outputs/splatam-candidate-selection-002'
    m=read(p/'metrics.json');records=list(csv.DictReader((p/'iterations.csv').open()))
    assert len(records)==600 and m['optimizer_steps']==600
    fig,axes=plt.subplots(1,3,figsize=(12,3.3),layout='constrained')
    checks=[]
    for ax,r in zip(axes,m['rows']):
        lines=[x for x in records if int(x['query_frame'])==r['frame']]
        assert [int(x['iteration']) for x in lines]==list(range(200))
        losses=np.array([float(x['loss']) for x in lines]);best=int(np.argmin(losses))
        assert best==r['best_pre_step_iteration']
        official=r['candidates']['official_selection'];consistent=r['candidates']['consistent_selection']
        assert consistent['loss']==losses[best]
        assert abs(official['loss']-losses[best+1])<.05
        for v in r['candidates'].values():assert np.isclose(v['loss_per_support'],v['loss']/v['support_pixels'],rtol=0,atol=1e-12)
        assert r['consistent_minus_official_loss']==consistent['loss']-official['loss']
        assert r['consistent_minus_official_position_error_m']==consistent['frozen_alignment_position_error_m']-official['frozen_alignment_position_error_m']
        ax.plot(np.arange(200),losses/losses[0],lw=1.1,color='#245c8d')
        ax.scatter([best],[losses[best]/losses[0]],color='#15806a',s=28,label='Selected pre-update pose')
        ax.set(title=f'room frame {r["frame"]}',xlabel='Tracking iteration',ylabel='Loss / initial loss')
        ax.grid(alpha=.2)
        checks.append(dict(frame=r['frame'],best_iteration=best,loss_change=r['consistent_minus_official_loss'],position_error_change_mm=1000*r['consistent_minus_official_position_error_m']))
    axes[0].legend(fontsize=8,loc='upper right')
    fig.savefig(p/'candidate-loss-curves.png',dpi=180);plt.close(fig)
    save(p/'validation.json',dict(status='passed_CSV_selection_and_report_arithmetic',rows=checks,
        source_metrics_sha256=sha(p/'metrics.json'),iterations_sha256=sha(p/'iterations.csv'),review_source_sha256=sha(Path(__file__)),
        limit='Independent arithmetic review, not a second GPU replay. Lower objective is not proof of trajectory improvement.'))
    # Bound effects of six-significant-digit coordinate logging on paired errors.
    import fast_livo2_phase_comparison_001 as phase
    fp=phase.OUT;out,groups=phase.state_file(phase.LOG/'mat_out.txt',20)
    before=[];after=[]
    def bound(v):
        return np.where(v!=0,.5*10.**(np.floor(np.log10(np.maximum(np.abs(v),1e-300)))-5),0.)
    for ids in groups.values():
        if len(ids)==2:before.append(bound(out[ids[0],4:7]));after.append(bound(out[ids[1],4:7]))
    before=np.array(before);after=np.array(after)
    T=np.array(read(phase.BASE/'evaluation/metrics.json')['metrics']['livo']['SE3_alignment_estimated_to_reference'])
    assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-12)
    rows=list(csv.DictReader((fp/'paired-phase-errors.csv').open()));bounds=[];eb=[];ea=[]
    for r in rows:
        l=int(r['interpolation_left']);u=int(r['interpolation_right']);a=float(r['interpolation_fraction'])
        bounds.append([np.linalg.norm((1-a)*before[l]+a*before[u]),np.linalg.norm((1-a)*after[l]+a*after[u])])
        eb.append(float(r['pre_position_error_m']));ea.append(float(r['post_position_error_m']))
    bounds=np.array(bounds);eb=np.array(eb);ea=np.array(ea);delta=ea-eb;unc=bounds.sum(1)
    rmse=lambda x:float(np.sqrt(np.mean(x*x)))
    pre_interval=[rmse(np.maximum(0,eb-bounds[:,0])),rmse(eb+bounds[:,0])]
    post_interval=[rmse(np.maximum(0,ea-bounds[:,1])),rmse(ea+bounds[:,1])]
    result=dict(status='passed_coordinate_rounding_sensitivity_bound',paired_reference_points=len(rows),
        robust_error_decrease_count=int((delta+unc<0).sum()),robust_error_increase_count=int((delta-unc>0).sum()),
        rounding_ambiguous_count=int(((delta-unc<=0)&(delta+unc>=0)).sum()),
        mean_error_change_interval_mm=[float(1000*np.mean(delta-unc)),float(1000*np.mean(delta+unc))],
        rmse_change_interval_mm=[1000*(post_interval[0]-pre_interval[1]),1000*(post_interval[1]-pre_interval[0])],
        max_paired_error_rounding_bound_mm=float(1000*unc.max()),
        method='Default six significant digits gives component half-bin; positive linear interpolation propagates component bounds. Euclidean norm is 1-Lipschitz, frozen rotation preserves norms. Combine pre/post bounds conservatively.',
        limitation='Only state text rounding, conditional on unchanged alignment, timestamps and reference. Not GT accuracy or statistical uncertainty.',
        source_metrics_sha256=sha(fp/'metrics.json'),source_csv_sha256=sha(fp/'paired-phase-errors.csv'),review_source_sha256=sha(Path(__file__)))
    save(fp/'validation.json',result)
    print(json.dumps({'splatam':checks,'fast':result},ensure_ascii=False))

if __name__=='__main__':main()
