"""Recompute full-view timing summaries from individual recorded measurements."""
from pathlib import Path
import json,hashlib
import numpy as np
LAB=Path(__file__).resolve().parents[1]
SRC=LAB/'outputs/large-scene-gs-001/research/city-render-stages-022'
OUT=LAB/'outputs/city-lod-timing-001'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=False)
    raw=read(SRC/'measurements.json');warm=read(SRC/'warmups.json');reported=read(SRC/'metrics.json')
    assert len(raw)==702 and len(warm)==234 and len(reported['summaries'])==234
    assert all(r['rgb_bitwise_equal'] and r['max_abs_rgb_difference']==0 for r in raw+warm)
    assert all(np.isclose(r['synchronized_stage_sum_ms'],r['cut_host_ms']+r['weights_host_ms']+r['synchronized_host_ms'],rtol=0,atol=1e-10) for r in raw)
    summary={};plot={}
    keys=('cut_host_ms','weights_host_ms','synchronized_stage_sum_ms','full_render_stream_ms','native_rasterizer_stream_ms','surrounding_stream_ms')
    for setting in ('leaves','tau3'):
        rows=[];groups=[]
        for j in range(117):
            group=[r for r in raw if r['setting']==setting and r['index']==j]
            assert sorted(r['repeat'] for r in group)==[0,1,2]
            entry={k:float(np.median([r[k] for r in group])) for k in keys};entry['index']=j;rows.append(entry);groups.append(group)
            old=next(r for r in reported['summaries'] if r['setting']==setting and r['index']==j)
            assert all(old[k]['median']==entry[k] for k in keys)
        summary[setting]={k:float(np.median([r[k] for r in rows])) for k in keys}
        plot[setting]=rows
    leaf=np.array([r['synchronized_stage_sum_ms'] for r in plot['leaves']]);lod=np.array([r['synchronized_stage_sum_ms'] for r in plot['tau3']])
    result=dict(status='verified_all117_paired_stage_timing',views=117,repetitions=3,measured_renders=702,warmup_renders=234,
        all_rendered_RGB_bitwise_equal_to_archived_reference=True,all_cut_and_weights_exact_checks_passed=True,
        statistics='Median across 117 per-view medians of 3 repetitions. Sum stages per repetition before taking medians; do not sum separate stage medians.',
        unit='ms',timing_by_setting=summary,views_tau3_faster=int((lod<leaf).sum()),
        median_paired_stage_speed_ratio=float(np.median(leaf/lod)),ratio_of_overall_stage_medians=float(np.median(leaf)/np.median(lod)),
        rasterizer_fraction_definition='Per-view median native stream span divided by per-view median full render stream span; CUDA event spans include dispatch gaps.',
        tau3_median_native_fraction=float(np.median([r['native_rasterizer_stream_ms']/r['full_render_stream_ms'] for r in plot['tau3']])),
        previous_record='Prior one-pass 117-view timings remain unchanged. New measurement interleaves settings with three repeats per view. No single cause of the old discrepancy is established.',
        model_changed=False,renderer_changed=False,quality_scores_recomputed=False,streaming=False,
        scope='SmallCity local post006, 256 x 172, RTX 5080, frozen 117 heldout views, whole model resident, cached immutable properties. Not live playback FPS or a new method.',
        input_records={str(p.relative_to(LAB)).replace('\\','/'):sha(p) for p in (SRC/'source.py',SRC/'metrics.json',SRC/'measurements.json',SRC/'warmups.json',SRC/'input-bindings.json',SRC/'source-audit.json')})
    save(OUT/'metrics.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    for setting,color,label in [('leaves','#3a5873','Complete leaf layer'),('tau3','#008878','LOD 3 px')]:
        axes[0].plot(range(117),[r['synchronized_stage_sum_ms'] for r in plot[setting]],label=label,color=color,lw=1.3)
        axes[1].plot(range(117),[r['native_rasterizer_stream_ms'] for r in plot[setting]],label=label,color=color,lw=1.3)
    axes[0].set(ylabel='Selected stages total / ms',title='117 fixed views; median of 3 interleaved repetitions per view')
    axes[1].set(ylabel='Native rasterizer span / ms',xlabel='Held-out camera index (frozen order)')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'paired-timing.png',dpi=180);plt.close(fig)
    save(OUT/'validation.json',dict(status='passed_independent_row_reduction',rows=len(raw),warmups=len(warm),
        unique_view_setting_repeat_keys=len({(r['index'],r['setting'],r['repeat']) for r in raw}),
        source_review_sha256=sha(Path(__file__)),summary_sha256=sha(OUT/'metrics.json'),
        limit='Timing arithmetic and archived exact-image controls independently checked; not another GPU run.'))
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
