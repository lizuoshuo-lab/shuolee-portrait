"""Fixed-query, common-coverage comparison. No quality acceptance by a scalar."""
from pathlib import Path
import hashlib,json,numpy as np

P=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/static-ncc-paired-fresh-093')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def compare(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if not len(a):return dict(common_queries=0,metrics_available=False)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    am,bm=float(np.median(a)),float(np.median(b))
    return dict(common_queries=len(a),control_median=am,no_ncc_median=bm,
        control_p90=float(np.quantile(a,.9)),no_ncc_p90=float(np.quantile(b,.9)),
        median_relative_change=bm/am-1 if am else None,paired_mean_change=float((b-a).mean()),
        queries_decreased=int((b<a).sum()),metrics_available=True)

def main():
    result={};files={}
    paths=[P/a/'review/correspondence/correspondence-pairs.json' for a in ['control','no_ncc']]
    rows=[read(p)['rows'] for p in paths];files.update({str(p):sha(p) for p in paths})
    key=lambda r:(r['track'],r['source'],r['target'])
    ds=[{key(r):r for r in rr} for rr in rows]
    assert all(len(d)==len(rr) for d,rr in zip(ds,rows)), 'Duplicate correspondence keys'
    groups={}
    for group,name in [(1,'face_lower'),(2,'face_upper')]:
        ks=[{k for k,r in d.items() if r['group']==group} for d in ds];common=sorted(ks[0]&ks[1])
        measurements={}
        for mode in ['native','rgb_center','triangulated']:
            measurements[mode]=compare(*[[d[k]['errors_rectified_px'][mode] for k in common] for d in ds])
        groups[name]=dict(control_pairs=len(ks[0]),no_ncc_pairs=len(ks[1]),common_pairs=len(common),
            control_coverage_retained=len(common)/len(ks[0]) if ks[0] else None,
            control_only_keys=[list(k) for k in sorted(ks[0]-ks[1])],no_ncc_only_keys=[list(k) for k in sorted(ks[1]-ks[0])],metrics=measurements)
    result['surface_transfer']=groups
    paths=[P/a/'review/depth/result.json' for a in ['control','no_ncc']]
    depth=[read(p) for p in paths];files.update({str(p):sha(p) for p in paths})
    assert depth[0]['normalizer']==depth[1]['normalizer']
    ds=[{r['index']:r for r in d['records']} for d in depth];assert set(ds[0])==set(ds[1])
    for k in ds[0]:
        a,b=ds[0][k],ds[1][k]
        assert (a['raw_frame'],a['kind'],a['pixel'])==(b['raw_frame'],b['kind'],b['pixel'])
    groups={}
    for kind in sorted({r['kind'] for r in ds[0].values()}):
        keys=sorted(k for k,r in ds[0].items() if r['kind']==kind)
        g=dict(fixed_queries=len(keys),alpha=compare(*[[d[k]['alpha_after'] for k in keys] for d in ds]),
            control_alpha_above005=sum(ds[0][k]['alpha_after']>.05 for k in keys),
            no_ncc_alpha_above005=sum(ds[1][k]['alpha_after']>.05 for k in keys))
        for mode in ['center','plane']:
            valid=[k for k in keys if all('span90_body_fraction' in d[k].get(mode,{}) for d in ds)]
            common_visible=[k for k in valid if all(d[k]['alpha_after']>.05 for d in ds)]
            g[mode]=dict(all_finite=compare(*[[d[k][mode]['span90_body_fraction'] for k in valid] for d in ds]),
                visible_in_both=compare(*[[d[k][mode]['span90_body_fraction'] for k in common_visible] for d in ds]))
        groups[kind]=g
    result.update(depth_mixture=groups,depth_normalizer=depth[0]['normalizer'],source_sha256=files,
        correspondence_units='Full rectified image pixels; no synthetic view ground truth',
        correspondence_interpretation='Use rgb_center for the093 comparison. The helper also retains legacy uncorrected plane/NCC gate fields for historical comparison; those fields do not reproduce093 corrected training gates.',
        depth_units='90 percent weighted depth span divided by fixed077 robust person height; not metric depth',
        fixed_queries=True,common_coverage_reported=True,optimizer_updates=0,quality_accepted=False,
        interpretation='Transfer errors include observation and camera uncertainty. Thin depth mixtures or less alpha alone do not prove correct geometry. Inspect actual photographs and novel-view renders.')
    dest=P/'review/common-query-comparison.json';dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
