"""Compare fixed keys and coverage, avoiding selective alpha-based improvement."""
from pathlib import Path
import json,numpy as np
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current')
A=B/'fullscene-person-constraint-audit-076';R=B/'person-static-observed-correspondence-fresh-077'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    original=read(A/'correspondence-pairs.json')['rows'];candidate=read(R/'review/correspondence/correspondence-pairs.json')['rows']
    key=lambda r:(r['track'],r['source'],r['target'])
    d0={key(r):r for r in original};d1={key(r):r for r in candidate};common=set(d0)&set(d1);groups={}
    for group,name in [(1,'face_lower'),(2,'face_upper')]:
        before={k for k,r in d0.items() if r['group']==group};after={k for k,r in d1.items() if r['group']==group};ks=sorted(before&after)
        q={}
        for mode in ['native','rgb_center','triangulated']:
            a=np.array([d0[k]['errors_rectified_px'][mode] for k in ks]);b=np.array([d1[k]['errors_rectified_px'][mode] for k in ks])
            q[mode]=dict(before_median=float(np.median(a)),after_median=float(np.median(b)),before_p90=float(np.quantile(a,.9)),
                after_p90=float(np.quantile(b,.9)),median_relative_change=float(np.median(b)/np.median(a)-1),
                mean_paired_change=float((b-a).mean()),pairs_improved=int((b<a).sum()))
        groups[name]=dict(baseline_pairs=len(before),candidate_pairs=len(after),common_pairs=len(ks),
            retained_baseline_fraction=len(ks)/len(before),metrics=q)
    face=groups['face_lower'];up=groups['face_upper'];metrics=read(R/'review/same-camera-result.json')
    gate=dict(lower_face_transfer20percent=face['metrics']['rgb_center']['after_median']<=.8*face['metrics']['rgb_center']['before_median'],
        lower_face_coverage95percent=face['retained_baseline_fraction']>=.95,upper_face_coverage95percent=up['retained_baseline_fraction']>=.95,
        upper_face_no_more_than5percent_worse=up['metrics']['rgb_center']['after_median']<=1.05*up['metrics']['rgb_center']['before_median'],
        whole_person_protection=metrics['numeric_screening_passed'])
    output=dict(groups=groups,gates=gate,all_numeric_gates_passed=all(gate.values()),
        units='full075/077rectified image pixels; not1536trainingpixels',primary_endpoint='RGB pixel-center-consistent surface transfer',
        fixed_pair_keys=True,feature_disjoint_development=True,unseen_image_GT=False,
        camera_or_feature_error_not_ruled_out=True,visual_inspection_complete=False,quality_accepted=False)
    (R/'review/paired-transfer-comparison.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8');print(json.dumps(output))


if __name__=='__main__':main()
