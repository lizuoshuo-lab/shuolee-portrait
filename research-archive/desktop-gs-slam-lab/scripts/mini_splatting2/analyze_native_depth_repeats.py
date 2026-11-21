"""Aggregate matched seeds without treating correlated frames as independent runs."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def describe(values):
    return dict(mean=statistics.mean(values),sample_std=statistics.stdev(values),minimum=min(values),maximum=max(values),values=values)


def main():
    queue=json.loads((ROOT/'native-depth-repeat-queue.json').read_text());assert queue['status']=='complete'
    out=ROOT/'native-depth-matched-repeats.json';assert not out.exists(),'Preserve frozen repetition results'
    runs={key:[json.loads((ROOT/f'{prefix}-18000-s{seed}'/'evaluation-summary.json').read_text()) for seed in [0,1,2]]
        for key,prefix in [('baseline','native-motion-mip-depthmask'),('depth003','native-depthguide003'),('depth010','native-depthguide010')]}
    metrics=['full_psnr','crop_psnr','ssim','lpips_alex'];result={}
    for key,rows in runs.items():
        summary={m:describe([r['native_mean'][m] for r in rows]) for m in metrics}
        entry=dict(aggregate=summary,gaussians=[r['gaussians'] for r in rows])
        if key!='baseline':
            entry['paired_difference']={m:describe([a['native_mean'][m]-b['native_mean'][m] for a,b in zip(rows,runs['baseline'])]) for m in metrics}
            entry['positive_full_psnr_seeds']=sum(x>0 for x in entry['paired_difference']['full_psnr']['values'])
            entry['positive_crop_psnr_seeds']=sum(x>0 for x in entry['paired_difference']['crop_psnr']['values'])
            entry['lower_lpips_seeds']=sum(x<0 for x in entry['paired_difference']['lpips_alex']['values'])
            entry['per_frame_paired_mean']=[dict(frame=rows[0]['native_rows'][i]['frame'],
                full_psnr=statistics.mean(rows[s]['native_rows'][i]['full']['psnr']-runs['baseline'][s]['native_rows'][i]['full']['psnr'] for s in range(3)),
                crop_psnr=statistics.mean(rows[s]['native_rows'][i]['subject_crop']['psnr']-runs['baseline'][s]['native_rows'][i]['subject_crop']['psnr'] for s in range(3))) for i in range(8)]
        result[key]=entry
    data=dict(status='matched_native_appearance_repeats_complete',seeds=[0,1,2],models=result,independent_final_test=False,
        scope='eight repeatedly inspected development views; summarize runs and paired seed differences, not24independent samples',
        remaining=['extra24frame evaluation for new guided seeds','sparse-geometry and contradiction audit for new guided seeds',
            'visual review of full images and shared extrapolation paths','independent camera-distance/pitch and measured geometry coverage'])
    out.write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))


if __name__=='__main__':main()
