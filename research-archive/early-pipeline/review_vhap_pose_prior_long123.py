"""Reuse the native all59 review at a named saved long-fit checkpoint."""
import argparse
import numpy as np
import review_smplify_vhap_surface123 as review


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch',type=int,choices=[30,60,90,120],required=True)
    args=parser.parse_args()
    review.RUN=review.ROOT/'smplify-vhap-pose-prior-long'
    review.CHECKPOINT_EPOCH=args.epoch
    review.COMPARISON_DIR=f'comparison-epoch{args.epoch:03d}'
    review.REQUIRE_COMPLETE=args.epoch==120
    review.RECORD_PREFIX=f'SMPL-X_VHAP长训练{args.epoch:03d}轮'
    review.CANDIDATE_LABEL=f'long_fit_epoch{args.epoch}'
    review.CONTROL_RUN=review.ROOT/'smplify-vhap-pose-prior-sum'
    review.CONTROL_LABEL='matched_short_fit_epoch30'
    review.RECORD_SUFFIX='原图_SMPLify初始化_短训练30轮_长训练'
    review.main()
    result=review.read(review.RUN/review.COMPARISON_DIR/'result.json')
    before=review.read(review.CONTROL_RUN/'epoch-30/result.json')['records']
    assert [r['frame'] for r in before]==[r['frame_id'] for r in result['per_frame']]
    summary={}
    for metric in ('mask_iou','target_uncovered','mesh_outside','elbow_px','body','face','left_hand','right_hand','feet'):
        def value(row):
            return row[metric] if metric in ('mask_iou','target_uncovered','mesh_outside','elbow_px') else row['groups'][metric]['median']
        a=np.asarray([value(r) for r in before]);b=np.asarray([value(r['candidate']) for r in result['per_frame']])
        better=b>a if metric=='mask_iou' else b<a
        summary[metric]=dict(control_median=float(np.median(a)),candidate_median=float(np.median(b)),
            improved_frames=int(better.sum()),worsened_frames=int((~better & (a!=b)).sum()))
    result['matched_short_comparison']=summary
    result['comparison_scope']='Same59 inputs/priors/initialization; longer global budget with stretched LR decay. Not a steps-only change.'
    review.save(review.RUN/review.COMPARISON_DIR/'result.json',result)


if __name__=='__main__':main()
