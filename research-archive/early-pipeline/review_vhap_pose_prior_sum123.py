"""Whole59 matched review of the single VPoser-prior reduction control."""
import numpy as np
import review_smplify_vhap_surface123 as review


def main():
    review.RUN = review.ROOT / 'smplify-vhap-pose-prior-sum'
    review.RECORD_PREFIX = 'SMPL-X_VHAP人体先验求和对照'
    review.CANDIDATE_LABEL = 'vposer_dimension_sum32'
    review.CONTROL_RUN = review.ROOT / 'smplify-vhap-normalized-laplacian'
    review.CONTROL_LABEL = 'matched_vposer_dimension_mean32'
    review.RECORD_SUFFIX = '原图_SMPLify初始化_人体先验平均控制_人体先验求和'
    review.main()
    result = review.read(review.RUN / 'comparison/result.json')
    before = review.read(review.CONTROL_RUN / 'epoch-30/result.json')['records']
    assert [r['frame'] for r in before] == [r['frame_id'] for r in result['per_frame']]
    summary = {}
    for metric in ('mask_iou', 'target_uncovered', 'mesh_outside', 'elbow_px', 'body', 'face', 'left_hand', 'right_hand', 'feet'):
        def value(row):
            return row[metric] if metric in ('mask_iou','target_uncovered','mesh_outside','elbow_px') else row['groups'][metric]['median']
        a = np.asarray([value(r) for r in before])
        b = np.asarray([value(r['candidate']) for r in result['per_frame']])
        better = b > a if metric == 'mask_iou' else b < a
        summary[metric] = dict(control_median=float(np.median(a)), candidate_median=float(np.median(b)),
                               improved_frames=int(better.sum()), worsened_frames=int((~better & (a != b)).sum()))
    result['matched_prior_comparison'] = summary
    result['comparison_scope'] = 'Same59 SMPLify initial poses, K, masks, optimizer, LR and500+1770 steps. Only VPoser32 latent prior dimension mean changes to sum.'
    review.save(review.RUN / 'comparison/result.json', result)


if __name__ == '__main__':
    main()
