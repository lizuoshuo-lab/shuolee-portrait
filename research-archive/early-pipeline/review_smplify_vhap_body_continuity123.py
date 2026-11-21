"""Reuse the all59 review for body-only original sequential propagation."""
import numpy as np
import review_smplify_vhap_surface123 as review


def main():
    review.RUN = review.ROOT / 'smplify-vhap-body-continuity'
    review.RECORD_PREFIX = 'SMPL-X_VHAP仅身体连续初始化'
    review.CANDIDATE_LABEL = 'body_only_sequential_propagation'
    review.CONTROL_RUN = review.ROOT / 'smplify-vhap-continuous'
    review.CONTROL_LABEL = 'same_stages_original_full_copy_control'
    review.RECORD_SUFFIX = '原图_SMPLify初始化_全参数传递控制_仅身体传递'
    review.main()
    result = review.read(review.RUN / 'comparison/result.json')
    summaries = {}
    for label, folder in [('matched_full_copy', 'continuous'),
                          ('same_stages_no_copy', 'sequential-stage'),
                          ('direct_body_reference', 'direct-body')]:
        control = review.read(review.ROOT / ('smplify-vhap-'+folder) / 'epoch-30/result.json')
        assert [r['frame'] for r in control['records']] == [r['frame_id'] for r in result['per_frame']]
        metrics = {}
        for metric in ('mask_iou', 'target_uncovered', 'mesh_outside', 'elbow_px',
                       'body', 'face', 'left_hand', 'right_hand', 'feet'):
            def value(row):
                return row[metric] if metric in ('mask_iou','target_uncovered','mesh_outside','elbow_px') else row['groups'][metric]['median']
            before = np.array([value(r) for r in control['records']])
            after = np.array([value(r['candidate']) for r in result['per_frame']])
            improved = after > before if metric == 'mask_iou' else after < before
            metrics[metric] = dict(control_median=float(np.median(before)),
                candidate_median=float(np.median(after)), improved_frames=int(improved.sum()),
                worsened_frames=int((~improved & (before != after)).sum()))
        summaries[label] = metrics
    result['body_continuity_comparison'] = summaries
    result['comparison_scope'] = ('Matched full-copy control: only the body latent is propagated; all eight '
        'non-body values restored exactly. Same5220 updates and VPoser32/PCA12. The no-copy and direct63 '
        'results are quality references, not identical single-factor controls. All59 frames retained.')
    review.save(review.RUN / 'comparison/result.json', result)


if __name__ == '__main__':
    main()
