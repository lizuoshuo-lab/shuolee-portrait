"""All59 native same-camera review of the original sequential initialization."""
import numpy as np
import review_smplify_vhap_surface123 as review


def main():
    review.RUN = review.ROOT / 'smplify-vhap-continuous'
    review.RECORD_PREFIX = 'SMPL-X_VHAP原版连续初始化'
    review.CANDIDATE_LABEL = 'original_sequential_propagation'
    review.CONTROL_RUN = review.ROOT / 'smplify-vhap-sequential-stage'
    review.CONTROL_LABEL = 'same_stages_no_propagation_VPoser32_control'
    review.RECORD_SUFFIX = '原图_SMPLify初始化_逐帧不传递控制_原版连续初始化'
    review.main()
    result = review.read(review.RUN / 'comparison/result.json')
    control = review.read(review.CONTROL_RUN / 'epoch-30/result.json')
    assert [row['frame'] for row in control['records']] == [row['frame_id'] for row in result['per_frame']]
    comparison = {}
    for metric in ('mask_iou', 'target_uncovered', 'mesh_outside', 'elbow_px',
                   'body', 'face', 'left_hand', 'right_hand', 'feet'):
        if metric in ('mask_iou', 'target_uncovered', 'mesh_outside', 'elbow_px'):
            before = np.array([row[metric] for row in control['records']])
            after = np.array([row['candidate'][metric] for row in result['per_frame']])
        else:
            before = np.array([row['groups'][metric]['median'] for row in control['records']])
            after = np.array([row['candidate']['groups'][metric]['median'] for row in result['per_frame']])
        improved = after > before if metric == 'mask_iou' else after < before
        comparison[metric] = dict(control_median=float(np.median(before)),
                                  candidate_median=float(np.median(after)),
                                  improved_frames=int(improved.sum()),
                                  worsened_frames=int((~improved & (before != after)).sum()))
    result['continuous_initialization_comparison'] = comparison
    result['comparison_scope'] = (
        'All59; the same original 500+2950+1770 stages and VPoser32/PCA12 settings. '
        'Only call the original initialize_next_timtestep after each original RGB50 stage; '
        'retain its 59 calls, 57 copied adjacent pairs and original boundary omissions. '
        'Per-frame root, face and hand values can be copied; fixed K/extrinsics do not '
        'preserve each original body-root seed. No selected-frame replacement.')
    review.save(review.RUN / 'comparison/result.json', result)


if __name__ == '__main__':
    main()
