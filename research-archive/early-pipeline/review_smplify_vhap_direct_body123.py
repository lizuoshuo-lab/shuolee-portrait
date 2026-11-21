"""All59 fixed-camera comparison for the proposed original direct-body option."""
import numpy as np
import review_smplify_vhap_surface123 as review


def main():
    review.RUN=review.ROOT/'smplify-vhap-direct-body'
    review.RECORD_PREFIX='SMPL-X_VHAP原版直接身体'
    review.CANDIDATE_LABEL='original_direct_axis_angle63'
    review.CONTROL_RUN=review.ROOT/'smplify-vhap-normalized-laplacian'
    review.CONTROL_LABEL='normalized-laplacian_VPoser32_control'
    review.RECORD_SUFFIX='原图_SMPLify初始化_原版平滑控制_直接身体轴角'
    review.main()
    result=review.read(review.RUN/'comparison/result.json')
    control=review.read(review.CONTROL_RUN/'epoch-30/result.json')
    assert [r['frame'] for r in control['records']]==[r['frame_id'] for r in result['per_frame']]
    comparison={}
    for metric in ('mask_iou','target_uncovered','mesh_outside','elbow_px','body','face','left_hand','right_hand','feet'):
        if metric in ('mask_iou','target_uncovered','mesh_outside','elbow_px'):
            before=np.array([r[metric] for r in control['records']])
            after=np.array([r['candidate'][metric] for r in result['per_frame']])
        else:
            before=np.array([r['groups'][metric]['median'] for r in control['records']])
            after=np.array([r['candidate']['groups'][metric]['median'] for r in result['per_frame']])
        improved=after>before if metric=='mask_iou' else after<before
        comparison[metric]=dict(control_median=float(np.median(before)),candidate_median=float(np.median(after)),
            improved_frames=int(improved.sum()),worsened_frames=int((~improved&(before!=after)).sum()))
    result['original_direct_body_option_comparison']=comparison
    result['option_changes_prior_and_temporal_metric']=True
    result['not_strict_wrist_only_single_factor']=True
    review.save(review.RUN/'comparison/result.json',result)


if __name__=='__main__':main()
