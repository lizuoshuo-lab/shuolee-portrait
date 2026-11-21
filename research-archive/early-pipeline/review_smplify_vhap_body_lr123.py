"""All59 same-camera review: only the global body LR changes."""
import numpy as np
import review_smplify_vhap_surface123 as review

def main():
    review.RUN=review.ROOT/'smplify-vhap-body-lr'
    review.RECORD_PREFIX='SMPL-X_VHAP身体学习率对照'
    review.CANDIDATE_LABEL='global_body_lr_x10'
    review.CONTROL_RUN=review.ROOT/'smplify-vhap-direct-body'
    review.CONTROL_LABEL='original_direct_axis_angle63_control'
    review.RECORD_SUFFIX='原图_SMPLify初始化_直接身体控制_身体学习率十倍'
    review.main()
    result=review.read(review.RUN/'comparison/result.json')
    control=review.read(review.CONTROL_RUN/'epoch-30/result.json')
    assert [r['frame'] for r in control['records']]==[r['frame_id'] for r in result['per_frame']]
    comparison={}
    for metric in ('mask_iou','target_uncovered','mesh_outside','elbow_px','body','face','left_hand','right_hand','feet'):
        if metric in ('mask_iou','target_uncovered','mesh_outside','elbow_px'):
            before=np.array([r[metric] for r in control['records']]);after=np.array([r['candidate'][metric] for r in result['per_frame']])
        else:
            before=np.array([r['groups'][metric]['median'] for r in control['records']]);after=np.array([r['candidate']['groups'][metric]['median'] for r in result['per_frame']])
        improved=after>before if metric=='mask_iou' else after<before
        comparison[metric]=dict(control_median=float(np.median(before)),candidate_median=float(np.median(after)),improved_frames=int(improved.sum()),worsened_frames=int((~improved&(before!=after)).sum()))
    result['body_lr_comparison']=comparison
    result['only_global_body_lr_multiplier']=10
    review.save(review.RUN/'comparison/result.json',result)

if __name__=='__main__':main()
