"""Use the same native CPU review for the separate all59 joint-stage trial."""
import review_smplify_vhap_surface123 as review

review.RUN=review.ROOT/'smplify-vhap-joint'
review.RECORD_PREFIX='SMPL-X_VHAP联合'
review.CANDIDATE_LABEL='VHAP_original_joint_groups'

if __name__=='__main__':
    review.main()
