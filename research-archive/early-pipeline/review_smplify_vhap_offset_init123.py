import review_smplify_vhap_surface123 as review
review.RUN=review.ROOT/'smplify-vhap-offset-init'
review.RECORD_PREFIX='SMPL-X_VHAP原版外扩初始'
review.CANDIDATE_LABEL='VHAP_restored_offset_initialization'
if __name__=='__main__':review.main()
