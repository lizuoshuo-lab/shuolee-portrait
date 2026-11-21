import review_smplify_vhap_surface123 as review
review.RUN=review.ROOT/'smplify-vhap-mask'
review.RECORD_PREFIX='SMPL-X_VHAP轮廓'
review.CANDIDATE_LABEL='VHAP_foreground_normalized_alpha'
if __name__=='__main__':review.main()
