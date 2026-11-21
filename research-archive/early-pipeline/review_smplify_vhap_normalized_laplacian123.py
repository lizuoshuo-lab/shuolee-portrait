import review_smplify_vhap_surface123 as review
review.RUN=review.ROOT/'smplify-vhap-normalized-laplacian'
review.RECORD_PREFIX='SMPL-X_VHAP原版平滑'
review.CANDIDATE_LABEL='VHAP_original_normalized_laplacian'
if __name__=='__main__':review.main()
