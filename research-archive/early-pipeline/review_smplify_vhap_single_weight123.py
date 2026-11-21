import review_smplify_vhap_surface123 as review
review.RUN = review.ROOT / 'smplify-vhap-single-landmark-weight'
review.RECORD_PREFIX = 'SMPL-X_VHAP单次关键点权重'
review.CANDIDATE_LABEL = 'VHAP_single_landmark_weight'
if __name__ == '__main__':
    review.main()
    import numpy as np
    current=review.read(review.RUN/'comparison/result.json')
    control=review.read(review.ROOT/'smplify-vhap-normalized-laplacian/epoch-30/result.json')
    previous={r['frame']:r for r in control['records']}
    comparisons={}
    for key in ('mask_iou','target_uncovered','mesh_outside','elbow_px','body','face','left_hand','right_hand','feet'):
        def value(row):
            return row[key] if key in row else row['groups'][key]['median']
        a=np.array([value(previous[r['frame_id']]) for r in current['per_frame']])
        b=np.array([value(r['candidate']) for r in current['per_frame']])
        better=b>a if key=='mask_iou' else b<a
        comparisons[key]=dict(control_median=float(np.median(a)),candidate_median=float(np.median(b)),
                              improved=int(better.sum()),worsened=int((a!=b).sum()-better.sum()))
    current['preceding_normalized_laplacian_comparison']=comparisons
    current['frozen_control_checkpoint_sha256']=review.sha(review.ROOT/'smplify-vhap-normalized-laplacian/epoch-30/tracked-smplx-params.npz')
    review.save(review.RUN/'comparison/result.json',current)
