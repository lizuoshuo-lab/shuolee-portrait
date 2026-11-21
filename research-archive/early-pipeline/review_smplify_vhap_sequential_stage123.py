"""All59 review for the original per-frame RGB-stage restoration trial."""
import numpy as np
import review_smplify_vhap_surface123 as review

review.RUN=review.ROOT/'smplify-vhap-sequential-stage'
review.RECORD_PREFIX='SMPL-X_VHAP原版逐帧精修'
review.CANDIDATE_LABEL='VHAP_original_sequential_then_global'
review.CONTROL_RUN=review.ROOT/'smplify-vhap-normalized-laplacian'
review.CONTROL_LABEL='VHAP_normalized_laplacian_control'


def main():
    review.main()
    current=review.read(review.RUN/'comparison/result.json')
    control=review.read(review.CONTROL_RUN/'epoch-30/result.json')
    assert [r['frame'] for r in control['records']]==[r['frame_id'] for r in current['per_frame']]
    comparisons={}
    for key in ('mask_iou','target_uncovered','mesh_outside','elbow_px','body','face','left_hand','right_hand','feet'):
        def value(row):
            return row[key] if key in row else row['groups'][key]['median']
        a=np.array([value(r) for r in control['records']])
        b=np.array([value(r['candidate']) for r in current['per_frame']])
        better=b>a if key=='mask_iou' else b<a
        comparisons[key]=dict(control_median=float(np.median(a)),candidate_median=float(np.median(b)),
            improved=int(better.sum()),worsened=int((a!=b).sum()-better.sum()))
    current['preceding_normalized_laplacian_comparison']=comparisons
    current['comparison_scope']='One added original sequential stage, with all59 same-frame four-panel native crops; no selected-frame replacement.'
    review.save(review.RUN/'comparison/result.json',current)


if __name__=='__main__':
    main()
