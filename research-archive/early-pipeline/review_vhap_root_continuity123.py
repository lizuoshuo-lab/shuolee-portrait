"""Reuse the frozen all59 native review for the equivalent-root control."""
import numpy as np
from scipy.spatial.transform import Rotation
import review_smplify_vhap_surface123 as review


def temporal(path):
    with np.load(path,allow_pickle=False) as z:
        rotvec=z['global_orient'].astype(np.float64)
        frames=z['frame_ids'].tolist()
    rotations=Rotation.from_rotvec(rotvec)
    physical=np.degrees((rotations[:-1].inv()*rotations[1:]).magnitude())
    represented=np.degrees(np.linalg.norm(np.diff(rotvec,axis=0),axis=1))
    return dict(physical_median_deg=float(np.median(physical)),physical_max_deg=float(physical.max()),
                raw_vector_max_deg=float(represented.max()),
                pairs=[dict(previous=frames[i],current=frames[i+1],physical_deg=float(p),raw_vector_deg=float(v))
                       for i,(p,v) in enumerate(zip(physical,represented))])


def main():
    review.RUN=review.ROOT/'smplify-vhap-root-continuity'
    review.RECORD_PREFIX='SMPL-X_VHAP根旋转等价表示'
    review.CANDIDATE_LABEL='equivalent_root_branch'
    review.CONTROL_RUN=review.ROOT/'smplify-vhap-pose-prior-sum'
    review.CONTROL_LABEL='matched_short_original_root_branch'
    review.RECORD_SUFFIX='原图_SMPLify初始化_原表示控制_等价表示'
    assert review.read(review.RUN/'config.json')==review.read(review.CONTROL_RUN/'config.json')
    review.main()
    result=review.read(review.RUN/'comparison/result.json')
    before=review.read(review.CONTROL_RUN/'epoch-30/result.json')['records']
    assert [r['frame'] for r in before]==[r['frame_id'] for r in result['per_frame']]
    summary={}
    for metric in ('mask_iou','target_uncovered','mesh_outside','elbow_px','body','face','left_hand','right_hand','feet'):
        def value(row):
            return row[metric] if metric in ('mask_iou','target_uncovered','mesh_outside','elbow_px') else row['groups'][metric]['median']
        a=np.asarray([value(r) for r in before]);b=np.asarray([value(r['candidate']) for r in result['per_frame']])
        better=b>a if metric=='mask_iou' else b<a
        summary[metric]=dict(control_median=float(np.median(a)),candidate_median=float(np.median(b)),
            improved_frames=int(better.sum()),worsened_frames=int((~better & (a!=b)).sum()),
            control_per_frame=a.tolist(),candidate_per_frame=b.tolist())
    result['matched_root_comparison']=summary
    result['temporal_root']={label:temporal(run/'epoch-30/tracked-smplx-params.npz')
                            for label,run in [('control',review.CONTROL_RUN),('candidate',review.RUN)]}
    result['comparison_scope']='Same59 physical initial poses, K, images, masks, optimizer, losses, LR and500+1770 steps. Only equivalent global_orient branch chosen before fitting. Physical jumps may remain; no interpolation, frame exclusion or adjacent-pose copying.'
    review.save(review.RUN/'comparison/result.json',result)


if __name__=='__main__':main()
