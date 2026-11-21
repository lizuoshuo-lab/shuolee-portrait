"""Unannotated all-frame comparisons and transparent detector-fit statistics."""
import argparse
import numpy as np
from PIL import Image
import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',choices=['vhap-selected-reference','vhap-selected-reference-white'],default='vhap-selected-reference')
    args=parser.parse_args()
    root=a.REC/'demo-vhap-initialization'/args.run
    run=a.read(root/'result.json');assert run['full_sequence_complete'] and run['frames']==59
    source=root/'global-epoch-30';result=a.read(source/'result.json')
    out=root/'review';a.lp(out).mkdir(exist_ok=False)
    for region in ['body','face']:
        entries=[x for x in result['images'] if x['region']==region]
        assert len(entries)==59
        if region=='body':
            for start in range(0,59,10):
                page=Image.new('RGB',(1000,2250),'white')
                for slot,entry in enumerate(entries[start:start+10]):
                    im=Image.open(a.lp(source/entry['path'].split('/')[-1])).convert('RGB')
                    im.thumbnail((500,450));page.paste(im,((slot%2)*500,(slot//2)*450))
                page.save(a.lp(out/f'body-page-{start//10:02d}.jpg'),quality=94)
        else:
            page=Image.new('RGB',(1200,1440),'white')
            for slot,entry in enumerate(entries):
                im=Image.open(a.lp(source/entry['path'].split('/')[-1])).convert('RGB')
                im.thumbnail((240,120));page.paste(im,((slot%5)*240,(slot//5)*120))
            page.save(a.lp(out/'faces-all59.jpg'),quality=95)
    stages={}
    for stage in ['rgb_init_offset','bidirectional-complete','global-epoch-10','global-epoch-20','global-epoch-30']:
        d=a.read(root/stage/'result.json');groups={}
        for g in ['body','face','left_hand','right_hand']:
            v=[m['groups'][g] for m in d['metrics']]
            groups[g]=dict(frames=len(v),median_of_frame_medians=float(np.median([x['median'] for x in v])),
                maximum_frame_median=float(max(x['median'] for x in v)),largest_observation_error=float(max(x['max'] for x in v)))
        stages[stage]=dict(groups=groups,offset_mm=d['offset_mm'])
    with np.load(a.lp(source/'tracked-smplx-params.npz'),allow_pickle=False) as z:fit=dict(z)
    with np.load(a.lp(root/'rgb_init_offset/tracked-smplx-params.npz'),allow_pickle=False) as z:before=dict(z)
    with np.load(a.lp('D:/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz'),allow_pickle=False) as z:
        template=z['v_template'];shapedirs=z['shapedirs'][:,:,:len(fit['betas'])]
    base=template+np.einsum('vci,i->vc',shapedirs,fit['betas']);shifted=base+fit['static_offset'][0];faces=fit['faces']
    def normals(v):
        tri=v[faces];return np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    n0,n1=normals(base),normals(shifted);valid=(np.linalg.norm(n0,axis=1)>1e-12)&(np.linalg.norm(n1,axis=1)>1e-12)
    flips=int(((n0*n1).sum(1)<0)[valid].sum())
    summary=dict(updated=now(),state='awaiting_actual_all_frame_visual_review',frames=59,reference_slot=run['reference_slot'],
        total_fit_seconds=run['elapsed_seconds'],total_updates=run['total_updates'],
        stages=stages,shared_shape_change_L2=float(np.linalg.norm(fit['betas']-before['betas'])),
        shared_offset_change_L2_m=float(np.linalg.norm(fit['static_offset']-before['static_offset'])),
        canonical_triangle_normal_reversals=flips,valid_triangles=int(valid.sum()),
        normal_reversal_is_not_full_self_intersection_check=True,fit_residuals_are_not_independent_truth=True,
        hand_repair_deferred=True,quality_accepted=False,
        review_layout='body pages chronological ten frames each, row-major; each cell original/mesh. Faces five columns chronological. No image text.')
    a.save(out/'summary.json',summary)
    print(summary)


if __name__=='__main__':main()
