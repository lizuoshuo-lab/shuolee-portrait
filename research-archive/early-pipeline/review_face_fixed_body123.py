"""Read-only all-frame native face review; no fitting or GPU calls."""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
import replay_demo_limb_losses123 as b
from review_full_sequence123 import mesh_renderer, joined
from review_smplifyx_orientations123 import load

BODY=b.ROOT.parent
SOURCE=BODY/'body-weight-guarded-sequence'
OUT=BODY/'face-fixed-body-refinement'
REVIEW=OUT/'review'


def project(vertices,K):
    q=vertices@K.T
    return q[...,:2]/q[...,2:]


def main():
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not REVIEW.exists()
    result=b.read(OUT/'result.json'); assert result['frames']==59
    REVIEW.mkdir()
    renderer,renderer_source=mesh_renderer()
    records={r['frame']:r for r in b.read(b.ROOT/'data/training-inputs.json')['records']}
    with np.load(b.ASSETS/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=False) as z:
        weights=z['weights']
    limb=np.isin(weights.argmax(1),[1,2,4,5,7,8,10,11,13,14,16,17,18,19,20,21]+list(range(25,55)))
    images=[]; changes=[]
    for row in result['rows']:
        frame=row['frame']
        paths=[b.ROOT/'frames'/frame/'winner-camera-space.npz',SOURCE/'frames'/frame/'winner-camera-space.npz',OUT/'frames'/frame/'candidate-camera-space.npz']
        baseline,previous,candidate=[load(path) for path in paths]
        for name in ('K_integer_centers','camera_translation','betas','global_orient','body_pose_axis_angle','pose_embedding','left_hand_pose','right_hand_pose'):
            assert np.array_equal(previous[name],candidate[name]),(frame,name)
        raw=Image.open(records[frame]['image']).convert('RGB')
        crop=next(x['crop_integer_xyxy'] for x in b.read(b.ROOT/'review'/(frame+'_saved-review.json'))['files'] if x['region']=='face')
        native=raw.crop(crop)
        panels=[native]+[renderer(native,np.array(crop),data) for data in (baseline,previous,candidate)]
        path=REVIEW/(frame+'_face_raw_independent_bodyweight_faceonly.png')
        joined(panels).save(path)
        destination=Path('/mnt/d/00/记录/demo')/('SMPL-X_固定身体面部精修_'+path.name)
        assert not destination.exists()
        destination.write_bytes(path.read_bytes())
        images.append(dict(frame=frame,path=str(path),record_path=str(destination),sha256=b.sha(path),crop_integer_xyxy=crop,
            panel_wh=list(native.size),columns=['raw','independent_baseline','body_weight_selected','face_only_refined'],
            source_sha256={str(path):b.sha(path) for path in paths},no_resampling=True,no_text_added=True))
        K=candidate['K_integer_centers'].astype(np.float64)
        v0=previous['vertices_camera'].astype(np.float64); v1=candidate['vertices_camera'].astype(np.float64)
        delta=np.linalg.norm(v1-v0,axis=-1)[0];delta_uv=np.linalg.norm(project(v1,K)-project(v0,K),axis=-1)[0]
        changes.append(dict(frame=frame,limb_vertex_max_euclidean_m=float(delta[limb].max()),limb_vertex_max_projection_px=float(delta_uv[limb].max()),
            all_vertex_max_euclidean_m=float(delta.max()),all_vertex_max_projection_px=float(delta_uv.max()),
            body_joint_max_projection_px=float(np.linalg.norm(candidate['joints_projected_native'][:,:25]-previous['joints_projected_native'][:,:25],axis=-1).max())))
    sheets=[]
    for start in range(0,59,10):
        group=images[start:start+10];loaded=[Image.open(x['path']).convert('RGB') for x in group]
        cw=max(im.width for im in loaded);ch=max(im.height for im in loaded)
        canvas=Image.new('RGB',(cw*2,ch*((len(group)+1)//2)),(245,245,245))
        for i,im in enumerate(loaded):canvas.paste(im,((i%2)*cw,(i//2)*ch))
        path=REVIEW/f'faces_{start+1:02d}_{start+len(group):02d}_native.png';canvas.save(path)
        sheets.append(dict(path=str(path),sha256=b.sha(path),frames=[x['frame'] for x in group],layout='row-major two columns; four panels per frame',no_resampling=True))
    receipt=dict(updated=datetime.now(timezone.utc).isoformat(),state='generated_pending_actual_review',images=images,contact_sheets=sheets,
        input_result_sha256=b.sha(OUT/'result.json'),source_hashes={str(Path(__file__)):b.sha(__file__),str(renderer_source):b.sha(renderer_source)},
        saved_geometry_changes=changes,max_limb_projection_change_px=max(x['limb_vertex_max_projection_px'] for x in changes),
        max_limb_euclidean_change_m=max(x['limb_vertex_max_euclidean_m'] for x in changes),
        inherited_body_review=str(SOURCE/'review/visual-review.json'),inherited_body_review_sha256=b.sha(SOURCE/'review/visual-review.json'),
        limitations=['CPU depth-sorted mesh overlays are diagnostic, not exact z-buffer renders.',
            'All59 face crops newly generated. Prior whole-body review is inherited with fixed-parameter and tiny vertex-change verification, not claimed as295 newly viewed images.',
            'Detector residuals are not independent material ground truth; no threshold changed.'],quality_accepted=False)
    b.save(REVIEW/'manifest.json',receipt)
    print('ALL59_FACE_REVIEW_GENERATED',len(sheets),'limb projection max',receipt['max_limb_projection_change_px'])


if __name__=='__main__':main()
