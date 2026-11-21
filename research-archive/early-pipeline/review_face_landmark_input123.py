"""Read-only native face landmark evidence; model projection is not 3D truth."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import OUT as AUDIT,ROOT,SNAP

OUT=AUDIT/'face-landmark-boundary'
OBS=Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3/data/observations.npz')
MODEL=io.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
CACHE=AUDIT/'original-import-teacher-contract/original-import-lbs-cache.npz'


def summary(x):
    return dict(median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(np.max(x)))


def main():
    assert not (OUT/'landmark-input-review.json').exists()
    files=[Path(__file__),OBS,MODEL,CACHE,SNAP/'current-meshes.npz',ROOT/'data/training-inputs.json']
    hashes={str(p):io.sha(p) for p in files}
    with np.load(OBS,allow_pickle=False) as z:obs=dict(z)
    with np.load(MODEL,allow_pickle=True) as z:
        faces=z['f'];lmkfaces=z['lmk_faces_idx'];bary=z['lmk_bary_coords']
    with np.load(CACHE,allow_pickle=False) as z:cache=dict(z)
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z:snap=dict(z)
    rows=io.read(ROOT/'data/training-inputs.json')['records']
    slots={str(f):i for i,f in enumerate(obs['frame_ids'])};savedslots={str(f):i for i,f in enumerate(snap['frame_ids'])}
    assert set(slots)==set(savedslots)==set(map(str,cache['frame_ids']))
    records=[];images=[];all_errors=[];old_errors=[]
    for i,row in enumerate(rows):
        assert str(cache['frame_ids'][i])==row['frame']
        assert io.sha(row['image'])==row['source_sha256']
        detector=obs['keypoints'][slots[row['frame']],40:91]
        scores=obs['scores'][slots[row['frame']],40:91]
        E=np.array(row['w2c']);K=np.array(row['K_integer_centers'])
        base=cache['base_vertices'][i]@E[:3,:3].T+E[:3,3]
        current=snap['vertices_camera'][savedslots[row['frame']]]
        predictions=[];errors=[]
        for vertices in [base,current]:
            points=np.einsum('lvc,lv->lc',vertices[faces[lmkfaces]],bary)
            assert (points[:,2]>0).all()
            q=points@K.T;pixel=q[:,:2]/q[:,2:]
            assert np.isfinite(pixel).all()
            error=np.linalg.norm(pixel-detector,axis=-1)
            predictions.append(pixel);errors.append(error)
        valid=(scores>0)&np.isfinite(scores)&np.isfinite(detector).all(-1)
        # No residual-based rejection, confidence threshold scan, or frame removal.
        e0,e1=errors[0][valid],errors[1][valid]
        all_errors.extend(e0);old_errors.extend(e1)
        records.append(dict(frame=row['frame'],finite_positive_detector_scores=int(valid.sum()),
            import_to_detector_px=summary(e0),snapshot120k_to_detector_px=summary(e1),
            confidence_min=float(scores.min()),confidence_max=float(scores.max()),
            independent_3d_truth=False,visibility_independently_reviewed=False))
        if row['frame'] in ['frame_000033','frame_000015','frame_000059']:
            raw=Image.open(row['image']).convert('RGB')
            xy=obs['keypoints'][slots[row['frame']],23:91]
            x0,y0=np.floor(xy.min(0)-15).astype(int);x1,y1=np.ceil(xy.max(0)+15).astype(int)
            box=(max(x0,0),max(y0,0),min(x1,raw.width),min(y1,raw.height))
            panels=[raw.copy(),raw.copy(),raw.copy()]
            for panel,points,colour in [(panels[1],detector,'#00ff40'),(panels[2],predictions[0],'#ff30e0')]:
                draw=ImageDraw.Draw(panel)
                for x,y in points:draw.ellipse((x-1,y-1,x+1,y+1),outline=colour,width=1)
            crops=[p.crop(box) for p in panels]
            sheet=Image.new('RGB',(crops[0].width*3,crops[0].height))
            for j,crop in enumerate(crops):sheet.paste(crop,(j*crop.width,0))
            name=f'特征点_面部约束来源_{row["frame"]}_原图_二维检测_原VHAP模型三维点投影.png'
            target=Path('/mnt/d/00/记录/demo')/name;assert not target.exists();sheet.save(target)
            images.append(dict(frame=row['frame'],path=str(target),sha256=io.sha(target),no_resampling=True,no_image_text=True))
    result=dict(updated=io.now(),state='native_face_landmark_inputs_audited_no_fit',frames=59,per_frame=records,
        import_detector_disagreement_px=summary(all_errors),snapshot120k_detector_disagreement_px=summary(old_errors),
        metric_scope='Euclidean discrepancy to original2D detector outputs, including finite positive-score points whose physical visibility is unverified. Not independent accuracy or a subpixel claim.',
        input_3d='The static51 barycentric points come from the fitted SMPL-X mesh. They are model predictions. No independently triangulated/measured3D points are contained in the active observation archive.',
        index_contract='RTMW global40:91 = IBUG17:68 = SMPL-X static51; do not copy FLAME vertex IDs. Contour17 varies with view and is excluded from fixed material3D constraints.',
        images=images,hashes=hashes,actual_visual_review=False,optimizer_updates=0,new_face_fit=False,quality_accepted=False)
    for p,h in hashes.items():assert io.sha(p)==h
    io.save(OUT/'landmark-input-review.json',result)
    print({k:result[k] for k in ['state','frames','import_detector_disagreement_px','snapshot120k_detector_disagreement_px']},flush=True)


if __name__=='__main__':main()
