"""Read-only native and synthetic-side SMPL-X review; no fitting or GPU."""
import argparse, os
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read, save, sha, now
from review_full_sequence123 import mesh_renderer, joined, crop_box

def main(root):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import cv2
    cv2.setNumThreads(1)
    out = root/'pose-admission-review'
    assert not (out/'result.json').exists(), 'Review already exists; inspect it.'
    out.mkdir(exist_ok=True)
    p = read(root/'plan.json')
    record = Path('/mnt/d/00/记录')/p['record_id']
    record.mkdir(exist_ok=True)
    rows = read(root/'data/training-inputs.json')['records']
    selected = read(root/'selection.json')['selected_frames']
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as z:
        fit = dict(z)
    assert fit['frame_ids'].astype(str).tolist() == [r['frame'] for r in rows]
    render, source = mesh_renderer()
    report, pages, thumbs = [], [], []
    for i, r in enumerate(rows):
        assert sha(r['image']) == r['source_sha256']
        assert sha(r['mask']) == r['mask_sha256']
        raw = Image.open(r['image']).convert('RGB')
        mask = Image.open(r['mask']).convert('L')
        box = crop_box(mask.getbbox(), raw.size, 30)
        vc = fit['vertices_camera'][i]
        K = np.array(r['K_integer_centers'])
        overlay = render(raw.crop(box), np.array(box), dict(vertices_camera=vc[None], faces=fit['faces'], K_integer_centers=K))
        native = joined([raw.crop(box), overlay])
        path = record/f'smpl-x_跨人物124姿态审阅_{p["subject"]}_{r["frame"]}_原图_原生投影.png'
        native.save(path)
        # Real source pixels are not warped; the three other views have no GT.
        center = (vc.min(0)+vc.max(0))/2
        extent = float(np.max(np.ptp(vc, axis=0)))
        views = []
        for degrees in (0, 90, -90):
            a = np.deg2rad(degrees)
            rot = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
            v = (vc-center)@rot.T
            v[:,2] += 3*extent
            k = np.array([[1150,0,240],[0,1150,240],[0,0,1]])
            views.append(render(Image.new('RGB',(480,480),'black'), np.array([0,0,480,480]), dict(vertices_camera=v[None],faces=fit['faces'],K_integer_centers=k)))
        side = joined(views)
        sidepath = record/f'分析_跨人物124姿态审阅_{p["subject"]}_{r["frame"]}_仅mesh_正面_左转90_右转90.png'
        side.save(sidepath)
        thumb = Image.new('RGB',(1440,760),'black')
        n = native.copy(); n.thumbnail((1440,280)); thumb.paste(n,((1440-n.width)//2,0)); thumb.paste(side,(0,280))
        thumbs.append(thumb)
        j = fit['joints_camera'][i]; obs = fit['observed_native_keypoints'][i]
        err = np.linalg.norm(fit['joints_projected_native'][i]-obs[:,:2],axis=-1)
        report.append(dict(frame=r['frame'],time=r['time_seconds'],training=r['frame'] in selected,
            native_comparison=str(path),synthetic_views=str(sidepath),
            right_arm_error_px=err[[2,3,4]].tolist(),left_arm_error_px=err[[5,6,7]].tolist(),
            right_arm_confidence=obs[[2,3,4],2].tolist(),left_arm_confidence=obs[[5,6,7],2].tolist(),
            right_wrist_minus_shoulder_depth_model_m=float(j[4,2]-j[2,2]),
            left_wrist_minus_shoulder_depth_model_m=float(j[7,2]-j[5,2])))
    for begin in range(0,len(thumbs),4):
        group=thumbs[begin:begin+4]
        sheet=Image.new('RGB',(1440,760*len(group)),'black')
        for k,im in enumerate(group):sheet.paste(im,(0,760*k))
        path=out/f'page-{begin//4+1:02d}.png';sheet.save(path)
        pages.append(dict(path=str(path),frames=[r['frame'] for r in report[begin:begin+4]]))
    save(out/'result.json',dict(updated=now(),fit_sha256=sha(root/'fit/fitted.npz'),renderer_sha256=sha(source),
        read_only=True,all_fitted_frames=len(rows),all_selected_and_review_frames=True,
        native_crops_unwarped=True,synthetic_views_not_ground_truth=True,
        no_error_based_filtering=True,actual_visual_review=False,quality_accepted=False,
        records=report,pages=pages))
    print('POSE_REVIEW_READY',str(out),len(rows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);main(ap.parse_args().root)
