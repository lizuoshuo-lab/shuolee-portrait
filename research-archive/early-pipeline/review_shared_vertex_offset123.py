"""Same-pixel original/baseline/shared-offset review assets; no training."""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from plyfile import PlyData
from review_full_sequence123 import read, sha, relocate

T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O = T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle'
ROOT = O/'shared-vertex-offset-1000'
SNAP = ROOT/'gaussian-run/step-001000'
OLD = O/'native-resolution-1000/gaussian-run/step-001000'
D = O/'limb-and-reference-review/selected-reference'
OUT = ROOT/'comparison'

def save(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def stats(x): return dict(median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(np.max(x)))
def join(images):
    out=Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)),(245,245,245))
    x=0
    for im in images: out.paste(im,(x,0)); x+=im.width
    return out
def clip_box(xy,pad,w,h):
    lo=np.floor(np.min(xy,axis=0)-pad).astype(int);hi=np.ceil(np.max(xy,axis=0)+pad).astype(int)
    return (int(max(0,lo[0])),int(max(0,lo[1])),int(min(w,hi[0])),int(min(h,hi[1])))

def review():
    assert read(SNAP/'native-ply-parity.json')['passed']
    OUT.mkdir(exist_ok=True)
    with np.load(ROOT/'fit/fitted.npz',allow_pickle=False) as f: fit=dict(f)
    rows=read(ROOT/'data/training-inputs.json')['records']
    comparisons=[]
    for i in [0,18,37,50,55,74,93]:
        row=rows[i];fid=row['frame']
        new=SNAP/f'{fid}_render.png'
        old=(D/f'{fid}-pose51-render.png') if i==50 else OLD/f'{fid}_render.png'
        raw=Image.open(relocate(row['image'])).convert('RGB')
        a,b=Image.open(old).convert('RGB'),Image.open(new).convert('RGB')
        mask=np.asarray(Image.open(relocate(row['mask'])).convert('L'))>252
        assert raw.size==a.size==b.size==(2704,1520)
        y,x=np.nonzero(mask)
        boxes={'body':clip_box(np.c_[x,y],30,*raw.size),
               'face':clip_box(fit['observed_native_keypoints'][i,67:118,:2],40,*raw.size),
               'feet':clip_box(fit['observed_native_keypoints'][i,19:25,:2],45,*raw.size)}
        metrics={}
        r,aa,bb=[np.asarray(im,dtype=float)/255 for im in (raw,a,b)]
        for label,box in boxes.items():
            path=OUT/f'{fid}_{label}_original_baseline_shared.png'
            join([im.crop(box) for im in (raw,a,b)]).save(path)
            x0,y0,x1,y1=box;selected=mask[y0:y1,x0:x1]
            error=lambda rendered:float(abs(rendered[y0:y1,x0:x1]-r[y0:y1,x0:x1])[selected].mean())
            metrics[label]=dict(crop=list(box),baseline_l1=error(aa),shared_offset_l1=error(bb),
                path=str(path),sha256=sha(path))
        comparisons.append(dict(frame_id=fid,source_frame_index=row['source_frame_index'],regions=metrics))
    with np.load(SNAP/'shared-static-offset.npz',allow_pickle=False) as f: offset=f['static_offset']
    center_metrics=[]
    for name,path in [('baseline',D/'world.ply'),('shared_offset',SNAP/'world.ply')]:
        v=PlyData.read(path)['vertex'].data
        xyz=np.column_stack([v[k] for k in ('x','y','z')])
        distance=cKDTree(fit['vertices_world'][50]).query(xyz,k=1,workers=4)[0]
        scale=np.exp(np.column_stack([v[f'scale_{j}'] for j in range(3)]))
        center_metrics.append(dict(name=name,points=len(xyz),distance_to_original_bare_vertices_m=stats(distance),
                                   Gaussian_std_max_m=float(scale.max())))
    result=dict(state='ready_for_actual_visual_review',steps=1000,source_sha256=sha(SNAP/'world.ply'),
        comparisons=comparisons,offset_norm_m=stats(np.linalg.norm(offset,axis=1)),
        center_metrics=center_metrics,
        metric_scope='L1 from8-bit saved same-pose native renders inside masks; not independent geometric accuracy. Centre distance is to nearest bare vertex, not clothing ground truth.',
        end_to_end_benchmark=False,quality_accepted=False)
    save(OUT/'comparison.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},ensure_ascii=False))

def novel():
    paths=sorted((ROOT/'export/review').glob('novel_*.png'))
    assert len(paths)==36
    thumbnails=[]
    for p in paths:
        im=Image.open(p).convert('RGB'); im.thumbnail((160,240),Image.Resampling.LANCZOS);thumbnails.append(im)
    contact=Image.new('RGB',(960,1440),(242,242,242))
    for i,im in enumerate(thumbnails):contact.paste(im,((i%6)*160,(i//6)*240))
    contact.save(OUT/'novel_ordered_000_035.png')
    record=Path('D:/00/记录/005')
    import shutil
    for p in OUT.glob('*_original_baseline_shared.png'):
        target=record/('3DGS' if '_body_' in p.name else '分析')
        target.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target/('20261030_原版共享顶点偏移1000步_左原图_中旧版_右共享_'+p.name))
    shutil.copy2(OUT/'novel_ordered_000_035.png',record/'分析/20261030_原版共享顶点偏移1000步_连续未知视角000至035_无质量通过.png')
    print('RECORD_IMAGES_WRITTEN_NO_TEXT_OVERLAYS')

def stop_after_review():
    from datetime import datetime, timezone
    request=read(ROOT/'reviews/gaussian-step-001000-request.json')
    assert request['step']==1000 and not (ROOT/'reviews/gaussian-step-001000-decision.json').exists()
    review=dict(updated=datetime.now(timezone.utc).isoformat(),actual_visual_review=True,
        reviewed_same_pose_frames=[1,19,38,51,56,75,94],
        reviewed_native_reference_crops=['body','face','feet'],
        findings=['Original and shared-offset1000step images remain visually near-identical.',
                  'Clothing/hair overspread, fuzzy extremities and profile spikes persist.',
                  'Learned shared vertex offset median0.0693mm/P90 0.1922mm/max0.9092mm; original lr1e-6 was retained.',
                  'Zero-offset parity and native PLY checks validate interface, not reconstructed quality.'],
        quality_accepted=False,snapshot_sha256=request['snapshot_sha256'])
    save(OUT/'actual-step1000-review.json',review)
    save(ROOT/'reviews/gaussian-step-001000-decision.json',dict(step=1000,
        snapshot_sha256=request['snapshot_sha256'],decision='stop',actual_visual_review=True,
        reason='Completed bounded original-mechanism control; no visible clothing-surface improvement at1000. Export this exact result; no parameter scan.',
        evidence=str(OUT/'actual-step1000-review.json'),reviewed_at=review['updated']))
    print('EXACT_STEP1000_REVIEW_RECORDED_EXPORT_CONTINUES')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--novel',action='store_true');parser.add_argument('--stop',action='store_true');args=parser.parse_args()
    stop_after_review() if args.stop else novel() if args.novel else review()
