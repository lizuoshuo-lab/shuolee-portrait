"""Bounded CPU face-presence check using only bundled OpenCV cascades.

This does not replace facial landmarks or alter fitting inputs. Absence of a
detection means unsupported/unknown, never proof that a face is invisible.
"""
import os,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from run_official_gaussians123 import read,save,sha,now
from review_full_sequence123 import crop_box,joined


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';cv2.setNumThreads(1)
    root=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
    out=root/'face-presence-audit';out.mkdir(exist_ok=False)
    files=[Path(cv2.data.haarcascades)/name for name in ['haarcascade_frontalface_default.xml','haarcascade_profileface.xml']]
    sources=[Path(__file__),*files];schedule=[]
    for subject in ['sofa','indoor']:
        src=root/(subject+'-input');sources.extend([src/'selection.json',src/'data/inputs.json'])
        ids=read(src/'selection.json')['fit_frames']
        schedule.extend([dict(subject=subject,**r) for r in read(src/'data/inputs.json')['records'] if r['frame'] in ids])
    assert len(schedule)==24
    save(out/'plan.json',dict(updated=now(),state='registered',hypothesis='An independent face-presence detector may reject hallucinated facial measurements on a back-facing head.',
        frames=[dict(subject=r['subject'],frame=r['frame']) for r in schedule],
        pipeline='One frontal cascade plus the same profile cascade on original/mirrored native ROI. Fixed scaleFactor1.1,minNeighbors3,minSize30x30; no threshold or crop sweep.',
        roi='Existing pre-fit face box expanded by half its larger dimension on all sides, clipped at integer raster boundaries; no resizing or warping.',
        budget=dict(images=24,cascade_calls_per_image=3,max_cpu_wall_seconds=180,new_downloads=0),
        no_input_replacement=True,no_fitting=True,source_hashes={str(p):sha(p) for p in sources},
        stop='Finish the one fixed audit; review visible and back-head cases. Never automatically use detection presence as full landmark visibility or geometry truth.'))
    detectors=[cv2.CascadeClassifier(str(p)) for p in files];assert all(not x.empty() for x in detectors)
    records=[];pictures=[];started=time.monotonic()
    for row in schedule:
        assert time.monotonic()-started<180
        assert sha(row['image'])==row['source_sha256']
        raw=Image.open(row['image']).convert('RGB');b=np.array(row['face_box_xyxy'])
        box=crop_box(b,raw.size,float(max(b[2]-b[0],b[3]-b[1]))*.5)
        crop=raw.crop(box);grey=cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2GRAY);found=[]
        for kind,detector,im in [('frontal',detectors[0],grey),('profile',detectors[1],grey),('profile_mirrored',detectors[1],grey[:,::-1].copy())]:
            for x,y,w,h in detector.detectMultiScale(im,scaleFactor=1.1,minNeighbors=3,minSize=(30,30)):
                if kind=='profile_mirrored':x=grey.shape[1]-x-w
                found.append(dict(kind=kind,xyxy=[int(x+box[0]),int(y+box[1]),int(x+w+box[0]),int(y+h+box[1])]))
        drawn=crop.copy();draw=ImageDraw.Draw(drawn)
        for detection in found:
            q=np.array(detection['xyxy'])-np.array([box[0],box[1],box[0],box[1]])
            draw.rectangle(q.tolist(),outline=(0,240,90) if detection['kind']=='frontal' else (255,150,0),width=2)
        record=Path('/mnt/d/00/记录')/('006' if row['subject']=='sofa' else '008')
        path=record/f'特征点_124_{row["frame"]}_人脸存在性_原图_独立检测.png'
        assert not path.exists();joined([crop,drawn]).save(path)
        panel=Image.open(path);panel.thumbnail((640,340));pictures.append(panel)
        records.append(dict(subject=row['subject'],frame=row['frame'],crop_integer=box,detections=found,image=str(path),face_detected=bool(found),visibility_of_each_landmark_unknown=True))
    pages=[]
    for start in range(0,len(pictures),6):
        page=Image.new('RGB',(1280,1020),'black')
        for j,pic in enumerate(pictures[start:start+6]):page.paste(pic,((j%2)*640,(j//2)*340))
        path=out/f'page-{start//6+1:02d}.png';page.save(path);pages.append(str(path))
    save(out/'result.json',dict(updated=now(),state='complete_pending_visual_review',records=records,pages=pages,
        seconds=time.monotonic()-started,actual_visual_review=False,input_gate_admitted=False,quality_accepted=False))
    print('FACE_PRESENCE_AUDIT_READY',len(records),'detections',sum(r['face_detected'] for r in records),flush=True)


if __name__=='__main__':main()
