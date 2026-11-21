"""Review preregistered native2D input landmarks, without fitting or filtering."""
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_official_gaussians123 import read,save,sha,now
from review_full_sequence123 import joined,crop_box
from smplify_score_contract123 import relative_confidence


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    root=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
    # COCO17 image-space skeleton. Dots are detector measurements, not truth.
    links=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
           (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for subject,number in [('sofa','006'),('indoor','008')]:
        src=root/(subject+'-input');out=src/'observation-review';out.mkdir(exist_ok=False)
        rows=read(src/'data/inputs.json')['records'];sel=read(src/'selection.json')
        ids=sel['fit_frames'];assert len(ids)==12
        with np.load(src/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
        assert obs['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]
        confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
        save(out/'plan.json',dict(updated=now(),subject=subject,frames=ids,source_sha256=sha(src/'data/observations.npz'),
             observation_type='RTMW133 native detections; relative weights are not anatomical visibility or 3D truth.',
             color='body/feet green, face magenta, hands orange; lower relative weight blends toward grey',
             no_coordinate_changes=True,no_fitting=True,no_error_based_selection=True))
        records=[];panels=[]
        for row in rows:
            if row['frame'] not in ids:continue
            slot=[r['frame'] for r in rows].index(row['frame']);xy=obs['keypoints'][slot];c=confidence[slot]
            assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
            raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
            plotted=raw.copy();draw=ImageDraw.Draw(plotted)
            for a,b in links:
                weight=float(min(c[a],c[b]));colour=tuple(int(100*(1-weight)+v*weight) for v in (0,255,80))
                draw.line([tuple(xy[a]),tuple(xy[b])],fill=colour,width=2)
            for j,(x,y) in enumerate(xy):
                color=(0,255,80) if j<23 else ((255,60,220) if j<91 else (255,150,0))
                color=tuple(int(100*(1-float(c[j]))+v*float(c[j])) for v in color)
                radius=3 if j<23 else 2
                draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
            box=crop_box(mask.getbbox(),raw.size,30)
            face=crop_box(row['face_box_xyxy'],raw.size,25)
            hands=crop_box(np.r_[xy[91:133].min(0)-25,xy[91:133].max(0)+25],raw.size,0)
            images={}
            for region,b in [('body',box),('face',face),('hands',hands)]:
                path=Path('/mnt/d/00/记录')/number/f'特征点_124_{row["frame"]}_{region}_原图_原生二维观测.png'
                assert not path.exists();joined([raw.crop(b),plotted.crop(b)]).save(path);images[region]=str(path)
            body=Image.open(images['body']);body.thumbnail((700,480))
            close=Image.open(images['hands']);close.thumbnail((700,360))
            panel=Image.new('RGB',(1400,480),'black');panel.paste(body,(0,0));panel.paste(close,(700,0));panels.append(panel)
            outside=(xy[:,0]<-.5)|(xy[:,1]<-.5)|(xy[:,0]>=raw.width-.5)|(xy[:,1]>=raw.height-.5)
            records.append(dict(frame=row['frame'],images=images,body_confidence=c[:23].tolist(),
                hand_root_confidence=c[[91,112]].tolist(),
                wrist_duplicate_distance_px=np.linalg.norm(xy[[9,10]]-xy[[91,112]],axis=-1).tolist(),
                outside_body_feet_indices=np.flatnonzero(outside[:23]).tolist(),
                no_3d_landmarks_or_fit_exist=True))
        pages=[]
        for start in range(0,len(panels),4):
            page=Image.new('RGB',(1400,1920),'black')
            for j,p in enumerate(panels[start:start+4]):page.paste(p,(0,j*480))
            p=out/f'page-{start//4+1:02d}.png';page.save(p);pages.append(str(p))
        save(out/'result.json',dict(updated=now(),subject=subject,records=records,pages=pages,
            actual_visual_review=False,quality_accepted=False,gaussian_admitted=False,source_unchanged=sha(src/'data/observations.npz')==read(out/'plan.json')['source_sha256']))
        print('CORE_NATIVE_2D_REVIEW_READY',subject,len(records),flush=True)


if __name__=='__main__':main()
