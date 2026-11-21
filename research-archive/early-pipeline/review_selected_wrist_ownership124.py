"""Inspect native source ownership, distinct from same-joint alias consistency."""
from pathlib import Path
from datetime import datetime,timezone
import time,hashlib
import numpy as np
from PIL import Image,ImageDraw
import audit_gaussianavatars_params123 as a

ROOT=a.REC/'cross-subject-sh0-124'
FIT=ROOT/'grass-selected-shape124'
OUT=ROOT/'selected-wrist-ownership124'


def local(p):
    p=str(p)
    if p.startswith('/opt/'):return Path('//wsl.localhost/Ubuntu-20.04'+p)
    if p.startswith('/mnt/'):return Path(p[5].upper()+':/'+p[7:])
    return Path(p)


def digest(p):return hashlib.sha256(local(p).read_bytes()).hexdigest()


def main():
    assert not OUT.exists(),'Review the saved native evidence; do not repeat.'
    started=time.monotonic();stamp=datetime.now(timezone.utc).isoformat()
    plan=a.read(FIT/'plan.json');rows=plan['rows']
    frozen={str(p):a.sha(p) for p in [Path(__file__),FIT/'plan.json',FIT/'selected-fit.npz',
        ROOT/'grass-temporal-refinement/occluded-arm-audit/result.json',
        ROOT/'selected-observation-audit124/wrist-preselection/result.json']}
    a.save(OUT/'plan.json',dict(updated=stamp,frames=plan['frames'],source_hashes=frozen,
        hypothesis='Two anatomical wrists can collapse to one visible hand while each body/handroot alias pair remains internally consistent. Prior input audit did not visually decide ownership of selected frame85.',
        scope='Read only the same eight original images and saved detector targets. Native source ownership/visibility review; not new fitting or independent metric3D truth.',
        budget=dict(CPU_seconds=120,GPU_fits=0,new_detector_calls=0),
        rule='No deletion, confidence rewriting or manual wrist placement; actual visibility decision pending rendered review. Hand proximity alone cannot reject genuine hand contact.'))
    with np.load(FIT/'selected-fit.npz',allow_pickle=False) as z:
        ids=z['frame_ids'].astype(str).tolist();obs=z['observed_native_keypoints'].copy();proj=z['joints_projected_native'].copy()
    assert ids==plan['frames']
    audit={r['frame']:r for r in a.read(ROOT/'grass-temporal-refinement/occluded-arm-audit/result.json')['records']}
    records=[]
    for i,row in enumerate(rows):
        f=row['frame'];assert digest(row['image'])==row['source_sha256']
        im=Image.open(local(row['image'])).convert('RGB');assert im.size==tuple(row['wh'])
        b=row['initial_box_xyxy'];height=b[3]-b[1]
        crop=[max(0,int(np.floor(b[0]))-15),max(0,int(np.floor(b[1]))-10),
            min(im.width,int(np.ceil(b[2]))+15),min(im.height,int(np.ceil(b[1]+.60*height)))]
        raw=im.crop(crop);panels=[raw.copy(),raw.copy(),raw.copy()]
        for panel,points,is_target in [(panels[1],obs[i,:,:2],True),(panels[2],proj[i],False)]:
            draw=ImageDraw.Draw(panel)
            for chain,color in [([5,6,7],(255,70,40)),([2,3,4],(0,170,255))]:
                xy=[tuple(points[k]-np.asarray(crop[:2])) for k in chain]
                draw.line(xy,fill=color,width=2)
                for x,y in xy:draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
            for indexes,color in [(range(25,46),(255,165,0)),(range(46,67),(0,245,245))]:
                for j in indexes:
                    if is_target and obs[i,j,2]**2==0:continue
                    x,y=points[j]-np.asarray(crop[:2]);draw.ellipse((x-2,y-2,x+2,y+2),outline=color,width=1)
        canvas=Image.new('RGB',(raw.width*3,raw.height))
        for k,p in enumerate(panels):canvas.paste(p,(raw.width*k,0))
        out=Path('D:/00/记录/005')/('特征点_选帧手腕归属124_'+f+'_原图_2D红左橙左手蓝右青右手_3D投影.png')
        assert not out.exists();canvas.save(out)
        gap=float(np.linalg.norm(obs[i,7,:2]-obs[i,4,:2]))
        assert abs(gap-audit[f]['wrist_pair_distance_px'])<1e-4
        records.append(dict(frame=f,path=str(out),sha256=digest(out),native_crop=crop,resized=False,text_on_image=False,
            source_image=row['image'],source_sha256=row['source_sha256'],wrist_distance_px=gap,
            left_body_wrist=obs[i,7].tolist(),right_body_wrist=obs[i,4].tolist(),
            left_handroot=obs[i,25].tolist(),right_handroot=obs[i,46].tolist(),
            hand_centroid_distance_px=audit[f]['hand_centroid_distance_px'],
            left_hand_mean_confidence=audit[f]['left_hand_mean_confidence'],
            right_hand_mean_confidence=audit[f]['right_hand_mean_confidence'],
            source_visibility_reviewed=False))
    assert time.monotonic()-started<120
    assert all(a.sha(Path(p))==h for p,h in frozen.items())
    a.save(OUT/'result.json',dict(updated=datetime.now(timezone.utc).isoformat(),state='native_ownership_images_ready',
        seconds=time.monotonic()-started,records=records,source_hashes_unchanged=True,
        detector_calls=0,fit_calls=0,training_steps=0,original_targets_unchanged=True,
        observations_filtered=False,actual_visual_review=False,quality_accepted=False))
    print('NATIVE_OWNERSHIP_EVIDENCE_READY8_NO_FILTER')


if __name__=='__main__':main()
