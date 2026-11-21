"""Frozen four-person, native-pixel hand identity audit. No filtering or fitting."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from run_official_gaussians123 import read, save, sha, now
from smplify_score_contract123 import relative_confidence

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
OUT=ROOT/'cross-hand-identity-audit'
SUBJECTS={'grass':'005','winterhat':'001','sofa':'006','indoor':'008'}

def prepare():
    assert not OUT.exists(), 'Preserve the one registered audit.'
    frozen={str(p):sha(p) for s in SUBJECTS for p in [ROOT/(s+'-input')/'data/inputs.json',ROOT/(s+'-input')/'data/observations.npz',ROOT/(s+'-input')/'selection.json']}
    history=[ROOT/'duplicate-wrist-anchor-audit/result.json',ROOT/'hand-observation-coherence/control-verdict.json',
        ROOT/'unique-wrist-input124/sofa16-control/conclusion.json',ROOT/'grass-temporal-refinement/conclusion.json']
    frozen.update({str(p):sha(p) for p in history})
    frozen[str(Path(__file__))]=sha(__file__)
    save(OUT/'plan.json',dict(updated=now(),state='registered_not_started',
        hypothesis='Left and right21-point hand chains may collapse onto the same visible hand during occlusion; this is distinct from two noisy estimates of the same wrist.',
        metrics=['centroid separation divided by mean hand diameter','same-index21-point RMS separation divided by mean hand diameter',
            'body-to-own-hand-wrist residual divided by mean hand diameter','body-to-opposite-hand-wrist residual','existing relative confidence and real timestamps'],
        selection='For each subject exactly three deterministic anchors: smallest normalized hand-centroid distance; largest absolute asymmetry between left/right body-to-hand-root distances; frozen original reference frame. Include immediate chronological predecessor of each anchor. Deduplicate identical frames. No fit residual used.',
        budget=dict(subjects=4,frames=373,cpu_seconds=120,selected_images_max=24,gpu_calls=0,fitter_calls=0,feature_extraction_calls=0),
        genuine_contact_protection='Close or crossed hands can be valid. Inspect raw pixels and temporal context; never call proximity alone an identity error or suppress a side solely for overlap.',
        limits=['No acceptance threshold, confidence rule or pose update is applied.','Reuse prior duplicate-wrist audit; do not rerun root suppression or parent-confidence controls.',
            'Selected cases explain mechanisms; they are not an unbiased error-rate estimate or independent3Dtruth.'],
        sources=frozen))
    print('CROSS_HAND_IDENTITY_AUDIT_REGISTERED4',flush=True)

def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan=read(OUT/'plan.json');assert not (OUT/'result.json').exists()
    for path,h in plan['sources'].items():assert sha(path)==h,path
    started=time.monotonic();subjects={};total=0
    for subject,record_id in SUBJECTS.items():
        src=ROOT/(subject+'-input');rows=read(src/'data/inputs.json')['records']
        with np.load(src/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
        assert list(obs['frame_ids'].astype(str))==[r['frame'] for r in rows]
        xy=obs['keypoints'];conf,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
        left=xy[:,91:112];right=xy[:,112:133]
        def diameter(p):return np.sqrt(((p[:,:,None]-p[:,None,:])**2).sum(-1)).max(axis=(1,2))
        scale=.5*(diameter(left)+diameter(right));assert np.all(scale>0)
        separation=np.linalg.norm(left.mean(1)-right.mean(1),axis=-1)/scale
        lrms=np.sqrt(((left-right)**2).sum(-1).mean(-1))/scale
        dl=np.linalg.norm(xy[:,9]-left[:,0],axis=-1)
        dr=np.linalg.norm(xy[:,10]-right[:,0],axis=-1)
        asym=np.abs(dl-dr)/scale
        reference=read(src/'selection.json')['reference_frame'];ids=[r['frame'] for r in rows]
        anchors=[dict(index=int(np.argmin(separation)),reason='minimum_normalized_centroid_separation'),
            dict(index=int(np.argmax(asym)),reason='maximum_wrist_conflict_asymmetry'),dict(index=ids.index(reference),reason='frozen_reference')]
        selected=sorted(set(i for a in anchors for i in [max(0,a['index']-1),a['index']]))
        records=[];images=[]
        for i,row in enumerate(rows):
            assert time.monotonic()-started<plan['budget']['cpu_seconds']
            records.append(dict(frame=row['frame'],time_seconds=row['time_seconds'],hand_scale_px=float(scale[i]),
                centroid_distance_px=float(separation[i]*scale[i]),centroid_distance_normalized=float(separation[i]),
                same_index_rms_normalized=float(lrms[i]),body_hand_wrist_conflict_px=[float(dl[i]),float(dr[i])],
                own_conflict_normalized=[float(dl[i]/scale[i]),float(dr[i]/scale[i])],
                opposite_wrist_distance_px=[float(np.linalg.norm(xy[i,9]-right[i,0])),float(np.linalg.norm(xy[i,10]-left[i,0]))],
                body_wrist_confidence=conf[i,[9,10]].tolist(),hand_mean_confidence=[float(conf[i,91:112].mean()),float(conf[i,112:133].mean())]))
            if i not in selected:continue
            assert sha(row['image'])==row['source_sha256']
            raw=Image.open(row['image']).convert('RGB');w,h=raw.size
            pts=np.concatenate([xy[i,5:11],left[i],right[i]])
            valid=(pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)
            if not valid.any():raise RuntimeError('No in-raster hand context; unavailable rather than inventing crop.')
            pts=pts[valid];margin=max(30,int(.06*(obs['boxes'][i,3]-obs['boxes'][i,1])))
            box=[max(0,int(np.floor(pts[:,0].min()))-margin),max(0,int(np.floor(pts[:,1].min()))-margin),
                min(w,int(np.ceil(pts[:,0].max()))+margin),min(h,int(np.ceil(pts[:,1].max()))+margin)]
            crop=raw.crop(box);marked=crop.copy();draw=ImageDraw.Draw(marked)
            for indexes,color,radius in [(range(91,112),(255,150,0),2),(range(112,133),(0,220,255),2),([9],(255,0,0),6),([10],(0,70,255),6)]:
                for k in indexes:
                    x,y=xy[i,k]-np.array(box[:2]);draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=2)
            pair=Image.new('RGB',(2*crop.width,crop.height));pair.paste(crop,(0,0));pair.paste(marked,(crop.width,0))
            path=Path('/mnt/d/00/记录')/record_id/f'分析_跨人物手部身份124_{subject}_{row["frame"]}_原图_红左腕橙左手蓝右腕青右手.png'
            assert not path.exists();path.parent.mkdir(exist_ok=True,parents=True);pair.save(path)
            images.append(dict(frame=row['frame'],path=str(path),sha256=sha(path),native_crop=box,resized=False,text_on_image=False))
        subjects[subject]=dict(frames=len(rows),reference=reference,anchors=[dict(**v,frame=ids[v['index']]) for v in anchors],
            selected_frames=[ids[i] for i in selected],records=records,images=images)
        total+=len(rows);print(subject,len(rows),[x['frame'] for x in subjects[subject]['anchors']],flush=True)
    assert total==373 and sum(len(x['images']) for x in subjects.values())<=24
    for path,h in plan['sources'].items():assert sha(path)==h,path
    save(OUT/'result.json',dict(updated=now(),state='read_only_complete',subjects=subjects,frames=total,
        elapsed_seconds=time.monotonic()-started,observation_or_pose_changed=False,source_hashes_unchanged=True,
        quality_accepted=False,actual_visual_review=False,images_are_posthoc_explanatory_not_ground_truth=True))
    print('CROSS_HAND_IDENTITY_AUDIT_COMPLETE',total,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);args=p.parse_args()
    {'prepare':prepare,'run':run}[args.mode]()
