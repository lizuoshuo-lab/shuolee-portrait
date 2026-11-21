"""Read-only full-roster mask/observation audit; no fitting or target filtering."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

import replay_demo_limb_losses123 as base
from smplify_score_contract123 import relative_confidence

BODY = base.ROOT.parent
OUT = BODY / 'visibility-contract-audit'
PLAN = BODY / 'visibility-contract-plan.json'
DATA = Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3')
UPSTREAM = Path(__file__).parent / 'references/smplifyx-68f8536-data_parser.py'
COCO = np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
DEST = np.r_[COCO, np.arange(19,25), np.arange(25,67), np.arange(67,118)]
SOURCE = np.r_[np.arange(23), np.arange(91,133), np.arange(40,91)]
GROUPS = {'body': [i for i in COCO.tolist() if i not in (9,12)],
          'feet': list(range(19,25)), 'left_hand': list(range(25,46)),
          'right_hand': list(range(46,67)), 'face': list(range(67,118))}


def register():
    assert not PLAN.exists() and not OUT.exists(), 'Keep previous audit.'
    rows = base.read(base.ROOT/'data/training-inputs.json')['records']
    rows = sorted(rows, key=lambda r:(r['source_frame_index'],r['frame']))
    assert len(rows)==59 and len({r['frame'] for r in rows})==59
    bound = [Path(__file__), Path(__file__).with_name('smplify_score_contract123.py'),
             Path(__file__).with_name('official_smplifyx_entry123.py'),
             DATA/'data/observations.npz', base.ROOT/'fit/fitted.npz',
             base.ROOT/'data/training-inputs.json', UPSTREAM,
             base.OFFICIAL/'smplifyx/fitting.py',base.OFFICIAL/'cfg_files/fit_smplx.yaml']
    plan = dict(state='registered_readonly', created=datetime.now(timezone.utc).isoformat(),
        hypothesis='The existing relative RTMW confidence is not a visibility measurement; test whether existing observations outside the saved silhouette still contribute to the original objective, without removing any points.',
        source_sha256=base.sha(__file__), bound_hashes={str(p):base.sha(p) for p in bound},
        roster=[dict(frame=r['frame'],source_frame_index=r['source_frame_index'],
                     image_sha256=r['source_sha256'],mask_sha256=r['mask_sha256']) for r in rows],
        observations='All59 frames, all133 raw observations, all114 active original118 targets. No best-frame selection.',
        method='Nearest integer pixel floor(x+0.5) on existing binary masks; full per-point record and original final-stage robust-data-loss contribution at frozen saved projections.',
        image_rule='Every frame: raw / unchanged native2D observations / saved3D projections. Native crop only, no text drawn. No fit result selected.',
        budget=dict(cpu_wall_seconds=180,frames=59,new_detections=0,optimizer_updates=0,gpu_forwards=0),
        limits=['Mask is a segmentation estimate, not a physical visibility label.',
                'A point inside the silhouette may be occluded by clothing or an object.',
                'Outside-mask status is descriptive only; no automatic rejection or confidence update.',
                'Fitting residuals are not independent ground truth.'],
        stop_conditions=['Input identity/hash/size mismatch','Nonbinary mask','Nonfinite observation','CPU budget exceeded'],
        training_authorized=False,parameter_changes=False,quality_accepted=False)
    base.save(PLAN,plan)
    print(json.dumps(dict(state=plan['state'],frames=59,source_sha256=plan['source_sha256'])))


def summary(values):
    a=np.asarray(values,dtype=float)
    return dict(n=int(a.size),min=float(a.min()),median=float(np.median(a)),max=float(a.max())) if a.size else dict(n=0,min=None,median=None,max=None)


def run():
    import cv2
    start=time.monotonic()
    plan=base.read(PLAN)
    assert plan['state']=='registered_readonly' and not OUT.exists()
    for path,digest in plan['bound_hashes'].items(): assert base.sha(path)==digest,path
    OUT.mkdir()
    base.save(OUT/'progress.json',dict(state='running_readonly',pid=os.getpid(),frames_done=0))
    try:
        rows=base.read(base.ROOT/'data/training-inputs.json')['records']
        rows=sorted(rows,key=lambda r:(r['source_frame_index'],r['frame']))
        assert [r['frame'] for r in rows]==[r['frame'] for r in plan['roster']]
        with np.load(DATA/'data/observations.npz',allow_pickle=False) as z: obs=dict(z)
        with np.load(base.ROOT/'fit/fitted.npz',allow_pickle=False) as z:
            fit={k:z[k].copy() for k in ('frame_ids','observed_native_keypoints','joints_projected_native')}
        observed_ids=obs['frame_ids'].astype(str).tolist(); fitted_ids=fit['frame_ids'].astype(str).tolist()
        assert set(observed_ids)==set(fitted_ids)=={r['frame'] for r in rows}
        conf,q90=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
        names=obs['joint_names'].astype(str).tolist()
        record=Path('/mnt/d/00/记录/demo'); record.mkdir(parents=True,exist_ok=True)
        image_dir=OUT/'images'; image_dir.mkdir()
        perframe=[]; allpoints=[]; rawpoints=[]; images=[]
        joint_weight=np.ones(118);joint_weight[[1,9,12]]=0; joint_weight[25:]=2
        source_to_destination={int(s):int(d) for s,d in zip(SOURCE,DEST)}
        group_of={i:name for name,indices in GROUPS.items() for i in indices}
        for ordinal,row in enumerate(rows):
            assert time.monotonic()-start<plan['budget']['cpu_wall_seconds']
            frame=row['frame'];oi=observed_ids.index(frame);fi=fitted_ids.index(frame)
            assert base.sha(row['image'])==row['source_sha256']
            assert base.sha(row['mask'])==row['mask_sha256']
            image=Image.open(row['image']).convert('RGB')
            mask=np.asarray(Image.open(row['mask']).convert('L'))
            assert image.size==tuple(row['wh']) and mask.shape==(image.height,image.width)
            assert set(np.unique(mask).tolist())<=set([0,255]),frame
            targets=fit['observed_native_keypoints'][fi]
            xy=obs['keypoints'][oi]
            assert np.array_equal(xy[SOURCE].astype(np.float32),targets[DEST,:2])
            assert np.array_equal(conf[oi,SOURCE],targets[DEST,2])
            pixel=np.floor(xy+.5).astype(np.int64)
            within=(pixel[:,0]>=0)&(pixel[:,0]<image.width)&(pixel[:,1]>=0)&(pixel[:,1]<image.height)
            inside=np.zeros(133,bool);inside[within]=mask[pixel[within,1],pixel[within,0]]!=0
            # Distance is descriptive, never used to select or alter observations.
            outside_distance=cv2.distanceTransform((mask==0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            dist=np.full(133,np.nan);dist[within]=outside_distance[pixel[within,1],pixel[within,0]]
            residual=fit['joints_projected_native'][fi].astype(float)-targets[:,:2]
            effective=(joint_weight*targets[:,2])**2*(1000/image.height)**2
            terms=effective*(10000*residual**2/(residual**2+10000)).sum(axis=1)
            frame_points=[]
            for source in range(133):
                destination=source_to_destination.get(source)
                raw=dict(frame=frame,source_index=source,name=names[source],xy=xy[source].astype(float).tolist(),
                         raw_score=float(obs['scores'][oi,source]),relative_confidence=float(conf[oi,source]),
                         flip_disagreement_px=float(obs['flip_disagreement'][oi,source]),inside_image=bool(within[source]),
                         inside_saved_mask=bool(inside[source]),outside_distance_px=float(dist[source]) if within[source] else None,
                         destination=destination,active=destination in group_of)
                rawpoints.append(raw)
                if destination in group_of:
                    point=dict(raw,group=group_of[destination],fit_residual_px=float(np.linalg.norm(residual[destination])),
                               original_final_data_loss=float(terms[destination]),effective_weight=float(effective[destination]))
                    frame_points.append(point);allpoints.append(point)
            groups={}
            for name,indices in GROUPS.items():
                points=[p for p in frame_points if p['group']==name]
                outside=[p for p in points if not p['inside_saved_mask']]
                groups[name]=dict(count=len(points),outside_count=len(outside),
                                 relative_confidence=summary([p['relative_confidence'] for p in points]),
                                 outside_confidence=summary([p['relative_confidence'] for p in outside]),
                                 data_loss=float(sum(p['original_final_data_loss'] for p in points)),
                                 outside_data_loss=float(sum(p['original_final_data_loss'] for p in outside)))
            perframe.append(dict(frame=frame,source_frame_index=row['source_frame_index'],groups=groups,
                                 elbow_residual_px=float(np.linalg.norm(residual[6])),
                                 saved_mask_sha256=row['mask_sha256']))
            x1,y1,x2,y2=row['initial_box_xyxy']
            crop=(max(0,int(x1)-24),max(0,int(y1)-24),min(image.width,int(x2)+25),min(image.height,int(y2)+25))
            parts=[image.crop(crop) for _ in range(3)]
            for pi,coordinates,color in [(1,targets[:,:2],(0,190,255)),(2,fit['joints_projected_native'][fi],(40,220,70))]:
                draw=ImageDraw.Draw(parts[pi]); pt=lambda p:(float(p[0]-crop[0]),float(p[1]-crop[1]))
                for chain in [(2,3,4),(5,6,7),(9,10,11),(12,13,14)]:draw.line([pt(coordinates[i]) for i in chain],fill=color,width=2)
                for d in group_of:
                    x,y=pt(coordinates[d]);draw.ellipse((x-1,y-1,x+1,y+1),fill=color)
            canvas=Image.new('RGB',(parts[0].width*3,parts[0].height))
            for i,part in enumerate(parts):canvas.paste(part,(i*part.width,0))
            name=f'特征点_全59帧遮挡审计_{frame}_原图_2D观测_3D投影.png'
            output=image_dir/name;dest=record/name
            assert not dest.exists(),'Preserve existing record image.'
            canvas.save(output);dest.write_bytes(output.read_bytes())
            images.append(dict(frame=frame,path=str(output),record_path=str(dest),sha256=base.sha(output),crop=list(crop),resized=False,text_drawn=False))
            if (ordinal+1)%10==0:base.save(OUT/'progress.json',dict(state='running_readonly',pid=os.getpid(),frames_done=ordinal+1))
        groups={}
        for name in GROUPS:
            points=[p for p in allpoints if p['group']==name];outside=[p for p in points if not p['inside_saved_mask']]
            groups[name]=dict(count=len(points),outside_count=len(outside),frames_with_outside=sum(r['groups'][name]['outside_count']>0 for r in perframe),
                             relative_confidence=summary([p['relative_confidence'] for p in points]),
                             outside_confidence=summary([p['relative_confidence'] for p in outside]),
                             outside_distance_px=summary([p['outside_distance_px'] for p in outside if p['outside_distance_px'] is not None]),
                             data_loss=float(sum(p['original_final_data_loss'] for p in points)),
                             outside_data_loss=float(sum(p['original_final_data_loss'] for p in outside)))
        assert len(rawpoints)==59*133 and len(allpoints)==59*114 and len(images)==59
        src=UPSTREAM.read_text(encoding='utf-8');tree=ast.parse(src)
        excerpts={n.name:dict(start_line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(src,n)) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ('read_keypoints','get_joint_weights')}
        upstream=dict(commit=base.entry.COMMIT,url=f'https://github.com/vchoutas/smplify-x/blob/{base.entry.COMMIT}/smplifyx/data_parser.py',
                      source_sha256=base.sha(UPSTREAM),source_executed=False,excerpts=excerpts,
                      finding='Original parser concatenates body25, both complete21-point hands including roots, and face51. It passes input confidence unchanged; weights ignore configured neck/hips. No mask/visibility test occurs in these functions.')
        for path,digest in plan['bound_hashes'].items():assert base.sha(path)==digest,path
        result=dict(state='complete_readonly_full59_visibility_audit',elapsed_seconds=time.monotonic()-start,
                    frames=59,raw_observations=59*133,active_targets=59*114,groups=groups,per_frame=perframe,
                    images=images,upstream=upstream,bound_hashes=plan['bound_hashes'],
                    all_frames_included=True,frame_filter_used=False,observations_changed=False,weights_changed=False,
                    fitting_started=False,gaussian_training_started=False,quality_accepted=False,limits=plan['limits'])
        base.save(OUT/'all-observations.json',dict(raw=rawpoints,active=allpoints))
        base.save(OUT/'result.json',result)
        base.save(OUT/'progress.json',dict(state='complete',pid=None,frames_done=59,result_sha256=base.sha(OUT/'result.json')))
        plan.update(state='complete_readonly',result_sha256=base.sha(OUT/'result.json'));base.save(PLAN,plan)
        print(json.dumps({k:result[k] for k in ('state','elapsed_seconds','frames','raw_observations','active_targets','groups')},indent=2))
    except BaseException as error:
        base.save(OUT/'progress.json',dict(state='failed',pid=None,error=repr(error),training_started=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    register() if args.register else run()
