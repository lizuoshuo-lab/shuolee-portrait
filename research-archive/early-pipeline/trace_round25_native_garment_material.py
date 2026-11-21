"""Check whether fixed-view contributor rows are visible on labeled native material."""

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import json
import numpy as np
from PIL import Image, ImageDraw
import torch

from trace_round25_garment_pixels import OUT, PROJECT, R24_REVIEW, loaded_drive, np_tensor
from trace_round22_face_pixels import checked, code, need, pin, pixel, preprocess, read, sha


# Interior reference polygons traced on original signed photographs. Each
# four-tuple is an inclusive rectangle; a longer point list is a polygon.
REGIONS = {
    '016': {
        'frame_000027': dict(upper_garment=[[(1260,755),(1418,744),(1427,819),(1431,1015),(1280,1026),(1250,963)]],
            lower_garment=[(1250,1086,1420,1330)], footwear=[(1300,1464,1330,1488),(1360,1465,1395,1492)]),
        'frame_000002': dict(upper_garment=[[(1452,780),(1536,744),(1543,849),(1534,1001),(1415,1028),(1418,902)]],
            lower_garment=[(1400,1170,1513,1348),(1430,1350,1498,1400)], footwear=[(1371,1415,1414,1438),(1453,1436,1490,1450)]),
        'frame_000051': dict(upper_garment=[[(1290,734),(1390,763),(1430,858),(1440,1005),(1310,1054),(1269,924)]],
            lower_garment=[(1317,1215,1434,1370),(1405,1160,1453,1210)], footwear=[(1400,1448,1440,1470)]),
    },
    '006': {
        'frame_000025': dict(upper_garment=[(1280,684,1450,860),(1187,745,1240,845),(1515,715,1542,848)],
            lower_garment=[(1132,1060,1205,1255),(1464,1090,1510,1300)], footwear=[(1115,1447,1181,1479),(1463,1469,1548,1502)],
            background_seat=[(1245,1125,1392,1205)]),
        'frame_000041': dict(upper_garment=[[(1215,637),(1337,620),(1410,727),(1401,815),(1227,848),(1176,755)]],
            lower_garment=[(1379,1060,1478,1310),(1635,1050,1687,1220)], footwear=[(1380,1439,1480,1492),(1668,1340,1745,1367)],
            background_seat=[(1178,1090,1260,1168)]),
        'frame_000049': dict(upper_garment=[[(1191,636),(1307,628),(1390,714),(1380,842),(1195,927),(1153,772)]],
            lower_garment=[(1210,1000,1450,1080),(1540,1130,1635,1290)], footwear=[(1478,1464,1560,1504),(1685,1321,1745,1344)],
            background_seat=[(1138,1123,1295,1205)]),
    },
}


def annotate(wh, polygons):
    canvas=Image.new('1',tuple(wh))
    draw=ImageDraw.Draw(canvas)
    for path in polygons:
        if isinstance(path,tuple):
            draw.rectangle(path,fill=1)
        else:
            draw.polygon(path,fill=1)
    return np.asarray(canvas,dtype=bool)


def native_scene(scene, frame, opacity, rgb):
    scene.select_mesh_by_timestep(frame)
    return (np_tensor(scene.get_xyz),np_tensor(scene.get_scaling),np_tensor(scene.get_rotation),opacity,rgb)


def evidence(context, scene, info, candidates):
    result,plan,arm=map(context.__getitem__,('review','plan','arm'))
    native=result['native'][arm]
    drives={row['frame']:row for row in plan['drives'][arm]['records']}
    original={row['frame']:row for row in plan['frozen']['rows']}
    config=read(checked(plan['training_config']))
    names=info['frame_ids']
    source_bins=dict(zip(names[:24],config['training_source_bins'],strict=True))
    opacity=np_tensor(scene.get_opacity)[:,0]
    rgb=np.maximum(np.float32(.28209479177387814)*np_tensor(scene.get_features)[:,0,:]+np.float32(.5),0.)
    records, by_row={},defaultdict(list)
    for frame,regions in REGIONS[context['subject']].items():
        slot=names.index(frame)
        row,qa,drive=original[frame],native[frame],drives[frame]
        camera=qa['camera']
        need(all(camera[k]==info['cameras'][slot][k] for k in ('frame','w2c','K','wh','pose_index')),
             'The photographed material probe has a different frozen camera.')
        need(qa['source_sha256']==row['source_sha256'] and qa['mask_sha256']==row['mask_sha256'],
             'The signed native color or mask changed.')
        with Image.open(pin(row['image'],row['source_sha256'])) as im:
            observed=np.asarray(im.convert('RGB'))
        with Image.open(pin(row['mask'],row['mask_sha256'])) as im:
            matte=np.asarray(im.convert('L'))
        x0,y0,x1,y1=drive['box_xyxy']
        bands=[]
        for name in ('black','white','alpha'):
            with Image.open(checked(qa['delivered_media'][name])) as im:
                bands.append(np.asarray(im.convert('RGB'),dtype=np.int16))
        renders=np.stack(bands)
        need(renders.shape==(3,y1-y0,x1-x0,3), 'The signed native render crop changed dimensions.')
        projected=preprocess(native_scene(scene,slot,opacity,rgb),camera)
        centers=np.rint(projected['xy'][candidates]).astype(np.int64)
        roi_receipts,probes=[],{}
        for name,polygons in regions.items():
            annotated=annotate(camera['wh'],polygons)
            mask=annotated & ((matte<=2) if name=='background_seat' else (matte>=253))
            admitted = int(mask.sum())
            need(admitted>=25, 'The native annotated material area is too small.')
            roi_receipts.append(dict(material=name,polygons=polygons,annotated_pixels=int(annotated.sum()),
                inside_matte_policy_pixels=admitted))
            for i,(x,y) in enumerate(centers):
                if x0<=x<x1 and y0<=y<y1 and mask[y,x]:
                    probes.setdefault((int(x),int(y)),[]).append((int(candidates[i]),name))
        errors=[]
        for (x,y),row_set in probes.items():
            item=pixel(projected,(x,y))
            colors=np.stack((code(item['rgb']),code(item['rgb']+1-item['opacity']),
                             np.full(3,code(item['opacity']),dtype=np.int16)))
            error=int(abs(colors-renders[:,y-y0,x-x0]).max())
            errors.append(error)
            if error>1:
                continue
            weights=dict(zip(item['rows'],item['weights'],strict=True))
            for row_id,material in row_set:
                weight=float(weights.get(row_id,0.))
                if weight>=.01:
                    by_row[row_id].append(dict(frame=frame,source_bin=source_bins[frame],material=material,
                        native_center_xy=[x,y],native_rgb=observed[y,x].tolist(),native_center_alpha=weight))
        records[frame]=dict(source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],
            camera=camera,poses_match=True,material_regions=roi_receipts,comparison_pixels=len(errors),
            max_code_error=max(errors,default=None),exact_code_pixels=errors.count(0),excluded_pixels=sum(e>1 for e in errors))
    return by_row,records


def run(round_number,review_sha):
    base=OUT/f'smpl-delivery-round{round_number}'
    idx=read(pin(base/'review-receipt.json',review_sha))
    sys.path.insert(0,PROJECT+str(round_number)+'/src')
    torch.set_num_threads(2)
    trace=base/'input-audit/side-waist-hem-attribution.json'
    attribution=read(trace)
    need(attribution['review_sha256']==review_sha,'The face/hem source review changed.')
    output={}
    for subject,details in attribution['subjects'].items():
        review=read(checked(idx['subjects'][subject]['review']))
        plan=read(checked(review['plan']))
        arm=f'r{round_number}'
        manifest=checked(plan['graphs'][arm]['manifest'])
        need(sha(manifest)==details['input_proof']['drive_sha256'],'A different drive was used for source material.')
        scene,info=loaded_drive(manifest.parent)
        candidates=np.array(sorted({r['row'] for v in details['views'].values() for p in v['pairs']
                                    if p['parity']['passed'] for r in p['contributors']}),dtype=np.int64)
        material,native=evidence(dict(subject=subject,review=review,plan=plan,arm=arm),scene,info,candidates)
        views={}
        for key,v in details['views'].items():
            rows=[]
            for p in v['pairs']:
                if not p['parity']['passed']:
                    continue
                groups=defaultdict(lambda:np.zeros(2))
                contributors=[]
                for r in p['contributors']:
                    ev=material.get(r['row'],[])
                    names={i['material'] for i in ev}
                    bins={i['source_bin'] for i in ev}
                    name=next(iter(names)) if len(names)==1 else 'mixed' if names else 'unclassified'
                    groups[f'{r["part"]}/{name}']+=(r['alpha_weight'],r['direct_luma'])
                    if ev or r['alpha_weight']>=.02:
                        contributors.append(dict(stable_id=r['stable_id'],layer=r['part'],weight=r['alpha_weight'],
                            luma=r['direct_luma'],source_material=name,source_bins=len(bins),native_evidence=ev))
                rows.append(dict(region=p['region'],artifact_xy_px=p['artifact_xy_px'],
                    native_groups={k:dict(zip(('artifact_alpha','direct_luma'),x.tolist())) for k,x in groups.items()},
                    contributors=contributors))
            views[key]=rows
        output[subject]=dict(native=native,views=views)
    target=base/'input-audit/side-waist-hem-native-material.json'
    with target.open('x',encoding='utf-8') as f:
        json.dump(dict(schema='shuolee-garment-native-material-support/1',round=round_number,review_sha256=review_sha,
                       side_attribution_sha256=sha(trace),source_sha256=sha(Path(__file__)),subjects=output,
                       scope='Positive support requires the center in a source polygon and matte, >=.01 actual raster weight and native PNG validation. Missing observations remain unclassified. These observations do not independently establish material identity in a novel view.'),
                  f,indent=2,allow_nan=False)
        f.write('\n')
    print(dict(output=str(target),sha256=sha(target)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round',type=int,choices=(24,25))
    parser.add_argument('--review-sha')
    args=parser.parse_args()
    if args.round==25 and args.review_sha is None:
        parser.error('R25 needs an exact signed review SHA')
    run(args.round,args.review_sha or R24_REVIEW)
