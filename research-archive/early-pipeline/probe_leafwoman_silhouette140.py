"""Compare source-only 022 scanlines to the saved full-K head silhouette."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from official_smplifyx_entry123 import sha
from prepare_leafwoman_ear139 import MAP
from probe_leafwoman_shared_shape140 import ARCHIVE, EAR, OUT, atomic, project, read
from review_full_sequence123 import joined


def intersections(corners,y):
    a = corners
    b = corners[:,[1,2,0]]
    hit = ((a[...,1]<=y)&(b[...,1]>y))|((b[...,1]<=y)&(a[...,1]>y))
    ids,edge = np.nonzero(hit)
    p,q = a[ids,edge],b[ids,edge]
    x = p[:,0]+(y-p[:,1])*(q[:,0]-p[:,0])/(q[:,1]-p[:,1])
    assert len(ids)%2==0 and np.array_equal(ids[::2],ids[1::2])
    return ids[::2],np.sort(x.reshape(-1,2),axis=1)


def review():
    plan_path = OUT/'silhouette-plan.json'
    plan = read(plan_path)
    assert sha(ARCHIVE)==plan['frozen_archive_sha256'] and sha(EAR/'plan.json')==plan['source_plan_sha256']
    assert sha(MAP)==read(EAR/'plan.json')['correspondence_sha256']
    marks = [read(OUT/('silhouette-annotation-'+name+'.json')) for name in 'ab']
    assert all(x['plan_sha256']==sha(plan_path) for x in marks)
    assert not (OUT/'silhouette-result.json').exists()
    head = np.load(MAP,allow_pickle=False)
    rows = []
    with np.load(ARCHIVE,allow_pickle=False) as data:
        faces, K = data['faces'],data['K_integer_centers']
        xyz = data['vertices_camera']
        ids = data['frame_ids'].tolist()
        part = np.isin(faces,head).all(1)
        for slot,(item,side) in enumerate(zip(plan['frames'],plan['edge'],strict=True)):
            frame = item['frame']
            path = Path(item['original_crop'])
            assert sha(path)==item['crop_sha256'] and xyz[ids.index(frame),:,2].min()>.05
            with Image.open(path) as im:
                raw=im.convert('RGB')
            ox,oy = item['crop_xyxy'][:2]
            corners = (project(xyz[ids.index(frame)].astype(float),K)-[ox,oy])[faces]
            overlay=raw.copy();pen=ImageDraw.Draw(overlay)
            report=[]
            for index,y in enumerate(plan['rows_y_native_crop']):
                source=[]
                for m in marks:
                    annotation=m['rows'][slot]
                    assert annotation['frame']==frame
                    x=annotation['x'][index]
                    source.append(dict(x=x,uncertainty_px=annotation['uncertainty_px'][index]))
                report_row=dict(native_image_y=y+oy,source=source,admitted=False)
                report.append(report_row)
                if any(x['x'] is None for x in source):
                    report_row['excluded']='one reading found ambiguous skin/background'
                    continue
                owners, spans = intersections(corners,y)
                clip=(spans[:,1]>0)&(spans[:,0]<raw.width-1)
                spans,owners=spans[clip],owners[clip]
                assert len(spans)
                col=0 if side=='image_left' else 1
                winner=int((np.argmin if col==0 else np.argmax)(spans[:,col]))
                model=float(spans[winner,col])
                if model<=0 or model>=raw.width-1 or not part[owners[winner]]:
                    report_row['excluded']='outer edge clipped or belongs to non-head triangle'
                    continue
                report_row.update(admitted=True, mesh_x_crop=model,
                    mesh_minus_reading_px=[model-r['x'] for r in source])
                for s,c in zip(source,('#ffe341','#ff9e43'),strict=True):
                    u=s['x'];pen.line((u,y-2,u,y+2),fill=c,width=1)
                pen.line((model,y-2,model,y+2),fill='#0cf1ff',width=1)
            dest=OUT/(frame+'-jaw-scanline.png')
            joined([raw,overlay]).save(dest)
            rows.append(dict(frame=frame,edge=side,rows=report,image=str(dest),sha256=sha(dest)))
    atomic(OUT/'silhouette-result.json',dict(plan_sha256=sha(plan_path),rows=rows,
        camera_source='existing conditional K, same native integer crop',evaluates_dynamic_landmark_ids=False,
        ear_subject_3D_truth=False,fit_started=False,quality_accepted=False))
    for row in rows:
        print(row['frame'],[(r['native_image_y'],r.get('mesh_minus_reading_px',r.get('excluded'))) for r in row['rows']])


if __name__=='__main__':
    review()
