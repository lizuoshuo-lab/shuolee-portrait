"""All-node native-pixel review of the frozen 25 FIT/3 HELD face components.
No optimizer, new feature detector, residual selection or data mutation.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,time,hashlib
import numpy as np
from PIL import Image,ImageDraw
from audit_grayT123_camera_support import REC,ROOT,read,save,sha,native

OUT=REC/'grayT-material-identity-review'

def main():
    started=time.monotonic()
    if OUT.exists():raise RuntimeError('Review already exists; do not regenerate.')
    OUT.mkdir()
    follows=read(REC/'grayT-material-followup-plan.json');assert follows['status']=='planned_not_executed'
    fit=read(REC/'grayT-track-attrition/FIT-face-components.json');held=read(REC/'grayT-track-attrition/HELD-face-components.json')
    specs=[('FIT',r['track_id']) for r in fit if r['all_nodes_face']]+[('HELD',r['track_id']) for r in held if r['all_nodes_face']]
    specs=sorted(specs);assert sum(a=='FIT' for a,b in specs)==25 and sum(a=='HELD' for a,b in specs)==3
    sources=[ROOT/'camera/appearance-tracks.json',ROOT/'data/training-inputs.json',REC/'grayT-track-attrition/FIT-face-components.json',REC/'grayT-track-attrition/HELD-face-components.json']
    plan=dict(created=datetime.now(timezone.utc).isoformat(),selection='All25FIT and3HELD frozen whole-face-region components, selected before reading residuals.',
        pixel_policy='128x128 native pixels around each saved subpixel feature; all observations chronological in8columns, no resize/warp. Plain empty cells only after final observation.',
        point_locator='Separate hollow cyan5px circle at actual subpixel feature coordinate; no text or identity judgment in pixels.',
        raw_images_unchanged=True,held_errors_not_read_for_selection=True,all_observations_kept=True,
        budget_seconds=120,no_feature_extraction=True,no_geometry_fit=True,
        sources=[dict(path=str(p),sha256=sha(p)) for p in sources],tracks=[dict(split=a,id=b) for a,b in specs])
    save(OUT/'plan.json',plan)
    tracks={r['id']:r for r in read(sources[0])['records']};rows=read(sources[1])['records'];sheets={};requests={};images=[]
    for k,(split,ident) in enumerate(specs):
        nodes=tracks[ident]['nodes'];assert tracks[ident]['held']==(split=='HELD')
        w=128*min(8,len(nodes));h=128*int(np.ceil(len(nodes)/8));sheets[ident]=[Image.new('RGB',(w,h),(245,245,245)),Image.new('RGB',(w,h),(245,245,245))]
        ss=[]
        for n,(i,j) in enumerate(nodes):
            requests.setdefault(i,[]).append((ident,n,j));ss.append(dict(frame=rows[i]['frame'],source_frame_index=rows[i]['source_frame_index'],node_index=n,cell=[n//8,n%8]))
        base=f'{split}_{k+1:02d}_{ident[:12]}'
        images.append(dict(frame=ident,region=split+'_全部原始节点',file=base+'_native.png',locator=base+'_locator.png',
            columns=[s['frame'] for s in ss],sources=ss,all_original_nodes=len(nodes),native_pixels_no_warp=True,labels_in_pixels=False))
    imap={im['frame']:im for im in images}
    for i,reqs in sorted(requests.items()):
        xy=np.load(ROOT/f'camera/features-{i:03d}.npz')['xy']
        with Image.open(native(rows[i]['image'])) as raw:
            rgb=raw.convert('RGB')
            for ident,n,j in reqs:
                u,v=map(float,xy[j]);cx,cy=int(round(u)),int(round(v));crop=(cx-64,cy-64,cx+64,cy+64);p=rgb.crop(crop);x=(n%8)*128;y=(n//8)*128
                original,marked=sheets[ident];original.paste(p,(x,y));marked.paste(p,(x,y));d=ImageDraw.Draw(marked);mx=x+u-crop[0];my=y+v-crop[1]
                d.ellipse((mx-5,my-5,mx+5,my+5),outline=(0,255,255),width=1)
                imap[ident]['sources'][n].update(xy=[u,v],crop=list(crop),source_sha256=rows[i]['source_sha256'])
        if time.monotonic()-started>120:raise RuntimeError('Bounded image preparation exceeded120seconds.')
    for im in images:
        for image,key in zip(sheets[im['frame']],['file','locator']):image.save(OUT/im[key])
        im['sha256']=sha(OUT/im['file']);im['locator_sha256']=sha(OUT/im['locator'])
    save(OUT/'index.json',dict(subject='grayT',images=images,actual_review_pending=True,read_order='left to right, top to bottom'))
    assert all(sha(Path(s['path']))==s['sha256'] for s in plan['sources'])
    print(json.dumps(dict(tracks=len(images),observations=sum(im['all_original_nodes'] for im in images),seconds=time.monotonic()-started,
        images=[dict(file=im['locator'],split=im['region'],nodes=im['all_original_nodes'],frames=im['columns']) for im in images]),ensure_ascii=False))

if __name__=='__main__':main()
